"""Gymnasium environment for toppling-safe pushing (M1).

Formulation follows docs/project_outline.md §3:

- observation  o = (d, psi, xdot, psidot, w, L, H); hidden theta never given
- action       a = (h, F, tau), normalized to [-1, 1]^3 for the RL policy and
               denormalized here to h in [0.1H, 0.9H], F in ACTION_RANGES.force,
               tau in [0.05, 0.50] s
- episode      at most T_max = 10 decisions; success and topple terminate early
"""

from dataclasses import dataclass
from typing import Any, Final

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
from numpy.typing import ArrayLike, NDArray

from src.simulation.environment.reward import RewardModel
from src.simulation.execution.settling import RestConfig, wait_for_rest
from src.simulation.execution.force_push import (
    DenseTrajectory,
    ForcePushExecutor,
    PushExecutor,
)
from src.simulation.execution.robot import (
    BlockToppledError,
    ExecutionReport,
    GantryPushExecutor,
    OutsidePlaneError,
    TravelLimitError,
)
from src.simulation.modelling.action_mapping import normalized_to_physical
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.constants import DT, GRAVITY
from src.simulation.modelling.object import Cuboid, CuboidSpec
from src.simulation.modelling.records import ActionRecord
from src.simulation.modelling.sampling import ObjectSampler


@dataclass(frozen=True)
class ActionRanges:
    """
    Physical action bounds (outline §3.3).

    Height is a fraction of object height; force uses newtons and duration
    uses seconds. The force upper bound is provisional pending pilot tests.
    """

    height_fraction: tuple[float, float] = (0.1, 0.9)
    force: tuple[float, float] = (0.0, 50.0)
    duration: tuple[float, float] = (0.05, 0.50)


ACTION_RANGES: Final = ActionRanges()


class ExecutionFailure(RuntimeError):
    """A physical push failed; step() reports the status as a terminal outcome."""

    def __init__(self, status: str) -> None:
        super().__init__(status)
        self.status = status


class PushEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """Target environment: an upright cuboid pushed toward a target on a plane.

    The push primitive is executed by `executor` (direct force by default;
    robot=True creates a contact gantry on this environment's client). The dense
    trajectory of the last action is kept in `self.last_trajectory` for the
    system-identification stage (M2).

    The default maximum force of 50 N exceeds the approximate sliding threshold at the joint-OOD
    upper bounds (mu * m * g = 1.1 * 3.0 * 9.81 ~= 32.4 N; outline §9.1).
    This leaves headroom for unsafe high-force actions (outline §5.3), but
    does not guarantee safe sliding. PROVISIONAL pending pilot tests.

    With wait_for_rest enabled, passive motion is simulated after each push for
    at most rest_timeout seconds (zero disables the cap). Observations use that
    state; time costs include the wait. last_trajectory and action_record remain
    push-only for system identification. The info dict reports settling_steps,
    settling_time, settled, and settling_timeout. The wait consumes no decision.
    """

    def __init__(
        self,
        target_distance: float = 1.0,
        epsilon: float = 0.05, psi_succ_tol: float = 0.02,
        v_succ_tol: float = 0.05,
        T_max: int = 10,
        reward: RewardModel | None = None,
        executor: PushExecutor | None = None,
        fixed_spec: CuboidSpec | None = None,
        near_topple_frac: float = 0.8,  # psi_max / crit_tilt (filter, S5)
        realtime_scale: float = 0.0,  # >0: pace physics at real time (demo)
        gui: bool = False,
        action_ranges: ActionRanges = ACTION_RANGES,
        object_sampler: ObjectSampler | None = None,
        wait_for_rest: bool = False, rest_timeout: float = 2.0, robot: bool = False,
    ) -> None:
        super().__init__()
        self.action_ranges = action_ranges
        self.target_distance, self.epsilon = target_distance, epsilon
        self.psi_succ_tol = psi_succ_tol
        self.v_succ_tol = v_succ_tol
        self.T_max = T_max
        self.reward = reward or RewardModel()
        self.near_topple_frac = near_topple_frac
        self.rest_config = RestConfig(wait_for_rest, rest_timeout)
        if robot and executor is not None:
            raise ValueError("provide robot or executor, not both")
        self.robot = robot
        self._initialize_world(gui, executor, realtime_scale)
        self._initialize_episode_state(fixed_spec, object_sampler)

    def _initialize_world(
        self,
        gui: bool,
        executor: PushExecutor | None,
        realtime_scale: float,
    ) -> None:
        self.realtime_scale = realtime_scale
        self.client: int | None = p.connect(p.GUI if gui else p.DIRECT)
        p.setGravity(0, 0, GRAVITY, physicsClientId=self.client)
        p.setTimeStep(DT, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Floor with mu = 1.0 so the cuboid's lateralFriction IS the effective
        # object-floor coefficient (see object.py friction convention).
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
        p.changeDynamics(
            self.plane, -1, lateralFriction=1.0, physicsClientId=self.client
        )

        executor_class = GantryPushExecutor if self.robot else ForcePushExecutor
        self.executor = executor or executor_class(
            self.client, realtime_scale=realtime_scale)

    def _initialize_episode_state(
        self, fixed_spec: CuboidSpec | None, object_sampler: ObjectSampler | None
    ) -> None:
        self._rng = np.random.default_rng()
        self._object_sampler = object_sampler or ObjectSampler()
        self._fixed_spec = fixed_spec
        self.cuboid: Cuboid | None = None
        self.last_trajectory: DenseTrajectory | None = None
        self.last_execution: ExecutionReport | None = None
        self._d_prev = self.target_distance
        self.steps = 0
        self.psi_max = 0.0  # near-topple threshold; set per-object in reset()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )  # (d, psi, xdot, psidot, w, L, H)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )  # (h, F, tau)

    def _sample_spec(self) -> CuboidSpec:
        """Delegate object selection to the caller's sampler using the reset RNG."""
        return self._object_sampler.sample(self._rng)

    def reset(
        self, seed: int | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset the object and home the gantry before starting a new episode."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        spec = self._fixed_spec or self._sample_spec()
        if self.cuboid is not None:
            self.cuboid.remove()
            self.cuboid = None
        home_steps = self.executor.home() if self.robot else 0
        self.replace_object(spec=spec)
        num_seconds_to_settle = 0.5
        for _ in range(round(num_seconds_to_settle / DT)):
            p.stepSimulation(physicsClientId=self.client)
        self._d_prev = self.target_distance
        self.steps = 0
        self.last_trajectory = None
        self.last_execution = None
        self.psi_max = self.near_topple_frac * spec.crit_tilt
        info = {"true_theta": (spec.m, spec.mu, spec.h_c)}
        if self.robot:
            info["home_steps"] = home_steps
        return self._get_observation(), info

    def _get_observation(self) -> NDArray[np.float32]:
        """
        Return the current observation vector (d, psi, xdot, psidot, w, L, H).
        """
        x, psi, xdot, psidot = self.cuboid.state()
        spec = self.cuboid.spec
        d = self.target_distance - x
        return np.array(
            [d, psi, xdot, psidot, spec.w, spec.L, spec.H], dtype=np.float32
        )

    def step(
        self, action: ArrayLike
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        physical_action = self._denormalize_action(action)
        try:
            record = self.push(physical_action)
        except ExecutionFailure as failure:
            return self._failed_step(physical_action, failure.status)
        return self._completed_step(physical_action, record)

    def _completed_step(
        self, action: PushAction, record: ActionRecord
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        rest = wait_for_rest(
            self.cuboid,
            self.client,
            self.rest_config,
            realtime_scale=self.realtime_scale,
        )
        reward, terminated, truncated, info = self._compute_step_result(
            action, rest.steps * DT
        )
        info["action_record"] = record
        info["max_tilt"] = max(info["max_tilt"], rest.max_tilt)
        info.update(
            settling_steps=rest.steps,
            settling_time=rest.steps * DT,
            settled=rest.rested,
            settling_timeout=bool(
                self.rest_config.enabled and not (rest.rested or rest.toppled)
            ),
        )
        info.update(self._execution_info("completed"))
        return self._get_observation(), reward, terminated, truncated, info

    def _failed_step(
        self, action: PushAction, status: str
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        duration = self.last_execution.execution_time if self.last_execution else 0.0
        reward, _, _, info = self._compute_step_result(
            action, duration - action.tau, execution_failed=True)
        info.update(self._execution_info(status))
        info["TimeLimit.truncated"] = False
        info["success"] = info["is_success"] = False
        return self._get_observation(), reward, True, False, info

    def _execution_info(self, status: str) -> dict[str, Any]:
        info: dict[str, Any] = {
            "execution_status": status, "execution_failed": status != "completed"}
        if self.last_execution is not None:
            for name in ("contact_made", "h_real", "F_real", "tau_real",
                         "execution_time", "stray_contact_steps"):
                info[name] = getattr(self.last_execution, name)
        return info

    def _denormalize_action(self, action: ArrayLike) -> PushAction:
        """Clip the normalized action and map it to physical push bounds."""
        spec = self.cuboid.spec
        return normalized_to_physical(
            action, (spec.w, spec.L, spec.H), self.action_ranges, clip=True
        )

    def replace_object(self, x: float = 0.0, spec: CuboidSpec | None = None) -> None:
        """
        Spawn an upright, stationary object without settling or resetting an episode.

        Reuse the current spec unless one is supplied. Study protocols can
        replace a fallen object at its current x without resampling physics.

        Args:
            x: The x-coordinate to place the object at.
            spec: The CuboidSpec to use for the object. If None, the current spec will be reused.
        """
        if spec is None:
            if self.cuboid is None:
                raise RuntimeError("Call reset() or supply a spec first.")
            spec = self.cuboid.spec
        if self.cuboid is not None:
            self.cuboid.remove()
        self.cuboid = Cuboid(spec, self.client)
        self.cuboid.spawn(
            position=(x, 0.0, spec.H / 2.0 + 0.005)
        )  # spawn slightly above floor to avoid initial penetration with the plane
        self.last_trajectory = None
        self.psi_max = self.near_topple_frac * spec.crit_tilt

    def push(
        self,
        action: PushAction,
        *,
        warmup_steps: int = 0,
    ) -> ActionRecord:
        """Execute a physical push; record its measured start and applied forces."""
        if self.cuboid is None:
            raise RuntimeError("Call reset() before pushing.")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be nonnegative.")
        if abs(self.cuboid.state()[1]) > self.cuboid.spec.crit_tilt:
            self.last_trajectory, self.last_execution = DenseTrajectory(), None
            raise ExecutionFailure("block_toppled")
        start = self.cuboid.state()
        try:
            self.last_trajectory = self.executor.execute(
                self.cuboid, h=action.h, F=action.F, tau=action.tau,
                warmup_steps=warmup_steps)
        except (TimeoutError, TravelLimitError, OutsidePlaneError, BlockToppledError) as exc:
            self.last_trajectory = DenseTrajectory()
            self.last_execution = self.executor.last_execution if self.robot else None
            raise ExecutionFailure(self._execution_status(exc)) from exc
        if self.robot:
            return self._robot_record(action)
        return ActionRecord(start, action, self.last_trajectory, warmup_steps)

    def _robot_record(self, action: PushAction) -> ActionRecord:
        self.last_execution = self.executor.last_execution
        return ActionRecord(
            self.executor.timed_start, action, self.last_trajectory,
            applied_forces=tuple(self.executor.applied_forces))

    @staticmethod
    def _execution_status(exc: Exception) -> str:
        if isinstance(exc, TravelLimitError):
            return "travel_limit"
        if isinstance(exc, OutsidePlaneError):
            return "outside_plane"
        if isinstance(exc, BlockToppledError):
            return "block_toppled"
        return "contact_timeout"

    def _compute_step_result(
        self, action: PushAction, settling_time: float = 0.0,
        *, execution_failed: bool = False,
    ) -> tuple[float, bool, bool, dict[str, Any]]:
        """Evaluate the completed push and advance episode bookkeeping."""
        x, tilt, velocity, _ = self.cuboid.state()
        distance = self.target_distance - x
        toppled = abs(tilt) > self.cuboid.spec.crit_tilt
        success = not execution_failed and (
            abs(distance) <= self.epsilon
            and abs(tilt) <= self.psi_succ_tol
            and abs(velocity) <= self.v_succ_tol
        )
        reward = self.reward.compute_reward(
            previous_distance=self._d_prev,
            distance=distance,
            duration=action.tau + settling_time,
            success=success,
            toppled=toppled,
            goal_tolerance=self.epsilon,
        )
        self._d_prev = distance
        self.steps += 1
        terminated = bool(success or toppled)
        truncated = bool(self.steps >= self.T_max and not terminated)
        info = self._build_step_info(action, tilt, success, toppled, truncated)
        return reward, terminated, truncated, info

    def _build_step_info(
        self,
        action: PushAction,
        tilt: float,
        success: bool,
        toppled: bool,
        truncated: bool,
    ) -> dict[str, Any]:
        return {
            "true_theta": self.cuboid.spec.theta,
            "executed": (action.h, action.F, action.tau),
            "toppled": toppled,
            "success": success,
            "trajectory": self.last_trajectory,
            "max_tilt": max(
                (abs(value) for value in self.last_trajectory.psi), default=abs(tilt)
            ),
            "TimeLimit.truncated": truncated,
            "is_success": bool(success),
        }

    def close(self) -> None:
        if getattr(self, "client", None) is not None:
            p.disconnect(physicsClientId=self.client)
            self.client = None
