"""Dyna-style imagination: short policy rollouts in the internal simulator.

Start states come from recorded real target-environment states, never from
the mixed replay buffer. The policy picks an action, the internal simulator
forecasts it under the provider's candidate physics, and the target
environment's shared scoring rules reward it, so imagined transitions match
the target's definitions. Imagined transitions go into the policy's own
replay buffer.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.agents.rl.models.sac import SAC
from src.agents.rl.physics_providers import PhysicsProvider
from src.agents.simulator.internal_sim import InternalSim
from src.simulation.environment.env import ActionRanges
from src.simulation.environment.reward import RewardModel
from src.simulation.modelling.action_mapping import normalized_to_physical
from src.simulation.modelling.object import CuboidSpec, ObjectState
from src.simulation.modelling.records import ActionRecord


@dataclass
def _score_push_outcome(
    end: ObjectState,
    spec: CuboidSpec,
    distance: float,
    previous_distance: float,
    duration: float,
    reward_model: RewardModel,
    epsilon: float,
    psi_succ_tol: float,
    v_succ_tol: float,
) -> tuple[float, bool, bool]:
    """Score a simulated end state with the target environment's rules."""
    _x, tilt, velocity, _ = end
    toppled = abs(tilt) > spec.crit_tilt
    success = (
        abs(distance) <= epsilon
        and abs(tilt) <= psi_succ_tol
        and abs(velocity) <= v_succ_tol
    )
    reward = reward_model.compute_reward(
        previous_distance=previous_distance,
        distance=distance,
        duration=duration,
        success=success,
        toppled=toppled,
        goal_tolerance=epsilon,
    )
    return reward, toppled, success


class ImaginationStats:
    transitions: int = 0
    topples: int = 0
    successes: int = 0


@dataclass
class Imagination:
    """Roll the current policy out in the simulator under the provider's physics.

    Rollout start states come from recorded real target-environment states,
    never from the replay buffer: the buffer mixes imagined transitions, and
    starting rollouts from imagined states compounds the model's error.
    """

    sim: InternalSim
    provider: PhysicsProvider
    reward: RewardModel
    action_ranges: ActionRanges
    target_distance: float
    epsilon: float
    psi_succ_tol: float
    v_succ_tol: float
    horizon: int = 2
    warmup_steps: int = 0
    seed: int = 0
    rng: np.random.Generator = field(init=False)
    real_states: list[np.ndarray] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1")
        self.rng = np.random.default_rng(self.seed)

    def observe(self, record: ActionRecord) -> None:
        """Feed a real push to the provider (estimators refit on it)."""
        self.provider.observe(record)

    def record_state(self, obs: ArrayLike) -> None:
        """Register a real target-environment state as a rollout starting point."""
        self.real_states.append(np.asarray(obs, dtype=np.float32))

    def imagine_step(
        self, obs: NDArray[np.float32], action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool]:
        """Simulate one decision from `obs`; returns (next_obs, reward, toppled, success)."""
        d, psi, xdot, psidot, w, L, H = (float(v) for v in obs)
        m, mu, h_c = self.provider.theta(w, L, H)
        spec = CuboidSpec(w=w, L=L, H=H, m=m, mu=mu, h_c=h_c)
        push = normalized_to_physical(action, (w, L, H), self.action_ranges)
        start: ObjectState = (self.target_distance - d, psi, xdot, psidot)
        traj = self.sim.predict(spec, start, push, self.warmup_steps)
        end: ObjectState = (traj.x[-1], traj.psi[-1], traj.xdot[-1], traj.psidot[-1])
        distance = self.target_distance - end[0]
        reward, toppled, success = _score_push_outcome(
            end, spec, distance, d, push.tau, self.reward,
            self.epsilon, self.psi_succ_tol, self.v_succ_tol,
        )
        next_obs = np.array(
            [distance, end[1], end[2], end[3], w, L, H], dtype=np.float32
        )
        return next_obs, reward, toppled, success

    def generate(self, model: SAC, n_rollouts: int) -> ImaginationStats:
        """Roll the current policy from real recorded states into its own replay."""
        stats = ImaginationStats()
        replay = model.replay
        if not self.real_states or replay is None or n_rollouts < 1:
            return stats
        indices = self.rng.integers(len(self.real_states), size=n_rollouts)
        starts = [self.real_states[int(i)] for i in indices]
        for obs in starts:
            for _ in range(self.horizon):
                action = model.predict(obs, deterministic=False)
                next_obs, reward, toppled, success = self.imagine_step(obs, action)
                replay.add(obs, action, reward, next_obs, toppled)
                stats.transitions += 1
                if toppled:
                    stats.topples += 1
                    break
                if success:
                    stats.successes += 1
                obs = next_obs
        return stats
