"""The external-force action primitive and the shared executor interface.

`PushExecutor` is the contract between the environment and *how* a commanded
primitive a = (h, F, tau) becomes object motion. `ForcePushExecutor` applies
the requested force directly; `robot.GantryPushExecutor` tracks it through
physical pad contact. Identification can replay the measured horizontal force.
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
import pybullet as p

from src.simulation.modelling.constants import DT
from src.simulation.modelling.object import Cuboid


@dataclass(frozen=True)
class AppliedForce:
    """One physics step's horizontal object force and body-relative height.

    The height is measured from the cuboid's bottom along its vertical axis.
    This planar replay omits vertical/lateral forces and contact-patch couples;
    it does not reconstruct the complete gantry contact wrench.
    """

    force_x: float
    height: float


@dataclass
class DenseTrajectory:
    """Within-action object trajectory, recorded every physics step (outline
    §3.2/§6). System identification replays the same action in the internal
    simulator and compares against this.

    Attributes:
        t: Sample timestamps in seconds, relative to the start of this
            execution's recording, including any optional warmup. Samples
            are recorded after each physics step, but timestamps are
            zero-based: the first recorded sample is labelled t = 0.
        x: Object position along the world-frame push axis, in metres.
            Positive x points toward the target; values are absolute world
            coordinates, not displacement since the start of the push.
        psi: Object pitch (tilt about the y axis), in radians, extracted
            from its orientation as an Euler angle. Zero means upright.
        xdot: World-frame linear velocity along the x axis, in metres per
            second (dx/dt).
        psidot: World-frame angular velocity about the y axis, in radians
            per second. This represents dpsi/dt for the intended planar
            pitching motion; it is not a finite difference of psi samples.

    All lists have the same length, and entries at the same index describe
    the same physics step. Recording stops when the push ends; subsequent
    settling motion is not included by ForcePushExecutor.execute. An imagined
    forecast may explicitly append passive settling samples afterwards.
    """

    t: list[float] = field(default_factory=list)
    x: list[float] = field(default_factory=list)
    psi: list[float] = field(default_factory=list)
    xdot: list[float] = field(default_factory=list)
    psidot: list[float] = field(default_factory=list)
    NUM_FEATURES: int = 5

    def to_numpy(self) -> NDArray[np.float64]:
        """
        Return raw samples as (N, 5) in t, x, psi, xdot, psidot order.

        Values retain their recorded units and coordinates. Empty trajectories
        produce (0, 5); channels must be aligned one-dimensional sequences.
        """
        try:
            samples = np.array(
                [self.t, self.x, self.psi, self.xdot, self.psidot], dtype=np.float64
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "trajectory must have five aligned numeric channels"
            ) from exc
        if samples.ndim != 2:
            raise ValueError("trajectory channels must be one-dimensional and aligned")
        return samples.T

    def append(
        self, t: float, x: float, psi: float, xdot: float, psidot: float
    ) -> None:
        self.t.append(t)
        self.x.append(x)
        self.psi.append(psi)
        self.xdot.append(xdot)
        self.psidot.append(psidot)


class PushExecutor(ABC):
    @abstractmethod
    def execute(
        self, cuboid: Cuboid, h: float, F: float, tau: float, warmup_steps: int = 0
    ) -> DenseTrajectory:
        """Record optional free warm-up steps followed by the commanded push."""


class ForcePushExecutor(PushExecutor):
    """Direct application of the horizontal force (outline §3.3).

    The force acts in the fixed world-frame push direction (+x, toward the
    target) at the rear face, body-relative height h, with the application
    point recomputed from the object's current pose every physics step. It is
    removed after tau seconds; the object is NOT required to settle afterwards.
    """

    def __init__(self, client_id: int, realtime_scale: float = 0.0) -> None:
        """realtime_scale > 0 paces the physics at real time (1.0) or slow
        motion (< 1.0) by sleeping between steps, for GUI viewing. Wall-clock
        only — trajectories are identical. 0.0 runs as fast as possible."""
        self.client = client_id
        p.setTimeStep(DT, physicsClientId=self.client)
        self.realtime_scale = realtime_scale

    @staticmethod
    def _pace(deadline: float, interval: float) -> float:
        """Include computation time and small sleep overruns in the schedule."""
        deadline += interval
        now = time.perf_counter()
        remaining = deadline - now
        if remaining > 0:
            time.sleep(remaining)
        elif -remaining > max(0.1, 4 * interval):
            # A long interruption should not trigger a burst of catch-up steps.
            # Reset the clock without skipping any physics or recorded samples.
            deadline = now
        return deadline

    def execute(
        self,
        cuboid: Cuboid,
        h: float,
        F: float,
        tau: float,
        warmup_steps: int = 0,
    ) -> DenseTrajectory:
        """Record free contact warm-up followed by the requested constant force."""
        forces = (AppliedForce(F, h) for _ in range(max(1, round(tau / DT))))
        return self.execute_forces(cuboid, forces, warmup_steps)

    def execute_forces(
        self, cuboid: Cuboid, forces: Iterable[AppliedForce], warmup_steps: int = 0
    ) -> DenseTrajectory:
        """Replay one measured horizontal force per step, after free warm-up.

        Zero-force samples preserve contact gaps; the requested action's force
        and duration do not override this measured trace.
        """
        trajectory = DenseTrajectory()
        paced = self.realtime_scale > 0
        interval = DT / self.realtime_scale if paced else 0.0
        # Warm-up and pushing share one clock; each action starts a fresh schedule.
        deadline = time.perf_counter() if paced else 0.0
        for step in range(warmup_steps):
            deadline = self.advance_and_record(
                cuboid, trajectory, step, deadline, interval
            )
        for step, applied in enumerate(forces):
            self._apply_force(cuboid, applied.height, applied.force_x)
            deadline = self.advance_and_record(
                cuboid, trajectory, warmup_steps + step, deadline, interval
            )
        return trajectory

    def _apply_force(self, cuboid: Cuboid, height: float, force: float) -> None:
        position, orientation = cuboid.pose()
        contact_point = cuboid.rear_face_point(position, orientation, height)
        p.applyExternalForce(
            cuboid.body,
            -1,
            forceObj=[force, 0.0, 0.0],
            posObj=contact_point,
            flags=p.WORLD_FRAME,
            physicsClientId=self.client,
        )

    def advance_and_record(
        self,
        cuboid: Cuboid,
        trajectory: DenseTrajectory,
        step: int,
        deadline: float,
        interval: float,
    ) -> float:
        p.stepSimulation(physicsClientId=self.client)
        trajectory.append(step * DT, *cuboid.state())
        if self.realtime_scale > 0:
            return self._pace(deadline, interval)
        return deadline
