"""Headless action replay under candidate object physics."""

import math
from collections.abc import Sequence

import pybullet as p
import pybullet_data

from src.simulation.execution.force_push import (
    AppliedForce,
    DenseTrajectory,
    ForcePushExecutor,
)
from src.simulation.execution.settling import RestConfig, wait_for_rest
from src.simulation.modelling.constants import DT, GRAVITY
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.object import Cuboid, CuboidSpec, ObjectState
from src.simulation.modelling.records import ActionRecord


class InternalSim:
    """
    Headless internal PyBullet client — the agent's imagination engine
    (outline §5.1). Replays recorded actions under candidate physics.
    """

    def __init__(self) -> None:
        self.client = p.connect(p.DIRECT)
        p.setGravity(0, 0, GRAVITY, physicsClientId=self.client)
        p.setTimeStep(DT, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
        p.changeDynamics(
            self.plane, -1, lateralFriction=1.0, physicsClientId=self.client
        )
        self.executor = ForcePushExecutor(self.client)

    def replay(self, spec: CuboidSpec, record: ActionRecord) -> DenseTrajectory:
        """Identify physics using recorded actual force, when available."""
        if record.applied_forces is not None:
            return self._simulate(
                spec,
                record.start,
                record.action,
                record.warmup_steps,
                applied_forces=record.applied_forces,
            )
        return self.predict(spec, record.start, record.action, record.warmup_steps)

    def predict(
        self,
        spec: CuboidSpec,
        start: ObjectState,
        action: PushAction,
        warmup_steps: int = 0,
        rest: RestConfig | None = None,
    ) -> DenseTrajectory:
        """Forecast one physical command from observable state under a hypothesis.

        Optional rest is simulated after the push, never before it. Reconstruct the
        floor-contact pose from the planar state; this idealized reconstruction
        can differ from the target's contact state. The candidate body is removed
        even when execution raises, leaving the client ready for another call.
        """
        return self._simulate(spec, start, action, warmup_steps, rest)

    def _simulate(
        self,
        spec: CuboidSpec,
        start: ObjectState,
        action: PushAction,
        warmup_steps: int,
        rest: RestConfig | None = None,
        applied_forces: Sequence[AppliedForce] | None = None,
    ) -> DenseTrajectory:
        """Reconstruct the measured planar state, then execute a command or trace."""
        if self.client is None:
            raise RuntimeError("internal simulator is closed")
        cuboid = Cuboid(spec, self.client)
        try:
            self._spawn_from_state(cuboid, start)
            return self._execute_prediction(
                cuboid, action, warmup_steps, rest, applied_forces
            )
        finally:
            cuboid.remove()

    def _spawn_from_state(self, cuboid: Cuboid, start: ObjectState) -> None:
        x0, psi0, vx0, wy0 = start
        spec = cuboid.spec
        # State x is the COM position; spawn() takes the geometric/link origin.
        x_center = x0 - (spec.h_c - spec.H / 2.0) * math.sin(psi0)
        # Reconstruct floor contact from the observable planar state.
        z0 = (spec.H / 2.0) * math.cos(psi0) + (spec.w / 2.0) * abs(math.sin(psi0))
        orientation = p.getQuaternionFromEuler([0.0, psi0, 0.0])
        cuboid.spawn(position=(x_center, 0.0, z0), orientation=orientation)
        p.resetBaseVelocity(
            cuboid.body, [vx0, 0.0, 0.0], [0.0, wy0, 0.0], physicsClientId=self.client
        )

    def _execute_prediction(
        self,
        cuboid: Cuboid,
        action: PushAction,
        warmup_steps: int,
        rest: RestConfig | None,
        applied_forces: Sequence[AppliedForce] | None = None,
    ) -> DenseTrajectory:
        if applied_forces is None:
            trajectory = self.executor.execute(
                cuboid, h=action.h, F=action.F, tau=action.tau,
                warmup_steps=warmup_steps,
            )
        else:
            trajectory = self.executor.execute_forces(cuboid, applied_forces, warmup_steps)
        if rest is not None:
            wait_for_rest(cuboid, self.client, rest, trajectory)
        return trajectory

    def close(self) -> None:
        if self.client is not None:
            p.disconnect(physicsClientId=self.client)
            self.client = None
