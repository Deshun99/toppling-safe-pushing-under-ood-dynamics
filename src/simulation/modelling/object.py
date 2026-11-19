"""Cuboid object with hidden physical parameters (M1).

Body-frame convention: the base-link origin sits at the *geometric centre* of
the cuboid. The COM is offset with `baseInertialFramePosition`, so a hidden
vertical COM height h_c is representable while the body frame stays
geometry-aligned. A lateral COM offset in a future iteration (plan §7.2) slots
into the same mechanism.

Friction convention: Bullet combines the two contacting bodies' lateralFriction
values by multiplication, so the floor is given mu = 1.0 and the cuboid's
lateralFriction IS the effective object-floor coefficient (outline §5.3).
Rolling/spinning friction are zero by default — the outline treats them as
unmodelled G2 dynamics, so keep them off at G0.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import pybullet as p

Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
ObjectState = tuple[float, float, float, float]
Geometry = tuple[float, float, float]
PhysicsParameters = tuple[float, float, float]


@dataclass
class CuboidSpec:
    """Geometry + hidden physical parameters (outline §2.1, §3.1).

    w:   width along the push direction (m)
    L:   depth (m)
    H:   height (m)
    m:   mass (kg)
    mu:  object-floor friction coefficient
    h_c: vertical COM height (m)
    """

    w: float
    L: float
    H: float
    m: float
    mu: float
    h_c: float

    @property
    def theta(self) -> PhysicsParameters:
        """The hidden physical parameters (m, mu, h_c)."""
        return (self.m, self.mu, self.h_c)

    @property
    def crit_tilt(self) -> float:
        """Critical tilt about the tipping axis: COM projection reaches the
        front edge. The toppling condition (outline §3.4)."""
        return math.atan((self.w / 2.0) / self.h_c)


class Cuboid:
    """A spawned cuboid in one PyBullet client."""

    def __init__(self, spec: CuboidSpec, client_id: int) -> None:
        self.spec = spec
        self.client = client_id
        self.body: int | None = None  # created by spawn()

    def spawn(
        self,
        position: Sequence[float] = (0.0, 0.0, 0.0),
        orientation: Sequence[float] = (0, 0, 0, 1),
    ) -> int:
        spec = self.spec
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[spec.w / 2.0, spec.L / 2.0, spec.H / 2.0],
            physicsClientId=self.client,
        )
        com_offset = [0.0, 0.0, spec.h_c - spec.H / 2.0]
        self.body = p.createMultiBody(
            baseMass=spec.m,
            baseCollisionShapeIndex=collision_shape,
            basePosition=position,
            baseOrientation=orientation,
            baseInertialFramePosition=com_offset,
            physicsClientId=self.client,
        )
        p.changeDynamics(
            self.body,
            -1,
            lateralFriction=spec.mu,
            rollingFriction=0.0,
            spinningFriction=0.0,
            restitution=0.0,
            physicsClientId=self.client,
        )
        return self.body

    def remove(self) -> None:
        if self.body is not None:
            p.removeBody(self.body, physicsClientId=self.client)
            self.body = None

    def pose(self) -> tuple[Vector3, Quaternion]:
        return p.getBasePositionAndOrientation(self.body, physicsClientId=self.client)

    def velocity(self) -> tuple[Vector3, Vector3]:
        return p.getBaseVelocity(self.body, physicsClientId=self.client)

    def state(self) -> ObjectState:
        """(x, psi, xdot, psidot) — the dynamic part of the observation.

        psi is the pitch (tilt about the tipping axis), from Euler angles.
        """
        (x, _, _), orn = self.pose()
        (vx, _, _), (_, wy, _) = self.velocity()
        _, psi, _ = p.getEulerFromQuaternion(orn)
        return x, psi, vx, wy

    def rear_face_point(
        self, position: Sequence[float], orientation: Sequence[float], h: float
    ) -> Vector3:
        """
        Locate the centred rear-face push point in world coordinates.

        Height is measured along the body's vertical axis from its base, not
        above the floor. The caller supplies the current pose each physics step
        so the point follows the moving object. This method applies no force.
        """
        # Bullet's base pose is the COM pose, even though spawn() takes the
        # geometric/link origin. Measure the application point from the COM.
        local = (-self.spec.w / 2.0, 0.0, h - self.spec.h_c)
        world, _ = p.multiplyTransforms(
            position,
            orientation,
            local,
            [0, 0, 0, 1],
            physicsClientId=self.client,
        )
        return world
