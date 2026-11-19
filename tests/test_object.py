import math
from dataclasses import replace

import numpy as np
import pybullet as p
import pytest

from src.simulation.environment.env import PushEnv
from src.simulation.modelling.object import Cuboid, CuboidSpec


def test_spec(spec):
    assert spec.theta == (1.0, 0.3, 0.08)
    assert spec.crit_tilt == pytest.approx(math.atan(0.04 / 0.08))


def test_spawn_dynamics_and_remove(env, spec):
    box = env.cuboid
    dynamics = p.getDynamicsInfo(box.body, -1, physicsClientId=env.client)
    assert dynamics[0] == spec.m
    assert dynamics[1] == spec.mu
    assert dynamics[3] == pytest.approx((0, 0, spec.h_c - spec.H / 2))
    assert p.getDynamicsInfo(env.plane, -1, physicsClientId=env.client)[1] == 1
    count = p.getNumBodies(physicsClientId=env.client)
    box.remove()
    box.remove()
    assert box.body is None
    assert p.getNumBodies(physicsClientId=env.client) == count - 1


@pytest.mark.parametrize("pitch", [0, 0.3, -0.3])
def test_rear_face_tracks_pose(env: PushEnv, spec: CuboidSpec, pitch: float) -> None:
    position = (1, 2, 3)
    orientation = p.getQuaternionFromEuler((0, pitch, 0))
    local = np.array((-spec.w / 2, 0, 0.03 - spec.h_c))
    rotation = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    expected = np.array(position) + rotation @ local
    assert env.cuboid.rear_face_point(position, orientation, 0.03) == pytest.approx(
        expected
    )


@pytest.mark.parametrize("com_height", [0.04, 0.16])
def test_command_height_uses_object_base_with_offset_com(
    env: PushEnv, spec: CuboidSpec, com_height: float
) -> None:
    box = Cuboid(replace(spec, H=0.2, h_c=com_height), env.client)
    box.spawn(position=(0.0, 0.0, box.spec.H / 2))
    try:
        position, orientation = box.pose()
        point = box.rear_face_point(position, orientation, 0.03)
        assert point == pytest.approx((-box.spec.w / 2, 0.0, 0.03))
    finally:
        box.remove()


def test_state_exposes_pitch_and_world_velocities(env):
    box = env.cuboid
    p.resetBasePositionAndOrientation(
        box.body,
        (0.4, 0, 0.2),
        p.getQuaternionFromEuler((0, 0.2, 0)),
        physicsClientId=env.client,
    )
    p.resetBaseVelocity(box.body, (0.7, 0, 0), (0, -0.6, 0), physicsClientId=env.client)
    assert box.state() == pytest.approx((0.4, 0.2, 0.7, -0.6))


def test_unspawned_remove(env, spec):
    Cuboid(spec, env.client).remove()
