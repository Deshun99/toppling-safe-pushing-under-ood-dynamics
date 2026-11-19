"""Gantry environment records timed contact pushes and reports failed execution."""

import numpy as np
from collections.abc import Iterator
import pybullet as p
import pytest

from src.simulation.environment.env import ActionRanges, ExecutionFailure, PushEnv
from src.simulation.execution.robot import GantryPushExecutor, HOME_POSITIONS, TravelLimitError
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.object import CuboidSpec

SPEC = CuboidSpec(w=.08, L=.08, H=.20, m=1., mu=.3, h_c=.08)


@pytest.fixture
def robot_env() -> Iterator[PushEnv]:
    env = PushEnv(robot=True, fixed_spec=SPEC, target_distance=.5,
                  action_ranges=ActionRanges(force=(0., 15.)))
    try:
        env.reset(seed=0)
        yield env
    finally:
        env.close()


def test_robot_shares_client_and_preserves_pose_between_pushes(robot_env: PushEnv) -> None:
    executor = robot_env.executor
    assert isinstance(executor, GantryPushExecutor)
    assert executor.client == robot_env.client
    _, _, _, _, info = robot_env.step(np.array([-.75, .3, 0.]))
    assert info['execution_status'] == 'completed'
    assert executor.joint_positions()[0] > HOME_POSITIONS[0]
    position = executor.joint_positions()
    robot_env.step(np.array([-.75, -1., -1.]))
    assert executor.joint_positions()[0] >= position[0] - .005
    assert executor.joint_positions()[1] == pytest.approx(position[1], abs=.002)


def test_record_starts_after_approach_and_contains_actual_force(robot_env: PushEnv) -> None:
    before = robot_env.cuboid.state()
    record = robot_env.push(PushAction(.04, 10., .25), warmup_steps=8)
    assert record.start == robot_env.executor.timed_start
    assert record.start != before
    assert record.warmup_steps == 0
    assert len(record.applied_forces) == len(record.traj.t) == 60
    assert any(sample.force_x > 0. for sample in record.applied_forces)
    assert any(abs(sample.force_x - record.action.F) > .1 for sample in record.applied_forces)
    assert robot_env.last_execution.approach_steps > 0
    assert record.traj is robot_env.last_trajectory


def test_reset_homes_and_clears_contact_report(robot_env: PushEnv) -> None:
    robot_env.step(np.array([-.75, .3, 0.]))
    obs, info = robot_env.reset(seed=0)
    assert info['home_steps'] > 0
    assert robot_env.executor.joint_positions() == pytest.approx(HOME_POSITIONS, abs=.002)
    assert robot_env.last_execution is None
    assert robot_env.last_trajectory is None
    assert obs[0] == pytest.approx(.5, abs=.001)
    assert p.getNumBodies(physicsClientId=robot_env.client) == 3


def test_execution_failure_has_no_fabricated_record(
    robot_env: PushEnv, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail() -> None:
        raise TravelLimitError('test travel limit')
    monkeypatch.setattr(robot_env.executor, '_check_forward_travel', fail)
    _, reward, terminated, truncated, info = robot_env.step([-.75, .3, 0.])
    assert terminated and not truncated
    assert np.isfinite(reward)
    assert info['execution_status'] == 'travel_limit'
    assert info['execution_failed']
    assert 'action_record' not in info
    assert not info['TimeLimit.truncated']
    assert robot_env.steps == 1


def test_raw_push_preserves_failure_exception(
    robot_env: PushEnv, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail() -> None:
        raise TimeoutError('test contact timeout')
    monkeypatch.setattr(robot_env.executor, '_check_forward_travel', fail)
    with pytest.raises(ExecutionFailure, match='contact_timeout'):
        robot_env.push(PushAction(.04, 10., .25))


def test_prepush_toppling_is_terminal_without_execution(robot_env: PushEnv) -> None:
    position, _ = p.getBasePositionAndOrientation(
        robot_env.cuboid.body, physicsClientId=robot_env.client)
    p.resetBasePositionAndOrientation(
        robot_env.cuboid.body, position,
        p.getQuaternionFromEuler([0., SPEC.crit_tilt + .1, 0.]),
        physicsClientId=robot_env.client)
    _, _, terminated, truncated, info = robot_env.step([-.75, .3, 0.])
    assert terminated and not truncated
    assert info['toppled']
    assert info['execution_status'] == 'block_toppled'
    assert 'action_record' not in info


def test_robot_and_custom_executor_are_exclusive() -> None:
    with pytest.raises(ValueError, match='not both'):
        PushEnv(robot=True, executor=object())
