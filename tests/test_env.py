from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pybullet as p
import pytest

from src.simulation.environment.env import PushEnv
from src.simulation.environment.reward import RewardConfig, RewardModel
from src.simulation.execution.force_push import DenseTrajectory
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.sampling import DEFAULT_OBJECT_RANGES


@pytest.mark.parametrize(
    "value,expected",
    [
        (-2, (0.02, 0, 0.05)),
        (-1, (0.02, 0, 0.05)),
        (0, (0.1, 25, 0.275)),
        (1, (0.18, 50, 0.5)),
        (2, (0.18, 50, 0.5)),
    ],
)
def test_action_mapping(env, value, expected):
    action = env._denormalize_action([value] * 3)
    assert (action.h, action.F, action.tau) == pytest.approx(expected)


def test_reset_clears_episode_and_replaces_body(env, spec):
    env.step([-1, -1, -1])
    obs, info = env.reset(seed=42)
    assert env.observation_space.contains(obs)
    assert obs[4:] == pytest.approx((spec.w, spec.L, spec.H))
    assert info == {"true_theta": spec.theta}
    assert env.steps == 0
    assert env.last_trajectory is None
    assert env.psi_max == pytest.approx(env.near_topple_frac * spec.crit_tilt)
    assert p.getNumBodies(physicsClientId=env.client) == 2


def test_sampling_bounds_and_seed():
    env = PushEnv()
    try:
        first, _ = env.reset(seed=7)
        theta = env.cuboid.spec.theta
        second, _ = env.reset(seed=7)
        np.testing.assert_array_equal(first, second)
        assert env.cuboid.spec.theta == theta
        for _ in range(5_000):
            spec = env._sample_spec()
            for name in ("w", "L", "H", "m", "mu"):
                low, high = getattr(DEFAULT_OBJECT_RANGES, name)
                assert low <= getattr(spec, name) <= high
            assert (
                DEFAULT_OBJECT_RANGES.h_c_frac[0]
                <= spec.h_c / spec.H
                <= DEFAULT_OBJECT_RANGES.h_c_frac[1]
            )
    finally:
        env.close()


@pytest.mark.parametrize(
    "state,steps,success,toppled,terminated,truncated",
    [
        ((0.2, 0, 0, 0), 0, False, False, False, False),
        ((1, 0, 0, 0), 9, True, False, True, False),
        ((1, 0, 0.1, 0), 0, False, False, False, False),
        ((1, 0.03, 0, 0), 0, False, False, False, False),
        ((0, 0.6, 0, 0), 9, False, True, True, False),
        ((0, -0.6, 0, 0), 0, False, True, True, False),
        ((0, 0, 0, 0), 9, False, False, False, True),
    ],
)
def test_reward_and_episode_rules(
    env: PushEnv,
    monkeypatch: pytest.MonkeyPatch,
    state: tuple[float, float, float, float],
    steps: int,
    success: bool,
    toppled: bool,
    terminated: bool,
    truncated: bool,
) -> None:
    monkeypatch.setattr(env.cuboid, "state", lambda: state)
    env.steps = steps
    env.last_trajectory = DenseTrajectory(psi=[0.1, -0.7])
    reward, term, trunc, info = env._compute_step_result(PushAction(0.04, 4, 0.2))
    expected = 10 * state[0] - 0.2 - 0.1 + 100 * success - 100 * toppled
    assert reward == pytest.approx(expected)
    assert (term, trunc) == (terminated, truncated)
    assert (info["success"], info["toppled"]) == (success, toppled)
    assert info["max_tilt"] == 0.7
    assert env.steps == steps + 1
    assert env._d_prev == pytest.approx(1 - state[0])


def test_custom_executor_and_observation(env, monkeypatch):
    trajectory = DenseTrajectory()
    env.executor = Mock()
    env.executor.execute.return_value = trajectory
    monkeypatch.setattr(env.cuboid, "state", lambda: (0.25, 0.01, 0.02, 0.03))
    obs, _, _, _, info = env.step([-1, -1, -1])
    h, F, tau = info["executed"]
    env.executor.execute.assert_called_once_with(
        env.cuboid, h=h, F=F, tau=tau, warmup_steps=0
    )
    assert info["trajectory"] is env.last_trajectory is trajectory
    assert info["max_tilt"] == 0.01
    assert obs[:4] == pytest.approx((0.75, 0.01, 0.02, 0.03))
    assert obs.dtype == np.float32


def test_episode_runs_to_budget(env):
    for index in range(env.T_max):
        obs, reward, terminated, truncated, info = env.step([-1, -1, -1])
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        assert not terminated
        assert truncated == (index == env.T_max - 1)
        assert info["executed"] == pytest.approx((0.02, 0, 0.05))


def test_clients_are_isolated_and_close_is_idempotent(env):
    other = PushEnv()
    try:
        other.reset(seed=1)
        other.close()
        other.close()
        assert other.client is None
        assert p.isConnected(env.client)
        env.step([-1, -1, -1])
    finally:
        other.close()


def test_reset_without_seed_and_custom_configuration(spec):
    executor = Mock()
    reward = RewardModel(RewardConfig(alpha=2, c_time=3, c_step=4))
    env = PushEnv(fixed_spec=spec, executor=executor, reward=reward)
    try:
        env.reset()
        assert env.executor is executor
        assert env.reward is reward
        executor.execute.return_value = DenseTrajectory()
        _, value, terminated, truncated, _ = env.step([-1, -1, -1])
        assert value == pytest.approx(-3 * 0.05 - 4, abs=1e-3)
        assert not terminated and not truncated
    finally:
        env.close()


@pytest.mark.parametrize("warmup", [0, 8])
def test_physical_push_records_start_and_replay_fidelity(env, spec, warmup):
    from src.agents.simulator.internal_sim import InternalSim
    from src.simulation.modelling.constants import DT

    start = env.cuboid.state()
    previous_distance = env._d_prev
    record = env.push(PushAction(h=0.04, F=4, tau=0.3), warmup_steps=warmup)
    assert record.start == start
    assert record.action == PushAction(h=0.04, F=4, tau=0.3)
    assert record.traj is env.last_trajectory
    assert env.steps == 0 and env._d_prev == previous_distance
    sim = InternalSim()
    try:
        replay = sim.replay(spec, record)
        assert len(record.traj.t) == len(replay.t) == warmup + round(0.3 / DT)
        assert record.traj.t == replay.t
        # Low pushes stay away from the bistable tipping boundary. Allow
        # contact transients from reconstructing the state in a fresh world;
        # these are physical error budgets, not exact trace snapshots.
        rms_limits = {"x": 0.001, "psi": 0.002, "xdot": 0.01, "psidot": 0.15}
        for quantity, limit in rms_limits.items():
            recorded = np.asarray(getattr(record.traj, quantity))
            simulated = np.asarray(getattr(replay, quantity))
            rms_error = float(np.sqrt(np.mean((recorded - simulated) ** 2)))
            assert rms_error < limit, f"{quantity}: RMS error {rms_error} >= {limit}"

        # Demonstrate that the position budget catches incorrect dynamics,
        # even when replay produces exactly the same timestamps.
        wrong_mass = sim.replay(replace(spec, m=1.5 * spec.m), record)
        assert wrong_mass.t == replay.t
        position_error = np.asarray(record.traj.x) - wrong_mass.x
        assert np.sqrt(np.mean(position_error**2)) > rms_limits["x"]
    finally:
        sim.close()


def test_step_and_physical_push_share_trajectory(env):
    physical = env._denormalize_action([-1, -0.84, -1])
    record = env.push(physical)
    env.reset(seed=42)
    _, _, _, _, info = env.step([-1, -0.84, -1])
    assert info["trajectory"] == record.traj
    assert env.steps == 1


def test_replacement_preserves_spec_and_client(env, spec):
    env.replace_object(x=0.4)
    assert env.cuboid.spec == spec
    assert env.cuboid.state() == pytest.approx((0.4, 0, 0, 0))
    assert p.getNumBodies(physicsClientId=env.client) == 2
    assert env.last_trajectory is None


def test_action_ranges_are_per_environment(spec):
    from src.simulation.environment.env import ActionRanges

    limited = PushEnv(fixed_spec=spec, action_ranges=ActionRanges(force=(0, 15)))
    default = PushEnv(fixed_spec=spec)
    try:
        limited.reset()
        default.reset()
        assert limited._denormalize_action([0, 1, 0]).F == 15
        assert default._denormalize_action([0, 1, 0]).F == 50
    finally:
        limited.close()
        default.close()


@pytest.mark.parametrize(
    "previous_x,current_x,progress",
    [
        (0.8, 0.9, 0.1),
        (1.1, 1.2, -0.1),
        (0.9, 1.2, -0.1),
    ],
)
def test_reward_tracks_unsigned_distance(
    env: PushEnv,
    monkeypatch: pytest.MonkeyPatch,
    previous_x: float,
    current_x: float,
    progress: float,
) -> None:
    monkeypatch.setattr(env.cuboid, "state", lambda: (current_x, 0, 0, 0))
    env._d_prev = env.target_distance - previous_x
    env.last_trajectory = DenseTrajectory()
    reward, _, _, info = env._compute_step_result(PushAction(0.04, 4, 0.2))
    assert reward == pytest.approx(env.reward.config.alpha * progress - 0.2 - 0.1)
    assert info["is_success"] == info["success"]
    assert not info["TimeLimit.truncated"]
