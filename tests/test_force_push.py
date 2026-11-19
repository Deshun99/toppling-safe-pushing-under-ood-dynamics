from unittest.mock import Mock

import numpy as np
import pytest

from src.simulation.execution.force_push import DenseTrajectory, ForcePushExecutor
from src.simulation.modelling.constants import DT, PHYSICS_HZ


def test_trajectory_to_numpy_preserves_raw_samples() -> None:
    trajectory = DenseTrajectory(
        t=[0.0, 0.2],
        x=[5.0, 5.4],
        psi=[0.0, 0.04],
        xdot=[0.0, 1.0],
        psidot=[0.0, 0.2],
    )
    samples = trajectory.to_numpy()
    np.testing.assert_array_equal(
        samples, [[0.0, 5.0, 0.0, 0.0, 0.0], [0.2, 5.4, 0.04, 1.0, 0.2]]
    )
    assert samples.dtype == np.float64
    samples[0, 1] = 100.0
    assert trajectory.x[0] == 5.0


def test_empty_trajectory_to_numpy_has_five_columns() -> None:
    assert DenseTrajectory().to_numpy().shape == (0, 5)


def test_trajectory_to_numpy_rejects_misaligned_channels() -> None:
    with pytest.raises(ValueError):
        DenseTrajectory(t=[0.0], x=[1.0, 2.0]).to_numpy()


def test_trajectory_lists_are_independent():
    first, second = DenseTrajectory(), DenseTrajectory()
    first.append(0, 1, 2, 3, 4)
    assert (first.t, first.x, first.psi, first.xdot, first.psidot) == (
        [0],
        [1],
        [2],
        [3],
        [4],
    )
    assert second.x == []


@pytest.mark.parametrize("tau,warmup", [(0, 0), (0.05, 0), (0.1, 8)])
def test_step_count_and_timestamps(env, tau, warmup):
    trajectory = env.executor.execute(env.cuboid, 0.04, 0, tau, warmup_steps=warmup)
    count = warmup + max(1, round(tau / DT))
    assert DT == 1 / PHYSICS_HZ
    assert trajectory.t == pytest.approx(np.arange(count) * DT)
    for field in ("x", "psi", "xdot", "psidot"):
        assert len(getattr(trajectory, field)) == count
        assert np.isfinite(getattr(trajectory, field)).all()


def test_force_tracks_current_face_and_world_direction(env, monkeypatch):
    apply = Mock()
    point = Mock(wraps=env.cuboid.rear_face_point)
    monkeypatch.setattr(
        "src.simulation.execution.force_push.p.applyExternalForce", apply
    )
    monkeypatch.setattr(env.cuboid, "rear_face_point", point)
    env.executor.execute(env.cuboid, 0.04, 4, 3 * DT, warmup_steps=2)
    assert apply.call_count == point.call_count == 3
    for call in apply.call_args_list:
        assert call.kwargs["forceObj"] == [4, 0, 0]
        assert call.kwargs["physicsClientId"] == env.client


@pytest.mark.parametrize(
    "now,expected,sleep", [(1.02, 1.1, 0.08), (1.15, 1.1, None), (2, 2, None)]
)
def test_deadline_pacing(monkeypatch, now, expected, sleep):
    sleeper = Mock()
    monkeypatch.setattr(
        "src.simulation.execution.force_push.time.perf_counter", lambda: now
    )
    monkeypatch.setattr("src.simulation.execution.force_push.time.sleep", sleeper)
    assert ForcePushExecutor._pace(1, 0.1) == pytest.approx(expected)
    if sleep is None:
        sleeper.assert_not_called()
    else:
        sleeper.assert_called_once_with(pytest.approx(sleep))


def test_pacing_preserves_physics(env, monkeypatch):
    env.reset(seed=3)
    fast = env.executor.execute(env.cuboid, 0.04, 4, 0.1, warmup_steps=2)
    env.reset(seed=3)
    monkeypatch.setattr(
        "src.simulation.execution.force_push.time.sleep", lambda _: None
    )
    env.executor.realtime_scale = 0.5
    paced = env.executor.execute(env.cuboid, 0.04, 4, 0.1, warmup_steps=2)
    assert paced == fast


@pytest.mark.parametrize("height,force,safe", [(0.05, 4, True), (0.18, 8, False)])
def test_sliding_and_toppling_regimes(env, height, force, safe):
    # Stay away from the bistable boundary; exact traces vary across platforms.
    trajectory = env.executor.execute(env.cuboid, height, force, 0.4)
    peak = max(abs(value) for value in trajectory.psi)
    if safe:
        assert trajectory.x[-1] - trajectory.x[0] > 0.01
        assert peak < env.cuboid.spec.crit_tilt
    else:
        assert peak > env.cuboid.spec.crit_tilt
