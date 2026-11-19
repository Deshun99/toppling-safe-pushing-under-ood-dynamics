"""Optional post-push rest, matched forecasts, and persisted task settings."""

from math import cos, sin
from pathlib import Path

import numpy as np
import pybullet as p
import pytest

from src.agents.rl.training.delta_utils import parse_args
from src.agents.rl.training.train_delta import run, environment_settings
from src.agents.simulator.internal_sim import InternalSim
from src.simulation.environment import PushEnv
from src.simulation.execution.settling import (
    RestConfig,
    REST_HOLD_TIME,
    REST_LINEAR_SPEED,
    is_at_rest,
    wait_for_rest,
)
from src.simulation.modelling.constants import DT
from src.simulation.modelling.object import CuboidSpec


def test_wait_observes_rest_and_keeps_sysid_record_push_only(spec: CuboidSpec) -> None:
    env = PushEnv(fixed_spec=spec, wait_for_rest=True)
    try:
        env.reset(seed=42)
        obs, reward, _, _, info = env.step([-1.0, -0.8, -0.6])
        assert info["settled"] and not info["settling_timeout"]
        assert info["settling_time"] >= REST_HOLD_TIME
        assert is_at_rest(env.cuboid, env.client)
        assert abs(obs[2]) <= REST_LINEAR_SPEED
        record = info["action_record"]
        assert len(record.traj.t) == round(record.action.tau / DT)
        assert env.target_distance - obs[0] > record.traj.x[-1]
        expected = env.reward.compute_reward(
            previous_distance=1.0,
            distance=obs[0],
            duration=record.action.tau + info["settling_time"],
            success=info["success"],
            toppled=info["toppled"],
        )
        assert reward == pytest.approx(expected, abs=1e-5)
        assert env.steps == 1
    finally:
        env.close()


def test_wait_is_bounded_and_detects_full_angular_velocity(env: PushEnv) -> None:
    p.resetBaseVelocity(
        env.cuboid.body, angularVelocity=[0, 0, 1], physicsClientId=env.client
    )
    assert not is_at_rest(env.cuboid, env.client)
    result = wait_for_rest(env.cuboid, env.client, RestConfig(True, DT))
    assert result.steps == 1 and not result.rested


def test_wait_stops_when_object_topples(env: PushEnv) -> None:
    spec = env.cuboid.spec
    tilt = spec.crit_tilt * 0.99
    height = spec.H / 2 * cos(tilt) + spec.w / 2 * sin(tilt)
    p.resetBasePositionAndOrientation(
        env.cuboid.body,
        [0, 0, height],
        p.getQuaternionFromEuler([0, tilt, 0]),
        physicsClientId=env.client,
    )
    p.resetBaseVelocity(
        env.cuboid.body, angularVelocity=[0, 5, 0], physicsClientId=env.client
    )
    result = wait_for_rest(env.cuboid, env.client, RestConfig(True))
    assert result.toppled and not result.rested
    assert 0 < result.steps < 2 / DT
    assert result.max_tilt > spec.crit_tilt


def test_forecasts_include_wait_but_sysid_replay_does_not(spec: CuboidSpec) -> None:
    env, simulator = PushEnv(fixed_spec=spec), InternalSim()
    try:
        env.reset(seed=42)
        _, _, _, _, info = env.step([-1.0, -0.8, -0.6])
        record = info["action_record"]
        replay = simulator.replay(spec, record)
        forecast = simulator.predict(
            spec, record.start, record.action, rest=RestConfig(True)
        )
        assert len(replay.t) == len(record.traj.t)
        assert len(forecast.t) > len(replay.t)
        assert abs(forecast.xdot[-1]) <= REST_LINEAR_SPEED
        np.testing.assert_allclose(np.diff(forecast.t), DT, atol=1e-12)
    finally:
        env.close()
        simulator.close()


def test_delta_checkpoint_restores_wait_settings(tmp_path: Path) -> None:
    settings = "--smoke --backend known --wait-for-rest --rest-timeout 0.1".split()
    args = parse_args(settings + ["--out-dir", str(tmp_path / "source")])
    trained = run(args)
    checkpoint = args.out_dir / "delta_final.pt"
    settings = "--smoke --mode evaluate --backend known".split()
    evaluate = parse_args(
        settings
        + ["--checkpoint", str(checkpoint), "--out-dir", str(tmp_path / "evaluate")]
    )
    result = run(evaluate)
    assert result["metadata"]["environment"]["wait_for_rest"]
    assert result["metadata"]["environment"]["rest_timeout"] == 0.1
    assert result["metadata"]["controller"]["wait_for_rest"]
    assert trained["metadata"]["environment"] == result["metadata"]["environment"]
    evaluate.wait_for_rest = False
    with pytest.raises(ValueError, match="Checkpoint settling settings"):
        environment_settings(evaluate, trained["metadata"])


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "0.0001"])
def test_invalid_rest_timeout_is_rejected(value: str) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--wait-for-rest", "--rest-timeout", value])


def test_sac_checkpoint_evaluation_restores_wait_settings(
    tmp_path: Path, spec: CuboidSpec
) -> None:
    from dataclasses import asdict
    from src.agents.rl.models.sac import SAC, SACConfig
    from src.evaluation.rl.evaluator import Evaluator
    from src.simulation.environment import ACTION_RANGES

    model = SAC(SACConfig(7, 3, hidden_dim=16), inference_only=True)
    model.training_metadata = dict(
        action_ranges=asdict(ACTION_RANGES),
        environment=dict(wait_for_rest=True, rest_timeout=0.1),
    )
    path = tmp_path / "sac.pt"
    model.save(path)
    _, env = Evaluator._load_model_and_create_env(path, spec, 42, False, 1.0)
    try:
        assert env.rest_config == RestConfig(True, 0.1)
    finally:
        env.close()


def test_unlimited_wait_can_exceed_the_default_time_cap(env: PushEnv) -> None:
    from unittest.mock import patch
    from math import ceil
    from src.simulation.execution.settling import REST_HOLD_TIME

    moving_steps = round(2.5 / DT)
    quiet_steps = ceil(REST_HOLD_TIME / DT)
    resting = [False] * moving_steps + [True] * quiet_steps
    with patch("src.simulation.execution.settling.is_at_rest", side_effect=resting):
        result = wait_for_rest(env.cuboid, env.client, RestConfig(True, 0.0))
    assert result.rested and not result.toppled
    assert result.steps == moving_steps + quiet_steps
    assert result.steps * DT > 2.0


def test_unlimited_wait_is_accepted_and_saved_as_zero() -> None:
    args = parse_args(["--wait-for-rest", "--rest-timeout", "0"])
    settings = environment_settings(args, None)
    assert settings["wait_for_rest"] and settings["rest_timeout"] == 0.0
    args.checkpoint = Path("checkpoint.pt")
    args.wait_for_rest, args.rest_timeout = None, None
    assert environment_settings(args, {"environment": settings})["rest_timeout"] == 0.0
