"""Progress strategies, reward bounds, and checkpoint reward selection."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from src.agents.rl.training.delta_utils import parse_args
from src.agents.rl.training.train_delta import _resolve_reward_config, run
from src.simulation.environment.reward import (
    GoalProximityReward,
    LinearProgressReward,
    RewardConfig,
    RewardModel,
)


def compute_reward(model: RewardModel, previous: float, distance: float) -> float:
    return model.compute_reward(
        previous_distance=previous,
        distance=distance,
        duration=0.1,
        success=False,
        toppled=False,
    )


def test_linear_strategy_preserves_original_reward() -> None:
    model = RewardModel()
    assert isinstance(model.strategy, LinearProgressReward)
    assert compute_reward(model, 1.0, 0.99) == pytest.approx(-0.1)
    assert compute_reward(model, 0.0, -0.1) == pytest.approx(-1.2)


def test_proximity_reward_encourages_movement_at_start_and_more_near_goal() -> None:
    model = RewardModel(RewardConfig(strategy="goal_proximity"))
    assert isinstance(model.strategy, GoalProximityReward)
    far = compute_reward(model, 1.0, 0.99)
    near = compute_reward(model, 0.1, 0.09)
    assert far == pytest.approx(0.06129032258)
    assert near > far > 0
    assert compute_reward(model, 0, 0) == pytest.approx(-0.2)
    assert compute_reward(model, 1, 1) == pytest.approx(-0.2)


def test_proximity_reward_is_finite_symmetric_and_reverses_on_retreat() -> None:
    model = RewardModel(RewardConfig(strategy="goal_proximity"))
    strategy = model.strategy
    gain = strategy.compute_progress(1, 0)
    assert np.isfinite(gain) and gain == pytest.approx(90.0)
    assert strategy.compute_progress(0, 1) == pytest.approx(-gain)
    assert strategy.compute_progress(-1, 0) == pytest.approx(gain)
    assert strategy.compute_progress(0.1, -0.1) == pytest.approx(0.0)
    assert strategy.compute_progress(0, -0.1) < 0


@pytest.mark.parametrize("strategy", ["linear", "goal_proximity"])
def test_terminal_bonus_and_penalty_are_shared(strategy: str) -> None:
    model = RewardModel(RewardConfig(strategy=strategy))
    arguments = dict(previous_distance=0.0, distance=0.0, duration=0.1)
    assert model.compute_reward(
        **arguments, success=True, toppled=False
    ) == pytest.approx(99.8)
    assert model.compute_reward(
        **arguments, success=False, toppled=True
    ) == pytest.approx(-100.2)


@pytest.mark.parametrize(
    "settings",
    [
        {"strategy": "unknown"},
        {"proximity_scale": 0},
        {"proximity_scale": float("inf")},
        {"proximity_bonus": -1},
        {"alpha": float("nan")},
    ],
)
def test_invalid_reward_configuration_is_rejected(settings: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RewardConfig(**settings)


def test_checkpoint_restores_strategy_and_rejects_reward_changes(
    tmp_path: Path,
) -> None:
    settings = (
        "--smoke --backend known --reward-strategy goal_proximity "
        "--proximity-bonus 80 --proximity-scale 0.2 "
        "--no-progress-penalty 1 --no-progress-threshold 0.002"
    ).split()
    args = parse_args(settings + ["--out-dir", str(tmp_path / "source")])
    result = run(args)
    checkpoint = args.out_dir / "delta_final.pt"
    evaluate = parse_args(
        "--smoke --mode evaluate".split()
        + ["--checkpoint", str(checkpoint), "--out-dir", str(tmp_path / "evaluate")]
    )
    evaluated = run(evaluate)
    assert evaluated["metadata"]["reward"] == result["metadata"]["reward"]
    assert result["metadata"]["reward"] == asdict(
        RewardConfig(
            strategy="goal_proximity",
            proximity_bonus=80,
            proximity_scale=0.2,
            no_progress_penalty=1,
            no_progress_threshold=0.002,
        )
    )
    evaluate.reward_strategy = "linear"
    with pytest.raises(ValueError, match="Checkpoint reward settings"):
        _resolve_reward_config(evaluate, result["metadata"])


def test_legacy_checkpoint_defaults_to_linear_reward() -> None:
    args = parse_args([])
    legacy = dict(alpha=10.0, c_time=1.0, c_step=0.1, R_succ=100.0, R_topple=100.0)
    config = _resolve_reward_config(args, {"reward": legacy})
    assert config.strategy == "linear"
    assert config.no_progress_penalty == 0


def test_topple_penalty_flag_overrides_reward_config() -> None:
    args = parse_args(["--topple-penalty", "30"])
    config = _resolve_reward_config(args, None)
    assert config.R_topple == 30.0


def test_topple_penalty_defaults_to_standard_value_when_omitted() -> None:
    config = _resolve_reward_config(parse_args([]), None)
    assert config.R_topple == RewardConfig().R_topple


def test_negative_topple_penalty_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--topple-penalty", "-5"])


@pytest.mark.parametrize(
    "option,value",
    [
        ("--proximity-scale", "0"),
        ("--proximity-bonus", "-1"),
        ("--proximity-scale", "nan"),
    ],
)
def test_invalid_cli_coefficients_are_rejected(option: str, value: str) -> None:
    with pytest.raises(SystemExit):
        parse_args([option, value])


@pytest.mark.parametrize(
    "previous,distance,penalized",
    [
        (1.0, 1.0, True),
        (1.0, 1.01, True),
        (1.0, -1.0, True),
        (1.0, 0.9995, True),
        (1.0, 0.99, False),
        (-1.0, -0.99, False),
        (0.04, 0.04, False),
    ],
)
def test_no_progress_penalty_uses_net_distance_reduction(
    previous: float,
    distance: float,
    penalized: bool,
) -> None:
    baseline = RewardModel(RewardConfig(strategy="goal_proximity"))
    penalized_model = RewardModel(
        RewardConfig(strategy="goal_proximity", no_progress_penalty=1.0)
    )
    difference = compute_reward(baseline, previous, distance) - compute_reward(
        penalized_model, previous, distance
    )
    assert difference == pytest.approx(1.0 if penalized else 0.0)


def test_no_progress_threshold_is_inclusive() -> None:
    model = RewardModel(
        RewardConfig(
            strategy="goal_proximity",
            no_progress_penalty=1.0,
            no_progress_threshold=0.125,
        )
    )
    baseline = RewardModel(RewardConfig(strategy="goal_proximity"))
    assert compute_reward(baseline, 1.0, 0.875) - compute_reward(
        model, 1.0, 0.875
    ) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "success,toppled,distance,tolerance",
    [
        (True, False, 0.0, 0.05),
        (False, True, 1.0, 0.05),
        (False, False, 0.08, 0.1),
    ],
)
def test_no_progress_penalty_respects_terminal_outcomes_and_goal_tolerance(
    success: bool,
    toppled: bool,
    distance: float,
    tolerance: float,
) -> None:
    model = RewardModel(
        RewardConfig(strategy="goal_proximity", no_progress_penalty=1.0)
    )
    baseline = RewardModel(RewardConfig(strategy="goal_proximity"))
    kwargs = dict(
        previous_distance=distance,
        distance=distance,
        duration=0.1,
        success=success,
        toppled=toppled,
        goal_tolerance=tolerance,
    )
    assert model.compute_reward(**kwargs) == baseline.compute_reward(**kwargs)


def test_linear_reward_ignores_no_progress_settings() -> None:
    model = RewardModel(RewardConfig(no_progress_penalty=1.0))
    assert compute_reward(model, 1.0, 1.0) == pytest.approx(-0.2)


@pytest.mark.parametrize(
    "option,value",
    [
        ("--no-progress-penalty", "-1"),
        ("--no-progress-threshold", "-1"),
        ("--no-progress-penalty", "inf"),
        ("--no-progress-threshold", "nan"),
    ],
)
def test_invalid_no_progress_options_are_rejected(option: str, value: str) -> None:
    with pytest.raises(SystemExit):
        parse_args([option, value])
