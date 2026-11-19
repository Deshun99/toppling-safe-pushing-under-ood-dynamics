"""Validate outcome accounting and isolation from SAC training state."""

import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import pytest
import torch

from src.agents.rl.models.sac import SAC, SACConfig
from src.evaluation.rl import (
    EpisodeRecorder,
    EpisodeResult,
    EvaluationMetrics,
    Evaluator,
)
from src.simulation.environment import ActionRanges, PushEnv
from src.simulation.modelling.object import CuboidSpec


def make_episode(
    outcome: Literal["success", "toppled", "timeout"] = "success", **changes: Any
) -> EpisodeResult:
    episode = EpisodeResult(42, None, outcome, 100.0, 2, 0.3, 0.01, 0.1)
    return replace(episode, **changes)


def test_summary_uses_all_outcomes_and_conditions_efficiency_on_success() -> None:
    result = EvaluationMetrics(
        [
            make_episode(),
            make_episode(
                "toppled", total_reward=-100, push_count=1, final_position_error_m=0.8
            ),
            make_episode(
                "timeout", total_reward=0, push_count=10, final_position_error_m=0.4
            ),
        ]
    ).summarize()
    assert result.episode_count == 3
    assert result.success_count == result.topple_count == result.timeout_count == 1
    assert (
        result.success_rate + result.topple_rate + result.timeout_rate
        == pytest.approx(1)
    )
    assert result.mean_reward == 0
    assert result.std_reward == pytest.approx(np.std([100, -100, 0]))
    assert result.mean_final_position_error_m == pytest.approx(1.21 / 3)
    assert result.mean_successful_push_count == 2
    assert result.mean_successful_push_time_s == 0.3


def test_unavailable_metrics_are_null_and_omitted_from_scalars() -> None:
    summary = EvaluationMetrics([make_episode("timeout")]).summarize()
    assert summary.mean_successful_push_count is None
    assert "mean_successful_push_count" not in summary.get_scalars()
    assert (
        json.loads(json.dumps(asdict(summary), allow_nan=False))[
            "mean_successful_push_time_s"
        ]
        is None
    )
    partial = EvaluationMetrics(
        [make_episode(), make_episode(push_time_s=None)]
    ).summarize()
    assert partial.mean_successful_push_time_s is None
    with pytest.raises(ValueError, match="At least one"):
        EvaluationMetrics([]).summarize()


def test_fine_tuning_includes_failures_and_retains_never_succeeded() -> None:
    episodes = [
        make_episode("toppled", push_count=3),
        make_episode(),
        make_episode("timeout"),
    ]
    summary = EvaluationMetrics(episodes).summarize_progress()
    assert summary.interactions_to_first_success == 5
    assert summary.push_time_to_first_success_s == pytest.approx(0.6)
    assert summary.cumulative_target_interactions == (3, 5, 7)
    assert summary.cumulative_topples == (1, 1, 1)
    assert summary.cumulative_successes == (0, 1, 1)
    never = EvaluationMetrics([make_episode("timeout")]).summarize_progress()
    assert never.interactions_to_first_success is None
    assert never.push_time_to_first_success_s is None


@pytest.mark.parametrize(
    ("durations", "expected"),
    [([None, 0.3], None), ([0.3, None], None), ([0.3, 0.4, None], 0.7)],
)
def test_progress_only_requires_durations_through_first_success(
    durations: list[float | None], expected: float | None
) -> None:
    episodes = [
        make_episode("success" if index == 1 else "timeout", push_time_s=duration)
        for index, duration in enumerate(durations)
    ]
    progress = EvaluationMetrics(episodes).summarize_progress()
    assert progress.interactions_to_first_success == 4
    assert progress.push_time_to_first_success_s == pytest.approx(expected)


def test_recorder_accumulates_peak_and_duration_and_prioritizes_toppling() -> None:
    recorder = EpisodeRecorder(42, None)
    recorder.record_step(2, dict(executed=(0.1, 5, 0.2), max_tilt=0.4))
    info = dict(executed=(0.1, 5, 0.3), max_tilt=0.1, success=True, toppled=True)
    recorder.record_step(-1, info)
    result = recorder.finish([-0.02], True, False, info)
    assert result.outcome == "toppled"
    assert result.max_abs_tilt_rad == 0.4
    assert result.push_time_s == pytest.approx(0.5)
    assert result.push_count == 2
    assert result.total_reward == 1
    assert result.final_position_error_m == 0.02
    with pytest.raises(ValueError, match="ongoing"):
        recorder.finish([0], False, False, info)


def test_evaluator_preserves_model_and_training_rng(env: PushEnv) -> None:
    torch.set_num_threads(1)
    model = SAC(SACConfig(7, 3, buffer_size=10, batch_size=2))
    model.normalizer.update(np.zeros((2, 7), dtype=np.float32))
    parameters = {
        name: deepcopy(getattr(model, name).state_dict())
        for name in ("actor", "critic", "target_critic")
    }
    moments = deepcopy(model.normalizer.state_dict())
    rng = torch.get_rng_state().clone()
    replay_size, steps = model.replay.size, model.num_timesteps
    Evaluator(env, [42, 43]).evaluate(model)
    assert model.replay.size == replay_size
    assert model.num_timesteps == steps
    for name, state in parameters.items():
        for key, value in state.items():
            torch.testing.assert_close(getattr(model, name).state_dict()[key], value)
    for key, value in moments.items():
        np.testing.assert_array_equal(model.normalizer.state_dict()[key], value)
    assert torch.equal(torch.get_rng_state(), rng)


def test_evaluator_repeats_cases_and_calls_back_before_next_reset(env: PushEnv) -> None:
    model = SAC(SACConfig(7, 3, hidden_dim=16), inference_only=True)
    reported: list[EpisodeResult] = []

    def record_episode(episode: EpisodeResult) -> None:
        assert env.cuboid.spec == CuboidSpec(**episode.object_spec)
        assert abs(float(env._get_observation()[0])) == pytest.approx(
            episode.final_position_error_m
        )
        reported.append(episode)

    evaluator = Evaluator(env, [42, 43])
    first = evaluator.evaluate(model, on_episode_end=record_episode)
    assert first == evaluator.evaluate(model)
    assert tuple(reported) == first.episodes
    assert first.episodes[0].object_spec is not None
    assert all(episode.push_time_s is not None for episode in first.episodes)
    json.dumps(asdict(first), allow_nan=False)
    env.reset(seed=42)  # The caller still owns a usable environment.


def test_evaluator_rejects_empty_cases() -> None:
    with pytest.raises(ValueError, match="At least one"):
        Evaluator(None, [])


@pytest.fixture
def training_output(
    tmp_path: Path,
) -> Path:
    from src.agents.rl.training.train_sac import train

    args = SimpleNamespace(
        out_dir=tmp_path,
        tb_dir=tmp_path / "logs",
        f_max=15.0,
        n_envs=1,
        seed=42,
        lr=3e-4,
        buffer_size=100,
        learning_starts=1,
        batch_size=8,
        gradient_steps=1,
        eval_freq=2,
        n_eval_episodes=2,
        total_steps=4,
        run_name="evaluation-test",
    )
    train(args)
    return tmp_path


def test_training_exports_episode_records_and_all_summary_scalars(
    training_output: Path,
) -> None:
    tmp_path = training_output
    history = json.loads((tmp_path / "evaluations.json").read_text())
    assert [row["steps"] for row in history] == [2, 4]
    details = json.loads(
        (tmp_path / "evaluation_episodes/steps_000004.json").read_text()
    )
    assert details["schema_version"] == 1
    assert details["steps"] == 4
    assert details["metadata"]["episode_seeds"] == [10042, 10043]
    assert details["metadata"]["action_ranges"]["force"] == [0, 15]
    assert len(details["episodes"]) == 2
    assert history[-1]["mean_reward"] == details["summary"]["mean_reward"]
    run_config = json.loads((tmp_path / "config.json").read_text())
    log = (
        tmp_path / "logs/evaluation-test" / run_config["run_id"] / "metrics.csv"
    ).read_text()
    for metric in ("mean_reward", "success_rate", "topple_rate", "timeout_rate"):
        assert f"eval/{metric}" in log
    assert (tmp_path / "best_model/best_model.pt").is_file()


@pytest.fixture
def checkpoint_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[SAC, Path, list[SAC]]:
    model = SAC(SACConfig(7, 3, hidden_dim=16, buffer_size=100, batch_size=8))
    path = tmp_path / "model.pt"
    model.save(path)
    bounds = ActionRanges(force=(0, 15))
    (tmp_path / "config.json").write_text(json.dumps({"action_ranges": asdict(bounds)}))
    loaded_models: list[SAC] = []
    original_load = SAC.load

    def load_checkpoint(*args: Any, **kwargs: Any) -> SAC:
        loaded = original_load(*args, **kwargs)
        loaded_models.append(loaded)
        return loaded

    monkeypatch.setattr(SAC, "load", load_checkpoint)
    return model, path, loaded_models


def test_checkpoint_evaluation_is_independent_and_preserves_reset_schedule(
    checkpoint_models: tuple[SAC, Path, list[SAC]], spec: CuboidSpec
) -> None:
    model, path, loaded_models = checkpoint_models
    frozen = Evaluator.evaluate_checkpoint(path, spec, 2, seed=7)
    first = Evaluator.evaluate_checkpoint(
        path, spec, 2, seed=7, fine_tune=True, lr_scale=0.5
    )
    repeated = Evaluator.evaluate_checkpoint(
        path, spec, 2, seed=7, fine_tune=True, lr_scale=0.5
    )
    assert first == repeated
    assert first.metadata["episode_seeds"] == [7, None]
    assert [episode.seed for episode in first.episodes] == [7, None]
    assert first.metadata["action_ranges"] == asdict(ActionRanges(force=(0, 15)))
    assert not frozen.metadata["fine_tuning_enabled"]
    assert first.metadata["fine_tuning_enabled"]
    frozen_model, tuned_model, repeated_model = loaded_models
    assert frozen_model.replay is None
    assert frozen_model.num_timesteps == 0
    interactions = sum(episode.push_count for episode in first.episodes)
    assert tuned_model.replay.size == tuned_model.num_timesteps == interactions
    assert repeated_model is not tuned_model
    assert all(
        loaded.normalizer.count == model.normalizer.count for loaded in loaded_models
    )
    assert tuned_model.config.learning_rate == model.config.learning_rate * 0.5


def test_fine_tuning_rejects_inference_model_before_environment_interaction(
    env: PushEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = SAC(SACConfig(7, 3, hidden_dim=16), inference_only=True)

    def reject_reset(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Must reject fine-tuning before resetting the environment")

    monkeypatch.setattr(env, "reset", reject_reset)
    with pytest.raises(RuntimeError, match="inference-only"):
        Evaluator(env, [42]).evaluate(model, fine_tune=True)
