"""Tests for the --world-model ablation: none, physics, and learned conditions."""

import types
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.agents.rl.models.delta_actor_critic import (
    DeltaActorCritic,
    DeltaActorCriticConfig,
    TrainingExample,
)
from src.agents.rl.models.delta_controller import (
    DeltaController,
    DeltaControllerConfig,
    ForecastContext,
)
from src.agents.rl.models.latent_world_model import (
    LatentWorldModel,
    interpolate_model_trajectory,
)
from src.agents.rl.training.delta_utils import parse_args
from src.simulation.environment.env import ACTION_RANGES
from src.simulation.execution.force_push import DenseTrajectory

_P = 4
_GEOMETRY = (0.08, 0.08, 0.20)
_OBS = np.array([0.8, 0.01, 0.2, 0.03, *_GEOMETRY], dtype=np.float32)


def _make_model(**settings):
    torch.set_num_threads(1)
    return DeltaActorCritic(
        DeltaActorCriticConfig(trajectory_points=_P, hidden_dim=16, **settings)
    )


def _make_example(*, terminated=True, next_context=None):
    return TrainingExample(
        observation=_OBS.copy(),
        real_trajectory=np.zeros((_P, 5), np.float32),
        delta=np.array([0.1, 0.05, -0.1], np.float32),
        forecasts=np.stack([np.zeros((_P, 5)), np.ones((_P, 5))]).astype(np.float32),
        weights=np.array([0.75, 0.25]),
        reward=1.0,
        terminated=terminated,
        truncated=False,
        reference_action=np.zeros(3, np.float32),
        next_context=next_context,
    )


def _make_forecast_context():
    """Minimal valid ForecastContext for direct _forecast tests."""
    return ForecastContext(
        observation=_OBS.astype(np.float64).copy(),
        start=(0.1, 0.01, 0.2, 0.03),
        geometry=_GEOMETRY,
        reference_action=np.zeros(3, dtype=np.float64),
        particles=np.array([[0.5, 0.4, 0.1]], dtype=np.float64),
        weights=np.array([1.0], dtype=np.float64),
        real_trajectory=np.zeros((_P, 5), dtype=np.float64),
        action_ranges=ACTION_RANGES,
    )


def _make_controller(world_model: str, model=None):
    if model is None:
        model = _make_model(world_model=world_model)
    config = DeltaControllerConfig(
        world_model=world_model,
        trajectory_points=_P,
        n_particles=2,
        candidates_per_batch=2,
    )
    return DeltaController(model, config)


# --- LatentWorldModel ---


def test_latent_world_model_forward_output_shape() -> None:
    model = LatentWorldModel(trajectory_points=_P, hidden_dim=16)
    batch = 3
    obs = torch.zeros(batch, 7)
    action_ref = torch.zeros(batch, 3)
    delta = torch.zeros(batch, 3)
    real_traj = torch.zeros(batch, _P, 5)
    out = model(obs, action_ref, delta, real_traj)
    assert out.shape == (batch, _P, 5)


def test_latent_world_model_predict_numpy_returns_correct_shape() -> None:
    model = LatentWorldModel(trajectory_points=_P, hidden_dim=16)
    out = model.predict_numpy(
        np.zeros(7, np.float32),
        np.zeros(3, np.float32),
        np.zeros(3, np.float32),
        np.zeros((_P, 5), np.float32),
    )
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, _P, 5)


def test_latent_world_model_predict_numpy_handles_readonly_arrays() -> None:
    model = LatentWorldModel(trajectory_points=_P, hidden_dim=16)
    obs = np.zeros(7, np.float32)
    obs.setflags(write=False)
    action_ref = np.zeros(3, np.float32)
    action_ref.setflags(write=False)
    delta = np.zeros(3, np.float32)
    delta.setflags(write=False)
    real_traj = np.zeros((_P, 5), np.float32)
    real_traj.setflags(write=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = model.predict_numpy(obs, action_ref, delta, real_traj)
    assert out.shape == (1, _P, 5)


# --- DeltaActorCriticConfig ---


def test_delta_actor_critic_config_rejects_invalid_world_model() -> None:
    with pytest.raises(ValueError, match="world_model"):
        DeltaActorCriticConfig(world_model="unknown")


# --- DeltaActorCritic instantiation ---


def test_learned_mode_creates_latent_world_model_and_optimizer() -> None:
    model = _make_model(world_model="learned")
    assert model.latent_world_model is not None
    assert model.world_model_optimizer is not None


@pytest.mark.parametrize("mode", ("none", "physics"))
def test_non_learned_modes_have_no_world_model(mode: str) -> None:
    model = _make_model(world_model=mode)
    assert model.latent_world_model is None
    assert model.world_model_optimizer is None


# --- train_step with learned mode ---


def test_train_step_learned_mode_reports_world_model_loss() -> None:
    model = _make_model(world_model="learned")
    next_ctx = types.SimpleNamespace(real_trajectory=np.zeros((_P, 5), np.float32))
    example = _make_example(terminated=True, next_context=next_ctx)
    metrics = model.train_step([example] * 8, lambda e, d: e.forecasts)
    assert "world_model_loss" in metrics


def test_train_step_learned_mode_without_next_context_omits_world_model_loss() -> None:
    model = _make_model(world_model="learned")
    example = _make_example(terminated=True, next_context=None)
    metrics = model.train_step([example] * 8, lambda e, d: e.forecasts)
    assert "world_model_loss" not in metrics


@pytest.mark.parametrize("mode", ("none", "physics"))
def test_train_step_non_learned_modes_never_report_world_model_loss(mode: str) -> None:
    model = _make_model(world_model=mode)
    next_ctx = types.SimpleNamespace(real_trajectory=np.zeros((_P, 5), np.float32))
    example = _make_example(terminated=True, next_context=next_ctx)
    metrics = model.train_step([example] * 8, lambda e, d: e.forecasts)
    assert "world_model_loss" not in metrics


# --- Checkpoint round-trip ---


def test_learned_mode_checkpoint_roundtrip_preserves_world_model(
    tmp_path: Path,
) -> None:
    model = _make_model(world_model="learned")
    # Record initial weights so we can confirm they survive the round-trip.
    params_before = {
        k: v.clone() for k, v in model.latent_world_model.state_dict().items()
    }
    path = tmp_path / "delta.pt"
    model.save(path)
    loaded = DeltaActorCritic.load(path)
    assert loaded.latent_world_model is not None
    for key, before in params_before.items():
        torch.testing.assert_close(
            loaded.latent_world_model.state_dict()[key], before
        )


# --- DeltaControllerConfig ---


def test_delta_controller_config_rejects_invalid_world_model() -> None:
    with pytest.raises(ValueError, match="world_model"):
        DeltaControllerConfig(world_model="simulation")


# --- DeltaController in none mode ---


def test_delta_controller_none_mode_has_no_simulator_or_estimator() -> None:
    controller = _make_controller("none")
    assert controller.simulator is None
    controller.initialize_for_object(_GEOMETRY)
    assert controller.estimator is None


def test_delta_controller_none_mode_forecast_returns_zeros() -> None:
    controller = _make_controller("none")
    controller.initialize_for_object(_GEOMETRY)
    result = controller._forecast(_make_forecast_context(), np.zeros(3))
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, _P, DenseTrajectory.NUM_FEATURES)
    assert np.all(result == 0)


# --- DeltaController in learned mode ---


def test_delta_controller_learned_mode_forecast_uses_world_model() -> None:
    model = _make_model(world_model="learned")
    controller = _make_controller("learned", model=model)
    controller.initialize_for_object(_GEOMETRY)
    result = controller._forecast(_make_forecast_context(), np.zeros(3))
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, _P, DenseTrajectory.NUM_FEATURES)


# --- parse_args ---


@pytest.mark.parametrize(
    "extra_args, expected",
    [
        ([], "physics"),
        (["--world-model", "none"], "none"),
        (["--world-model", "learned"], "learned"),
        (["--world-model", "physics"], "physics"),
    ],
)
def test_parse_args_world_model_flag(extra_args: list[str], expected: str) -> None:
    args = parse_args(extra_args)
    assert args.world_model == expected


def test_parse_args_forecast_workers_warns_for_non_physics() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parse_args(["--world-model", "none", "--forecast-workers", "2"])
    messages = [str(w.message) for w in caught]
    assert any("forecast-workers" in m or "forecast_workers" in m for m in messages)


def _scaled_motion(samples: int) -> np.ndarray:
    time = np.linspace(0.0, 0.8, samples)
    return np.column_stack((time, time**2, time / 2, time * 2, -time)).astype(np.float32)


def test_world_model_interpolates_scaled_channels_on_even_timestamps() -> None:
    trajectory = _scaled_motion(7)
    result = interpolate_model_trajectory(trajectory)
    grid = np.linspace(0.0, 0.8, 16)
    assert result.shape == (16, 5)
    np.testing.assert_allclose(result[:, 0], grid, atol=1e-7)
    np.testing.assert_allclose(
        result[:, 1], np.interp(grid, trajectory[:, 0], trajectory[:, 1]), atol=1e-7
    )
    np.testing.assert_array_equal(result[[0, -1]], trajectory[[0, -1]])


def test_attention_learned_model_forecasts_from_full_policy_trajectory() -> None:
    model = DeltaActorCritic(DeltaActorCriticConfig(
        hidden_dim=16, trajectory_feature_strategy="self-attention", world_model="learned"
    ))
    controller = DeltaController(model, DeltaControllerConfig(world_model="learned"))
    trajectory = _scaled_motion(23)
    context = replace(_make_forecast_context(), real_trajectory=trajectory)
    prediction = controller._forecast(context, np.zeros(3))
    assert prediction.shape == (1, 16, 5)
    assert context.real_trajectory.shape == (23, 5)
    controller.close()


def test_attention_learned_model_trains_on_unequal_history_lengths() -> None:
    model = DeltaActorCritic(DeltaActorCriticConfig(
        hidden_dim=16, trajectory_feature_strategy="self-attention", world_model="learned"
    ))
    examples = []
    for samples in (7, 23):
        next_context = replace(
            _make_forecast_context(), real_trajectory=_scaled_motion(samples + 2)
        )
        example = _make_example(terminated=False, next_context=next_context)
        example.truncated = True
        example.real_trajectory = _scaled_motion(samples)
        example.forecasts = np.zeros((1, 16, 5), np.float32)
        example.weights = np.ones(1)
        examples.append(example)
    batch = model._prepare_world_model_batch(examples)
    assert batch[3].shape == batch[4].shape == (2, 16, 5)
    np.testing.assert_allclose(batch[4][1], interpolate_model_trajectory(_scaled_motion(25)))
    metrics = model.train_step(examples, lambda e, d: e.forecasts)
    assert np.isfinite(metrics["world_model_loss"])
    assert examples[1].real_trajectory.shape == (23, 5)
