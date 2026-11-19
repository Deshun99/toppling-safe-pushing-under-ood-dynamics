"""Delta contracts: soft TD targets, input isolation, and fresh forecasts."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.agents.rl.models.delta_actor_critic import (
    DeltaActorCritic,
    DeltaActorCriticConfig,
    PreparedExample,
    TrainingExample,
    TrajectoryEncoder,
    TRAJECTORY_SCALES,
)
from src.agents.rl.models.sac import SAC, SACConfig
from src.simulation.execution.force_push import DenseTrajectory


def make_model(**settings):
    torch.set_num_threads(1)
    return DeltaActorCritic(
        DeltaActorCriticConfig(trajectory_points=4, hidden_dim=16, **settings)
    )


def make_example(target=3.0, reference=None):
    return TrainingExample(
        observation=np.array([0.8, 0.01, 0.2, 0.03, 0.06, 0.08, 0.2], np.float32),
        real_trajectory=np.zeros((4, 5), np.float32),
        delta=np.array([0.1, 0.05, -0.1], np.float32),
        forecasts=np.stack([np.zeros((4, 5)), np.ones((4, 5))]).astype(np.float32),
        weights=np.array([0.75, 0.25]),
        reward=target,
        terminated=True,
        truncated=False,
        reference_action=np.zeros(3, np.float32)
        if reference is None
        else np.asarray(reference, np.float32),
        metadata=object(),
    )


def test_features_remove_global_position_and_time_offset_but_keep_duration() -> None:
    trajectory = DenseTrajectory(
        t=[0.0, 0.2, 0.4],
        x=[5, 5.1, 5.4],
        psi=[0, 0.02, 0.04],
        xdot=[0, 0.5, 1],
        psidot=[0, 0.1, 0.2],
    )
    features = TrajectoryEncoder.extract_trajectory_features(trajectory, points=5)
    translated = DenseTrajectory(
        t=[9, 9.2, 9.4],
        x=[15, 15.1, 15.4],
        psi=trajectory.psi,
        xdot=trajectory.xdot,
        psidot=trajectory.psidot,
    )
    np.testing.assert_allclose(
        features,
        TrajectoryEncoder.extract_trajectory_features(translated, 5),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        features[-1] * TRAJECTORY_SCALES, [0.4, 0.4, 0.04, 1, 0.2]
    )
    assert features.shape == (5, 5) and features.dtype == np.float32


def test_features_repeat_single_sample() -> None:
    one = DenseTrajectory(t=[0], x=[10], psi=[0.1], xdot=[0.2], psidot=[0.3])
    encoded = TrajectoryEncoder.extract_trajectory_features(one, 4)
    np.testing.assert_array_equal(encoded[:, :2], np.zeros((4, 2)))
    assert np.all(encoded == encoded[0])


@pytest.mark.parametrize(
    "trajectory",
    [
        DenseTrajectory(),
        DenseTrajectory(t=[0, 0], x=[0, 1], psi=[0, 0], xdot=[0, 0], psidot=[0, 0]),
        DenseTrajectory(t=[0], x=[float("nan")], psi=[0], xdot=[0], psidot=[0]),
        DenseTrajectory(t=[0], x=[0, 1], psi=[0], xdot=[0], psidot=[0]),
    ],
)
def test_features_reject_invalid_records(trajectory: DenseTrajectory) -> None:
    with pytest.raises(ValueError, match="trajectory"):
        TrajectoryEncoder.extract_trajectory_features(trajectory)


def test_collection_reuses_one_actor_forward_and_local_rng():
    torch.manual_seed(123)
    global_state = torch.random.get_rng_state().clone()
    model = make_model()
    example = make_example()
    with patch.object(model.actor, "forward", wraps=model.actor.forward) as forward:
        distribution = model.distribution(example.observation, example.real_trajectory)
        first = distribution.sample(128)
        second = distribution.sample(128)
        assert forward.call_count == 1
    assert not np.array_equal(first, second)
    assert np.all(np.abs(first) <= np.array(model.config.delta_scale))
    assert distribution.mean.shape == (3,)
    torch.testing.assert_close(global_state, torch.random.get_rng_state())
    duplicate = make_model()
    np.testing.assert_array_equal(
        first,
        duplicate.distribution(example.observation, example.real_trajectory).sample(
            128
        ),
    )


def test_score_is_posterior_weighted_and_permutation_invariant():
    model, example = make_model(), make_example()
    deltas = np.stack([example.delta, -example.delta])
    forecasts = np.stack([example.forecasts, example.forecasts * 0.5])
    scores = model.score(
        example.observation, example.real_trajectory, deltas, forecasts, example.weights
    )
    shuffled = model.score(
        example.observation,
        example.real_trajectory,
        deltas,
        forecasts[:, ::-1].copy(),
        example.weights[::-1].copy(),
    )
    np.testing.assert_allclose(scores, shuffled)
    obs = torch.tensor(example.observation)[None].expand(2, -1)
    real = torch.tensor(example.real_trajectory)[None].expand(2, -1, -1)
    delta_tensor = torch.tensor(deltas, requires_grad=True)
    q1, q2 = model.critic(obs, real, delta_tensor, torch.tensor(forecasts))
    weights = torch.tensor(example.weights)
    expected = torch.minimum((q1 * weights).sum(-1), (q2 * weights).sum(-1))
    np.testing.assert_allclose(scores, expected.detach().numpy(), rtol=1e-6)
    expected.sum().backward()
    assert delta_tensor.grad is not None
    assert torch.isfinite(delta_tensor.grad).all() and delta_tensor.grad.abs().sum() > 0


def test_reparameterized_density_includes_squash_and_delta_scale():
    model, example = make_model(), make_example()
    obs = torch.tensor(example.observation)[None]
    real = torch.tensor(example.real_trajectory)[None]
    delta, log_prob = model.actor.rsample(obs, real, generator=model.rng)
    mean, log_std = model.actor(obs, real)
    unit_delta = delta / model.actor.delta_scale
    raw = torch.atanh(unit_delta)
    expected = (
        torch.distributions.Normal(mean, log_std.exp()).log_prob(raw)
        - torch.log1p(-unit_delta.square())
        - model.actor.delta_scale.log()
    ).sum(-1)
    torch.testing.assert_close(log_prob, expected)
    with torch.no_grad():
        model.actor.net.model[-1].bias[:3].fill_(100)
    _, saturated_density = model.actor.rsample(obs, real, generator=model.rng)
    assert torch.isfinite(saturated_density).all()
    saturated_density.mean().backward()
    assert all(
        torch.isfinite(parameter.grad).all() for parameter in model.actor.parameters()
    )


def test_fixed_sample_density_has_normal_score_even_when_tanh_saturates() -> None:
    model, example = make_model(), make_example()
    obs, real = model._prepare_context(example.observation, example.real_trajectory)
    mean = torch.tensor([[0.2, -0.3, 100.0]], requires_grad=True)
    log_std = torch.tensor([[0.0, -0.5, 0.5]], requires_grad=True)
    noise = torch.randn((1, 3), generator=torch.Generator().manual_seed(42))
    with patch.object(model.actor, "forward", return_value=(mean, log_std)):
        delta, pathwise, score = model.actor.sample_with_score(
            obs, real, generator=torch.Generator().manual_seed(42)
        )
    torch.testing.assert_close(pathwise, score)
    score_gradients = torch.autograd.grad(score.sum(), (mean, log_std))
    torch.testing.assert_close(score_gradients[0], noise / log_std.exp())
    torch.testing.assert_close(score_gradients[1], noise.square() - 1)
    assert delta[0, 2] == model.actor.delta_scale[2]
    assert all(torch.isfinite(value).all() for value in score_gradients)


@pytest.mark.parametrize("reference", [0.0, -1.0, 0.9])
def test_score_value_gradient_matches_complete_action_value(reference: float) -> None:
    model, example = make_model(), make_example()
    obs, real = model._prepare_context(example.observation, example.real_trajectory)
    mean = torch.zeros((1, 3), requires_grad=True)
    expanded = mean.expand(65536, -1)
    with patch.object(
        model.actor, "forward", return_value=(expanded, torch.zeros_like(expanded))
    ):
        delta, _, score = model.actor.sample_with_score(
            obs,
            real,
            generator=model.rng,
            reference_action=torch.full((3,), reference),
        )
    score_gradient = torch.autograd.grad(
        -(delta[:, 1].detach() * score).mean(), mean, retain_graph=True
    )[0]
    complete_gradient = torch.autograd.grad(-delta[:, 1].mean(), mean)[0]
    torch.testing.assert_close(score_gradient, complete_gradient, atol=0.006, rtol=0.02)


@pytest.mark.parametrize("forecast_gain,direct_gain", [(1.0, 0.0), (2.0, -1.0)])
def test_actor_follows_forecast_value_despite_wrong_direct_delta_gradient(
    forecast_gain: float, direct_gain: float
) -> None:
    model = make_model(entropy_coefficient=0.0, learning_rate=0.01)
    example = replace(make_example(), delta=np.zeros(3), forecasts=np.zeros((2, 4, 5)))
    with torch.no_grad():
        for parameter in model.actor.net.parameters():
            parameter.zero_()
    prepared = [model._prepare_example(example)] * 512

    def evaluate(
        items: list[PreparedExample], critic: object | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.stack(
            [
                forecast_gain * i.forecasts.mean() + direct_gain * i.delta[0, 1]
                for i in items
            ]
        )
        return values, values

    def forecast(record: TrainingExample, delta: np.ndarray) -> np.ndarray:
        return np.full_like(record.forecasts, delta[1])

    with patch.object(model, "_evaluate_batch", side_effect=evaluate):
        metrics = model._update_actor([example] * 512, prepared, forecast)
    assert model.distribution(example.observation, example.real_trajectory).mean[1] > 0
    assert metrics["actor_gradient_norm"] > 0


def test_actor_loss_detaches_values_and_uses_independent_replay_baseline() -> None:
    model = make_model(entropy_coefficient=0.2)
    actor_items, replay_items = [object()], [object()]
    values = torch.tensor([3.0, 5.0], requires_grad=True)
    baseline = torch.tensor([7.0, -2.0], requires_grad=True)
    log_probs = torch.tensor([0.1, 0.2], requires_grad=True)
    scores = torch.tensor([0.1, 0.2], requires_grad=True)
    with patch.object(
        model, "_evaluate_batch", side_effect=[(values, values), (baseline, baseline)]
    ) as evaluate:
        loss, _ = model._actor_loss(actor_items, replay_items, log_probs, scores)
    assert evaluate.call_args_list[0].args[0] is actor_items
    assert evaluate.call_args_list[1].args[0] is replay_items
    entropy_grad, score_grad, value_grad, baseline_grad = torch.autograd.grad(
        loss, (log_probs, scores, values, baseline), allow_unused=True
    )
    torch.testing.assert_close(entropy_grad, torch.full((2,), 0.1))
    torch.testing.assert_close(score_grad, -(values - baseline).detach() / 2)
    assert value_grad is None and baseline_grad is None


def test_zero_value_signal_with_disabled_entropy_keeps_actor_fixed() -> None:
    model, example = make_model(entropy_coefficient=0.0), make_example()
    prepared = [model._prepare_example(example)]
    parameters = [value.detach().clone() for value in model.actor.net.parameters()]
    with patch.object(
        model, "_evaluate_batch", return_value=(torch.zeros(1), torch.zeros(1))
    ):
        metrics = model._update_actor([example], prepared, lambda e, d: e.forecasts)
    assert metrics["actor_gradient_norm"] == 0
    for previous, current in zip(parameters, model.actor.net.parameters()):
        torch.testing.assert_close(previous, current, atol=0, rtol=0)


def test_immutable_controller_snapshots_do_not_produce_torch_write_warnings():
    import warnings

    model, example = make_model(), make_example()
    for array in (
        example.observation,
        example.real_trajectory,
        example.delta,
        example.forecasts,
        example.weights,
        example.reference_action,
    ):
        array.setflags(write=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.distribution(example.observation, example.real_trajectory)
        model.score(
            example.observation,
            example.real_trajectory,
            example.delta[None],
            example.forecasts[None],
            example.weights,
        )
        model.train_step([example], lambda record, delta: record.forecasts)
    assert not any("not writable" in str(warning.message) for warning in caught)


def test_terminal_td_loss_fits_posterior_average() -> None:
    model, example = make_model(), make_example(target=1)
    forward = model.critic.forward
    calls = 0

    def controlled(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            return torch.tensor([[0.0, 10.0]], requires_grad=True), torch.tensor(
                [[2.0, 12.0]], requires_grad=True
            )
        return forward(*args)

    with patch.object(model.critic, "forward", side_effect=controlled):
        metrics = model.train_step([example], lambda e, d: e.forecasts)
    # Weighted twin predictions are 2.5 and 4.5; the terminal target is reward 1.
    assert metrics["critic_loss"] == pytest.approx(0.5 * (1.5**2 + 3.5**2))


def test_training_updates_actor_and_critic_with_fresh_matching_forecasts() -> None:
    model, example = make_model(learning_rate=0.005), make_example()
    original_actor = [value.detach().clone() for value in model.actor.parameters()]
    score_inputs = (
        example.observation,
        example.real_trajectory,
        example.delta[None],
        example.forecasts[None],
        example.weights,
    )
    initial = model.score(*score_inputs)[0]
    sampled = []

    def forecast(record: TrainingExample, delta: np.ndarray) -> np.ndarray:
        assert record is example
        sampled.append(delta.copy())
        return np.full_like(record.forecasts, delta.sum())

    for _ in range(35):
        metrics = model.train_step([example], forecast)
        assert all(np.isfinite(value) for value in metrics.values())
    final = model.score(*score_inputs)[0]
    assert abs(final - example.reward) < abs(initial - example.reward)
    assert len(sampled) == 35 and not np.array_equal(sampled[0], sampled[-1])
    assert any(
        not torch.equal(before, after)
        for before, after in zip(original_actor, model.actor.parameters())
    )
    assert all(parameter.requires_grad for parameter in model.critic.parameters())
    assert all(parameter.grad is None for parameter in model.critic.parameters())


@pytest.mark.parametrize("reference", [-1.0, -0.95, 0.60000002, 0.95, 1.0])
def test_actor_and_collection_samples_respect_final_action_bounds(
    reference: float,
) -> None:
    model = make_model()
    example = make_example(reference=[reference] * 3)
    distribution = model.distribution(
        example.observation, example.real_trajectory, example.reference_action
    )
    samples = distribution.sample(1000)
    assert np.all(np.abs(samples.astype(np.float64) + example.reference_action) <= 1)
    obs, trajectory = model._prepare_context(
        example.observation, example.real_trajectory
    )
    delta, log_prob = model.actor.rsample(
        obs, trajectory, reference_action=torch.from_numpy(example.reference_action)
    )
    assert np.all(
        np.abs(delta.detach().numpy().astype(np.float64) + example.reference_action)
        <= 1
    )
    assert torch.isfinite(log_prob).all()


def test_float64_reference_rejects_invalid_stored_executed_action():
    model, example = make_model(), make_example()
    example.reference_action = np.full(3, 0.60000002, dtype=np.float64)
    example.delta = np.full(3, 0.4, dtype=np.float32)
    with pytest.raises(ValueError, match="valid executed final action"):
        model.train_step([example], lambda record, delta: record.forecasts)


@pytest.mark.parametrize("strategy", ["interpolate", "self-attention"])
def test_checkpoint_roundtrip_metadata_optimizer_and_sampling(
    tmp_path: Path,
    strategy: str,
) -> None:
    model = make_model(trajectory_feature_strategy=strategy)
    example = make_example()
    model.train_step([example], lambda record, delta: record.forecasts)
    model.training_metadata = {"action_ranges": {"force": [0.0, 15.0]}}
    path = tmp_path / "delta.pt"
    model.save(path)
    restored = DeltaActorCritic.load(path)
    assert restored.config == model.config and restored.num_updates == 1
    assert restored.training_metadata == model.training_metadata
    left = model.distribution(example.observation, example.real_trajectory)
    right = restored.distribution(example.observation, example.real_trajectory)
    np.testing.assert_array_equal(left.mean, right.mean)
    np.testing.assert_array_equal(left.sample(10), right.sample(10))
    score_inputs = (
        example.observation,
        example.real_trajectory,
        example.delta[None],
        example.forecasts[None],
        example.weights,
    )
    np.testing.assert_array_equal(
        model.score(*score_inputs), restored.score(*score_inputs)
    )
    SAC(SACConfig(7, 3, hidden_dim=8, buffer_size=1)).save(path)
    with pytest.raises(ValueError, match="vanilla SAC"):
        DeltaActorCritic.load(path)


@pytest.mark.parametrize(
    "setting",
    [
        {"gamma": -1},
        {"gamma": float("nan")},
        {"learning_rate": 0},
        {"entropy_coefficient": -0.1},
        {"delta_scale": (0, 0.1, 0.2)},
        {"trajectory_points": True},
        {"obs_dim": 8},
        {"gradient_steps": 0},
        {"batch_size": 0},
        {"seed": -1},
    ],
)
def test_invalid_config_fails_before_allocation(setting):
    with pytest.raises(ValueError):
        DeltaActorCriticConfig(**setting)


def test_rejects_misaligned_forecasts_weights_and_bad_training_data():
    model, example = make_model(), make_example()
    with pytest.raises(ValueError, match="weights"):
        model.score(
            example.observation,
            example.real_trajectory,
            example.delta[None],
            example.forecasts[None],
            [0, 0],
        )
    with pytest.raises(ValueError, match="forecasts"):
        model.score(
            example.observation,
            example.real_trajectory,
            example.delta[None],
            np.zeros((1, 2, 5, 5)),
            example.weights,
        )
    invalid = replace(example, reference_action=np.array([2, 0, 0]))
    with pytest.raises(ValueError, match="valid executed"):
        model.train_step([invalid], lambda *args: example.forecasts)
    with pytest.raises(ValueError, match="reward"):
        model.train_step(
            [replace(example, reward=float("nan"))],
            lambda *args: example.forecasts,
        )


@pytest.mark.parametrize("hidden_dim", [1, 16, 22])
def test_attention_strategy_encodes_time_and_backpropagates(hidden_dim: int) -> None:
    from src.agents.rl.models.nn_modules import AttentionTrajectoryEncoder

    config = DeltaActorCriticConfig(
        hidden_dim=hidden_dim,
        trajectory_points=4,
        trajectory_feature_strategy="self-attention",
    )
    encoder = TrajectoryEncoder(config)
    assert isinstance(encoder.net, AttentionTrajectoryEncoder)
    trajectory = torch.randn(2, 4, 5, requires_grad=True)
    with patch.object(encoder.net, "forward", wraps=encoder.net.forward) as forward:
        output = encoder(trajectory)
    physical, seconds, mask = forward.call_args.args
    assert mask is None
    torch.testing.assert_close(physical, trajectory[..., 1:])
    torch.testing.assert_close(seconds, trajectory[..., 0] * 0.5)
    assert output.shape == (2, encoder.output_dim)
    output[:, 0].sum().backward()
    assert torch.isfinite(trajectory.grad).all()
    assert trajectory.grad.abs().sum() > 0


def test_interpolation_strategy_does_not_construct_attention() -> None:
    with patch(
        "src.agents.rl.models.delta_actor_critic.AttentionTrajectoryEncoder"
    ) as attention:
        encoder = TrajectoryEncoder(DeltaActorCriticConfig())
        encoder(torch.zeros(2, 16, 5))
    attention.assert_not_called()


def test_attention_collection_preserves_rng_and_repeats_scores() -> None:
    model = make_model(trajectory_feature_strategy="self-attention")
    example = make_example()
    state = torch.random.get_rng_state().clone()
    first = model.distribution(example.observation, example.real_trajectory)
    second = model.distribution(example.observation, example.real_trajectory)
    np.testing.assert_array_equal(first.mean, second.mean)
    arguments = (
        example.observation,
        example.real_trajectory,
        example.delta[None],
        example.forecasts[None],
        example.weights,
    )
    np.testing.assert_array_equal(model.score(*arguments), model.score(*arguments))
    torch.testing.assert_close(state, torch.random.get_rng_state())


def test_attention_extracts_every_raw_sample_without_interpolation() -> None:
    trajectory = DenseTrajectory(
        t=[9, 9.01, 9.2],
        x=[5, 5.15, 5.2],
        psi=[0, 0.02, 0.01],
        xdot=[0, 2, 1],
        psidot=[0, 0.2, 0.1],
    )
    with patch(
        "numpy.interp", side_effect=AssertionError("attention must not interpolate")
    ):
        features = TrajectoryEncoder.extract_trajectory_features(
            trajectory, points=16, strategy="self-attention"
        )
    expected = trajectory.to_numpy()
    expected[:, :2] -= expected[0, :2].copy()
    np.testing.assert_allclose(features * TRAJECTORY_SCALES, expected, atol=1e-7)
    assert features.shape == (3, 5)
    assert features.dtype == np.float32


def make_variable_length_attention_example() -> (
    tuple[DeltaActorCritic, TrainingExample]
):
    model = make_model(trajectory_feature_strategy="self-attention")
    example = make_example()
    example.real_trajectory = np.ones((7, 5), dtype=np.float32)
    example.forecasts = (np.zeros((1, 5), np.float32), np.ones((9, 5), np.float32))
    return model, example


def test_attention_scores_candidates_with_different_trajectory_lengths() -> None:
    model, example = make_variable_length_attention_example()
    candidates = (
        example.forecasts,
        (np.ones((3, 5), np.float32), np.zeros((2, 5), np.float32)),
    )
    deltas = np.stack((example.delta, -example.delta))
    scores = model.score(
        example.observation,
        example.real_trajectory,
        deltas,
        candidates,
        example.weights,
    )
    for index in range(2):
        individual = model.score(
            example.observation,
            example.real_trajectory,
            deltas[index : index + 1],
            candidates[index : index + 1],
            example.weights,
        )
        np.testing.assert_allclose(scores[index], individual[0], atol=1e-6)


def test_attention_trains_and_restores_variable_length_forecasts(
    tmp_path: Path,
) -> None:
    model, example = make_variable_length_attention_example()
    metrics = model.train_step([example], lambda record, delta: record.forecasts)
    assert np.isfinite(list(metrics.values())).all()
    model.save(tmp_path / "attention.pt")
    restored = DeltaActorCritic.load(tmp_path / "attention.pt")
    arguments = (
        example.observation,
        example.real_trajectory,
        example.delta[None],
        (example.forecasts,),
        example.weights,
    )
    np.testing.assert_array_equal(model.score(*arguments), restored.score(*arguments))


def test_attention_forecast_padding_mask_preserves_each_prediction() -> None:
    model = make_model(trajectory_feature_strategy="self-attention")
    forecasts = (np.zeros((1, 5), np.float32), np.ones((6, 5), np.float32))
    padded, _, mask = model._prepare_forecasts(forecasts, [0.5, 0.5])
    encoder = model.critic.q1.encoder
    combined = encoder(padded[0], mask[0])
    for index, forecast in enumerate(forecasts):
        expected = encoder(torch.from_numpy(forecast)[None])
        torch.testing.assert_close(combined[index], expected[0], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("strategy", ["interpolate", "self-attention"])
def test_all_networks_share_one_critic_owned_encoder(strategy: str) -> None:
    model = make_model(trajectory_feature_strategy=strategy)
    assert model.actor.encoder is model.encoder
    assert model.critic.q1.encoder is model.encoder
    assert model.critic.q2.encoder is model.encoder
    actor_ids = {
        id(p) for group in model.actor_optimizer.param_groups for p in group["params"]
    }
    critic_ids = [
        id(p) for group in model.critic_optimizer.param_groups for p in group["params"]
    ]
    assert actor_ids.isdisjoint(critic_ids)
    assert len(critic_ids) == len(set(critic_ids))
    assert all(id(p) in critic_ids for p in model.encoder.parameters())


@pytest.mark.parametrize("strategy", ["interpolate", "self-attention"])
def test_critic_trains_shared_encoder_and_actor_keeps_it_fixed(strategy: str) -> None:
    model = make_model(trajectory_feature_strategy=strategy)
    example = make_example()
    example.real_trajectory[:] = 0.5
    prepared = [model._prepare_example(example)]
    before = [p.detach().clone() for p in model.encoder.parameters()]
    model._update_critic(prepared, [torch.tensor(example.reward)])
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.encoder.parameters())
    )
    before = [p.detach().clone() for p in model.encoder.parameters()]
    actor_before = [p.detach().clone() for p in model.actor.net.parameters()]
    model.critic.requires_grad_(False)
    try:
        model._update_actor([example], prepared, lambda record, delta: record.forecasts)
    finally:
        model.critic.requires_grad_(True)
    assert all(
        torch.equal(old, new) for old, new in zip(before, model.encoder.parameters())
    )
    assert any(
        not torch.equal(old, new)
        for old, new in zip(actor_before, model.actor.net.parameters())
    )
    assert all(p.grad is None for p in model.encoder.parameters())


def make_nonterminal_example(*, truncated: bool = False) -> TrainingExample:
    from types import SimpleNamespace

    example = make_example()
    context = SimpleNamespace(
        observation=example.observation.copy(),
        real_trajectory=np.ones((4, 5), np.float32),
        reference_action=np.full(3, 0.2, np.float64),
        weights=np.array([0.2, 0.8]),
    )
    return replace(example, terminated=False, truncated=truncated, next_context=context)


@pytest.mark.parametrize("truncated", [False, True])
def test_td_target_bootstraps_next_context_and_entropy(truncated: bool) -> None:
    model = make_model(entropy_coefficient=0.05)
    example = make_nonterminal_example(truncated=truncated)
    seen = []

    def forecast(record: TrainingExample, delta: np.ndarray) -> np.ndarray:
        seen.append(record.metadata)
        np.testing.assert_array_equal(
            record.reference_action, example.next_context.reference_action
        )
        return example.forecasts

    q_values = (torch.tensor([[10.0, 20.0]]), torch.tensor([[12.0, 18.0]]))
    with (
        patch.object(model.target_critic, "forward", return_value=q_values),
        patch.object(
            model.actor,
            "rsample",
            return_value=(torch.zeros(1, 3), torch.tensor([2.0])),
        ),
    ):
        target = model._compute_targets([example], forecast)[0]
    expected = example.reward + model.config.gamma * (
        min(0.2 * 10 + 0.8 * 20, 0.2 * 12 + 0.8 * 18) - 0.05 * 2
    )
    assert target.item() == pytest.approx(expected)
    assert seen == [example.next_context]
    assert not target.requires_grad


def test_td_terminal_target_never_queries_actor_simulator_or_target() -> None:
    model, example = make_model(), make_example()
    with patch.object(
        model.actor, "rsample", side_effect=AssertionError("terminal bootstrap")
    ):
        target = model._compute_targets(
            [example], lambda *args: pytest.fail("terminal simulation")
        )[0]
    assert target.item() == example.reward


@pytest.mark.parametrize("strategy", ["interpolate", "self-attention"])
def test_td_target_copy_is_separate_and_soft_updated(strategy: str) -> None:
    model = make_model(trajectory_feature_strategy=strategy, tau=0.1)
    example = make_example()
    assert model.target_critic.q1.encoder is model.target_critic.q2.encoder
    assert model.target_critic.q1.encoder is not model.encoder
    before = [p.clone() for p in model.target_critic.parameters()]
    model.train_step([example], lambda record, delta: record.forecasts)
    for old, target, online in zip(
        before, model.target_critic.parameters(), model.critic.parameters()
    ):
        torch.testing.assert_close(target, old.lerp(online.detach(), model.config.tau))
        assert not target.requires_grad and target.grad is None


@pytest.mark.parametrize("strategy", ["interpolate", "self-attention"])
def test_td_checkpoint_restores_targets_temperature_and_next_update(
    tmp_path: Path, strategy: str
) -> None:
    model = make_model(trajectory_feature_strategy=strategy)
    example = make_nonterminal_example()

    def forecast(record: TrainingExample, delta: np.ndarray) -> np.ndarray:
        return record.forecasts

    model.train_step([example], forecast)
    model.num_timesteps = 7
    model.save(tmp_path / "td.pt")
    restored = DeltaActorCritic.load(tmp_path / "td.pt")
    assert restored.num_timesteps == 7
    assert model.train_step([example], forecast) == restored.train_step(
        [example], forecast
    )
    for left, right in zip(
        model.target_critic.parameters(), restored.target_critic.parameters()
    ):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    torch.testing.assert_close(model.log_alpha, restored.log_alpha, atol=0, rtol=0)


@pytest.mark.parametrize("strategy", ["interpolate", "self-attention"])
def test_twin_encoder_reuse_preserves_predictions_and_gradients(strategy: str) -> None:
    model = make_model(trajectory_feature_strategy=strategy)
    obs = torch.randn(2, 7)
    real = torch.randn(2, 4, 5)
    deltas = torch.randn(2, 3, requires_grad=True)
    forecasts = torch.randn(2, 3, 4, 5, requires_grad=True)
    inputs = (
        obs.repeat_interleave(3, 0),
        real.repeat_interleave(3, 0),
        deltas.repeat_interleave(3, 0),
        forecasts.flatten(0, 1),
    )
    expected = tuple(
        head(*inputs).reshape(2, 3) for head in (model.critic.q1, model.critic.q2)
    )
    parameters = (*model.critic.parameters(), deltas, forecasts)
    expected_gradients = torch.autograd.grad(sum(q.sum() for q in expected), parameters)
    with patch.object(model.encoder, "forward", wraps=model.encoder.forward) as encode:
        actual = model.critic(obs, real, deltas, forecasts)
    assert encode.call_count == 2
    gradients = torch.autograd.grad(sum(q.sum() for q in actual), parameters)
    for result, reference in zip(
        (*actual, *gradients), (*expected, *expected_gradients)
    ):
        torch.testing.assert_close(result, reference, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("strategy", ["interpolate", "self-attention"])
def test_replay_batch_matches_individual_values_and_gradients(strategy: str) -> None:
    model = make_model(trajectory_feature_strategy=strategy)
    examples = [make_example(), make_example()]
    examples[1].forecasts = np.ones((3, 4, 5), np.float32)
    examples[1].weights = np.array([0.2, 0.3, 0.5])
    if strategy == "self-attention":
        examples[1].real_trajectory = np.ones((7, 5), np.float32)
        examples[1].forecasts = tuple(np.ones((n, 5), np.float32) for n in (2, 5, 8))
    prepared = [model._prepare_example(example) for example in examples]
    individual = [model._evaluate_batch([example]) for example in prepared]
    expected = tuple(torch.cat(values) for values in zip(*individual))
    parameters = tuple(model.critic.parameters())
    expected_gradients = torch.autograd.grad(sum(q.sum() for q in expected), parameters)
    actual = model._evaluate_batch(prepared)
    gradients = torch.autograd.grad(sum(q.sum() for q in actual), parameters)
    for result, reference in zip(
        (*actual, *gradients), (*expected, *expected_gradients)
    ):
        torch.testing.assert_close(result, reference, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("normalized_log_prob", [-2.0, 5.0])
def test_temperature_update_is_independent_of_feasible_interval_width(
    normalized_log_prob: float,
) -> None:
    temperatures = []
    for log_scale in (0.0, -3.0, -7.0):
        model = make_model()
        model._update_temperature(
            torch.tensor([normalized_log_prob - log_scale]), torch.tensor([log_scale])
        )
        expected_gradient = 3.0 - normalized_log_prob
        assert model.log_alpha.grad.item() == pytest.approx(expected_gradient)
        temperatures.append(model.alpha.item())
    np.testing.assert_allclose(temperatures, temperatures[0])
    if normalized_log_prob < 3.0:
        assert temperatures[0] < 1.0  # Entropy exceeds the target of -3.
    else:
        assert temperatures[0] > 1.0


def test_logged_normalized_entropy_removes_feasible_interval_scale() -> None:
    model, example = make_model(), make_example()
    metrics = model.train_step([example], lambda record, delta: record.forecasts)
    log_scale = np.log(model.config.delta_scale).sum()
    assert metrics["normalized_entropy"] == pytest.approx(
        metrics["entropy"] - log_scale, abs=1e-6
    )


def test_actor_std_metrics_are_per_coordinate_and_do_not_change_rng() -> None:
    model = make_model()
    prepared = [model._prepare_example(make_example()) for _ in range(2)]
    values = torch.tensor([[model.actor.MIN_LOG_STD, 0.0, -1.0], [0.0, 1.0, 0.0]])
    before = model.rng.get_state().clone()
    responses = [(torch.zeros(1, 3), row[None]) for row in values]
    with patch.object(model.actor, "forward", side_effect=responses):
        metrics = model._measure_actor_std(prepared)
    assert torch.equal(before, model.rng.get_state())
    assert len(metrics) == 21
    for index, name in enumerate(("height", "force", "duration")):
        std = values[:, index].exp().numpy()
        for label, q in (
            ("min", 0.0),
            ("p10", 0.1),
            ("median", 0.5),
            ("p90", 0.9),
            ("max", 1.0),
        ):
            assert metrics[f"actor_std_{name}_{label}"] == pytest.approx(
                np.quantile(std, q)
            )
        assert metrics[f"actor_std_{name}_mean"] == pytest.approx(std.mean())
    assert metrics["actor_std_height_floor_fraction"] == 0.5
    assert metrics["actor_std_force_floor_fraction"] == 0.0
    assert all(parameter.grad is None for parameter in model.actor.parameters())


def test_training_logs_actor_std_distribution() -> None:
    model, example = make_model(), make_example()
    metrics = model.train_step([example], lambda record, delta: record.forecasts)
    for name in ("height", "force", "duration"):
        prefix = f"actor_std_{name}"
        assert np.exp(model.actor.MIN_LOG_STD) <= metrics[f"{prefix}_mean"]
        assert metrics[f"{prefix}_mean"] <= np.exp(model.actor.MAX_LOG_STD)
        assert (
            metrics[f"{prefix}_min"]
            <= metrics[f"{prefix}_median"]
            <= metrics[f"{prefix}_max"]
        )
