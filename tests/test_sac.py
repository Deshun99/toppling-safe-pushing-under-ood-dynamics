import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from src.agents.rl.models.sac import (
    SAC,
    Actor,
    ObservationNormalizer,
    ReplayBuffer,
    SACConfig,
)

from src.agents.rl.training.train_sac import _replay_next_observations


def make_model() -> SAC:
    torch.set_num_threads(1)
    model = SAC(SACConfig(4, 2, hidden_dim=16, buffer_size=32, batch_size=8))
    rng = np.random.default_rng(7)
    obs = rng.normal(size=(16, 4)).astype(np.float32)
    model.normalizer.update(obs)
    model.replay.add(
        obs,
        rng.uniform(-1, 1, (16, 2)),
        rng.normal(size=16),
        obs + 0.1,
        np.arange(16) % 2,
    )
    return model


class OneStepEnv:
    """Minimal environment exposing the evaluator's metrics interface."""

    def __init__(self) -> None:
        from src.simulation.environment import ActionRanges
        from src.simulation.environment.reward import RewardModel
        from src.simulation.modelling.object import CuboidSpec

        self.cuboid = SimpleNamespace(spec=CuboidSpec(0.08, 0.08, 0.2, 1.0, 0.3, 0.08))
        self.action_ranges = ActionRanges()
        self.reward = RewardModel()
        self.target_distance = 1.0
        self.epsilon = self.psi_succ_tol = self.v_succ_tol = 0.01
        self.T_max = 1

    def reset(
        self, seed: int | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        return np.zeros(4, dtype=np.float32), {}

    def step(
        self, action: ArrayLike
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        return (
            np.ones(4, dtype=np.float32),
            1.0,
            False,
            True,
            dict(success=True, toppled=False),
        )


class TestSAC(unittest.TestCase):
    def test_normalizer_matches_combined_moments(self) -> None:
        rng = np.random.default_rng(1)
        obs = rng.normal(size=(20, 4))
        a, b = ObservationNormalizer(4), ObservationNormalizer(4)
        a.update(obs)
        b.update(obs[:7])
        b.update(obs[7:])
        np.testing.assert_allclose(a.mean, b.mean)
        np.testing.assert_allclose(a.var, b.var)
        assert a.count == b.count
        assert np.max(np.abs(a(obs * 1e8))) <= 10

    def test_actor_log_density_and_saturation(self) -> None:
        actor = Actor(4, 2, 16)
        obs = torch.zeros((8, 4))
        actions, log_prob = actor(obs)
        mean, log_std = actor.net(obs).chunk(2, -1)
        raw = torch.atanh(actions)
        expected = (
            torch.distributions.Normal(mean, log_std.clamp(-20, 2).exp()).log_prob(raw)
            - torch.log1p(-actions.square())
        ).sum(-1, keepdim=True)
        torch.testing.assert_close(log_prob, expected)
        assert actions.shape == (8, 2)
        with torch.no_grad():
            actor.net.model[-1].bias[:2].fill_(100)
        actions, log_prob = actor(obs)
        assert torch.isfinite(log_prob).all()
        log_prob.mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in actor.parameters())

    def test_update_changes_networks_temperature_and_polyak_target(self) -> None:
        model = make_model()
        actor_before = [p.detach().clone() for p in model.actor.parameters()]
        critic_before = [p.detach().clone() for p in model.critic.parameters()]
        target_before = [p.detach().clone() for p in model.target_critic.parameters()]
        losses = model.update()
        assert all(np.isfinite(v) for v in losses.values())
        assert any(
            not torch.equal(a, b)
            for a, b in zip(actor_before, model.actor.parameters())
        )
        assert any(
            not torch.equal(a, b)
            for a, b in zip(critic_before, model.critic.parameters())
        )
        assert model.log_alpha.item() != 0
        for before, target, current in zip(
            target_before, model.target_critic.parameters(), model.critic.parameters()
        ):
            torch.testing.assert_close(target, before.lerp(current, model.config.tau))
            assert target.grad is None

    def test_replay_wraparound_and_vector_terminal_observations(self) -> None:
        replay = ReplayBuffer(2, 1, 1)
        nxt = np.array([[100.0], [200.0], [3.0]], dtype=np.float32)
        terminated = np.array([True, False, False])
        truncated = np.array([False, True, False])
        final = _replay_next_observations(
            nxt,
            terminated,
            truncated,
            {"final_observation": [np.array([1.0]), np.array([2.0]), None]},
        )
        np.testing.assert_array_equal(final[:, 0], [1, 2, 3])
        replay.add([[0], [1], [2]], [[0], [0], [0]], [0, 0, 0], final, terminated)
        assert replay.size == 2
        # Truncation retains its final observation and bootstraps (terminated=0).
        assert replay.next_observations[1, 0] == 2
        assert replay.terminated[1, 0] == 0
        assert replay.next_observations[0, 0] == 3
        assert nxt[0, 0] == 100

    def test_checkpoint_roundtrip_empty_replay_and_scaled_optimizer_lr(self) -> None:
        model = make_model()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        tmp_path = Path(directory.name)
        model.update()
        path = tmp_path / "model.pt"
        model.save(path)
        loaded = SAC.load(path, lr_scale=0.5, seed=9)
        obs = np.ones((2, 4), dtype=np.float32)
        np.testing.assert_array_equal(model.predict(obs), loaded.predict(obs))
        assert loaded.normalizer.count == model.normalizer.count
        assert loaded.replay.size == 0
        torch.testing.assert_close(model.log_alpha, loaded.log_alpha)
        for optimizer in (
            loaded.actor_optimizer,
            loaded.critic_optimizer,
            loaded.alpha_optimizer,
        ):
            assert optimizer.param_groups[0]["lr"] == model.config.learning_rate * 0.5
        with self.assertRaisesRegex(ValueError, "retraining"):
            SAC.load(tmp_path / "legacy.zip")

        # Old native checkpoints have no embedded split information.
        state = torch.load(path, weights_only=True)
        state.pop("training_metadata")
        torch.save(state, path)
        legacy = SAC.load(path, inference_only=True)
        assert legacy.training_metadata is None
        np.testing.assert_array_equal(model.predict(obs), legacy.predict(obs))

    def test_fine_tuning_updates_final_transition_and_freezes_stats(self) -> None:
        from src.evaluation.rl import EvaluationMetrics, Evaluator

        model = make_model()
        count = model.normalizer.count
        before = model.num_timesteps
        result = Evaluator(OneStepEnv(), [42, None, None]).evaluate(
            model, fine_tune=True
        )
        assert len(result.episodes) == 3
        assert all(episode.outcome == "success" for episode in result.episodes)
        progress = EvaluationMetrics(result.episodes).summarize_progress()
        assert progress.interactions_to_first_success == 1
        assert model.num_timesteps == before + 3
        assert model.normalizer.count == count
        assert model.replay.size == 19

    def test_bellman_target_bootstraps_truncations_only(self) -> None:
        from unittest.mock import patch

        from torch.nn import functional as F

        model = make_model()
        obs = torch.zeros(2, 4)
        actions = torch.zeros(2, 2)
        batch = (obs, actions, torch.ones(2, 1), obs, torch.tensor([[0.0], [1.0]]))
        actor_forward = model.actor.forward
        calls = 0

        def actor(observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal calls
            calls += 1
            if calls == 1:  # target actions: entropy contribution is zero
                return actions, torch.zeros(2, 1)
            return actor_forward(observations)

        with (
            patch.object(model.replay, "sample", return_value=batch),
            patch.object(model.actor, "forward", side_effect=actor),
            patch.object(
                model.target_critic,
                "forward",
                return_value=(torch.full((2, 1), 2.0), torch.full((2, 1), 4.0)),
            ),
            patch("src.agents.rl.models.sac.F.mse_loss", wraps=F.mse_loss) as loss,
        ):
            model.update()
            expected = torch.tensor([[1 + model.config.gamma * 2], [1.0]])
            for call in loss.call_args_list:
                torch.testing.assert_close(call.args[1], expected)

    def test_inference_loading_skips_replay_and_preserves_predictions(self) -> None:
        from unittest.mock import patch

        model = make_model()
        model.update()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "model.pt"
        model.save(path)
        with patch(
            "src.agents.rl.models.sac.ReplayBuffer",
            side_effect=AssertionError("Inference must not allocate replay"),
        ):
            loaded = SAC.load(path, inference_only=True)
        assert loaded.replay is None
        assert loaded.config.buffer_size == model.config.buffer_size
        obs = np.ones((2, model.config.obs_dim), dtype=np.float32)
        np.testing.assert_array_equal(model.predict(obs), loaded.predict(obs))
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            loaded.update()

        # Inference mode is a loading choice, not a change to checkpoint format.
        loaded.save(path)
        restored = SAC.load(path)
        assert restored.replay is not None
        assert restored.replay.size == 0
        assert restored.replay.capacity == model.config.buffer_size
        np.testing.assert_array_equal(model.predict(obs), restored.predict(obs))

    def test_invalid_configuration_is_rejected_before_allocation(self) -> None:
        from dataclasses import replace
        from unittest.mock import patch

        config = SACConfig(4, 2, hidden_dim=16, buffer_size=32, batch_size=8)
        invalid_values: dict[str, tuple[Any, ...]] = {
            "obs_dim": (0, -1, 2.5, True),
            "action_dim": (0, -1, 2.5, True),
            "hidden_dim": (0, -1, 2.5, True),
            "batch_size": (0, -1, 2.5, True),
            "buffer_size": (0, -1, 2.5, True),
            "learning_rate": (0, -0.1, float("nan"), float("inf"), "0.1", True),
            "gamma": (-0.1, 1.1, float("nan"), float("inf"), "0.9", True),
            "tau": (-0.1, 1.1, float("nan"), float("inf"), "0.1", True),
            "seed": (-1, int(np.iinfo(np.uint64).max) + 1, 1.5, True),
        }
        for field, values in invalid_values.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    with (
                        patch("src.agents.rl.models.sac.Actor") as actor,
                        patch("src.agents.rl.models.sac.ReplayBuffer") as replay,
                        patch("src.agents.rl.models.sac.torch.manual_seed") as seed,
                        self.assertRaisesRegex(ValueError, field),
                    ):
                        SAC(replace(config, **{field: value}))
                    actor.assert_not_called()
                    replay.assert_not_called()
                    seed.assert_not_called()
        with self.assertRaisesRegex(TypeError, "config"):
            SAC(None)
        with self.assertRaisesRegex(TypeError, "inference_only"):
            SAC(config, inference_only="yes")
        with self.assertRaisesRegex(ValueError, "buffer_size"):
            SAC(replace(config, buffer_size=0), inference_only=True)

    def test_configuration_allows_endpoints_and_sampling_with_replacement(self) -> None:
        for gamma, tau in ((0.0, 0.0), (1.0, 1.0)):
            with self.subTest(gamma=gamma, tau=tau):
                model = SAC(
                    SACConfig(
                        4,
                        2,
                        hidden_dim=16,
                        gamma=gamma,
                        tau=tau,
                        buffer_size=1,
                        batch_size=8,
                    )
                )
                model.replay.add(np.zeros(4), np.zeros(2), 0.0, np.ones(4), True)
                losses = model.update()
                assert all(np.isfinite(value) for value in losses.values())
