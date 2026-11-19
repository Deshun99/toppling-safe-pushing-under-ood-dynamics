"""
Native PyTorch SAC for continuous, normalized [-1, 1] actions.

Replay stores raw observations; normalization uses the current running moments
at sampling time. Only true terminations suppress Bellman bootstrapping.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch.nn import functional as F
from src.agents.rl.models.nn_modules import (
    Actor,
    ObservationNormalizer,
    TwinCritic,
    ReplayBuffer,
)


@dataclass
class SACConfig:
    obs_dim: int
    action_dim: int
    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 4
    buffer_size: int = 1_000_000
    seed: int = 42


class SAC:
    def __init__(
        self,
        config: SACConfig,
        device: str | torch.device = "cpu",
        *,
        inference_only: bool = False,
    ) -> None:
        """Build SAC; inference_only omits replay but keeps networks and optimizers."""
        self._validate_args(config, inference_only=inference_only)
        self.config = config
        self.device = torch.device(device)
        torch.manual_seed(config.seed)
        self._create_networks_and_optimizers()
        self.normalizer = ObservationNormalizer(config.obs_dim)
        self.replay: ReplayBuffer | None = (
            None
            if inference_only
            else ReplayBuffer(
                config.buffer_size, config.obs_dim, config.action_dim, config.seed
            )
        )
        self.num_timesteps = 0
        self.training_metadata: dict[str, Any] | None = None

    def _create_networks_and_optimizers(self) -> None:
        config = self.config
        self.actor = Actor(config.obs_dim, config.action_dim, config.hidden_dim).to(
            self.device
        )
        self.critic = TwinCritic(
            config.obs_dim, config.action_dim, config.hidden_dim
        ).to(self.device)
        self.target_critic = deepcopy(self.critic).requires_grad_(False)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.learning_rate
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=config.learning_rate
        )

    def _validate_args(self, config: SACConfig, *, inference_only: bool) -> None:
        """Validate before allocating memory or changing random state.

        Batch sampling uses replacement; batch size may exceed replay capacity.
        """
        if not isinstance(config, SACConfig):
            raise TypeError("config must be a SACConfig instance")
        if not isinstance(inference_only, bool):
            raise TypeError("inference_only must be a bool")
        for name in (
            "obs_dim",
            "action_dim",
            "hidden_dim",
            "batch_size",
            "buffer_size",
        ):
            value = getattr(config, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self._validate_optimization_settings(config)
        # Torch seeds must fit uint64; NumPy's replay RNG also requires nonnegative seeds.
        max_seed = int(np.iinfo(np.uint64).max)
        if (
            isinstance(config.seed, bool)
            or not isinstance(config.seed, Integral)
            or not 0 <= config.seed <= max_seed
        ):
            raise ValueError(f"seed must be an integer within [0, {max_seed}]")

    @staticmethod
    def _validate_optimization_settings(config: SACConfig) -> None:
        learning_rate = config.learning_rate
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, Real)
            or not np.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError("learning_rate must be positive and finite")
        for name in ("gamma", "tau"):
            value = getattr(config, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be finite and within [0, 1]")

    @torch.no_grad()
    def predict(
        self, observations: ArrayLike, deterministic: bool = True
    ) -> NDArray[np.float32]:
        obs = torch.as_tensor(
            self.normalizer(np.asarray(observations)), device=self.device
        )
        return self.actor(obs, deterministic)[0].cpu().numpy()

    def update(self) -> dict[str, float]:
        """Perform one SAC update; inference-only instances cannot learn."""
        if self.replay is None:
            raise RuntimeError(
                "Cannot update an inference-only SAC model; reload with inference_only=False."
            )
        obs, actions, rewards, next_obs, terminated = self.replay.sample(
            self.config.batch_size, self.normalizer, self.device
        )
        alpha = self.log_alpha.exp().detach()
        target = self._compute_target(rewards, next_obs, terminated, alpha)
        critic_loss = self._update_critic(obs, actions, target)
        actor_loss, log_prob = self._update_actor(obs, alpha)
        alpha_loss = self._update_temperature(log_prob)
        self._update_target_critic()
        return dict(
            critic_loss=critic_loss.item(),
            actor_loss=actor_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=self.log_alpha.exp().item(),
        )

    @torch.no_grad()
    def _compute_target(
        self,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        terminated: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        next_actions, next_log_prob = self.actor(next_obs)
        q1, q2 = self.target_critic(next_obs, next_actions)
        return rewards + self.config.gamma * (1 - terminated) * (
            torch.minimum(q1, q2) - alpha * next_log_prob
        )

    def _update_critic(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        q1, q2 = self.critic(obs, actions)
        critic_loss = 0.5 * (F.mse_loss(q1, target) + F.mse_loss(q2, target))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss

    def _update_actor(
        self,
        obs: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.critic.requires_grad_(False)
        sampled_actions, log_prob = self.actor(obs)
        q1, q2 = self.critic(obs, sampled_actions)
        # Actor's objective:
        # maximize [alpha * entropy + min(Q1, Q2)] = minimize [alpha * log_prob - min(Q1, Q2)]
        actor_loss = (alpha * log_prob - torch.minimum(q1, q2)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic.requires_grad_(True)
        return actor_loss, log_prob

    def _update_temperature(self, log_prob: torch.Tensor) -> torch.Tensor:
        # Target entropy is -action_dim; optimize log temperature.
        alpha_loss = -(
            self.log_alpha * (log_prob.detach() - self.config.action_dim)
        ).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        return alpha_loss

    def _update_target_critic(self) -> None:
        """
        Slow moving average of critic parameters for stable bootstrapping of target Q-values.
        """
        with torch.no_grad():
            for target_param, param in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_param.lerp_(param, self.config.tau)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                format_version=1,
                config=asdict(self.config),
                actor=self.actor.state_dict(),
                critic=self.critic.state_dict(),
                target_critic=self.target_critic.state_dict(),
                log_alpha=self.log_alpha.detach(),
                actor_optimizer=self.actor_optimizer.state_dict(),
                critic_optimizer=self.critic_optimizer.state_dict(),
                alpha_optimizer=self.alpha_optimizer.state_dict(),
                normalizer=self.normalizer.state_dict(),
                num_timesteps=self.num_timesteps,
                training_metadata=self.training_metadata,
            ),
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
        lr_scale: float = 1.0,
        buffer_size: int | None = None,
        seed: int | None = None,
        *,
        inference_only: bool = False,
    ) -> Self:
        """Restore networks and optimizers, with empty replay unless inference-only."""
        if Path(path).suffix != ".pt":
            raise ValueError(
                "Use a native .pt checkpoint; legacy zip checkpoints require retraining"
            )
        state = torch.load(path, map_location=device, weights_only=True)
        if state["format_version"] != 1:
            raise ValueError("Unsupported SAC checkpoint version")
        config = SACConfig(**state["config"])
        config.learning_rate *= lr_scale
        if buffer_size is not None:
            config.buffer_size = buffer_size
        if seed is not None:
            config.seed = seed
        model = cls(config, device, inference_only=inference_only)
        model._restore_checkpoint(state)
        return model

    def _restore_checkpoint(self, state: dict[str, Any]) -> None:
        for name in ("actor", "critic", "target_critic"):
            getattr(self, name).load_state_dict(state[name])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"])
        for name in ("actor_optimizer", "critic_optimizer", "alpha_optimizer"):
            optimizer = getattr(self, name)
            optimizer.load_state_dict(state[name])
            for group in optimizer.param_groups:
                group["lr"] = self.config.learning_rate
        self.normalizer.load_state_dict(state["normalizer"])
        self.num_timesteps = state["num_timesteps"]
        # Old checkpoints have unknown embedded metadata, not an implied empty split.
        self.training_metadata = state.get("training_metadata")
        # Training loads start with empty replay; inference-only loads omit it.
