import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from typing import Any

from numpy.typing import ArrayLike, NDArray


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.model = self._build_mlp(input_dim, output_dim, hidden_dim)

    def _build_mlp(self, input_dim: int, output_dim: int, hidden_dim: int):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)


class AttentionTrajectoryEncoder(nn.Module):
    """Encode physical samples with relative timestamps measured in seconds."""

    TIME_EMBEDDING_FREQUENCIES = (2.0, 10.0)
    TIME_SCALE = 0.1  # seconds

    def __init__(
        self,
        feature_dim: int = 4,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or d_model < 1 or nhead < 1 or num_layers < 1:
            raise ValueError("encoder dimensions and layer count must be positive")
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.feature_dim = feature_dim
        self.register_buffer(
            "frequencies", torch.tensor(self.TIME_EMBEDDING_FREQUENCIES)
        )
        input_dim = feature_dim + 1 + 2 * len(self.TIME_EMBEDDING_FREQUENCIES)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.transformer_layers = self._build_transformer_layers(
            d_model, nhead, num_layers, dropout
        )
        self.output_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _build_transformer_layers(
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
    ) -> nn.ModuleList:
        # Construct independently initialized layers rather than cloned weights.
        return nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model,
                    nhead,
                    dim_feedforward=4 * d_model,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map [B,N,D] samples and [B,N] times to [B,d_model].

        Boolean mask entries are True for padding. Masked values may be nonfinite;
        each trajectory must contain at least one finite, unmasked sample.
        """
        padding_mask = self._validate_inputs(x, timestamps, padding_mask)
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0)
        timestamps = timestamps.masked_fill(padding_mask, 0)
        tau = timestamps / self.TIME_SCALE
        angles = tau.unsqueeze(-1) * self.frequencies
        features = torch.cat((x, tau.unsqueeze(-1), angles.sin(), angles.cos()), dim=-1)
        h = self.input_projection(features)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        h = torch.cat((cls, h), dim=1)
        cls_mask = padding_mask.new_zeros((x.shape[0], 1))
        attention_mask = torch.cat((cls_mask, padding_mask), dim=1)
        for layer in self.transformer_layers:
            h = layer(h, src_key_padding_mask=attention_mask)
        return self.output_norm(h[:, 0])

    def _validate_inputs(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.feature_dim:
            raise ValueError("x must have shape [B, N, feature_dim]")
        if timestamps.shape != x.shape[:2]:
            raise ValueError("timestamps must have shape [B, N]")
        if x.device != timestamps.device or x.dtype != timestamps.dtype:
            raise ValueError("x and timestamps must share device and dtype")
        if padding_mask is None:
            padding_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        if padding_mask.shape != x.shape[:2] or padding_mask.dtype != torch.bool:
            raise ValueError("padding_mask must be boolean with shape [B, N]")
        if padding_mask.device != x.device:
            raise ValueError("padding_mask must share the input device")
        valid = ~padding_mask
        if not valid.any(dim=1).all():
            raise ValueError("Each trajectory must contain at least one valid point")
        if (
            not torch.isfinite(x[valid]).all()
            or not torch.isfinite(timestamps[valid]).all()
        ):
            raise ValueError("unmasked trajectory samples must be finite")
        return padding_mask


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = MLP(obs_dim, 2 * action_dim, hidden_dim)

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = log_std.clamp(-20, 2)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        raw = mean if deterministic else distribution.rsample()
        action = raw.tanh()
        # Stable log(1 - tanh(raw)^2), including saturated actions.
        correction = 2 * (np.log(2) - raw - F.softplus(-2 * raw))
        log_prob = (distribution.log_prob(raw) - correction).sum(-1, keepdim=True)
        return action, log_prob


class TwinCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.q1 = MLP(obs_dim + action_dim, 1, hidden_dim)
        self.q2 = MLP(obs_dim + action_dim, 1, hidden_dim)

    def forward(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat((obs, actions), dim=-1)
        return self.q1(inputs), self.q2(inputs)


class ObservationNormalizer:
    """
    Normalizes observations to have zero mean and unit variance.
    """

    def __init__(self, size: int) -> None:
        self.mean = np.zeros(size, dtype=np.float64)
        self.var = np.ones(size, dtype=np.float64)
        self.count = 1e-4

    def update(self, observations: ArrayLike) -> None:
        batch = np.asarray(observations, dtype=np.float64).reshape(-1, len(self.mean))
        delta = batch.mean(axis=0) - self.mean
        total = self.count + len(batch)
        self.var = (
            self.var * self.count
            + batch.var(axis=0) * len(batch)
            + delta**2 * self.count * len(batch) / total
        ) / total
        self.mean += delta * len(batch) / total
        self.count = total

    def __call__(self, observations: ArrayLike) -> NDArray[np.float32]:
        return np.clip(
            (observations - self.mean) / np.sqrt(self.var + 1e-8), -10, 10
        ).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return dict(
            mean=torch.tensor(self.mean), var=torch.tensor(self.var), count=self.count
        )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean = state["mean"].cpu().numpy().copy()
        self.var = state["var"].cpu().numpy().copy()
        self.count = state["count"]


class ReplayBuffer:
    """
    Stores transitions for SAC training. Samples are drawn with replacement.
    """

    def __init__(
        self, capacity: int, obs_dim: int, action_dim: int, seed: int = 0
    ) -> None:
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity = capacity
        self.position = self.size = 0
        self.rng = np.random.default_rng(seed)
        self.observations = np.empty((capacity, obs_dim), dtype=np.float32)
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.terminated = np.empty((capacity, 1), dtype=np.float32)

    def add(
        self,
        obs: ArrayLike,
        actions: ArrayLike,
        rewards: ArrayLike,
        next_obs: ArrayLike,
        terminated: ArrayLike,
    ) -> None:
        for observation, action, reward, next_observation, terminal in zip(
            np.atleast_2d(obs),
            np.atleast_2d(actions),
            np.atleast_1d(rewards),
            np.atleast_2d(next_obs),
            np.atleast_1d(terminated),
        ):
            i = self.position
            self.observations[i], self.actions[i] = observation, action
            self.rewards[i] = reward
            self.next_observations[i] = next_observation
            self.terminated[i] = terminal
            self.position = (i + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        normalizer: ObservationNormalizer,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        indices = self.rng.integers(self.size, size=batch_size)
        arrays = (
            normalizer(self.observations[indices]),
            self.actions[indices],
            self.rewards[indices],
            normalizer(self.next_observations[indices]),
            self.terminated[indices],
        )
        return tuple(torch.as_tensor(a, device=device) for a in arrays)
