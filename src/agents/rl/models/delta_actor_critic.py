"""Delta actor with one-step soft TD trajectory critics.

The actor and online critics share an encoder trained by critic loss. Target
critics have a separate, slowly updated copy. Forecasts condition value estimates;
a likelihood-ratio actor gradient includes their detached PyBullet outcomes.
Simulator ranking collects experience; the TD continuation samples the actor
directly, as in SAC.
"""

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from src.agents.rl.models.nn_modules import AttentionTrajectoryEncoder, MLP
from src.agents.rl.models.latent_world_model import interpolate_model_trajectory
from src.agents.rl.training.delta_utils import compute_delta_bounds
from src.simulation.execution.force_push import DenseTrajectory

CHECKPOINT_FORMAT = "delta-td-trajectory-v1"

TRAJECTORY_SCALES = np.array(
    [
        0.5,  # Elapsed time (s).
        0.25,  # Relative displacement along the push axis (m).
        0.2,  # Pitch angle (rad).
        1.0,  # Linear velocity along the push axis (m/s).
        2.0,  # Angular velocity about the pitch axis (rad/s).
    ],
    dtype=np.float64,
)
OBSERVATION_SCALES = np.array(
    [
        1.0,  # Target distance (m).
        0.2,  # Pitch angle (rad).
        1.0,  # Linear velocity along the push axis (m/s).
        2.0,  # Angular velocity about the pitch axis (rad/s).
        0.1,  # Object width (m).
        0.1,  # Object length (m).
        0.25,  # Object height (m).
    ]
)


@dataclass(frozen=True)
class DeltaActorCriticConfig:
    obs_dim: int = 7
    trajectory_points: int = 16
    hidden_dim: int = 256
    delta_scale: tuple[float, float, float] = (
        0.3,
        0.4,
        0.4,
    )  # The maximum absolute changes for height, force, and duration.
    learning_rate: float = 3e-4
    gamma: float = 0.99
    entropy_coefficient: float | None = None
    tau: float = 0.005
    batch_size: int = 256
    gradient_steps: int = 1
    max_grad_norm: float = 10.0
    seed: int = 42
    trajectory_feature_strategy: str = "interpolate"
    world_model: str = "physics"

    def __post_init__(self) -> None:
        self._validate_dimensions()
        self._validate_learning_settings()
        self._validate_sampling_settings()
        if self.world_model not in ("none", "physics", "learned"):
            raise ValueError("world_model must be 'none', 'physics', or 'learned'")

    def _validate_dimensions(self) -> None:
        for name in (
            "obs_dim",
            "trajectory_points",
            "hidden_dim",
            "batch_size",
            "gradient_steps",
        ):
            value = getattr(self, name)
            minimum = 2 if name == "trajectory_points" else 1
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.obs_dim != len(OBSERVATION_SCALES):
            raise ValueError("obs_dim must be 7 for the observable pushing state")

    def _validate_learning_settings(self) -> None:
        for name in ("learning_rate", "max_grad_norm", "tau", "gamma"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
            minimum = 0 if name in ("tau", "gamma") else np.nextafter(0.0, 1.0)
            if value < minimum or (name in ("gamma", "tau") and value > 1):
                raise ValueError(f"invalid {name}")
        if self.entropy_coefficient is not None and (
            isinstance(self.entropy_coefficient, bool)
            or not np.isfinite(self.entropy_coefficient)
            or self.entropy_coefficient < 0
        ):
            raise ValueError(
                "entropy_coefficient must be nonnegative or None for automatic tuning"
            )
        if (
            self.trajectory_feature_strategy
            not in TrajectoryEncoder.FEATURE_EXTRACTION_STRATEGIES
        ):
            raise ValueError(
                f"trajectory_feature_strategy must be one of {TrajectoryEncoder.FEATURE_EXTRACTION_STRATEGIES}"
            )

    def _validate_sampling_settings(self) -> None:
        scale = np.asarray(self.delta_scale)
        if (
            scale.shape != (3,)
            or not np.issubdtype(scale.dtype, np.number)
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0)
            or np.any(scale > 2)
        ):
            raise ValueError("delta_scale must contain three finite values in (0, 2]")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, Integral)
            or not 0 <= self.seed <= np.iinfo(np.int64).max
        ):
            raise ValueError("seed must be a nonnegative int64 integer")


Forecasts = NDArray[np.float32] | tuple[NDArray[np.float32], ...]


@dataclass
class TrainingExample:
    """One executed delta transition; next_context includes the updated posterior."""

    observation: NDArray[np.float32]
    real_trajectory: NDArray[np.float32]
    delta: NDArray[np.float32]
    forecasts: Forecasts
    weights: NDArray[np.float64]
    reward: float
    reference_action: NDArray[np.float32]
    terminated: bool
    truncated: bool
    metadata: Any = None
    next_context: Any = None


@dataclass
class PreparedExample:
    """Validated current-state tensors with a full-precision action reference."""

    observation: torch.Tensor
    real_trajectory: torch.Tensor
    delta: torch.Tensor
    forecasts: torch.Tensor
    weights: torch.Tensor
    reference_action: NDArray[np.float64]
    forecast_mask: torch.Tensor | None


@dataclass
class CriticBatch:
    observations: torch.Tensor
    real_trajectories: torch.Tensor
    deltas: torch.Tensor
    forecasts: torch.Tensor
    weights: torch.Tensor
    forecast_mask: torch.Tensor | None
    real_mask: torch.Tensor | None


ForecastFunction = Callable[[TrainingExample, NDArray[np.float32]], Forecasts]


class TrajectoryEncoder(nn.Module):
    MIN_OUTPUT_DIM = 8
    FEATURE_EXTRACTION_STRATEGIES = ("interpolate", "self-attention")

    def __init__(self, config: DeltaActorCriticConfig) -> None:
        super().__init__()
        self.strategy = config.trajectory_feature_strategy
        self.output_dim = max(self.MIN_OUTPUT_DIM, config.hidden_dim // 2)
        if self.strategy == "interpolate":
            self.net = MLP(
                config.trajectory_points * DenseTrajectory.NUM_FEATURES,
                self.output_dim,
                config.hidden_dim,
            )
        elif self.strategy == "self-attention":
            self.net = AttentionTrajectoryEncoder(
                feature_dim=DenseTrajectory.NUM_FEATURES - 1,
                d_model=self.output_dim,
                nhead=4 if self.output_dim % 4 == 0 else 1,
                dropout=0.0,
            )
        else:
            raise NotImplementedError(
                f"trajectory_feature_strategy {config.trajectory_feature_strategy} is not implemented yet"
            )

    def forward(
        self,
        trajectory: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode a [batch, samples, 5] tensor of scaled trajectory channels.

        Channels: elapsed time, displacement, pitch, linear velocity,
        and angular velocity, in that order.
        """
        if self.strategy == "self-attention":
            timestamps = trajectory[:, :, 0] * float(TRAJECTORY_SCALES[0])
            physical_features = trajectory[:, :, 1:]
            return self.net(physical_features, timestamps, padding_mask)
        return F.relu(self.net(trajectory.flatten(start_dim=-2)))

    @staticmethod
    def extract_trajectory_features(
        trajectory: DenseTrajectory, points: int = 16, strategy: str = "interpolate"
    ) -> NDArray[np.float32]:
        if strategy not in TrajectoryEncoder.FEATURE_EXTRACTION_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {TrajectoryEncoder.FEATURE_EXTRACTION_STRATEGIES}"
            )
        if strategy == "self-attention":
            return TrajectoryEncoder._extract_attention_trajectory_features(trajectory)
        return TrajectoryEncoder._extract_interpolated_trajectory_features(
            trajectory, points
        )

    @staticmethod
    def _extract_interpolated_trajectory_features(
        trajectory: DenseTrajectory, points: int = 16
    ) -> NDArray[np.float32]:
        """Interpolate a push to ``[points, 5]`` using fixed physical scales.

        Channels are elapsed time, displacement from the first recorded position,
        tilt, forward velocity, and angular velocity. PyBullet records the first
        sample after a physics step at timestamp zero; retain that convention.
        A one-sample push is repeated, with zero recorded elapsed duration.
        """
        if isinstance(points, bool) or not isinstance(points, Integral) or points < 2:
            raise ValueError("points must be an integer >= 2")
        samples = trajectory.to_numpy()
        TrajectoryEncoder._validate_trajectory_samples(samples)
        times, x_positions, tilts, forward_velocities, angular_velocities = samples.T
        displacements = x_positions - x_positions[0]
        elapsed = times - times[0]
        grid = np.linspace(0.0, elapsed[-1], points)
        interpolated_channels = [grid]
        for channel in (displacements, tilts, forward_velocities, angular_velocities):
            interpolated_channels.append(np.interp(grid, elapsed, channel))
        encoded = np.column_stack(interpolated_channels)
        with np.errstate(over="ignore", invalid="ignore"):
            result = (encoded / TRAJECTORY_SCALES).astype(np.float32)
        if not np.all(np.isfinite(result)):
            raise ValueError("trajectory features exceed the finite float32 range")
        return result

    @staticmethod
    def _extract_attention_trajectory_features(
        trajectory: DenseTrajectory,
    ) -> NDArray[np.float32]:
        """Scale all recorded samples without resampling or truncating them."""
        samples = trajectory.to_numpy()
        TrajectoryEncoder._validate_trajectory_samples(samples)
        samples[:, :2] -= samples[0, :2].copy()
        with np.errstate(over="ignore", invalid="ignore"):
            result = (samples / TRAJECTORY_SCALES).astype(np.float32)
        if not np.all(np.isfinite(result)):
            raise ValueError("trajectory features exceed the finite float32 range")
        return result

    @staticmethod
    def _validate_trajectory_samples(samples: NDArray[np.float64]) -> None:
        """Require finite recorded samples with strictly increasing timestamps."""
        if not len(samples) or not np.all(np.isfinite(samples)):
            raise ValueError("trajectory must have nonempty finite aligned channels")
        times = samples[:, 0]
        if np.any(np.diff(times) <= 0):
            raise ValueError("trajectory timestamps must be strictly increasing")


class DeltaActor(nn.Module):
    # Bound Gaussian spread before tanh and delta scaling: std = exp(log_std).
    MIN_LOG_STD = -5.0
    MAX_LOG_STD = 1.0

    def __init__(
        self,
        config: DeltaActorCriticConfig,
        encoder: TrajectoryEncoder,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        action_dim = len(config.delta_scale)
        # Predict a mean and log standard deviation for each action dimension.
        self.net = MLP(
            config.obs_dim + self.encoder.output_dim,
            2 * action_dim,
            config.hidden_dim,
        )
        self.register_buffer(
            "obs_scale", torch.tensor(OBSERVATION_SCALES, dtype=torch.float32)
        )
        self.register_buffer(
            "delta_scale", torch.tensor(config.delta_scale, dtype=torch.float32)
        )

    def forward(
        self, observations: torch.Tensor, real_trajectory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return means and log standard deviations for height, force, and duration."""
        scaled_observations = observations / self.obs_scale
        encoded_trajectory = self.encoder(real_trajectory)
        actor_input = torch.cat((scaled_observations, encoded_trajectory), -1)
        mean, log_std = self.net(actor_input).chunk(2, -1)
        return mean, log_std.clamp(self.MIN_LOG_STD, self.MAX_LOG_STD)

    def rsample(
        self,
        observations: torch.Tensor,
        real_trajectory: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        reference_action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return feasible reparameterized deltas and transformed log densities."""
        delta, log_prob, _ = self.sample_with_score(
            observations,
            real_trajectory,
            generator=generator,
            reference_action=reference_action,
        )
        return delta, log_prob

    def sample_with_score(
        self,
        observations: torch.Tensor,
        real_trajectory: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        reference_action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return delta, pathwise log density, and fixed-sample score log density."""
        mean, log_std = self(observations, real_trajectory)
        noise = torch.randn(
            mean.shape, dtype=mean.dtype, device=mean.device, generator=generator
        )
        raw_delta = mean + log_std.exp() * noise
        reference = (
            torch.zeros_like(mean) if reference_action is None else reference_action
        )
        lower, upper = compute_delta_bounds(reference, self.delta_scale)
        scale, center = (upper - lower) / 2, (upper + lower) / 2
        delta = (center + raw_delta.tanh() * scale).clamp(lower, upper)
        correction = 2 * (np.log(2.0) - raw_delta - F.softplus(-2 * raw_delta))
        normal = torch.distributions.Normal(mean, log_std.exp())
        log_prob = (normal.log_prob(raw_delta) - correction - scale.log()).sum(-1)
        score_log_prob = (
            normal.log_prob(raw_delta.detach()) - correction.detach() - scale.log()
        ).sum(-1)
        return delta, log_prob, score_log_prob


class TrajectoryCritic(nn.Module):
    def __init__(
        self,
        config: DeltaActorCriticConfig,
        encoder: TrajectoryEncoder,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        action_dim = len(config.delta_scale)
        inputs = config.obs_dim + action_dim + 2 * self.encoder.output_dim
        self.net = MLP(inputs, 1, config.hidden_dim)
        self.register_buffer(
            "obs_scale", torch.tensor(OBSERVATION_SCALES, dtype=torch.float32)
        )

    def forward(
        self,
        observations: torch.Tensor,
        real: torch.Tensor,
        deltas: torch.Tensor,
        forecasts: torch.Tensor,
        forecast_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.predict_encoded(
            observations,
            self.encoder(real),
            deltas,
            self.encoder(forecasts, forecast_mask),
        )

    def predict_encoded(
        self,
        observations: torch.Tensor,
        encoded_real: torch.Tensor,
        deltas: torch.Tensor,
        encoded_forecasts: torch.Tensor,
    ) -> torch.Tensor:
        critic_input = torch.cat(
            (observations / self.obs_scale, encoded_real, deltas, encoded_forecasts),
            dim=-1,
        )
        return self.net(critic_input).squeeze(-1)


class TrajectoryTwinCritic(nn.Module):
    def __init__(
        self,
        config: DeltaActorCriticConfig,
        encoder: TrajectoryEncoder,
    ) -> None:
        super().__init__()
        self.q1 = TrajectoryCritic(config, encoder)
        self.q2 = TrajectoryCritic(config, encoder)

    def forward(
        self,
        observations: torch.Tensor,
        real_trajectory: torch.Tensor,
        deltas: torch.Tensor,
        forecasts: torch.Tensor,
        forecast_mask: torch.Tensor | None = None,
        real_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode once per trajectory, sharing features between both critic heads."""
        batch_size, hypothesis_count = forecasts.shape[:2]
        encoder = self.q1.encoder
        encoded_real = encoder(real_trajectory, real_mask)
        mask = None if forecast_mask is None else forecast_mask.flatten(0, 1)
        encoded_forecasts = encoder(forecasts.flatten(0, 1), mask)
        inputs = (
            observations.repeat_interleave(hypothesis_count, dim=0),
            encoded_real.repeat_interleave(hypothesis_count, dim=0),
            deltas.repeat_interleave(hypothesis_count, dim=0),
            encoded_forecasts,
        )
        return tuple(
            critic.predict_encoded(*inputs).reshape(batch_size, hypothesis_count)
            for critic in (self.q1, self.q2)
        )


class ActionDeltaDistribution:
    """Frozen proposal parameters for Gaussian or uniform feasible delta sampling.

    ``mean`` is the squashed Gaussian location (the deterministic proposal), not
    the exact expectation of the transformed distribution. Sampling does not
    call the actor again. A model-local generator provides reproducibility.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        scale: torch.Tensor,
        generator: torch.Generator,
        reference_action: torch.Tensor | None = None,
        uniform: bool = False,
    ) -> None:
        self._mean, self._log_std, self._scale = (
            mean.detach().clone(),
            log_std.detach().clone(),
            scale.detach().clone(),
        )
        reference = (
            torch.zeros_like(mean) if reference_action is None else reference_action
        )
        self._lower, self._upper = compute_delta_bounds(reference, scale)
        self._scale = (self._upper - self._lower) / 2
        self._center = (self._upper + self._lower) / 2
        self._uniform = uniform
        self._generator = generator

    @property
    def mean(self) -> NDArray[np.float32]:
        return (
            (self._center + self._mean.tanh() * self._scale)
            .clamp(self._lower, self._upper)
            .cpu()
            .numpy()
            .copy()
        )

    def sample(self, count: int) -> NDArray[np.float32]:
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
            raise ValueError("count must be a positive integer")
        shape = (count, len(self._mean))
        if self._uniform:
            unit_delta = (
                2
                * torch.rand(shape, device=self._mean.device, generator=self._generator)
                - 1
            )
        else:
            noise = torch.randn(
                shape, device=self._mean.device, generator=self._generator
            )
            unit_delta = (self._mean + self._log_std.exp() * noise).tanh()
        delta = (self._center + unit_delta * self._scale).clamp(
            self._lower, self._upper
        )
        return delta.cpu().numpy()


class DeltaActorCritic:
    """
    Learn normalized action adjustments using soft TD trajectory critics.

    Delta is added to the last executed action by the controller.
    """

    def __init__(
        self, config: DeltaActorCriticConfig, device: str | torch.device = "cpu"
    ) -> None:
        if not isinstance(config, DeltaActorCriticConfig):
            raise TypeError("config must be a DeltaActorCriticConfig")
        self.config, self.device = config, torch.device(device)
        # Initialization preserves the caller's CPU RNG; sampling uses only a
        # private device generator. Do not perturb baseline SAC experiment RNGs.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            self.encoder = TrajectoryEncoder(config).to(self.device)
            self.actor = DeltaActor(config, self.encoder).to(self.device)
            self.critic = TrajectoryTwinCritic(config, self.encoder).to(self.device)
            self.target_critic = deepcopy(self.critic).requires_grad_(False)
        self.rng = torch.Generator(device=self.device).manual_seed(config.seed)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.net.parameters(), lr=config.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.learning_rate
        )
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=config.learning_rate
        )
        if config.world_model == "learned":
            from src.agents.rl.models.latent_world_model import LatentWorldModel
            self.latent_world_model = LatentWorldModel(
                trajectory_points=config.trajectory_points,
                hidden_dim=config.hidden_dim,
            ).to(self.device)
            self.world_model_optimizer: torch.optim.Adam | None = torch.optim.Adam(
                self.latent_world_model.parameters(), lr=config.learning_rate
            )
        else:
            self.latent_world_model = None
            self.world_model_optimizer = None
        self.training_metadata: dict[str, Any] | None = None
        self.num_updates = 0
        self.num_timesteps = 0

    def _validate_array(
        self,
        value: Any,
        shape: tuple[int, ...],
        name: str,
        *,
        dtype: type = np.float32,
    ) -> np.ndarray:
        # Historical controller snapshots are immutable NumPy arrays. Copy so
        # PyTorch never wraps a read-only buffer with unsupported write access.
        array = np.array(value, dtype=dtype, copy=True)
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite with shape {shape}")
        return array

    def _prepare_context(
        self, observation: Any, real: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs = self._validate_array(observation, (self.config.obs_dim,), "observation")
        trajectory = self._validate_trajectory(real, "real_trajectory")
        return tuple(
            torch.as_tensor(array, device=self.device).unsqueeze(0)
            for array in (obs, trajectory)
        )

    def _validate_trajectory(self, values: Any, name: str) -> NDArray[np.float32]:
        trajectory = np.array(values, dtype=np.float32, copy=True)
        if (
            trajectory.ndim != 2
            or trajectory.shape[1] != 5
            or not len(trajectory)
            or not np.all(np.isfinite(trajectory))
        ):
            raise ValueError(f"{name} must be finite nonempty [P, 5]")
        if (
            self.config.trajectory_feature_strategy == "interpolate"
            and len(trajectory) != self.config.trajectory_points
        ):
            raise ValueError(f"{name} must have trajectory_points samples")
        return trajectory

    def _prepare_forecasts(
        self,
        forecasts: Any,
        weights: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if not len(forecasts):
            raise ValueError("forecasts must contain at least one trajectory")
        trajectories = [
            torch.as_tensor(
                self._validate_trajectory(item, "forecasts"), device=self.device
            )
            for item in forecasts
        ]
        mass = np.asarray(weights, dtype=np.float64)
        if (
            mass.shape != (len(trajectories),)
            or not np.all(np.isfinite(mass))
            or np.any(mass < 0)
            or not 0 < mass.sum() < np.inf
        ):
            raise ValueError(
                "weights must have nonnegative finite mass aligned with forecasts"
            )
        prediction, mask = self._pad_forecasts(trajectories)
        return (
            prediction.unsqueeze(0),
            torch.as_tensor(mass / mass.sum(), dtype=torch.float32, device=self.device),
            mask,
        )

    def _pad_forecasts(
        self,
        trajectories: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.config.trajectory_feature_strategy == "interpolate":
            return torch.stack(trajectories), None
        prediction = pad_sequence(trajectories, batch_first=True)
        lengths = torch.tensor([len(item) for item in trajectories], device=self.device)
        mask = (
            torch.arange(prediction.shape[1], device=self.device)[None]
            >= lengths[:, None]
        )
        return prediction, mask.unsqueeze(0)

    @torch.no_grad()
    def distribution(
        self,
        observation: Any,
        real_trajectory: Any,
        reference_action: Any = None,
        *,
        uniform: bool = False,
    ) -> ActionDeltaDistribution:
        obs, real = self._prepare_context(observation, real_trajectory)
        if uniform:
            mean = log_std = torch.zeros(
                (1, len(self.config.delta_scale)), device=self.device
            )
        else:
            mean, log_std = self.actor(obs, real)
        reference = (
            None
            if reference_action is None
            else torch.as_tensor(
                np.array(reference_action, copy=True), device=self.device
            )
        )
        return ActionDeltaDistribution(
            mean[0], log_std[0], self.actor.delta_scale, self.rng, reference, uniform
        )

    def score_tensor(
        self,
        observations: torch.Tensor,
        real: torch.Tensor,
        deltas: torch.Tensor,
        forecasts: torch.Tensor,
        weights: torch.Tensor,
        forecast_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Differentiable posterior-weighted twin minimum for validated tensors.

        Forecasts remain differentiable here.
        ``train_step`` explicitly detaches regenerated PyBullet features.
        """
        q1, q2 = self.critic(observations, real, deltas, forecasts, forecast_mask)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        q1_return = (q1 * weights).sum(-1)
        q2_return = (q2 * weights).sum(-1)
        return torch.minimum(q1_return, q2_return)

    def _validate_candidates(
        self, deltas: Any, forecasts: Any
    ) -> tuple[np.ndarray, Any]:
        adjustments = np.array(deltas, dtype=np.float32, copy=True)
        if (
            adjustments.ndim != 2
            or adjustments.shape[0] < 1
            or adjustments.shape[1] != 3
            or not np.all(np.isfinite(adjustments))
        ):
            raise ValueError("deltas must be finite [C, 3]")
        if len(forecasts) != len(adjustments):
            raise ValueError(
                "forecasts must contain one hypothesis collection per candidate"
            )
        return adjustments, forecasts

    @torch.no_grad()
    def score(
        self,
        observation: Any,
        real_trajectory: Any,
        deltas: Any,
        forecasts: Any,
        weights: Any,
    ) -> NDArray[np.float32]:
        obs, real = self._prepare_context(observation, real_trajectory)
        adjustments, prediction = self._validate_candidates(deltas, forecasts)
        examples = []
        for delta, candidate in zip(adjustments, prediction):
            forecasts_tensor, mass, mask = self._prepare_forecasts(candidate, weights)
            examples.append(
                PreparedExample(
                    obs,
                    real,
                    torch.as_tensor(delta, device=self.device)[None],
                    forecasts_tensor,
                    mass,
                    np.zeros(3),
                    mask,
                )
            )
        q1, q2 = self._evaluate_batch(examples)
        return torch.minimum(q1, q2).cpu().numpy()

    def train_step(
        self,
        examples: Sequence[TrainingExample],
        forecast_fn: ForecastFunction,
    ) -> dict[str, float]:
        """Fit one-step soft TD targets, then update the actor and target critics."""
        if not examples:
            raise ValueError("at least one TrainingExample is required")
        prepared = [self._prepare_example(example) for example in examples]
        targets = self._compute_targets(examples, forecast_fn)
        critic_metrics = self._update_critic(prepared, targets)
        # The critic optimizer owns the shared encoder. Freeze it along with
        # the critic heads while learning the actor's action mapping.
        self.critic.requires_grad_(False)
        try:
            actor_metrics = self._update_actor(examples, prepared, forecast_fn)
        finally:
            self.critic.requires_grad_(True)
        with torch.no_grad():
            for target, online in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target.lerp_(online, self.config.tau)
        world_model_metrics = (
            self._update_world_model(examples) if self.latent_world_model else {}
        )
        self.num_updates += 1
        return {
            **critic_metrics,
            **actor_metrics,
            **world_model_metrics,
            "num_updates": float(self.num_updates),
        }

    def _update_world_model(
        self, examples: Sequence[TrainingExample]
    ) -> dict[str, float]:
        """Train the latent world model to predict the next observed trajectory."""
        valid = [e for e in examples if e.next_context is not None]
        if not valid:
            return {}
        batch = self._prepare_world_model_batch(valid)
        predicted = self.latent_world_model(*batch[:4])
        loss = F.mse_loss(predicted, batch[4])
        self.world_model_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.latent_world_model.parameters(), self.config.max_grad_norm
        )
        self.world_model_optimizer.step()
        self.world_model_optimizer.zero_grad(set_to_none=True)
        return {"world_model_loss": float(loss.detach())}

    def _prepare_world_model_batch(
        self, examples: Sequence[TrainingExample]
    ) -> tuple[torch.Tensor, ...]:
        points = self.config.trajectory_points
        arrays = (
            np.stack([e.observation for e in examples]),
            np.stack([e.reference_action for e in examples]),
            np.stack([e.delta for e in examples]),
            np.stack([
                interpolate_model_trajectory(e.real_trajectory, points)
                for e in examples
            ]),
            np.stack([
                interpolate_model_trajectory(e.next_context.real_trajectory, points)
                for e in examples
            ]),
        )
        return tuple(
            torch.as_tensor(array, dtype=torch.float32, device=self.device)
            for array in arrays
        )

    def _prepare_example(self, example: TrainingExample) -> PreparedExample:
        obs, real = self._prepare_context(example.observation, example.real_trajectory)
        # Float32 rounding can hide an out-of-bounds final action at +/-1.
        delta = self._validate_array(example.delta, (3,), "delta", dtype=np.float64)
        reference = self._validate_array(
            example.reference_action, (3,), "reference_action", dtype=np.float64
        )
        if np.any(np.abs(reference) > 1) or np.any(np.abs(reference + delta) > 1):
            raise ValueError(
                "training example must contain a valid executed final action"
            )
        if not np.isfinite(example.reward):
            raise ValueError("reward must be finite")
        if not example.terminated and example.next_context is None:
            raise ValueError(
                "nonterminal transitions require next_context, including truncations"
            )
        forecasts, weights, mask = self._prepare_forecasts(
            example.forecasts, example.weights
        )
        return PreparedExample(
            obs,
            real,
            torch.as_tensor(delta, dtype=torch.float32, device=self.device)[None],
            forecasts,
            weights,
            reference,
            mask,
        )

    @property
    def alpha(self) -> torch.Tensor:
        if self.config.entropy_coefficient is not None:
            return torch.tensor(self.config.entropy_coefficient, device=self.device)
        return self.log_alpha.exp().detach()

    @staticmethod
    def _build_next_example(example: TrainingExample) -> TrainingExample:
        context = example.next_context
        return replace(
            example,
            observation=context.observation,
            real_trajectory=context.real_trajectory,
            reference_action=context.reference_action,
            weights=context.weights,
            metadata=context,
        )

    @torch.no_grad()
    def _compute_targets(
        self, examples: Sequence[TrainingExample], forecast_fn: ForecastFunction
    ) -> list[torch.Tensor]:
        targets = torch.tensor(
            [example.reward for example in examples],
            dtype=torch.float32,
            device=self.device,
        )
        indices = [index for index, item in enumerate(examples) if not item.terminated]
        if indices:
            prepared = [
                self._prepare_bootstrap(examples[i], forecast_fn) for i in indices
            ]
            next_examples, log_probs = zip(*prepared)
            q1, q2 = self._evaluate_batch(next_examples, self.target_critic)
            values = torch.minimum(q1, q2) - self.alpha * torch.stack(log_probs)
            targets[indices] += self.config.gamma * values
        return list(targets.unbind())

    def _prepare_bootstrap(
        self, example: TrainingExample, forecast_fn: ForecastFunction
    ) -> tuple[PreparedExample, torch.Tensor]:
        context = example.next_context
        observation, trajectory = self._prepare_context(
            context.observation, context.real_trajectory
        )
        reference = torch.as_tensor(
            np.array(context.reference_action, copy=True), device=self.device
        )
        delta, log_prob = self.actor.rsample(
            observation, trajectory, generator=self.rng, reference_action=reference
        )
        forecasts = forecast_fn(
            self._build_next_example(example), delta[0].cpu().numpy()
        )
        prediction, weights, mask = self._prepare_forecasts(forecasts, context.weights)
        prepared = PreparedExample(
            observation,
            trajectory,
            delta,
            prediction,
            weights,
            np.array(context.reference_action, copy=True),
            mask,
        )
        return prepared, log_prob[0]

    def _batch_examples(self, examples: Sequence[PreparedExample]) -> CriticBatch:
        hypothesis_count = max(item.forecasts.shape[1] for item in examples)
        trajectories, weights = self._collect_batch_forecasts(
            examples, hypothesis_count
        )
        forecasts, mask = self._pad_forecasts(trajectories)
        real = [item.real_trajectory[0] for item in examples]
        real_batch, real_mask = self._pad_forecasts(real)
        shape = (len(examples), hypothesis_count)
        return CriticBatch(
            torch.cat([item.observation for item in examples]),
            real_batch,
            torch.cat([item.delta for item in examples]),
            forecasts.reshape(*shape, *forecasts.shape[1:]),
            torch.stack(weights),
            None if mask is None else mask.reshape(*shape, -1),
            None if real_mask is None else real_mask[0],
        )

    def _collect_batch_forecasts(
        self, examples: Sequence[PreparedExample], hypothesis_count: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        trajectories, weights = [], []
        for item in examples:
            forecasts = list(item.forecasts[0].unbind())
            if item.forecast_mask is not None:
                forecasts = [
                    trajectory[~mask]
                    for trajectory, mask in zip(forecasts, item.forecast_mask[0])
                ]
            padding = hypothesis_count - len(forecasts)
            # Padded hypotheses duplicate a valid trajectory but carry zero weight.
            trajectories.extend(forecasts + [forecasts[0]] * padding)
            weights.append(F.pad(item.weights, (0, padding)))
        return trajectories, weights

    def _evaluate_batch(
        self,
        examples: Sequence[PreparedExample],
        critic: TrajectoryTwinCritic | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self._batch_examples(examples)
        critic = self.critic if critic is None else critic
        q1, q2 = critic(
            batch.observations,
            batch.real_trajectories,
            batch.deltas,
            batch.forecasts,
            batch.forecast_mask,
            batch.real_mask,
        )
        return (q1 * batch.weights).sum(-1), (q2 * batch.weights).sum(-1)

    def _update_critic(
        self,
        examples: list[PreparedExample],
        targets: list[torch.Tensor],
    ) -> dict[str, float]:
        q1, q2 = self._evaluate_batch(examples)
        target = torch.stack(targets)
        loss = 0.5 * ((q1 - target).square() + (q2 - target).square()).mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.config.max_grad_norm, error_if_nonfinite=True
        )
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)
        return dict(
            critic_loss=float(loss.detach()),
            critic_q_mean=float(torch.minimum(q1, q2).detach().mean()),
            target_mean=float(target.mean()),
            target_min=float(target.min()),
            target_max=float(target.max()),
            critic_gradient_norm=float(gradient_norm),
        )

    def _prepare_actor_example(
        self,
        example: TrainingExample,
        prepared: PreparedExample,
        forecast_fn: ForecastFunction,
    ) -> tuple[PreparedExample, torch.Tensor, torch.Tensor, torch.Tensor]:
        reference = torch.as_tensor(prepared.reference_action, device=self.device)
        delta, log_prob, score_log_prob = self.actor.sample_with_score(
            prepared.observation,
            prepared.real_trajectory,
            generator=self.rng,
            reference_action=reference,
        )
        forecasts = forecast_fn(example, delta[0].detach().cpu().numpy())
        prediction, _, mask = self._prepare_forecasts(forecasts, example.weights)
        lower, upper = compute_delta_bounds(reference, self.actor.delta_scale)
        log_scale = ((upper - lower) / 2).log().sum()
        actor_example = replace(
            prepared, delta=delta, forecasts=prediction.detach(), forecast_mask=mask
        )
        return actor_example, log_prob.mean(), score_log_prob.mean(), log_scale

    def _actor_loss(
        self,
        actor_examples: Sequence[PreparedExample],
        replay_examples: Sequence[PreparedExample],
        log_probs: torch.Tensor,
        score_log_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The replay action is fixed independently of the new policy sample,
        # making its value an unbiased state-dependent variance baseline.
        with torch.no_grad():
            q1, q2 = self._evaluate_batch(actor_examples)
            b1, b2 = self._evaluate_batch(replay_examples)
            values, baseline = torch.minimum(q1, q2), torch.minimum(b1, b2)
        advantage = (values - baseline).detach()
        # Score gradients capture the full sampled action/forecast value even
        # when PyBullet is nondifferentiable. Entropy retains its pathwise gradient.
        loss = (self.alpha * log_probs - advantage * score_log_probs).mean()
        return loss, values

    def _update_actor(
        self,
        examples: Sequence[TrainingExample],
        prepared: list[PreparedExample],
        forecast_fn: ForecastFunction,
    ) -> dict[str, float]:
        terms = [
            self._prepare_actor_example(example, tensors, forecast_fn)
            for example, tensors in zip(examples, prepared)
        ]
        actor_examples, probabilities, score_probabilities, scales = zip(*terms)
        log_probs, log_scales = torch.stack(probabilities), torch.stack(scales)
        loss, values = self._actor_loss(
            actor_examples, prepared, log_probs, torch.stack(score_probabilities)
        )
        objective = float((self.alpha * log_probs - values).detach().mean())
        gradient_norm = self._step_actor_optimizer(loss)
        alpha_loss = self._update_temperature(log_probs, log_scales)
        return dict(
            actor_loss=objective,
            alpha=float(self.alpha),
            alpha_loss=alpha_loss,
            actor_gradient_norm=gradient_norm,
            entropy=float(-log_probs.detach().mean()),
            normalized_entropy=float(-(log_probs + log_scales).detach().mean()),
            actor_q_mean=float(values.mean()),
            actor_saturation=self._measure_actor_saturation(actor_examples),
            **self._measure_actor_std(prepared),
        )

    @torch.no_grad()
    def _measure_actor_std(
        self, examples: Sequence[PreparedExample]
    ) -> dict[str, float]:
        """Summarize post-update, pre-tanh Gaussian std over replay states."""
        log_std = torch.cat(
            [self.actor(item.observation, item.real_trajectory)[1] for item in examples]
        )
        std = log_std.exp()
        quantiles = torch.quantile(
            std, std.new_tensor([0.0, 0.1, 0.5, 0.9, 1.0]), dim=0
        )
        metrics = {}
        for index, action_name in enumerate(("height", "force", "duration")):
            prefix = f"actor_std_{action_name}"
            for label, value in zip(
                ("min", "p10", "median", "p90", "max"), quantiles[:, index]
            ):
                metrics[f"{prefix}_{label}"] = float(value)
            metrics[f"{prefix}_mean"] = float(std[:, index].mean())
            metrics[f"{prefix}_floor_fraction"] = float(
                (log_std[:, index] <= self.actor.MIN_LOG_STD).float().mean()
            )
        return metrics

    def _step_actor_optimizer(self, loss: torch.Tensor) -> float:
        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.actor.net.parameters(),
            self.config.max_grad_norm,
            error_if_nonfinite=True,
        )
        self.actor_optimizer.step()
        return float(gradient_norm)

    @torch.no_grad()
    def _measure_actor_saturation(self, examples: Sequence[PreparedExample]) -> float:
        fractions = []
        for example in examples:
            reference = torch.as_tensor(example.reference_action, device=self.device)
            lower, upper = compute_delta_bounds(reference, self.actor.delta_scale)
            position = (2 * example.delta - upper - lower) / (upper - lower)
            fractions.append((position.abs() > 0.99).float().mean())
        return float(torch.stack(fractions).mean())

    def _update_temperature(
        self, log_prob: torch.Tensor, log_scale: torch.Tensor
    ) -> float:
        if self.config.entropy_coefficient is not None:
            return 0.0
        # Match SAC's -action_dim entropy target in normalized action coordinates.
        # Delta's density already subtracts log(scale); undo that change of units.
        normalized_log_prob = log_prob.detach() + log_scale.detach()
        loss = -(
            self.log_alpha * (normalized_log_prob - len(self.config.delta_scale))
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.alpha_optimizer.step()
        return float(loss.detach())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                checkpoint_format=CHECKPOINT_FORMAT,
                config=asdict(self.config),
                actor=self.actor.state_dict(),
                critic=self.critic.state_dict(),
                target_critic=self.target_critic.state_dict(),
                log_alpha=self.log_alpha.detach(),
                alpha_optimizer=self.alpha_optimizer.state_dict(),
                num_timesteps=self.num_timesteps,
                actor_optimizer=self.actor_optimizer.state_dict(),
                critic_optimizer=self.critic_optimizer.state_dict(),
                rng_state=self.rng.get_state(),
                num_updates=self.num_updates,
                training_metadata=self.training_metadata,
                latent_world_model=self.latent_world_model.state_dict()
                if self.latent_world_model is not None
                else None,
                world_model_optimizer=self.world_model_optimizer.state_dict()
                if self.world_model_optimizer is not None
                else None,
            ),
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> Self:
        state = torch.load(path, map_location=device, weights_only=True)
        if (
            not isinstance(state, dict)
            or state.get("checkpoint_format") != CHECKPOINT_FORMAT
        ):
            raise ValueError(
                "Expected a TD delta checkpoint; Monte Carlo delta and vanilla SAC weights require separate training"
            )
        config = DeltaActorCriticConfig(**state["config"])
        model = cls(config, device)
        model.actor.load_state_dict(state["actor"])
        model.critic.load_state_dict(state["critic"])
        model.target_critic.load_state_dict(state["target_critic"])
        with torch.no_grad():
            model.log_alpha.copy_(state["log_alpha"])
        model.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        model.num_timesteps = state["num_timesteps"]
        model.actor_optimizer.load_state_dict(state["actor_optimizer"])
        model.critic_optimizer.load_state_dict(state["critic_optimizer"])
        # CPU and CUDA generators use different state formats. Exact stochastic
        # continuation is preserved on the original device type; cross-device
        # loads retain learned behavior and begin the configured random stream.
        try:
            model.rng.set_state(state["rng_state"].cpu())
        except RuntimeError:
            pass
        model.num_updates = state["num_updates"]
        model.training_metadata = state.get("training_metadata")
        if state.get("latent_world_model") is not None and model.latent_world_model is not None:
            model.latent_world_model.load_state_dict(state["latent_world_model"])
        if state.get("world_model_optimizer") is not None and model.world_model_optimizer is not None:
            model.world_model_optimizer.load_state_dict(state["world_model_optimizer"])
        return model
