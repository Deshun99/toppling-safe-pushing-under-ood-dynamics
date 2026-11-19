"""Moving-reference delta proposals ranked using object-specific simulations.

Absolute actions and physics remain controller metadata. Actor inputs contain
only observations and the previous real trajectory; the scorer additionally
receives the delta and each forecast. A rejected action never moves the real
reference. STOP means no action was executed, not a fabricated terminal reward.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Final, Self

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from src.agents.simulator.forecast_pool import ForecastPool
    from src.agents.rl.models.delta_actor_critic import (
        ActionDeltaDistribution,
        DeltaActorCritic,
        TrainingExample,
    )

import numpy as np

from src.agents.rl.models.delta_actor_critic import Forecasts, TrajectoryEncoder

from src.agents.simulator.internal_sim import InternalSim
from src.agents.system_identification.smc2 import SMC2SysID
from src.agents.system_identification.sysid import (
    SysID,
    Z_HI,
    Z_LO,
    convert_z_to_cuboid_spec,
)
from src.simulation.environment.env import ACTION_RANGES, ActionRanges
from src.simulation.execution.force_push import DenseTrajectory
from src.simulation.modelling.action_mapping import (
    normalized_to_physical,
    physical_to_normalized,
)
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.object import CuboidSpec
from src.simulation.modelling.records import ActionRecord
from src.simulation.execution.settling import RestConfig


CANDIDATE_SELECTION: Final = "critic_after_bounds"
GEOMETRY_DIM: Final = 3  # Width, length, and height.


def _copy_readonly_array(
    values: ArrayLike, *, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    if (shape is not None and result.shape != shape) or not np.all(np.isfinite(result)):
        raise ValueError(f"Expected finite array with shape {shape}")
    result.setflags(write=False)
    return result


def _copy_readonly_forecasts(forecasts: Forecasts) -> Forecasts:
    if isinstance(forecasts, tuple):
        return tuple(_copy_readonly_array(item) for item in forecasts)
    return _copy_readonly_array(forecasts)


@dataclass(frozen=True)
class DeltaControllerConfig:
    probe_height_fraction: float = 0.1
    probe_force: float = 0.5
    probe_duration: float = 0.05
    candidates_per_batch: int = 8
    max_batches: int = 4
    trajectory_points: int = 16
    backend: str = "smc2"
    n_particles: int = 16
    seed: int = 42
    estimator_options: dict[str, Any] = field(default_factory=dict)
    cma_replay_budget: int = 128
    wait_for_rest: bool = False
    rest_timeout: float = 2.0
    world_model: str = "physics"

    def __post_init__(self) -> None:
        RestConfig(self.wait_for_rest, self.rest_timeout)
        self._validate_counts()
        self._validate_probe()
        self._validate_estimator()

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> Self:
        """Read earlier settings while discarding removed controller options."""
        values = dict(settings)
        values.pop("near_topple_fraction", None)
        values.pop("use_full_state", None)
        return cls(**values)

    def _validate_counts(self) -> None:
        for name, value in (
            ("candidates_per_batch", self.candidates_per_batch),
            ("max_batches", self.max_batches),
            ("trajectory_points", self.trajectory_points),
            ("n_particles", self.n_particles),
            ("cma_replay_budget", self.cma_replay_budget),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.trajectory_points < 2:
            raise ValueError("trajectory_points must be at least 2")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, Integral)
            or not 0 <= self.seed <= np.iinfo(np.uint64).max
        ):
            raise ValueError("seed must be a nonnegative uint64 integer")

    def _validate_probe(self) -> None:
        for name, value in (
            ("probe_height_fraction", self.probe_height_fraction),
            ("probe_force", self.probe_force),
            ("probe_duration", self.probe_duration),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if (
            not 0 <= self.probe_height_fraction <= 1
            or self.probe_force < 0
            or self.probe_duration <= 0
        ):
            raise ValueError("invalid probe configuration")

    def _validate_estimator(self) -> None:
        if self.backend not in ("smc2", "cmaes", "known"):
            raise ValueError("backend must be 'smc2', 'cmaes', or 'known'")
        if self.world_model not in ("none", "physics", "learned"):
            raise ValueError("world_model must be 'none', 'physics', or 'learned'")
        if not isinstance(self.estimator_options, dict):
            raise TypeError("estimator_options must be a dict")
        if {"geometry", "seed", "n_theta", "sim"} & self.estimator_options.keys():
            raise ValueError(
                "geometry, seed, n_theta and sim are controller-managed options"
            )


@dataclass(frozen=True)
class ForecastContext:
    """Historical inputs copied and validated by DeltaController.build_context.

    Particles are PHYSICAL (mass, friction, COM height), not SMC's z coordinates.
    Particles and full state configure simulation. Hidden parameters are never
    concatenated to either network.
    """

    observation: np.ndarray
    start: tuple[float, float, float, float]
    geometry: tuple[float, float, float]
    reference_action: np.ndarray
    particles: np.ndarray
    weights: np.ndarray
    real_trajectory: np.ndarray
    action_ranges: ActionRanges


def _validate_forecast_context(context: ForecastContext) -> None:
    for name, values, shape in (
        ("observation", context.observation, (7,)),
        ("reference_action", context.reference_action, (3,)),
        ("start", context.start, (4,)),
        ("geometry", context.geometry, (GEOMETRY_DIM,)),
    ):
        if np.shape(values) != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if np.any(np.asarray(context.geometry) <= 0):
        raise ValueError("geometry must be positive")
    if context.real_trajectory.ndim != 2 or context.real_trajectory.shape[1] != 5:
        raise ValueError("real_trajectory must have shape (P, 5)")
    particles, weights = context.particles, context.weights
    if particles.ndim != 2 or particles.shape[1:] != (3,) or not len(particles):
        raise ValueError("particles must have shape (N, 3), N > 0")
    if (
        np.any(particles[:, 0] <= 0)
        or np.any(particles[:, 1] < 0)
        or np.any(particles[:, 2] <= 0)
    ):
        raise ValueError("invalid physical particles")
    if weights.shape != (len(particles),):
        raise ValueError("weights must have one entry per particle")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("particle weights must be nonnegative and sum to one")


@dataclass(frozen=True)
class Proposal:
    delta: np.ndarray
    action: np.ndarray
    forecasts: Forecasts
    weights: np.ndarray
    context: ForecastContext


class DeltaController:
    """
    One object's inference and proposal lifecycle, with explicit ownership.

    Injected estimator/simulator instances remain caller-owned. An injected
    estimator is valid for one initialize_for_object call, since generic
    estimators cannot be reset safely. Default estimators are recreated for each
    new object.
    """

    def __init__(
        self,
        model: "DeltaActorCritic",
        config: DeltaControllerConfig | None = None,
        action_ranges: ActionRanges = ACTION_RANGES,
        target_distance: float = 1.0,
        estimator: SMC2SysID | SysID | None = None,
        simulator: InternalSim | None = None,
        forecast_pool: "ForecastPool | None" = None,
    ) -> None:
        self.model = model
        self.trajectory_feature_strategy = model.config.trajectory_feature_strategy
        self.config = config if config is not None else DeltaControllerConfig()
        if not isinstance(self.config, DeltaControllerConfig):
            raise TypeError("config must be a DeltaControllerConfig")
        if not np.isfinite(target_distance):
            raise ValueError("target_distance must be finite")
        self.action_ranges = deepcopy(action_ranges)
        self.target_distance = float(target_distance)
        self._injected_estimator = estimator
        self.estimator = None
        self._owns_estimator = estimator is None
        if self.config.world_model == "physics":
            self.simulator = simulator if simulator is not None else InternalSim()
            self._owns_simulator = simulator is None
        else:
            self.simulator = None
            self._owns_simulator = False
        self._forecast_pool = forecast_pool
        self._closed, self.geometry = False, None
        self.rest_config = RestConfig(
            self.config.wait_for_rest, self.config.rest_timeout
        )
        self._reset_history()
        self._prior_particles = None
        self.reset_episode()

    @property
    def history(self) -> list[ActionRecord]:
        """Copies of eligible append-only SysID records, across object episodes."""
        return deepcopy(self._history)

    @property
    def records(self) -> list[ActionRecord]:
        """Copies of all real records, including excluded toppling trajectories."""
        return deepcopy(self._records)

    def initialize_for_object(
        self, geometry: ArrayLike, *, known_parameters: ArrayLike | None = None
    ) -> None:
        """
        Initialize inference for a new object, discarding previous object history.

        Store geometry, initialize physics hypotheses, and reset episode feedback.
        The target environment creates and resets the physical object separately.
        """
        if self._closed:
            raise RuntimeError("controller is closed")
        dims = _copy_readonly_array(geometry, shape=(GEOMETRY_DIM,))
        if np.any(dims <= 0):
            raise ValueError("geometry must contain three positive dimensions")
        self._validate_known_parameters(known_parameters, float(dims[2]))
        if self.estimator is not None:
            if not self._owns_estimator:
                raise RuntimeError(
                    "Provide a fresh injected estimator for a new object"
                )
            self.estimator.close()
        self.geometry = tuple(dims)
        self._create_sysid_estimator()
        self._reset_history()
        self._initialize_hypotheses(known_parameters)
        self.reset_episode()

    def _validate_known_parameters(
        self, parameters: ArrayLike | None, height: float
    ) -> None:
        if self.config.backend != "known":
            if parameters is not None:
                raise ValueError("known_parameters requires backend='known'")
            return
        if parameters is None or self._injected_estimator is not None:
            raise ValueError("known physics requires parameters and no estimator")
        mass, friction, com_height = _copy_readonly_array(parameters, shape=(3,))
        if mass <= 0 or friction < 0 or not 0 < com_height < height:
            raise ValueError("invalid known physics parameters")

    def _initialize_hypotheses(self, known_parameters: ArrayLike | None) -> None:
        if self.config.world_model != "physics":
            self._prior_particles = None
            return
        if self.config.backend == "known":
            self._prior_particles = np.array(known_parameters, dtype=float).reshape(
                1, 3
            )
            return
        rng = np.random.default_rng(self.config.seed)
        z_prior = rng.uniform(Z_LO, Z_HI, size=(self.config.n_particles, 3))
        self._prior_particles = np.array(
            [convert_z_to_cuboid_spec(z, self.geometry).theta for z in z_prior]
        )

    def _create_sysid_estimator(self) -> None:
        if self.config.world_model != "physics" or self.config.backend == "known":
            self.estimator = None
            return
        options = deepcopy(self.config.estimator_options)
        if self._injected_estimator is not None:
            self.estimator = self._injected_estimator
        elif self.config.backend == "smc2":
            settings = dict(
                K=self.config.n_particles,
                n_theta=self.config.n_particles,
                n_state=1,
                start_std=(0.0, 0.0, 0.0, 0.0),
                seed=self.config.seed,
            )
            settings.update(options)
            self.estimator = SMC2SysID(self.geometry, **settings)
        else:
            settings = dict(K=self.config.n_particles, seed=self.config.seed)
            settings.update(options)
            self.estimator = SysID(self.geometry, **settings)

    def _reset_history(self) -> None:
        self._history = []
        self._records = []
        self._result = None

    def reset_episode(self) -> None:
        """Reset feedback context while retaining the current object's posterior."""
        self.action_reference = None
        self.real_trajectory = None
        self.stop_reason = None

    def _validate_observation(self, observation: ArrayLike) -> np.ndarray:
        """Validate that the observation is valid and matches the current object's geometry."""
        if self._closed:
            raise RuntimeError("controller is closed")
        if self.geometry is None:
            raise RuntimeError("Call initialize_for_object before running an episode")
        obs = _copy_readonly_array(observation, shape=(7,))
        if not np.allclose(obs[4:], self.geometry, rtol=1e-6, atol=1e-8):
            raise ValueError("observation geometry differs from the current object")
        return obs

    def probe(self, observation: ArrayLike) -> np.ndarray:
        """Return the configured physical probe in the object's action coordinates."""
        self._validate_observation(observation)
        physical = PushAction(
            self.geometry[2] * self.config.probe_height_fraction,
            self.config.probe_force,
            self.config.probe_duration,
        )
        return physical_to_normalized(physical, self.geometry, self.action_ranges)

    def observe(
        self, record: ActionRecord, next_observation: ArrayLike, toppled: bool = False
    ) -> None:
        """Assimilate an actually executed action, never a rejected proposal.

        Full-topple records remain available in records/metrics, but are excluded
        from the append-only upright SysID history without censoring any samples.
        """
        self._validate_observation(next_observation)
        reference = self._normalize_record_action(record.action)
        real = TrajectoryEncoder.extract_trajectory_features(
            record.traj,
            points=self.config.trajectory_points,
            strategy=self.trajectory_feature_strategy,
        )
        saved = deepcopy(record)
        self._records.append(saved)
        self.action_reference = np.asarray(reference, dtype=float).copy()
        self.real_trajectory = np.asarray(real, dtype=float).copy()
        self._update_estimator(saved, toppled)
        self.stop_reason = "observed_topple" if toppled else None

    def _normalize_record_action(self, action: PushAction) -> np.ndarray:
        # Geometry arrives through float32 observations, while the executed
        # command uses the target's full-precision visible geometry. Correct
        # only endpoint roundoff in the inverse *record* map. Candidate actions
        # are still rejected strictly, and the actual record stays unmodified.
        height_fraction = action.h / self.geometry[2]
        low, high = self.action_ranges.height_fraction
        tolerance = 4 * np.finfo(np.float32).eps
        if low - tolerance <= height_fraction < low:
            action = PushAction(low * self.geometry[2], action.F, action.tau)
        elif high < height_fraction <= high + tolerance:
            action = PushAction(high * self.geometry[2], action.F, action.tau)
        return physical_to_normalized(action, self.geometry, self.action_ranges)

    def _update_estimator(self, saved: ActionRecord, toppled: bool) -> None:
        if self.config.world_model != "physics" or self.config.backend == "known":
            return
        if not toppled:
            self._history.append(saved)
            if self.config.backend == "cmaes":
                # An explicit fitting budget remains a cap, not an ever-growing
                # minimum. If full-history fitting no longer fits, retain belief.
                if self.config.cma_replay_budget >= len(self._history):
                    self._result = self.estimator.update(
                        self._history, replay_budget=self.config.cma_replay_budget
                    )
            else:
                self._result = self.estimator.update(self._history)

    def _get_particles(self) -> tuple[np.ndarray, np.ndarray]:
        if self.geometry is None:
            raise RuntimeError("Call initialize_for_object first")
        if self.config.world_model != "physics":
            prior_mean = convert_z_to_cuboid_spec(
                (Z_LO + Z_HI) / 2, self.geometry
            ).theta
            return np.array([prior_mean], dtype=float), np.ones(1, dtype=float)
        if self.config.backend == "smc2":
            z_particles = np.asarray(self.estimator.particles, dtype=float)
            particles = np.array(
                [convert_z_to_cuboid_spec(z, self.geometry).theta for z in z_particles]
            )
            weights = np.asarray(self.estimator.weights, dtype=float).copy()
        elif self._result is not None:
            hypotheses = self._result.hypotheses or [self._result.point]
            particles = np.array([hyp.theta for hyp in hypotheses], dtype=float)
            weights = np.array([hyp.weight for hyp in hypotheses], dtype=float)
        else:
            particles = self._prior_particles.copy()
            weights = np.ones(len(particles), dtype=float)
        if (
            weights.shape != (len(particles),)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or weights.sum() <= 0
        ):
            raise ValueError("estimator returned invalid posterior weights")
        return particles, weights / weights.sum()

    def build_context(self, observation: ArrayLike) -> ForecastContext:
        obs = self._validate_observation(observation)
        if self.action_reference is None or self.real_trajectory is None:
            raise RuntimeError("Execute and observe the probe before proposing")
        particles, weights = self._get_particles()
        context = ForecastContext(
            observation=_copy_readonly_array(obs),
            start=(self.target_distance - obs[0], *obs[1:4]),
            geometry=tuple(_copy_readonly_array(self.geometry)),
            reference_action=_copy_readonly_array(self.action_reference),
            particles=_copy_readonly_array(particles),
            weights=_copy_readonly_array(weights),
            real_trajectory=_copy_readonly_array(self.real_trajectory),
            action_ranges=deepcopy(self.action_ranges),
        )
        _validate_forecast_context(context)
        return context

    def _forecast(self, context: ForecastContext, delta: ArrayLike) -> Forecasts:
        if self.config.world_model == "none":
            P = self.config.trajectory_points
            return np.zeros((1, P, DenseTrajectory.NUM_FEATURES), dtype=np.float32)
        if self.config.world_model == "learned":
            return self.model.latent_world_model.predict_numpy(
                context.observation,
                context.reference_action,
                np.asarray(delta, dtype=float),
                context.real_trajectory,
            )
        action = context.reference_action + np.asarray(delta, dtype=float)
        physical = normalized_to_physical(
            action, context.geometry, context.action_ranges
        )
        if self._forecast_pool is not None:
            features = self._forecast_pool.forecast_features(
                context.geometry,
                list(context.particles),
                context.start,
                physical,
                self.rest_config,
                self.config.trajectory_points,
                self.trajectory_feature_strategy,
            )
            if self.trajectory_feature_strategy == "self-attention":
                return tuple(features)
            return np.asarray(features, dtype=np.float32)
        features = []
        cached: dict[tuple[float, ...], np.ndarray] = {}
        for theta in context.particles:
            key = tuple(theta)
            if key not in cached:
                spec = CuboidSpec(*context.geometry, *theta)
                options = {"rest": self.rest_config} if self.rest_config.enabled else {}
                trajectory = self.simulator.predict(
                    spec, context.start, physical, **options
                )
                cached[key] = TrajectoryEncoder.extract_trajectory_features(
                    trajectory,
                    points=self.config.trajectory_points,
                    strategy=self.trajectory_feature_strategy,
                )
            features.append(cached[key])
        if self.trajectory_feature_strategy == "self-attention":
            return tuple(features)
        return np.asarray(features, dtype=np.float32)

    def propose(
        self, observation: ArrayLike, *, randomize: bool = False
    ) -> Proposal | None:
        """Rank bounded candidates and select the best finite score in the first valid batch."""
        self._validate_observation(observation)
        if self.stop_reason == "observed_topple":
            return None
        context = self.build_context(observation)
        distribution = self.model.distribution(
            context.observation,
            context.real_trajectory,
            context.reference_action,
            uniform=randomize,
        )
        self.stop_reason = None
        if randomize:
            delta = distribution.sample(1)[0]
            forecasts = self._forecast(context, delta)
            return self._build_proposal(context, delta, forecasts)
        return self._search_proposals(context, distribution)

    def _search_proposals(
        self,
        context: ForecastContext,
        distribution: "ActionDeltaDistribution",
    ) -> Proposal | None:
        for _ in range(self.config.max_batches):
            deltas = np.asarray(
                distribution.sample(self.config.candidates_per_batch), dtype=float
            )
            if deltas.shape != (self.config.candidates_per_batch, 3):
                raise ValueError("proposal distribution must sample shape (count, 3)")
            valid_deltas, forecasts = self._forecast_candidates(context, deltas)
            proposal = self._select_proposal(context, valid_deltas, forecasts)
            if proposal is not None:
                return proposal
        self.stop_reason = "search_budget_exhausted"
        return None

    def _forecast_candidates(
        self, context: ForecastContext, deltas: np.ndarray
    ) -> tuple[list[np.ndarray], list[Forecasts]]:
        valid_deltas, forecasts = [], []
        for delta in deltas:
            action = context.reference_action + delta
            if (
                not np.all(np.isfinite(action))
                or np.any(action < -1)
                or np.any(action > 1)
            ):
                continue
            forecast = self._forecast(context, delta)
            valid_deltas.append(delta)
            forecasts.append(forecast)
        return valid_deltas, forecasts

    def _score_candidates(
        self,
        context: ForecastContext,
        deltas: list[np.ndarray],
        forecasts: list[Forecasts],
    ) -> tuple[np.ndarray, np.ndarray]:
        scores = np.asarray(
            self.model.score(
                context.observation,
                context.real_trajectory,
                np.asarray(deltas),
                (
                    forecasts
                    if self.trajectory_feature_strategy == "self-attention"
                    else np.asarray(forecasts)
                ),
                context.weights,
            ),
            dtype=float,
        )
        if scores.shape != (len(deltas),):
            raise ValueError("model.score must return one scalar per candidate")
        finite = np.isfinite(scores)
        return scores, finite

    def _select_proposal(
        self,
        context: ForecastContext,
        deltas: list[np.ndarray],
        forecasts: list[Forecasts],
    ) -> Proposal | None:
        if not deltas:
            return None
        scores, finite = self._score_candidates(context, deltas, forecasts)
        if not np.any(finite):
            return None
        selected = int(np.argmax(np.where(finite, scores, -np.inf)))
        return self._build_proposal(context, deltas[selected], forecasts[selected])

    @staticmethod
    def _build_proposal(
        context: ForecastContext,
        delta: np.ndarray,
        forecasts: Forecasts,
    ) -> Proposal:
        return Proposal(
            _copy_readonly_array(delta, shape=(3,)),
            _copy_readonly_array(context.reference_action + delta, shape=(3,)),
            _copy_readonly_forecasts(forecasts),
            context.weights,
            context,
        )

    def forecast_for_training(
        self, example: "TrainingExample", delta: ArrayLike
    ) -> Forecasts:
        """Regenerate forecasts from the transition's saved physics context."""
        context = example.metadata
        if not isinstance(context, ForecastContext):
            raise TypeError("TrainingExample.metadata must be a ForecastContext")
        delta = np.asarray(delta, dtype=float)
        if delta.shape != (3,) or not np.all(np.isfinite(delta)):
            raise ValueError("delta must have three finite entries")
        if self.config.world_model == "physics":
            action = context.reference_action + delta
            if np.any(action < -1) or np.any(action > 1):
                raise ValueError("forecast action must lie within [-1, 1]")
        return self._forecast(context, delta)

    def close(self) -> None:
        if self._closed:
            return
        if self.estimator is not None and self._owns_estimator:
            self.estimator.close()
        if self.simulator is not None and self._owns_simulator:
            self.simulator.close()
        self._closed = True
