"""SMC² for the conditional action-replay model; see docs/smc2.tex.

Each action is a block observation. Inner particles integrate uncertainty in
its measured start state; outer particles infer static physics parameters.
This is not a continuous-state tracker across actions. No target theta is read.
"""

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, replace
from types import TracebackType
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

from src.agents.simulator.internal_sim import InternalSim
from src.agents.system_identification.scoring import get_scoring_indices
from src.simulation.execution.force_push import DenseTrajectory
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.constants import DT
from src.simulation.modelling.object import CuboidSpec, Geometry, PhysicsParameters
from src.simulation.modelling.records import ActionRecord

from .sysid import (
    LOSS_SCALES,
    N_PHYSICS_PARAMETERS,
    Z_HI,
    Z_LO,
    Hypothesis,
    SysIDResult,
    convert_z_to_cuboid_spec,
)

CHANNELS = ("x", "psi", "xdot", "psidot")
PARTICLE_HISTOGRAM_BINS = 20
PARAMETER_LABELS = ("Mass m (kg)", "Friction μ", "COM height h_c (m)")
MAX_TEMPERING_STAGES = 1000
TEMPERING_BISECTION_STEPS = 64
PROPOSAL_COVARIANCE_FLOOR = 1e-6  # Relative to mean variance in scaled coordinates.

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure


@dataclass
class InnerFilter:
    """Last block's start-state particles, posterior weights, and evidence."""

    starts: NDArray[
        np.float64
    ]  # sampled starting states for the latest actual push in real env
    weights: NDArray[np.float64]  # normalized likelihood weights for each start
    log_likelihood: float  # cumulative sum of log likelihood estimates


class SMC2SysID:
    """Sequential Bayesian alternative to SysID, with the same update(history).

    History must be append-only and contain upright action records for one
    object. Repeated calls with unchanged history do not assimilate it twice.
    Use a new instance for a different object. Defaults are exploratory noise
    assumptions, not calibrated uncertainty guarantees. A supplied replay sim
    is owned by its caller; otherwise close() releases the internal client.
    Warm-up is replayed but always excluded from scoring. For a known-start
    reference use start_std=(0, 0, 0, 0) and n_state=1 explicitly.

    geometry: (w, L, H) of the cuboid to identify.
    K: number of unique hypotheses to return in result().
    n_theta: number of outer particles (physics parameters).
    n_state: number of inner particles (start states).
    ess_fraction: resample when effective sample size falls below this fraction of n_theta.
    mcmc_steps: random-walk proposals per particle per resampling stage.
    proposal_scale: random-walk scale relative to physics prior ranges.
    proposal: weighted particle covariance by default, frozen before each
        resampling; "fixed" selects independent perturbations. Both have the
        same mean squared jump in prior-scaled coordinates. The covariance
        option changes shape, not the overall proposal scale.
    sample_stride: downsampling factor for trajectory likelihood evaluation.
        Warm-up is simulated and recorded but never scored.
    temper_updates: introduce each push gradually; requires zero start_std
        and n_state=1. Uses ess_fraction to control conditional ESS. The
        final target uses the full likelihood; proposal scale and moves
        per stage are unchanged.
    start_std: 4-tuple of stddevs for initial state uncertainty (x, psi, xdot, psidot).
    observation_std: 4-tuple of stddevs for trajectory likelihood evaluation; defaults to 1/sqrt(loss_scale).
    seed: random number generator seed for reproducibility.
    sim: optional InternalSim instance; if None, a new one is created and owned by this estimator.

    """

    def __init__(
        self,
        geometry: Geometry,
        K: int = 4,
        *,
        n_theta: int = 64,
        n_state: int = 8,
        ess_fraction: float = 0.5,
        mcmc_steps: int = 2,
        proposal_scale: float = 0.08,
        proposal: Literal["fixed", "covariance"] = "covariance",
        sample_stride: int = 12,
        temper_updates: bool = False,
        start_std: ArrayLike = (0.01, 0.002, 0.01, 0.02),
        observation_std: ArrayLike | None = None,
        seed: int = 0,
        sim: InternalSim | None = None,
    ) -> None:
        self.K, self.n_theta, self.n_state = K, n_theta, n_state
        self.proposal_scale, self.sample_stride = proposal_scale, sample_stride
        self.proposal, self.temper_updates = proposal, temper_updates
        self.ess_fraction, self.mcmc_steps = ess_fraction, mcmc_steps
        self.geometry = tuple(geometry)
        self._set_noise_scales(start_std, observation_std)
        self._validate_args()
        self._initialize_particles(seed)
        self._owns_sim = sim is None
        self.sim = InternalSim() if sim is None else sim
        self._closed = False

    def _set_noise_scales(
        self, start_std: ArrayLike, observation_std: ArrayLike | None
    ) -> None:
        self.start_std = np.asarray(start_std, dtype=float).copy()
        if observation_std is None:
            observation_std = [
                1 / np.sqrt(LOSS_SCALES[channel]) for channel in CHANNELS
            ]
        self.observation_std = np.asarray(observation_std, dtype=float).copy()

    def _initialize_particles(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        # Uniform in z: log-uniform mass, uniform friction and COM fraction.
        self._z = self.rng.uniform(
            Z_LO, Z_HI, size=(self.n_theta, N_PHYSICS_PARAMETERS)
        )
        self._log_weights = np.full(self.n_theta, np.log(1 / self.n_theta))
        self._filters = [
            InnerFilter(np.empty((0, len(CHANNELS))), np.empty(0), 0.0)
            for _ in range(self.n_theta)
        ]
        self._history: list[ActionRecord] = []
        self._proposal_transform: NDArray[np.float64] | None = None
        self.resampling_count = self.proposals = self.acceptances = 0
        self.last_ess = float(self.n_theta)
        self.tempering_history: list[dict[str, float | int | bool]] = []

    def _validate_args(self) -> None:
        if self.proposal not in ("fixed", "covariance"):
            raise ValueError("proposal must be 'fixed' or 'covariance'")
        for name, value in (
            ("K", self.K),
            ("n_theta", self.n_theta),
            ("n_state", self.n_state),
            ("sample_stride", self.sample_stride),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.mcmc_steps, int) or self.mcmc_steps < 0:
            raise ValueError("mcmc_steps must be a nonnegative integer")
        if (
            not 0 < self.ess_fraction <= 1
            or not np.isfinite(self.proposal_scale)
            or self.proposal_scale <= 0
        ):
            raise ValueError("invalid ESS fraction or proposal scale")
        self._validate_measurement_settings()

    def _validate_measurement_settings(self) -> None:
        if (
            np.shape(self.geometry) != (3,)
            or not np.all(np.isfinite(self.geometry))
            or np.any(np.asarray(self.geometry) <= 0)
        ):
            raise ValueError("geometry must contain three positive finite dimensions")

        for name, values, allow_zero in (
            ("start_std", self.start_std, True),
            ("observation_std", self.observation_std, False),
        ):
            if (
                values.shape != (len(CHANNELS),)
                or not np.all(np.isfinite(values))
                or np.any(values < 0 if allow_zero else values <= 0)
            ):
                raise ValueError(
                    f"{name} needs {len(CHANNELS)} finite {'nonnegative' if allow_zero else 'positive'} values"
                )
        if self.temper_updates and (
            self.n_state != 1 or np.any(self.start_std != 0) or self.ess_fraction >= 1
        ):
            raise ValueError(
                "Tempering requires n_state=1, zero start_std, and ess_fraction < 1"
            )

    @property
    def particles(self) -> NDArray[np.float64]:
        """Copy of all z particles (log mass, friction, relative COM)."""
        return self._z.copy()

    @property
    def weights(self) -> NDArray[np.float64]:
        return np.exp(self._log_weights)

    @property
    def inner_filters(self) -> list[InnerFilter]:
        """Copy of all inner filters."""
        return deepcopy(self._filters)

    def plot_particle_distribution(
        self,
        *,
        ground_truth: PhysicsParameters | None = None,
    ) -> "Figure":
        """Plot weighted posterior mass, with fixed prior bounds and optional truth.

        Repeated particles contribute their combined mass. The estimator is unchanged;
        callers own the returned figure and decide whether to display or save it.
        """
        import matplotlib.pyplot as plt

        if ground_truth is not None and (
            np.shape(ground_truth) != (N_PHYSICS_PARAMETERS,)
            or not np.all(np.isfinite(ground_truth))
        ):
            raise ValueError(
                "ground_truth must contain three finite physical parameters"
            )
        theta, edges = self._prepare_plot_data()
        figure, axes = plt.subplots(3, 3, figsize=(9, 8), layout="constrained")
        self._draw_marginal_histograms(axes, theta, edges, ground_truth)
        mesh, pair_axes = self._draw_pair_histograms(axes, theta, edges, ground_truth)
        if ground_truth is not None:
            axes[0, 0].lines[0].set_label("Ground truth")
            axes[0, 0].legend()
        figure.colorbar(mesh, ax=pair_axes, label="Probability mass per 2D bin")
        figure.suptitle(f"Physics posterior · {self.n_theta} particles")
        return figure

    def _prepare_plot_data(
        self,
    ) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
        theta = self.particles
        theta[:, 0] = np.exp(theta[:, 0])
        theta[:, 2] *= self.geometry[2]
        bounds = np.column_stack((Z_LO, Z_HI))
        bounds[0] = np.exp(bounds[0])
        bounds[2] *= self.geometry[2]
        edges = [
            np.linspace(low, high, PARTICLE_HISTOGRAM_BINS + 1) for low, high in bounds
        ]
        return theta, edges

    def _draw_marginal_histograms(
        self,
        axes: np.ndarray,
        theta: NDArray[np.float64],
        edges: list[NDArray[np.float64]],
        ground_truth: PhysicsParameters | None,
    ) -> None:
        for row in range(N_PHYSICS_PARAMETERS):
            for column in range(row + 1, N_PHYSICS_PARAMETERS):
                axes[row, column].set_visible(False)
            axis = axes[row, row]
            axis.set_xlim(edges[row][0], edges[row][-1])
            axis.hist(
                theta[:, row], bins=edges[row], weights=self.weights, color="steelblue"
            )
            axis.set(xlabel=PARAMETER_LABELS[row], ylabel="Probability mass")
            if ground_truth is not None:
                self._mark_ground_truth(axis, ground_truth, row, row)

    def _compute_pair_histograms(
        self,
        theta: NDArray[np.float64],
        edges: list[NDArray[np.float64]],
    ) -> dict[tuple[int, int], NDArray[np.float64]]:
        return {
            (row, col): np.histogram2d(
                theta[:, col],
                theta[:, row],
                bins=(edges[col], edges[row]),
                weights=self.weights,
            )[0]
            for row in range(N_PHYSICS_PARAMETERS)
            for col in range(row)
        }

    def _draw_pair_histograms(
        self,
        axes: np.ndarray,
        theta: NDArray[np.float64],
        edges: list[NDArray[np.float64]],
        ground_truth: PhysicsParameters | None,
    ) -> tuple["QuadMesh", list["Axes"]]:
        from matplotlib.colors import Normalize

        pairs = self._compute_pair_histograms(theta, edges)
        maximum = max(mass.max() for mass in pairs.values())
        color_scale = Normalize(vmin=0, vmax=maximum)
        plot_options = dict(cmap="viridis", norm=color_scale, shading="auto")
        pair_axes = []
        for (row, col), mass in pairs.items():
            axis = axes[row, col]
            mesh = axis.pcolormesh(edges[col], edges[row], mass.T, **plot_options)
            axis.set(
                xlim=(edges[col][0], edges[col][-1]),
                ylim=(edges[row][0], edges[row][-1]),
                xlabel=PARAMETER_LABELS[col],
                ylabel=PARAMETER_LABELS[row],
            )
            pair_axes.append(axis)
            if ground_truth is not None:
                self._mark_ground_truth(axis, ground_truth, row, col)
        return mesh, pair_axes

    @staticmethod
    def _mark_ground_truth(
        axis: "Axes", truth: PhysicsParameters, row: int, col: int
    ) -> None:
        axis.axvline(truth[col], color="crimson", linestyle="--")
        low, high = axis.get_xlim()
        axis.set_xlim(min(low, truth[col]), max(high, truth[col]))
        if row != col:
            axis.axhline(truth[row], color="crimson", linestyle="--")
            axis.plot(
                truth[col],
                truth[row],
                marker="*",
                color="crimson",
                markeredgecolor="white",
                markersize=12,
                clip_on=False,
            )
            low, high = axis.get_ylim()
            axis.set_ylim(min(low, truth[row]), max(high, truth[row]))

    @staticmethod
    def _trajectory_array(traj: DenseTrajectory) -> NDArray[np.float64]:
        """
        Convert a DenseTrajectory to a 2D array of shape (len(t), len(CHANNELS)),
        with columns in the order of CHANNELS. Raises ValueError if the trajectory
        is empty, has misaligned timestamps, or contains non-finite values.
        """
        data = np.asarray([getattr(traj, k) for k in CHANNELS], dtype=float).T
        if (
            data.ndim != 2
            or data.shape != (len(traj.t), len(CHANNELS))
            or not len(data)
            or not np.all(np.isfinite(data))
        ):
            raise ValueError("trajectory must have nonempty, finite, aligned channels")
        return data

    def _validate_record(self, record: ActionRecord) -> None:
        self._trajectory_array(record.traj)
        if np.shape(record.start) != (len(CHANNELS),) or not np.all(
            np.isfinite(record.start)
        ):
            raise ValueError("invalid start state")
        if (
            not isinstance(record.action, PushAction)
            or not np.all(
                np.isfinite([record.action.h, record.action.F, record.action.tau])
            )
            or record.action.tau <= 0
            or record.action.F < 0
            or not 0 <= record.action.h <= self.geometry[2]
        ):
            raise ValueError("invalid physical push action")
        if not isinstance(record.warmup_steps, int) or record.warmup_steps < 0:
            raise ValueError("invalid warmup_steps")
        n = record.warmup_steps + max(1, round(record.action.tau / DT))
        if len(record.traj.t) != n or not np.allclose(
            record.traj.t, np.arange(n) * DT, rtol=0, atol=1e-10
        ):
            raise ValueError(
                "trajectory timestamps must match the replay physics steps"
            )
        get_scoring_indices(
            len(record.traj.t),
            warmup_steps=record.warmup_steps,
            sample_stride=self.sample_stride,
        )

    def _compute_log_likelihood(
        self, observed: NDArray[np.float64], predicted: NDArray[np.float64]
    ) -> float:
        """Sum Gaussian log densities over aligned samples and state channels.

        Errors are independent with per-channel scales observation_std.
        Each log density is -0.5 * ((observed - predicted) / sigma)**2
        - log(sigma * sqrt(2*pi)); the constants normalize the Gaussian.
        """
        residual = (observed - predicted) / self.observation_std
        squared_error_term = -0.5 * np.sum(residual**2)
        normalization_term = len(observed) * np.sum(
            np.log(self.observation_std * np.sqrt(2 * np.pi))
        )
        return float(squared_error_term - normalization_term)

    def _evaluate_record(
        self,
        z: NDArray[np.float64],
        record: ActionRecord,
        cumulative: float = 0.0,
    ) -> tuple[InnerFilter, float]:
        """Integrate newly sampled starts; leave outer weights and history unchanged."""
        indices = get_scoring_indices(
            len(record.traj.t),
            warmup_steps=record.warmup_steps,
            sample_stride=self.sample_stride,
        )
        starts = self.rng.normal(
            record.start, self.start_std, size=(self.n_state, len(CHANNELS))
        )
        observed = self._trajectory_array(record.traj)[indices]
        spec = convert_z_to_cuboid_spec(z, self.geometry)
        scores = [
            self._score_replay(spec, record, start, observed, indices)
            for start in starts
        ]
        normalizer = logsumexp(scores)  # log(sum of likelihoods)
        if not np.isfinite(normalizer):
            raise FloatingPointError(
                "nonfinite inner likelihood; check data and noise scales"
            )
        increment = float(normalizer - np.log(self.n_state))
        return InnerFilter(
            starts, np.exp(scores - normalizer), cumulative + increment
        ), increment

    def _score_replay(
        self,
        spec: CuboidSpec,
        record: ActionRecord,
        start: NDArray[np.float64],
        observed: NDArray[np.float64],
        indices: NDArray[np.intp],
    ) -> float:
        predicted = self.sim.replay(spec, replace(record, start=tuple(start)))
        states = self._trajectory_array(predicted)
        if len(states) != len(record.traj.t) or not np.allclose(
            predicted.t, record.traj.t, rtol=0, atol=1e-10
        ):
            raise ValueError("replay and observation trajectories must align exactly")
        return self._compute_log_likelihood(observed, states[indices])

    def _run_filter(
        self, z: NDArray[np.float64], history: Iterable[ActionRecord]
    ) -> InnerFilter | None:
        state = None
        cumulative = 0.0
        for record in history:
            state, _ = self._evaluate_record(z, record, cumulative)
            cumulative = state.log_likelihood
        return state

    def _compute_proposal_transform(self) -> NDArray[np.float64]:
        """Shape the random walk using the weighted cloud before resampling.

        Estimate covariance in prior-scaled z coordinates, normalize its trace
        to the parameter count, and regularize to keep every direction mobile.
        A collapsed cloud falls back to the fixed isotropic shape. The common
        transform stays frozen throughout a stage, so each Gaussian proposal
        is symmetric; out-of-bounds proposals are still rejected, never clipped.
        Population adaptation is an SMC approximation, not a finite-population
        guarantee of exact posterior draws.
        """
        ranges = Z_HI - Z_LO
        scaled = (self._z - Z_LO) / ranges
        centered = scaled - self.weights @ scaled
        covariance = (centered * self.weights[:, None]).T @ centered
        mean_variance = np.trace(covariance) / N_PHYSICS_PARAMETERS
        if mean_variance <= np.finfo(float).eps:
            shape = np.eye(N_PHYSICS_PARAMETERS)
        else:
            shape = covariance / mean_variance
            shape += PROPOSAL_COVARIANCE_FLOOR * np.eye(N_PHYSICS_PARAMETERS)
            shape /= 1 + PROPOSAL_COVARIANCE_FLOOR
        return self.proposal_scale * ranges[:, None] * np.linalg.cholesky(shape)

    def _propose_parameters(self, current: NDArray[np.float64]) -> NDArray[np.float64]:
        """Draw one untruncated proposal from the stage's fixed Gaussian."""
        noise = self.rng.normal(size=N_PHYSICS_PARAMETERS)
        if self._proposal_transform is None:
            # Preserve the original arithmetic and RNG stream in fixed mode.
            return current + noise * self.proposal_scale * (Z_HI - Z_LO)
        return current + self._proposal_transform @ noise

    def _resample_particles(self) -> NDArray[np.intp]:
        """Systematically resample and return ancestry for aligned caches."""
        self._proposal_transform = (
            self._compute_proposal_transform()
            if self.proposal == "covariance"
            else None
        )
        positions = (self.rng.random() + np.arange(self.n_theta)) / self.n_theta
        cdf = np.cumsum(self.weights)
        cdf[-1] = 1.0
        ancestors = np.searchsorted(cdf, positions)
        self._z = self._z[ancestors].copy()
        self._filters = [deepcopy(self._filters[i]) for i in ancestors]
        self._log_weights.fill(-np.log(self.n_theta))
        self.resampling_count += 1
        return ancestors

    def _resample_move(self) -> None:
        """
        Resample outer particles and move them with a random walk, accepting
        or rejecting each proposal with a Metropolis-Hastings step using the
        inner filter's log likelihood as the target. The random walk is
        symmetric and untruncated; proposals outside the prior bounds are
        rejected. The inner filter is recomputed for each accepted proposal.
        """
        self._resample_particles()
        for i in range(self.n_theta):
            for _ in range(self.mcmc_steps):
                self.proposals += 1
                proposed = self._propose_parameters(self._z[i])
                # Reject, never clip: the untruncated random walk is symmetric.
                if np.any(proposed < Z_LO) or np.any(proposed > Z_HI):
                    continue
                candidate = self._run_filter(proposed, self._history)
                log_ratio = candidate.log_likelihood - self._filters[i].log_likelihood
                if np.log(self.rng.random()) < min(0.0, log_ratio):
                    self._z[i] = proposed
                    self._filters[i] = candidate
                    self.acceptances += 1
                # Rejection retains the CURRENT random likelihood estimate.

    def _compute_conditional_ess(
        self, increments: NDArray[np.float64], delta: float
    ) -> float:
        """Measure the concentration caused by this extra evidence step.

        CESS = N * E_w[exp(delta*l)]**2 / E_w[exp(2*delta*l)]. It starts at N
        even with unequal incoming weights. A common likelihood offset cancels.
        See Zhou, Johansen & Aston (2016), doi:10.1080/10618600.2015.1060885.
        """
        shifted = increments - np.max(increments)
        value = 2 * logsumexp(self._log_weights + delta * shifted) - logsumexp(
            self._log_weights + 2 * delta * shifted
        )
        return float(min(self.n_theta, self.n_theta * np.exp(value)))

    def _select_tempering_beta(
        self, beta: float, increments: NDArray[np.float64]
    ) -> float:
        """Choose the largest evidence step meeting the conditional-ESS target."""
        target = self.ess_fraction * self.n_theta
        if self._compute_conditional_ess(increments, 1 - beta) >= target:
            return 1.0
        low, high = beta, 1.0
        for _ in range(TEMPERING_BISECTION_STEPS):
            middle = (low + high) / 2
            if self._compute_conditional_ess(increments, middle - beta) >= target:
                low = middle
            else:
                high = middle
        if low <= beta:
            raise RuntimeError("Tempering cannot make numerical progress")
        return low

    def _move_tempered_particles(
        self,
        record: ActionRecord,
        beta: float,
        past: NDArray[np.float64],
        increments: NDArray[np.float64],
    ) -> None:
        """Move against past + beta*new; inner filters keep full likelihoods."""
        for i in range(self.n_theta):
            for _ in range(self.mcmc_steps):
                self.proposals += 1
                proposed = self._propose_parameters(self._z[i])
                if np.any(proposed < Z_LO) or np.any(proposed > Z_HI):
                    continue
                previous = self._run_filter(proposed, self._history)
                proposed_past = 0.0 if previous is None else previous.log_likelihood
                candidate, increment = self._evaluate_record(
                    proposed, record, proposed_past
                )
                log_ratio = (proposed_past - past[i]) + beta * (
                    increment - increments[i]
                )
                if np.log(self.rng.random()) < min(0.0, log_ratio):
                    self._z[i] = proposed
                    self._filters[i] = candidate
                    past[i], increments[i] = proposed_past, increment
                    self.acceptances += 1

    def _weight_tempering_stage(
        self,
        beta: float,
        next_beta: float,
        increments: NDArray[np.float64],
    ) -> tuple[dict, bool]:
        conditional_ess = self._compute_conditional_ess(increments, next_beta - beta)
        self._log_weights += (next_beta - beta) * increments
        self._log_weights -= logsumexp(self._log_weights)
        self.last_ess = float(1 / np.sum(self.weights**2))
        event = {
            "step": len(self._history) + 1,
            "beta_from": beta,
            "beta_to": next_beta,
            "conditional_ess": conditional_ess,
            "ess_before_resampling": self.last_ess,
        }
        resampled = next_beta < 1 or self.last_ess < self.ess_fraction * self.n_theta
        return event, resampled

    def _assimilate_tempered_record(self, record: ActionRecord) -> None:
        """Introduce the new push gradually using deterministic inner likelihoods."""
        past = np.array([state.log_likelihood for state in self._filters])
        states, values = zip(
            *(
                self._evaluate_record(z, record, value)
                for z, value in zip(self._z, past)
            )
        )
        increments = np.asarray(values)
        self._filters = list(states)
        beta = 0.0
        for _ in range(MAX_TEMPERING_STAGES):
            next_beta = self._select_tempering_beta(beta, increments)
            event, resampled = self._weight_tempering_stage(beta, next_beta, increments)
            if resampled:
                ancestors = self._resample_particles()
                past, increments = past[ancestors], increments[ancestors]
                self._move_tempered_particles(record, next_beta, past, increments)
            event.update(
                resampled=resampled, unique_particles=len(np.unique(self._z, axis=0))
            )
            self.tempering_history.append(event)
            beta = next_beta
            if beta == 1:
                self._history.append(record)
                return
        raise RuntimeError(
            "Tempering exceeded the stage limit before reaching the full likelihood"
        )

    def _assimilate_record(self, record: ActionRecord) -> None:
        new_filters, increments = zip(
            *(
                self._evaluate_record(z, record, state.log_likelihood)
                for z, state in zip(self._z, self._filters)
            )
        )
        log_weights = self._log_weights + increments
        log_weights -= logsumexp(log_weights)
        self._filters = list(new_filters)
        self._log_weights = log_weights
        self._history.append(record)
        self.last_ess = float(1 / np.sum(self.weights**2))
        if self.last_ess < self.ess_fraction * self.n_theta:
            self._resample_move()

    def update(self, history: Iterable[ActionRecord]) -> SysIDResult:
        """Assimilate only appended records, without recounting prior evidence.

        History must preserve all previously assimilated records in order. A new
        estimator is required to reset, change objects, or replace observations.
        The estimator stores a deep copy so later caller mutations are detectable.
        """
        if self._closed:
            raise RuntimeError("estimator is closed")
        history = deepcopy(list(history))
        if (
            len(history) < len(self._history)
            or history[: len(self._history)] != self._history
        ):
            raise ValueError(
                "history must be append-only; create a new estimator to reset"
            )
        for record in history[len(self._history) :]:
            self._validate_record(record)
        for record in history[len(self._history) :]:
            if self.temper_updates:
                self._assimilate_tempered_record(record)
                continue
            self._assimilate_record(record)
        return self.result()

    def result(self) -> SysIDResult:
        """Top-K unique particles with aggregated, renormalized posterior mass.

        `point` is the highest-mass retained particle, not a continuous MAP.
        `loss` is negative estimated log likelihood (can be negative); it is
        NOT the CMA-ES squared-error loss. Use the full ensemble for uncertainty.
        """
        unique, inverse = np.unique(self._z, axis=0, return_inverse=True)
        mass = np.bincount(inverse, weights=self.weights)
        selected = np.argsort(-mass, kind="stable")[: self.K]
        hypotheses = []
        for j in selected:
            index = np.flatnonzero(inverse == j)[0]
            hypotheses.append(
                Hypothesis(
                    theta=tuple(
                        convert_z_to_cuboid_spec(unique[j], self.geometry).theta
                    ),
                    loss=-self._filters[index].log_likelihood,
                    weight=float(mass[j] / mass[selected].sum()),
                )
            )
        return SysIDResult(point=deepcopy(hypotheses[0]), hypotheses=hypotheses)

    def close(self) -> None:
        if not self._closed and self._owns_sim:
            self.sim.close()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
