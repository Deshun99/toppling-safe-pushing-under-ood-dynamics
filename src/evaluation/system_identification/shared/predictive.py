"""Evaluate posterior predictions on records excluded from inference."""

import numpy as np
from scipy.special import logsumexp

from src.agents.system_identification.scoring import get_scoring_indices
from src.agents.system_identification.smc2 import SMC2SysID
from src.agents.system_identification.sysid import convert_z_to_cuboid_spec
from src.evaluation.system_identification.shared.constants import (
    PREDICTIVE_BATCH_SIZE,
    PREDICTIVE_BATCHES,
    Grid,
)
from src.evaluation.system_identification.shared.grid import select_posterior_mass
from src.simulation.modelling.object import CuboidSpec
from src.simulation.modelling.records import ActionRecord


def evaluate_predictive_mixture(
    points: np.ndarray,
    weights: np.ndarray,
    estimator: SMC2SysID,
    records: list[ActionRecord],
    spec: CuboidSpec,
    grid: Grid | None = None,
) -> dict:
    """Use >=99.9% of particle mass, or sample uniformly inside posterior grid cells.

    Scores condition on measured starts. Grid predictions use independent batches
    to estimate integration error; evaluation records never affect inference.
    """
    points, mixture_weights, retained_mass = _select_hypotheses(points, weights, grid)
    measurements = [
        _evaluate_record(
            points, mixture_weights, estimator, record, spec, grid is not None
        )
        for record in records
    ]
    return _summarize_predictions(
        measurements, retained_mass, len(points), grid is not None
    )


def _summarize_predictions(
    measurements: list[dict],
    retained_mass: float,
    hypothesis_count: int,
    uses_grid: bool,
) -> dict:
    return {
        "retained_mass": retained_mass,
        "replayed_hypotheses": hypothesis_count,
        "rmse_by_record_and_channel": [row["rmse"] for row in measurements],
        "log_predictive_density_by_record": [row["score"] for row in measurements],
        "toppling_probability_by_record": [
            row["topple_probability"] for row in measurements
        ],
        "integration": "posterior_monte_carlo"
        if uses_grid
        else "weighted_particle_mixture",
        "monte_carlo_errors_by_record": [row["errors"] for row in measurements]
        if uses_grid
        else [],
    }


def _select_hypotheses(
    points: np.ndarray,
    weights: np.ndarray,
    grid: Grid | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    if grid is not None:
        rng = np.random.default_rng(729)
        count = PREDICTIVE_BATCHES * PREDICTIVE_BATCH_SIZE
        cells = rng.choice(len(weights), size=count, p=weights / weights.sum())
        points = rng.uniform(grid.low[cells], grid.high[cells])
        return points, np.full(count, 1 / count), 1.0
    selected = select_posterior_mass(weights)
    retained_mass = float(weights[selected].sum())
    return points[selected], weights[selected] / retained_mass, retained_mass


def _replay_hypotheses(
    points: np.ndarray,
    estimator: SMC2SysID,
    record: ActionRecord,
    spec: CuboidSpec,
    observed: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions, likelihoods, topples = [], [], []
    for point in points:
        candidate = convert_z_to_cuboid_spec(point, (spec.w, spec.L, spec.H))
        trajectory = estimator.sim.replay(candidate, record)
        predicted = estimator._trajectory_array(trajectory)[indices]
        predictions.append(predicted)
        likelihoods.append(estimator._compute_log_likelihood(observed, predicted))
        topples.append(
            max(abs(value) for value in trajectory.psi) > candidate.crit_tilt
        )
    return np.asarray(predictions), np.asarray(likelihoods), np.asarray(topples)


def _evaluate_record(
    points: np.ndarray,
    weights: np.ndarray,
    estimator: SMC2SysID,
    record: ActionRecord,
    spec: CuboidSpec,
    estimate_integration_error: bool,
) -> dict:
    indices = get_scoring_indices(
        len(record.traj.t),
        warmup_steps=record.warmup_steps,
        sample_stride=estimator.sample_stride,
    )
    observed = estimator._trajectory_array(record.traj)[indices]
    predictions, likelihoods, topples = _replay_hypotheses(
        points, estimator, record, spec, observed, indices
    )
    mean = np.tensordot(weights, predictions, axes=1)
    return {
        "rmse": np.sqrt(np.mean((mean - observed) ** 2, axis=0)).tolist(),
        "score": float(logsumexp(np.log(weights) + likelihoods)),
        "topple_probability": float(weights @ topples),
        "errors": _estimate_monte_carlo_errors(predictions, likelihoods)
        if estimate_integration_error
        else None,
    }


def _estimate_monte_carlo_errors(
    predictions: np.ndarray, likelihoods: np.ndarray
) -> dict:
    batches = predictions.reshape(
        PREDICTIVE_BATCHES, PREDICTIVE_BATCH_SIZE, *predictions.shape[1:]
    ).mean(axis=1)
    mean_standard_error = np.std(batches, axis=0, ddof=1) / np.sqrt(PREDICTIVE_BATCHES)
    likelihood_batches = likelihoods.reshape(PREDICTIVE_BATCHES, PREDICTIVE_BATCH_SIZE)
    log_density_batches = logsumexp(likelihood_batches, axis=1) - np.log(
        PREDICTIVE_BATCH_SIZE
    )
    return {
        "rms_predictive_mean_standard_error": np.sqrt(
            np.mean(mean_standard_error**2, axis=0)
        ).tolist(),
        "log_predictive_density_batch_std": float(np.std(log_density_batches, ddof=1)),
    }
