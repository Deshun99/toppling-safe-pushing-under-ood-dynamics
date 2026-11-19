"""Posterior and replay measurements shared by SysID evaluation workflows."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from src.agents.simulator.internal_sim import InternalSim
from src.agents.system_identification.smc2 import CHANNELS, SMC2SysID
from src.evaluation.system_identification.shared.constants import PARAMETERS, SPEC
from src.simulation.execution.force_push import DenseTrajectory
from src.simulation.modelling.object import CuboidSpec
from src.simulation.modelling.records import ActionRecord


def compute_weighted_quantiles(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    probabilities: Sequence[float],
) -> NDArray[np.float64]:
    """Inverse empirical CDF, retaining discrete mass and repeated particles."""
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / np.sum(weights)
    indices = np.searchsorted(cumulative, probabilities, side="left")
    return values[order[np.clip(indices, 0, len(values) - 1)]]


def evaluate_predictions(
    sim: InternalSim, spec: CuboidSpec, records: Sequence[ActionRecord]
) -> dict[str, float]:
    """Report all-sample and upright-only errors; retain held-out topples."""
    errors = {channel: [] for channel in CHANNELS}
    upright_errors = {channel: [] for channel in CHANNELS}
    matches = []
    for record in records:
        predicted = sim.replay(spec, record)
        actual_topple = max(abs(value) for value in record.traj.psi) > SPEC.crit_tilt
        matches.append(
            actual_topple == (max(abs(v) for v in predicted.psi) > spec.crit_tilt)
        )
        for channel in CHANNELS:
            squared = _compute_squared_errors(record, predicted, channel)
            errors[channel].extend(squared)
            if not actual_topple:
                upright_errors[channel].extend(squared)
    return _summarize_prediction_errors(errors, upright_errors, matches)


def _compute_squared_errors(
    record: ActionRecord,
    predicted: DenseTrajectory,
    channel: str,
) -> NDArray[np.float64]:
    actual = np.asarray(getattr(record.traj, channel))
    simulated = np.asarray(getattr(predicted, channel))
    if actual.shape != simulated.shape or predicted.t != record.traj.t:
        raise ValueError("Held-out replay must align with target samples")
    return (actual - simulated) ** 2


def summarize_particles(estimator: SMC2SysID, spec: CuboidSpec = SPEC) -> dict:
    """Summarize the full posterior, not only the retained top-K set."""
    theta = convert_to_physical_parameters(estimator.particles, spec.H)
    weights = estimator.weights
    result = {}
    for index, (name, truth) in enumerate(zip(PARAMETERS, spec.theta)):
        low, median, high = compute_weighted_quantiles(
            theta[:, index], weights, (0.025, 0.5, 0.975)
        )
        result[name] = {
            "mean": float(weights @ theta[:, index]),
            "median": float(median),
            "low_95": float(low),
            "high_95": float(high),
            "covers_truth": bool(low <= truth <= high),
        }
    return result


def convert_to_physical_parameters(points: np.ndarray, height: float) -> np.ndarray:
    """Convert (log mass, friction, relative COM) to (mass, friction, COM height)."""
    theta = points.copy()
    theta[:, 0] = np.exp(theta[:, 0])
    theta[:, 2] *= height
    return theta


def summarize_distribution(
    points: np.ndarray,
    weights: np.ndarray,
    spec: CuboidSpec,
) -> dict:
    theta = convert_to_physical_parameters(points, spec.H)
    mean = weights @ theta
    centered = theta - mean
    covariance = (centered * weights[:, None]).T @ centered
    std = np.sqrt(np.diag(covariance))
    correlation = compute_correlation(covariance, std)
    intervals = {
        name: compute_weighted_quantiles(
            theta[:, col], weights, (0.025, 0.975)
        ).tolist()
        for col, name in enumerate(PARAMETERS)
    }
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "correlation": correlation.tolist(),
        "intervals_95": intervals,
        "covers_truth": {
            name: bool(intervals[name][0] <= truth <= intervals[name][1])
            for name, truth in zip(PARAMETERS, spec.theta)
        },
    }


def _summarize_prediction_errors(
    errors: dict[str, list],
    upright_errors: dict[str, list],
    matches: list[bool],
) -> dict[str, float]:
    result = {f"rmse_{c}": float(np.sqrt(np.mean(v))) for c, v in errors.items()}
    result.update(
        {
            f"upright_rmse_{c}": float(np.sqrt(np.mean(v)))
            for c, v in upright_errors.items()
            if v
        }
    )
    result["topple_accuracy"] = float(np.mean(matches))
    return result


def compute_correlation(covariance: np.ndarray, std: np.ndarray) -> np.ndarray:
    denominator = np.outer(std, std)
    return np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 0,
    )
