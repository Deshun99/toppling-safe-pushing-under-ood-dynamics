"""Numerical grid integration and cache validation for SysID diagnostics."""

from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

from src.agents.system_identification.smc2 import SMC2SysID
from src.agents.system_identification.sysid import Z_HI, Z_LO
from src.evaluation.system_identification.shared.constants import (
    CONVERGENCE_DISTANCE,
    CONVERGENCE_LOG_EVIDENCE,
    PARAMETERS,
    RETAINED_MASS,
    Grid,
)
from src.evaluation.system_identification.shared.metrics import (
    compute_correlation,
    summarize_distribution,
)
from src.simulation.modelling.object import CuboidSpec
from src.simulation.modelling.records import ActionRecord


def load_checked_grid(
    path: Path,
    spec: CuboidSpec,
    records: list[ActionRecord],
    stride: int,
) -> tuple[Grid, float]:
    with np.load(path) as saved:
        grid = Grid(saved["low"], saved["high"], saved["log_likelihood"])
    if grid.log_likelihood.shape != (len(grid.low), len(records)):
        raise ValueError("Saved grid likelihoods do not match the record count")
    difference = _measure_likelihood_difference(grid, spec, records, stride)
    if difference > 1e-8:
        raise ValueError("Saved grid likelihoods differ from the current model")
    return grid, difference


def _measure_likelihood_difference(
    grid: Grid,
    spec: CuboidSpec,
    records: list[ActionRecord],
    stride: int,
) -> float:
    with SMC2SysID(
        (spec.w, spec.L, spec.H),
        n_theta=1,
        n_state=1,
        start_std=(0, 0, 0, 0),
        sample_stride=stride,
    ) as probe:
        indices = np.linspace(0, len(grid.low) - 1, 32, dtype=int)
        replayed = np.array(
            [
                [
                    probe._evaluate_record(grid.points[i], record)[1]
                    for record in records
                ]
                for i in indices
            ]
        )
    return float(np.max(abs(replayed - grid.log_likelihood[indices])))


def create_uniform_cells(size: int) -> tuple[np.ndarray, np.ndarray]:
    axes = [np.arange(size) for _ in range(len(Z_LO))]
    indices = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
        -1, len(Z_LO)
    )
    widths = (Z_HI - Z_LO) / size
    low = Z_LO + indices * widths
    return (low, low + widths)


def split_cells(low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.stack(
        np.meshgrid(*[[0, 1]] * len(Z_LO), indexing="ij"), axis=-1
    ).reshape(-1, len(Z_LO))
    half = (high - low) / 2
    children = low[:, None, :] + offsets[None, :, :] * half[:, None, :]
    return (
        children.reshape(-1, len(Z_LO)),
        (children + half[:, None, :]).reshape(-1, len(Z_LO)),
    )


def select_posterior_mass(
    weights: np.ndarray, fraction: float = RETAINED_MASS
) -> np.ndarray:
    order = np.argsort(-weights)
    count = min(
        len(order), int(np.searchsorted(np.cumsum(weights[order]), fraction)) + 1
    )
    return order[:count]


def integrate_marginal(
    grid: Grid, step: int, col: int, height: float
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate cell marginals with eight Gauss-Legendre nodes per interval.

    The midpoint likelihood is constant within each cell. Distributing its mass
    uniformly in z avoids treating broad, well-integrated cells as point masses.
    """
    nodes, quadrature_weights = np.polynomial.legendre.leggauss(8)
    weights, _ = grid.posterior(step)
    low, high = (grid.low[:, col, None], grid.high[:, col, None])
    values = (low + high) / 2 + (high - low) / 2 * nodes
    if col == 0:
        values = np.exp(values)
    elif col == 2:
        values *= height
    return (
        values.ravel(),
        (weights[:, None] * quadrature_weights[None, :] / 2).ravel(),
    )


def compute_grid_quantiles(
    grid: Grid, step: int, col: int, height: float
) -> list[float]:
    """Invert the piecewise-uniform marginal CDF, without node quantization."""
    weights, _ = grid.posterior(step)
    low, high = (grid.low[:, col], grid.high[:, col])
    result = []
    for probability in (0.025, 0.975):
        left, right = (Z_LO[col], Z_HI[col])
        for _ in range(48):
            middle = (left + right) / 2
            cdf = np.sum(weights * np.clip((middle - low) / (high - low), 0, 1))
            if cdf < probability:
                left = middle
            else:
                right = middle
        value = float((left + right) / 2)
        if col == 0:
            value = float(np.exp(value))
        elif col == 2:
            value *= height
        result.append(value)
    return result


def summarize_grid(grid: Grid, step: int, spec: CuboidSpec) -> dict:
    weights, _ = grid.posterior(step)
    result = summarize_distribution(grid.points, weights, spec)
    means, variances, intervals = ([], [], {})
    for col, name in enumerate(PARAMETERS):
        values, mass = integrate_marginal(grid, step, col, spec.H)
        mean = float(mass @ values)
        means.append(mean)
        variances.append(float(mass @ (values - mean) ** 2))
        intervals[name] = compute_grid_quantiles(grid, step, col, spec.H)
    result.update(
        mean=means,
        std=np.sqrt(variances).tolist(),
        intervals_95=intervals,
        covers_truth={
            name: bool(intervals[name][0] <= truth <= intervals[name][1])
            for name, truth in zip(PARAMETERS, spec.theta)
        },
    )
    result["correlation"] = compute_grid_correlation(
        grid, weights, means, variances, spec.H
    )
    return result


def compute_grid_correlation(
    grid: Grid,
    weights: np.ndarray,
    means: list[float],
    variances: list[float],
    height: float,
) -> list[list[float]]:
    conditional_means = grid.points.copy()
    width = grid.high[:, 0] - grid.low[:, 0]
    conditional_means[:, 0] = (np.exp(grid.high[:, 0]) - np.exp(grid.low[:, 0])) / width
    conditional_means[:, 2] *= height
    centered = conditional_means - means
    covariance = (centered * weights[:, None]).T @ centered
    np.fill_diagonal(covariance, variances)
    return compute_correlation(covariance, np.sqrt(variances)).tolist()


def compare_grids(previous: Grid, current: Grid, step: int, spec: CuboidSpec) -> dict:
    _, old_evidence = previous.posterior(step)
    _, evidence = current.posterior(step)
    ranges = get_parameter_spans(spec.H)
    distances = []
    for col in range(len(PARAMETERS)):
        old_values, old_mass = integrate_marginal(previous, step, col, spec.H)
        values, mass = integrate_marginal(current, step, col, spec.H)
        distances.append(
            float(
                wasserstein_distance(old_values, values, old_mass, mass) / ranges[col]
            )
        )
    delta = abs(evidence - old_evidence)
    return {
        "normalized_marginal_wasserstein": distances,
        "log_evidence_change": delta,
        "stable": bool(
            max(distances) < CONVERGENCE_DISTANCE and delta < CONVERGENCE_LOG_EVIDENCE
        ),
    }


def evaluate_cells(
    low: np.ndarray,
    high: np.ndarray,
    estimator: SMC2SysID,
    records: list[ActionRecord],
) -> Grid:
    likelihood = np.empty((len(low), len(records)))
    for index, point in enumerate((low + high) / 2):
        for col, record in enumerate(records):
            likelihood[index, col] = estimator._evaluate_record(point, record)[1]
        if (index + 1) % 2000 == 0:
            print(f"  evaluated {index + 1}/{len(low)} cells", flush=True)
    return Grid(low, high, likelihood)


def refine_grid(
    grid: Grid,
    estimator: SMC2SysID,
    records: list[ActionRecord],
    max_new_cells: int = 20000,
) -> Grid:
    selected = select_refinement_cells(grid, len(records), max_new_cells)
    keep = np.ones(len(grid.low), dtype=bool)
    keep[selected] = False
    low, high = split_cells(grid.low[selected], grid.high[selected])
    children = evaluate_cells(low, high, estimator, records)
    return Grid(
        np.concatenate((grid.low[keep], children.low)),
        np.concatenate((grid.high[keep], children.high)),
        np.concatenate((grid.log_likelihood[keep], children.log_likelihood)),
    )


def get_parameter_spans(height: float) -> np.ndarray:
    return np.array(
        [
            np.exp(Z_HI[0]) - np.exp(Z_LO[0]),
            Z_HI[1] - Z_LO[1],
            (Z_HI[2] - Z_LO[2]) * height,
        ]
    )


def select_refinement_cells(grid: Grid, steps: int, max_new_cells: int) -> np.ndarray:
    selected = np.unique(
        np.concatenate(
            [
                select_posterior_mass(grid.posterior(step)[0])
                for step in range(1, steps + 1)
            ]
        )
    )
    if len(selected) * 8 > max_new_cells:
        priority = np.max(
            np.stack([grid.posterior(step)[0] for step in range(1, steps + 1)]),
            axis=0,
        )
        selected = selected[np.argsort(-priority[selected])[: max_new_cells // 8]]
    return selected
