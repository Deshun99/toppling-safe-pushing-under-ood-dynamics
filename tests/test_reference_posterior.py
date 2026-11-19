"""Check reference integration against known probability distributions."""

from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm, truncnorm

from src.agents.system_identification.smc2 import SMC2SysID
from src.agents.system_identification.sysid import Z_HI, Z_LO
from src.evaluation.system_identification.diagnostics.reference_posterior import (
    write_reference_report,
)
from src.evaluation.system_identification.shared.constants import Grid
from src.evaluation.system_identification.shared.grid import (
    create_uniform_cells,
    select_posterior_mass,
    split_cells,
    summarize_grid,
)
from src.evaluation.system_identification.shared.predictive import (
    evaluate_predictive_mixture,
)
from src.simulation.execution.force_push import DenseTrajectory
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.constants import DT
from src.simulation.modelling.object import CuboidSpec
from src.simulation.modelling.records import ActionRecord

SPEC = CuboidSpec(w=0.08, L=0.08, H=0.2, m=1, mu=0.3, h_c=0.1)


def test_grid_report_does_not_claim_failure_when_all_com_intervals_cover_truth(
    tmp_path: Path,
) -> None:
    distribution = {
        "intervals_95": {"m": [0.9, 1.1], "mu": [0.2, 0.4], "h_c": [0.05, 0.15]},
        "covers_truth": {"m": True, "mu": True, "h_c": True},
    }
    write_reference_report(
        [
            {
                "step": 1,
                "cross_grid": {"stable": True},
                "reference": distribution,
                "smc2": [
                    {"seed": 1, "unique_particles": 64, "distribution": distribution}
                ],
            }
        ],
        tmp_path,
    )
    report = (tmp_path / "report.md").read_text()
    assert "No final COM intervals exclude truth" in report
    assert "directly demonstrates loss" not in report


def test_selected_mass_uses_ranked_weights() -> None:
    weights = np.array([0.01, 0.02, 0.7, 0.27])
    selected = select_posterior_mass(weights, 0.9)
    np.testing.assert_array_equal(selected, [2, 3])
    assert weights[selected].sum() >= 0.9


def test_refining_constant_density_does_not_create_probability_mass() -> None:
    low, high = create_uniform_cells(2)
    child_low, child_high = split_cells(low[:1], high[:1])
    grid = Grid(
        np.concatenate((child_low, low[1:])),
        np.concatenate((child_high, high[1:])),
        np.zeros((15, 1)),
    )
    weights, log_evidence = grid.posterior(1)
    assert weights[:8].sum() == pytest.approx(1 / 8)
    assert weights.sum() == pytest.approx(1)
    assert log_evidence == pytest.approx(0, abs=1e-14)
    summary = summarize_grid(grid, 1, SPEC)
    expected_mass = (np.exp(Z_HI[0]) - np.exp(Z_LO[0])) / (Z_HI[0] - Z_LO[0])
    np.testing.assert_allclose(summary["mean"], [expected_mass, 0.6, 0.1], rtol=1e-10)
    # The prior is independent, including after a nonuniform partition.
    np.testing.assert_allclose(summary["correlation"], np.eye(3), atol=1e-12)


def test_grid_evidence_and_marginals_match_truncated_gaussian() -> None:
    low, high = create_uniform_cells(64)
    points = (low + high) / 2
    mean = np.array([0, 0.3, 0.5])
    sigma = np.array([0.18, 0.06, 0.12])
    log_likelihood = norm.logpdf(points, mean, sigma).sum(axis=1)[:, None]
    grid = Grid(low, high, log_likelihood)
    _, evidence = grid.posterior(1)
    normalization = norm.cdf((Z_HI - mean) / sigma) - norm.cdf((Z_LO - mean) / sigma)
    expected = np.log(normalization).sum() - np.log(Z_HI - Z_LO).sum()
    assert evidence == pytest.approx(expected, abs=0.002)
    summary = summarize_grid(grid, 1, SPEC)
    for col, name in ((1, "mu"), (2, "h_c")):
        distribution = truncnorm(
            (Z_LO[col] - mean[col]) / sigma[col],
            (Z_HI[col] - mean[col]) / sigma[col],
            loc=mean[col],
            scale=sigma[col],
        )
        scale = SPEC.H if col == 2 else 1
        assert summary["mean"][col] == pytest.approx(
            distribution.mean() * scale, abs=0.0005
        )
        np.testing.assert_allclose(
            summary["intervals_95"][name],
            distribution.ppf([0.025, 0.975]) * scale,
            atol=0.001,
        )


class FrictionSim:
    def replay(self, spec: CuboidSpec, record: ActionRecord) -> DenseTrajectory:
        return DenseTrajectory(t=[0], x=[spec.mu], psi=[0], xdot=[0], psidot=[0])


def test_predictive_grid_sampling_matches_uniform_gaussian_convolution() -> None:
    low, high = create_uniform_cells(2)
    grid = Grid(low, high, np.zeros((len(low), 1)))
    weights, _ = grid.posterior(1)
    fit = SMC2SysID(
        (SPEC.w, SPEC.L, SPEC.H),
        sim=FrictionSim(),
        observation_std=[1] * 4,
        sample_stride=1,
    )
    record = ActionRecord(
        (0, 0, 0, 0),
        PushAction(h=0.04, F=0, tau=DT),
        DenseTrajectory(t=[0], x=[0.6], psi=[0], xdot=[0], psidot=[0]),
    )
    result = evaluate_predictive_mixture(
        grid.points, weights, fit, [record], SPEC, grid=grid
    )
    expected_log_density = np.log(
        (norm.cdf(0.6 - Z_LO[1]) - norm.cdf(0.6 - Z_HI[1])) / (Z_HI[1] - Z_LO[1])
    ) + 3 * norm.logpdf(0)
    assert result["log_predictive_density_by_record"][0] == pytest.approx(
        expected_log_density, abs=0.03
    )
    assert result["rmse_by_record_and_channel"][0][0] < 0.04
    assert result["retained_mass"] == 1
    assert (
        result["monte_carlo_errors_by_record"][0]["log_predictive_density_batch_std"]
        >= 0
    )
