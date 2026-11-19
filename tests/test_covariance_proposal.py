"""Proposal geometry, stage freezing, and study replay caps."""

from copy import deepcopy
from typing import Any

import numpy as np
import pytest
from scipy.stats import norm

from src.agents.system_identification.smc2 import SMC2SysID
from src.agents.system_identification.sysid import N_PHYSICS_PARAMETERS, Z_HI, Z_LO
from src.evaluation.system_identification.comparisons.compare_proposals import (
    ReplayCappedSMC,
)
from src.simulation.execution.force_push import DenseTrajectory
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.constants import DT
from src.simulation.modelling.object import CuboidSpec
from src.simulation.modelling.records import ActionRecord


class GaussianSim:
    def __init__(self) -> None:
        self.replay_count = 0

    def replay(self, spec: CuboidSpec, record: ActionRecord) -> DenseTrajectory:
        self.replay_count += 1
        return DenseTrajectory(t=[0], x=[np.log(spec.m)], psi=[0], xdot=[0], psidot=[0])


def create_fit(estimator_type: type[SMC2SysID] = SMC2SysID, **kwargs: Any) -> SMC2SysID:
    config = dict(
        n_theta=64,
        n_state=1,
        start_std=(0, 0, 0, 0),
        observation_std=(0.05, 1, 1, 1),
        seed=7,
        sim=GaussianSim(),
        temper_updates=True,
    )
    config.update(kwargs)
    return estimator_type((0.08, 0.08, 0.2), **config)


def test_covariance_keeps_jump_length_and_particle_correlation() -> None:
    fit = create_fit(n_theta=4)
    scaled = np.array(
        [[0.1, 0.2, 0.4], [0.3, 0.4, 0.5], [0.5, 0.6, 0.6], [0.7, 0.8, 0.7]]
    )
    fit._z = Z_LO + scaled * (Z_HI - Z_LO)
    fit._log_weights = np.log([0.1, 0.2, 0.3, 0.4])
    transform = fit._compute_proposal_transform() / (Z_HI - Z_LO)[:, None]
    covariance = transform @ transform.T
    assert np.trace(covariance) == pytest.approx(
        N_PHYSICS_PARAMETERS * fit.proposal_scale**2
    )
    assert covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1]) > 0.999
    assert np.linalg.eigvalsh(covariance).min() > 0
    mean = fit.weights @ scaled
    expected = ((scaled - mean) * fit.weights[:, None]).T @ (scaled - mean)
    expected /= np.trace(expected) / N_PHYSICS_PARAMETERS
    np.testing.assert_allclose(covariance / fit.proposal_scale**2, expected, atol=3e-06)


def test_collapsed_population_falls_back_to_original_step_shape() -> None:
    fit = create_fit()
    fit._z[:] = (Z_LO + Z_HI) / 2
    np.testing.assert_allclose(
        fit._compute_proposal_transform(), np.diag(fit.proposal_scale * (Z_HI - Z_LO))
    )


def test_proposal_is_frozen_after_resampling_and_preserves_rng_draw_count() -> None:
    fit = create_fit()
    fit._resample_particles()
    transform = fit._proposal_transform.copy()
    rng = deepcopy(fit.rng)
    current = fit.particles[0]
    expected = current + transform @ rng.normal(size=N_PHYSICS_PARAMETERS)
    fit._z[:] = current
    np.testing.assert_array_equal(fit._propose_parameters(current), expected)
    np.testing.assert_array_equal(fit._proposal_transform, transform)
    np.testing.assert_array_equal(fit.rng.normal(size=10), rng.normal(size=10))


def test_invalid_proposal_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="proposal"):
        create_fit(proposal="unknown")


@pytest.mark.parametrize("proposal", ["fixed", "covariance"])
def test_replay_cap_finishes_full_evidence_without_partial_candidate_evaluations(
    proposal: str,
) -> None:
    fit = create_fit(ReplayCappedSMC, proposal=proposal)
    records = create_records((0.4, 0.1))
    for step in (1, 2):
        fit.replay_limit = step * fit.n_theta
        fit.update(records[:step])
        assert fit.sim.replay_count <= fit.replay_limit
        assert fit.tempering_history[-1]["beta_to"] == 1
        expected = sum(
            (
                norm.logpdf(record.traj.x[0], fit.particles[:, 0], 0.05)
                + 3 * norm.logpdf(0)
                for record in records[:step]
            )
        )
        np.testing.assert_allclose(
            [state.log_likelihood for state in fit.inner_filters], expected
        )
        assert len(fit._history) == step
    assert fit.blocked_stages > 0
    assert fit.acceptances == 0


def create_records(values: tuple[float, ...]) -> list[ActionRecord]:
    return [
        ActionRecord(
            (0, 0, 0, 0),
            PushAction(0.04, 1, DT),
            DenseTrajectory(t=[0], x=[value], psi=[0], xdot=[0], psidot=[0]),
        )
        for value in values
    ]
