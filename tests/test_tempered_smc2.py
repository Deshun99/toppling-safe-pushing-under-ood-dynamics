"""Validate gradual evidence updates against an analytic Gaussian model."""

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import norm

from src.agents.system_identification.smc2 import InnerFilter, SMC2SysID
from src.agents.system_identification.sysid import Z_HI, Z_LO
from src.simulation.execution.force_push import DenseTrajectory
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.constants import DT
from src.simulation.modelling.records import ActionRecord


class GaussianSim:
    def __init__(self):
        self.calls = 0

    def replay(self, spec, record):
        self.calls += 1
        return DenseTrajectory(t=[0], x=[np.log(spec.m)], psi=[0], xdot=[0], psidot=[0])


def create_record(value):
    return ActionRecord(
        (0, 0, 0, 0),
        PushAction(h=0.04, F=1, tau=DT),
        DenseTrajectory(t=[0], x=[value], psi=[0], xdot=[0], psidot=[0]),
    )


def create_estimator(**kwargs):
    config = dict(
        n_theta=64,
        n_state=1,
        start_std=[0] * 4,
        observation_std=[0.12, 1, 1, 1],
        temper_updates=True,
        sim=GaussianSim(),
        seed=7,
    )
    config.update(kwargs)
    return SMC2SysID((0.08, 0.08, 0.2), **config)


@pytest.mark.parametrize(
    "settings",
    [
        {"n_state": 2},
        {"start_std": [0.01, 0, 0, 0]},
        {"ess_fraction": 1},
    ],
)
def test_tempering_rejects_stochastic_inner_likelihoods(settings):
    with pytest.raises(ValueError, match="Tempering requires"):
        create_estimator(**settings)


def test_conditional_ess_handles_unequal_weights_and_likelihood_offsets():
    fit = create_estimator(n_theta=3)
    fit._log_weights = np.log([0.7, 0.2, 0.1])
    increments = np.array([-1000.0, 0.0, -10.0])
    assert fit._compute_conditional_ess(increments, 0) == pytest.approx(3)
    assert fit._compute_conditional_ess(increments, 0.1) == pytest.approx(
        fit._compute_conditional_ess(increments + 100000, 0.1)
    )
    beta = fit._select_tempering_beta(0, increments)
    assert 0 < beta < 1
    assert fit._compute_conditional_ess(increments, beta) == pytest.approx(1.5)
    assert fit._select_tempering_beta(0.2, np.ones(3)) == 1


def test_full_evidence_weights_and_likelihood_bookkeeping_are_exact_without_resampling():
    fit = create_estimator(ess_fraction=0.00001)
    initial = fit.particles
    history = [create_record(0.4), create_record(0.1)]
    fit.update(history)
    log_likelihood = norm.logpdf(0.4, initial[:, 0], 0.12) + norm.logpdf(
        0.1, initial[:, 0], 0.12
    )
    np.testing.assert_allclose(
        fit.weights, np.exp(log_likelihood - logsumexp(log_likelihood))
    )
    np.testing.assert_allclose(
        [state.log_likelihood for state in fit.inner_filters],
        log_likelihood + 6 * norm.logpdf(0),
    )
    assert fit.resampling_count == 0
    assert [event["beta_to"] for event in fit.tempering_history] == [1, 1]
    before = fit.sim.calls
    fit.update(history)
    assert fit.sim.calls == before and len(fit.tempering_history) == 2


@pytest.mark.parametrize("proposal", ["fixed", "covariance"])
def test_gradual_updates_preserve_the_gaussian_posterior_and_are_seeded(proposal):
    fits = [create_estimator(n_theta=1500, proposal=proposal) for _ in range(2)]
    history = [create_record(0.4), create_record(0.1)]
    for fit in fits:
        fit.update(history)
        samples = fit.particles[:, 0]
        mean = fit.weights @ samples
        std = np.sqrt(fit.weights @ (samples - mean) ** 2)
        assert mean == pytest.approx(0.25, abs=0.015)
        assert std == pytest.approx(0.12 / np.sqrt(2), abs=0.015)
        assert len(fit.tempering_history) > len(history)
        for step in (1, 2):
            events = [event for event in fit.tempering_history if event["step"] == step]
            assert events[0]["beta_from"] == 0
            assert events[-1]["beta_to"] == 1
            assert all(event["beta_to"] > event["beta_from"] for event in events)
            assert (
                min(event["conditional_ess"] for event in events)
                >= 0.5 * fit.n_theta - 1e-7
            )
        expected = sum(
            norm.logpdf(value, fit.particles[:, 0], 0.12) for value in (0.4, 0.1)
        ) + 6 * norm.logpdf(0)
        np.testing.assert_allclose(
            [state.log_likelihood for state in fit.inner_filters], expected
        )
    np.testing.assert_array_equal(fits[0].particles, fits[1].particles)
    np.testing.assert_array_equal(fits[0].weights, fits[1].weights)


def test_partial_target_keeps_past_evidence_at_full_strength(monkeypatch):
    fit = create_estimator(n_theta=1, mcmc_steps=1)
    fit._z[0] = (Z_LO + Z_HI) / 2
    initial = fit.particles
    state = InnerFilter(np.empty((0, 4)), np.empty(0), -20.0)
    monkeypatch.setattr(fit, "_run_filter", lambda *args: state)
    monkeypatch.setattr(
        fit,
        "_evaluate_record",
        lambda *args: (InnerFilter(state.starts, state.weights, 10.0), 30.0),
    )

    class FixedRandom:
        def normal(self, size):
            return np.array([0.1, 0, 0])

        def random(self):
            return 0.5

    fit.rng = FixedRandom()
    past, increments = np.array([0.0]), np.array([0.0])
    # Correct ratio: -20 + 0.1*30 = -17, so reject. Tempering past too
    # would give -2 + 3 = +1 and incorrectly accept this proposal.
    fit._move_tempered_particles(create_record(0), 0.1, past, increments)
    assert fit.acceptances == 0
    np.testing.assert_array_equal(fit.particles, initial)
    np.testing.assert_array_equal(past, [0])
    np.testing.assert_array_equal(increments, [0])
