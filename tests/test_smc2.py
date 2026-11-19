"""Statistical checks use an exactly solvable conditional Gaussian model."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import norm

from src.agents.simulator.internal_sim import InternalSim
from src.agents.system_identification.smc2 import SMC2SysID
from src.simulation.execution.force_push import DenseTrajectory
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.constants import DT
from src.simulation.modelling.records import ActionRecord


class LinearSim:
    calls = 0

    def replay(self, spec, record):
        self.calls += 1
        # y = log(m) + uncertain initial x + observation noise.
        value = np.log(spec.m) + record.start[0]
        return DenseTrajectory(t=[0.0], x=[value], psi=[0.0], xdot=[0.0], psidot=[0.0])


def record(y=0.4):
    return ActionRecord(
        (0.0, 0.0, 0.0, 0.0),
        PushAction(h=0.04, F=1.0, tau=DT),
        DenseTrajectory(t=[0.0], x=[y], psi=[0.0], xdot=[0.0], psidot=[0.0]),
    )


def estimator(**kwargs):
    defaults = dict(
        n_theta=64,
        n_state=1,
        start_std=[0] * 4,
        observation_std=[0.3, 1, 1, 1],
        ess_fraction=0.001,
        sim=LinearSim(),
        seed=7,
    )
    defaults.update(kwargs)
    return SMC2SysID((0.08, 0.08, 0.2), **defaults)


def test_exact_bayes_weights_and_append_only_history():
    fit = estimator()
    initial = fit.particles
    history = [record()]
    fit.update(history)
    logp = norm.logpdf(0.4, initial[:, 0], 0.3)
    np.testing.assert_allclose(fit.weights, np.exp(logp - logsumexp(logp)))
    calls = fit.sim.calls
    fit.update(history)
    assert fit.sim.calls == calls
    history.append(record(-0.1))
    result = fit.update(history)
    logp += norm.logpdf(-0.1, initial[:, 0], 0.3)
    np.testing.assert_allclose(fit.weights, np.exp(logp - logsumexp(logp)))
    assert sum(h.weight for h in result.hypotheses) == pytest.approx(1)
    with pytest.raises(ValueError, match="append-only"):
        fit.update(history[:1])
    history[0].traj.x[0] = 99
    with pytest.raises(ValueError, match="append-only"):
        fit.update(history)


def test_inner_marginal_likelihood_matches_gaussian_convolution():
    fit = estimator(n_theta=1, n_state=20000, start_std=[0.4, 0, 0, 0])
    fit.update([record()])
    expected = norm.pdf(0.4, fit.particles[0, 0], 0.5) * norm.pdf(0) ** 3
    inner = fit.inner_filters[0]
    assert np.exp(inner.log_likelihood) == pytest.approx(expected, rel=0.03)
    assert inner.weights.sum() == pytest.approx(1)
    # Analytic conditional mean for latent start given y.
    posterior_mean = (0.4 - fit.particles[0, 0]) * 0.4**2 / 0.5**2
    assert inner.weights @ inner.starts[:, 0] == pytest.approx(posterior_mean, abs=0.02)


def test_resampling_moves_preserve_posterior_and_are_seeded():
    fits = [estimator(n_theta=1500, ess_fraction=1, mcmc_steps=3) for _ in range(2)]
    for fit in fits:
        fit.update([record()])
        assert fit.resampling_count == 1
        assert 0 < fit.acceptances < fit.proposals
        assert fit.weights.sum() == pytest.approx(1)
        assert fit.particles[:, 0].mean() == pytest.approx(0.4, abs=0.05)
        assert fit.particles[:, 0].std() == pytest.approx(0.3, abs=0.04)
    np.testing.assert_array_equal(fits[0].particles, fits[1].particles)


def test_pmmh_rejection_retains_random_likelihood(monkeypatch):
    fit = estimator(n_theta=1, n_state=3, start_std=[0.1, 0, 0, 0], mcmc_steps=1)
    fit.update([record()])
    before = fit.inner_filters[0]
    old_z = fit.particles
    seen = []

    def impossible(z, history):
        seen.append(len(history))
        return replace(before, log_likelihood=-1e100)

    monkeypatch.setattr(fit, "_run_filter", impossible)
    # Tiny proposal makes staying inside prior support certain for this seed.
    fit.proposal_scale = 1e-8
    fit._resample_move()
    assert seen == [1]
    assert fit.acceptances == 0
    np.testing.assert_array_equal(fit.particles, old_z)
    np.testing.assert_array_equal(fit.inner_filters[0].starts, before.starts)
    assert fit.inner_filters[0].log_likelihood == before.log_likelihood


def test_unique_hypotheses_aggregate_duplicate_mass():
    fit = estimator(n_theta=3, K=3)
    fit._z[1] = fit._z[0]
    result = fit.update([])
    assert len(result.hypotheses) == 2
    assert result.point.weight == pytest.approx(2 / 3)
    assert sorted(h.weight for h in result.hypotheses) == pytest.approx([1 / 3, 2 / 3])


@pytest.mark.parametrize(
    "config",
    [
        dict(n_theta=0),
        dict(n_state=0),
        dict(K=0),
        dict(ess_fraction=0),
        dict(proposal_scale=float("nan")),
        dict(sample_stride=0),
        dict(start_std=[-1] * 4),
        dict(observation_std=[0] * 4),
        dict(mcmc_steps=-1),
    ],
)
def test_bad_configuration(config):
    with pytest.raises(ValueError):
        estimator(**config)


def test_bad_data_does_not_change_posterior():
    fit = estimator()
    before = fit.weights
    bad = record()
    bad.traj.t = [DT]
    with pytest.raises(ValueError, match="timestamps"):
        fit.update([record(), bad])
    np.testing.assert_array_equal(fit.weights, before)
    assert fit.sim.calls == 0


def test_pybullet_integration_and_cleanup(spec):
    import pybullet as p

    target = InternalSim()
    try:
        rec = ActionRecord(
            (0.0, 0.0, 0.0, 0.0),
            PushAction(h=0.04, F=4.0, tau=0.05),
            DenseTrajectory(),
            warmup_steps=2,
        )
        rec.traj = target.replay(spec, rec)
        with SMC2SysID(
            (spec.w, spec.L, spec.H),
            n_theta=6,
            n_state=2,
            mcmc_steps=1,
            ess_fraction=1,
            seed=12,
        ) as fit:
            client = fit.sim.client
            result = fit.update([rec])
            assert np.isfinite(result.point.loss)
            assert np.isfinite(fit.weights).all()
            assert p.getNumBodies(physicsClientId=client) == 1
            assert fit.resampling_count == 1
        assert not p.isConnected(client)
        fit.close()
        with pytest.raises(RuntimeError, match="closed"):
            fit.update([rec])
    finally:
        target.close()
