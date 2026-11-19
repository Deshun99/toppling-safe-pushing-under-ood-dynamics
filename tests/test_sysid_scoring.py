"""Observation-window regressions shared by SMC² and CMA-ES."""

from dataclasses import replace

import numpy as np
import pytest

from src.agents.system_identification.scoring import get_scoring_indices
from src.agents.system_identification.smc2 import SMC2SysID
from src.agents.system_identification.sysid import SysID
from src.simulation.execution.force_push import DenseTrajectory
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.constants import DT
from src.simulation.modelling.records import ActionRecord


def test_scoring_preserves_sampling_phase():
    np.testing.assert_array_equal(
        get_scoring_indices(25, warmup_steps=8, sample_stride=12), [12, 24]
    )


def test_zero_warmup_keeps_first_sample_and_passive_intervals():
    np.testing.assert_array_equal(get_scoring_indices(13, sample_stride=12), [0, 12])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_stride": 0},
        {"sample_stride": True},
        {"warmup_steps": -1},
        {"warmup_steps": 25},
        {"warmup_steps": 8, "sample_stride": 30},
    ],
)
def test_invalid_or_empty_windows_are_rejected(kwargs):
    with pytest.raises(ValueError):
        get_scoring_indices(25, **kwargs)


def test_estimators_share_window_and_preserve_full_replay(monkeypatch):
    n = 25
    zeros = [0.0] * n
    predicted = DenseTrajectory(
        t=(np.arange(n) * DT).tolist(),
        x=zeros.copy(),
        psi=zeros.copy(),
        xdot=zeros.copy(),
        psidot=zeros.copy(),
    )
    observed = replace(predicted, x=[1.0] * 8 + [0.0] * (n - 8))
    record = ActionRecord(
        (0, 0, 0, 0),
        PushAction(h=0.04, F=0, tau=17 * DT),
        observed,
        warmup_steps=8,
    )
    received = []

    class RecordingSim:
        def replay(self, spec, action_record):
            received.append(action_record)
            return predicted

    sim = RecordingSim()
    fit = SMC2SysID(
        (0.08, 0.08, 0.2),
        n_theta=1,
        n_state=1,
        start_std=(0, 0, 0, 0),
        sim=sim,
        sample_stride=12,
    )
    z = np.array([0, 0.3, 0.5])
    _, score = fit._evaluate_record(z, record)
    _, zero_score = fit._evaluate_record(z, replace(record, traj=predicted))
    assert -2 * (score - zero_score) == pytest.approx(0)

    cma = SysID((0.08, 0.08, 0.2), sample_stride=12)
    try:
        monkeypatch.setattr(cma.sim, "replay", sim.replay)
        assert cma.loss(z, [record]) == pytest.approx(0)
    finally:
        cma.close()
    fit.update([record])
    assert all(rec.warmup_steps == 8 and len(rec.traj.t) == n for rec in received)
    assert fit._history[0] == record
    assert record.traj.x[:8] == [1.0] * 8


def test_known_start_likelihood_is_repeatable_across_random_seeds():
    class StartSim:
        def replay(self, spec, record):
            return DenseTrajectory(
                t=[0], x=[record.start[0]], psi=[0], xdot=[0], psidot=[0]
            )

    record = ActionRecord(
        (0, 0, 0, 0),
        PushAction(h=0.04, F=0, tau=DT),
        DenseTrajectory(t=[0], x=[0.2], psi=[0], xdot=[0], psidot=[0]),
    )
    fit = SMC2SysID(
        (0.08, 0.08, 0.2), n_theta=1, n_state=1, start_std=(0, 0, 0, 0), sim=StartSim()
    )
    scores = []
    for seed in range(5):
        fit.rng = np.random.default_rng(seed)
        scores.append(fit._evaluate_record(np.array([0, 0.3, 0.5]), record)[1])
    assert len(set(scores)) == 1
