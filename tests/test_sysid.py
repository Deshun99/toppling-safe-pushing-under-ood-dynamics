import math

import numpy as np
import pytest

from src.agents.system_identification.sysid import (
    Z_HI,
    Z_LO,
    SysID,
    convert_z_to_cuboid_spec,
    trajectory_loss,
)
from src.simulation.execution.force_push import DenseTrajectory
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.records import ActionRecord


def test_parameter_transform(spec):
    candidate = convert_z_to_cuboid_spec(
        [math.log(spec.m), spec.mu, spec.h_c / spec.H], (spec.w, spec.L, spec.H)
    )
    assert candidate == spec


def test_loss_scales_truncation_and_empty():
    recorded = DenseTrajectory(x=[1, 2, 100], psi=[2, 4, 100])
    simulated = DenseTrajectory(x=[2, 4], psi=[3, 6])
    assert trajectory_loss(recorded, simulated, {"x": 2, "psi": 3}) == 25
    assert trajectory_loss(DenseTrajectory(), simulated) == 0
    assert trajectory_loss(simulated, simulated) == 0


@pytest.fixture
def sysid(spec):
    instance = SysID((spec.w, spec.L, spec.H), iterations=1, restarts=0, seed=7)
    try:
        yield instance
    finally:
        instance.close()


def test_history_loss_sums_replays(sysid, spec):
    z = [math.log(spec.m), spec.mu, spec.h_c / spec.H]
    record = ActionRecord(
        (0, 0, 0, 0), PushAction(h=0.04, F=4, tau=0.05), DenseTrajectory()
    )
    record.traj = sysid.sim.replay(spec, record)
    assert sysid.loss(z, [record, record]) == 0
    perturbed = [math.log(1.5), 0.1, 0.4]
    single = sysid.loss(perturbed, [record])
    assert single > 0
    assert sysid.loss(perturbed, [record, record]) == pytest.approx(2 * single)


def test_update_restarts_deduplicates_and_weights(sysid, monkeypatch):
    best = np.array([0, 0.3, 0.4])
    runner_up = np.array([0.1, 0.4, 0.5])
    seeds = []

    def run(history, seed):
        seeds.append(seed)
        if seed == 7:
            return [(best, 20.0)]
        return [(runner_up, 11.0), (best, 10.0), (best + 1e-5, 10.1)]

    monkeypatch.setattr(sysid, "_run_es", run)
    sysid.restarts = 1
    sysid.K = 2
    result = sysid.update([])
    assert seeds == [7, 8]
    assert result.point.loss == 10
    assert result.point.theta == pytest.approx((1, 0.3, 0.08))
    assert [item.loss for item in result.hypotheses] == [10, 11]
    weights = [item.weight for item in result.hypotheses]
    assert sum(weights) == pytest.approx(1)
    assert weights[0] > weights[1] > 0
    sysid.K = 10
    assert len(sysid.update([]).hypotheses) == 2


def test_real_cma_es_smoke(sysid, spec):
    record = ActionRecord(
        (0, 0, 0, 0), PushAction(h=0.04, F=4, tau=0.05), DenseTrajectory()
    )
    record.traj = sysid.sim.replay(spec, record)
    result = sysid.update([record])
    assert np.isfinite(result.point.loss)
    assert 1 <= len(result.hypotheses) <= sysid.K
    assert sum(h.weight for h in result.hypotheses) == pytest.approx(1)
    for hypothesis in result.hypotheses:
        mass, friction, com = hypothesis.theta
        z = np.array([math.log(mass), friction, com / spec.H])
        assert (Z_LO <= z).all() and (z <= Z_HI).all()
