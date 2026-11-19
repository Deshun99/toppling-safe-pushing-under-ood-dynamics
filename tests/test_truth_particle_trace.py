"""Trace instrumentation must preserve production particles, weights and RNG."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.agents.system_identification.smc2 import SMC2SysID
from src.evaluation.system_identification.diagnostics.trace_truth_particle import (
    TracedSMC2,
    TruthParticleDiagnostic,
    plot_truth_survival,
    write_truth_report,
)
from src.simulation.execution.force_push import DenseTrajectory
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.constants import DT
from src.simulation.modelling.object import CuboidSpec
from src.simulation.modelling.records import ActionRecord


class MassSim:
    def replay(self, spec: CuboidSpec, record: ActionRecord) -> DenseTrajectory:
        return DenseTrajectory(t=[0], x=[spec.m], psi=[0], xdot=[0], psidot=[0])


def create_tracing_pair(
    gradual: bool,
    injected: bool,
) -> tuple[SMC2SysID, TracedSMC2, list[ActionRecord]]:
    spec = CuboidSpec(w=0.08, L=0.08, H=0.2, m=1, mu=0.3, h_c=0.1)
    records = [
        ActionRecord(
            (0, 0, 0, 0),
            PushAction(0.04, 1, DT),
            DenseTrajectory(t=[0], x=[value], psi=[0], xdot=[0], psidot=[0]),
        )
        for value in (1, 0.98)
    ]
    config = dict(
        n_theta=64,
        n_state=1,
        start_std=(0, 0, 0, 0),
        observation_std=(0.05, 1, 1, 1),
        seed=7,
        sim=MassSim(),
        temper_updates=gradual,
    )
    plain = SMC2SysID((spec.w, spec.L, spec.H), **config)
    truth = np.array([0, 0.3, 0.5])
    probe = SMC2SysID((spec.w, spec.L, spec.H), **config)
    scores = np.array([probe._evaluate_record(truth, record)[1] for record in records])
    traced = TracedSMC2(spec, scores, inject_truth=injected, **config)
    if injected:
        plain._z[0] = truth
    return plain, traced, records


def assert_trace_events(plain: SMC2SysID, traced: TracedSMC2, injected: bool) -> None:
    resamples = [event for event in traced.events if event["kind"] == "resample"]
    assert resamples
    if injected:
        assert resamples[0]["before"]["exact_truth_count"] == 1
        assert resamples[0]["before"]["truth_target_rank"] == 1
    for event in resamples:
        assert len(event["ancestors"]) == plain.n_theta
        assert event["after"]["ess"] == pytest.approx(plain.n_theta)
    moves = [event for event in traced.events if event["kind"] == "move"]
    assert sum(event["accepted"] for event in moves) == plain.acceptances
    assert sum(event["proposals"] for event in moves) == plain.proposals
    assert len(moves) == len(resamples)


@pytest.mark.parametrize("gradual", [False, True])
@pytest.mark.parametrize("injected", [False, True])
def test_tracing_does_not_change_inference(gradual: bool, injected: bool) -> None:
    plain, traced, records = create_tracing_pair(gradual, injected)
    for step in (1, 2):
        plain.update(records[:step])
        traced.update(iter(records[:step]))
        np.testing.assert_array_equal(plain.particles, traced.particles)
        np.testing.assert_array_equal(plain.weights, traced.weights)
        assert plain.proposals == traced.proposals
        assert plain.acceptances == traced.acceptances
        np.testing.assert_array_equal(
            [s.log_likelihood for s in plain.inner_filters],
            [s.log_likelihood for s in traced.inner_filters],
        )
    np.testing.assert_array_equal(plain.rng.normal(size=10), traced.rng.normal(size=10))
    assert_trace_events(plain, traced, injected)


def test_truth_report_without_resampling(tmp_path: Path) -> None:
    _, traced, _ = create_tracing_pair(False, True)
    traced.sim.replay_count = 0
    runner = TruthParticleDiagnostic()
    runner.args = SimpleNamespace(out=tmp_path)
    row = {"regime": "empty history", "mode": "full", "seed": 1, "injected": True}
    row.update(runner._trace_updates(traced, [], "empty"))
    assert row["events"] == []
    assert row["initial_summary"]["exact_truth_count"] == 1
    plot_truth_survival([row], tmp_path)
    write_truth_report([row], tmp_path)
    assert (tmp_path / "truth_survival.png").stat().st_size > 0
    assert (
        "Runs without resampling are excluded" in (tmp_path / "report.md").read_text()
    )
