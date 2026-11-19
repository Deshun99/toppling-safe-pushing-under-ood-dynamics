"""Check benchmark statistics independently of the inference algorithms."""

from pathlib import Path

import numpy as np
import pytest

from src.agents.system_identification.smc2 import SMC2SysID
from src.evaluation.system_identification.diagnostics.diagnose_smc2 import (
    LikelihoodDiagnostics,
    run_ablation_trial,
)
from src.evaluation.system_identification.shared.common import load_action_records
from src.evaluation.system_identification.shared.constants import SPEC
from src.evaluation.system_identification.shared.metrics import (
    compute_weighted_quantiles,
    evaluate_predictions,
)
from src.simulation.execution.force_push import DenseTrajectory
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.object import CuboidSpec
from src.simulation.modelling.records import ActionRecord


def test_quantiles_preserve_weighted_duplicate_mass() -> None:
    values = np.array([9.0, 2.0, 2.0, 0.0])
    weights = np.array([0.1, 0.4, 0.5, 0.0])
    actual = compute_weighted_quantiles(values, weights, (0.025, 0.5, 0.975))
    np.testing.assert_array_equal(actual, [2, 2, 9])


def test_prediction_metrics_include_topples_and_separate_upright_errors() -> None:
    upright = DenseTrajectory(t=[0], x=[0], psi=[0], xdot=[0], psidot=[0])
    toppled = DenseTrajectory(t=[0], x=[2], psi=[1], xdot=[0], psidot=[0])
    records = [
        ActionRecord((0, 0, 0, 0), PushAction(h=0.02, F=4, tau=0.05), trajectory)
        for trajectory in (upright, toppled)
    ]

    class StationarySim:
        def replay(self, spec: CuboidSpec, record: ActionRecord) -> DenseTrajectory:
            return upright

    result = evaluate_predictions(StationarySim(), SPEC, records)
    assert result["rmse_x"] == pytest.approx(np.sqrt(2))
    assert result["upright_rmse_x"] == 0
    assert result["topple_accuracy"] == 0.5


def test_prediction_metrics_reject_misaligned_replays() -> None:
    observed = DenseTrajectory(t=[0], x=[0], psi=[0], xdot=[0], psidot=[0])
    record = ActionRecord((0, 0, 0, 0), PushAction(h=0.02, F=4, tau=0.05), observed)

    class MisalignedSim:
        def replay(self, spec: CuboidSpec, record: ActionRecord) -> DenseTrajectory:
            return DenseTrajectory(t=[1], x=[0], psi=[0], xdot=[0], psidot=[0])

    with pytest.raises(ValueError, match="align"):
        evaluate_predictions(MisalignedSim(), SPEC, [record])


def test_loading_excludes_rejected_records_and_restores_named_actions(
    tmp_path: Path,
) -> None:
    import json
    from dataclasses import asdict

    record = ActionRecord(
        (0, 0, 0, 0), PushAction(h=0.02, F=4, tau=0.05), DenseTrajectory()
    )
    path = tmp_path / "data.json"
    path.write_text(
        json.dumps(
            [
                {"accepted_for_fitting": False, "record": asdict(record)},
                {"accepted_for_fitting": True, "record": asdict(record)},
            ]
        )
    )
    assert load_action_records(path, accepted_only=True) == [record]


def test_collapse_probe_preserves_inference_random_stream_and_particles() -> None:
    from copy import deepcopy

    class StartStateSim:
        def replay(self, spec: CuboidSpec, record: ActionRecord) -> DenseTrajectory:
            return DenseTrajectory(
                t=[0], x=[record.start[0]], psi=[0], xdot=[0], psidot=[0]
            )

    fit = SMC2SysID(
        (SPEC.w, SPEC.L, SPEC.H),
        n_theta=4,
        n_state=4,
        observation_std=[1, 1, 1, 1],
        sim=StartStateSim(),
    )
    record = ActionRecord(
        (0, 0, 0, 0),
        PushAction(h=0.02, F=4, tau=0.05),
        DenseTrajectory(t=[0], x=[0.4], psi=[0], xdot=[0], psidot=[0]),
    )
    random_stream = deepcopy(fit.rng)
    particles, weights = fit.particles, fit.weights
    measured = LikelihoodDiagnostics.measure_initial_collapse(fit, record, particles[0])
    assert 1 <= measured["first_update_ess"] <= fit.n_theta
    np.testing.assert_array_equal(fit.particles, particles)
    np.testing.assert_array_equal(fit.weights, weights)
    np.testing.assert_array_equal(fit.rng.normal(size=5), random_stream.normal(size=5))


def test_ablation_reports_no_proposals_without_crashing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from src.simulation.modelling.constants import DT

    class StationarySim:
        replay_count = 0

        def replay(self, spec: CuboidSpec, record: ActionRecord) -> DenseTrajectory:
            self.replay_count += 1
            return record.traj

    record = ActionRecord(
        (0, 0, 0, 0),
        PushAction(h=0.02, F=0, tau=DT),
        DenseTrajectory(t=[0], x=[0], psi=[0], xdot=[0], psidot=[0]),
    )
    rows = run_ablation_trial(
        "no_resampling",
        {"n_theta": 1, "n_state": 1, "start_std": (0, 0, 0, 0)},
        1,
        SPEC,
        [record],
        StationarySim(),
    )
    assert rows[0]["acceptance_rate"] is None
    assert "accept=n/a" in capsys.readouterr().out
