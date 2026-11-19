"""Verify shared dataset semantics and reject incompatible replay caches."""

import json
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.evaluation.system_identification.comparisons.compare_proposals import (
    ProposalComparison,
)
from src.evaluation.system_identification.diagnostics.reference_posterior import (
    PosteriorReferenceComparison,
)
from src.evaluation.system_identification.diagnostics.trace_truth_particle import (
    TruthParticleDiagnostic,
)
from src.evaluation.system_identification.shared.common import (
    load_action_records,
)
from src.evaluation.system_identification.shared.constants import SPEC, Grid
from src.evaluation.system_identification.shared.grid import (
    create_uniform_cells,
    load_checked_grid,
)
from src.simulation.execution.force_push import AppliedForce, DenseTrajectory
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.records import ActionRecord


@pytest.fixture
def record() -> ActionRecord:
    return ActionRecord(
        (0.0, 0.0, 0.0, 0.0),
        PushAction(0.02, 4.0, 0.1),
        DenseTrajectory(t=[0.0], x=[0.0], psi=[0.0], xdot=[0.0], psidot=[0.0]),
        warmup_steps=8,
    )


def test_record_loader_keeps_rejected_evaluation_records(
    tmp_path: Path, record: ActionRecord
) -> None:
    path = tmp_path / "records.json"
    path.write_text(
        json.dumps(
            [
                {"accepted_for_fitting": accepted, "record": asdict(record)}
                for accepted in (True, False)
            ]
        )
    )
    assert load_action_records(path, accepted_only=True) == [record]
    assert load_action_records(path) == [record, record]
    assert load_action_records(path)[1].warmup_steps == 8


def test_record_loader_preserves_actual_force_for_identification(tmp_path: Path) -> None:
    record = ActionRecord(
        (0.0, 0.0, 0.0, 0.0), PushAction(.04, 15, .05),
        DenseTrajectory(t=[0.0], x=[0.0], psi=[0.0], xdot=[0.0], psidot=[0.0]),
        applied_forces=(AppliedForce(2.7, .035),),
    )
    path = tmp_path / "robot-records.json"
    path.write_text(json.dumps([{"record": asdict(record)}]))
    assert load_action_records(path) == [record]


class DeterministicProbe:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        assert kwargs["n_state"] == 1
        assert kwargs["start_std"] == (0, 0, 0, 0)

    def __enter__(self) -> "DeterministicProbe":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def _evaluate_record(
        self, point: np.ndarray, record: ActionRecord
    ) -> tuple[None, float]:
        return None, float(point.sum())


@pytest.mark.parametrize("changed", [False, True])
def test_grid_cache_checks_current_likelihoods(
    tmp_path: Path,
    record: ActionRecord,
    monkeypatch: pytest.MonkeyPatch,
    changed: bool,
) -> None:
    monkeypatch.setattr(
        "src.evaluation.system_identification.shared.grid.SMC2SysID", DeterministicProbe
    )
    low, high = create_uniform_cells(2)
    likelihood = ((low + high) / 2).sum(axis=1)[:, None]
    if changed:
        likelihood[0, 0] += 1.0
    path = tmp_path / "grid.npz"
    np.savez(path, low=low, high=high, log_likelihood=likelihood)
    if changed:
        with pytest.raises(ValueError, match="likelihoods differ"):
            load_checked_grid(path, SPEC, [record], 12)
    else:
        grid, difference = load_checked_grid(path, SPEC, [record], 12)
        assert difference == 0
        np.testing.assert_array_equal(grid.low, low)


def test_resuming_grid_rejects_a_different_record_count(
    tmp_path: Path, record: ActionRecord
) -> None:
    np.savez(
        tmp_path / "grid2_level0.npz",
        low=np.zeros((1, 3)),
        high=np.ones((1, 3)),
        log_likelihood=np.zeros((1, 2)),
    )
    runner = PosteriorReferenceComparison()
    runner.args = Namespace(out=tmp_path, resume=True, sample_stride=12)
    runner.spec, runner.records = SPEC, [record]
    with pytest.raises(ValueError, match="record count"):
        runner._load_or_refine_grid(2, 0, None, None)


@pytest.mark.parametrize("runner_type", [ProposalComparison, TruthParticleDiagnostic])
def test_optional_grid_is_skipped_without_matching_duration_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner_type: type[ProposalComparison] | type[TruthParticleDiagnostic],
) -> None:
    def reject_probe(*args: Any, **kwargs: Any) -> tuple[Grid, float]:
        raise AssertionError("A duration reference cannot validate another sequence")

    monkeypatch.setattr(Path, "exists", lambda path: True)
    monkeypatch.setattr(f"{runner_type.__module__}.load_checked_grid", reject_probe)
    runner = runner_type()
    runner.args = Namespace(out=tmp_path)
    runner.training_sequences = {"varied magnitudes": []}
    runner.spec = SPEC
    runner.config = {"sample_stride": 12}
    runner.output_config = {}
    runner._load_reference()
    assert runner.grid is None
    assert runner.output_config == {}
