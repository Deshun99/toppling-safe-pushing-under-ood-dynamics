"""Replay limits count complete-history evaluations, including all starts."""

from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from src.agents.system_identification.sysid import SysID
from src.simulation.execution.force_push import DenseTrajectory
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.constants import DT
from src.simulation.modelling.object import CuboidSpec
from src.simulation.modelling.records import ActionRecord


@pytest.mark.parametrize("history_size,budget", [(1, 31), (3, 17), (6, 7)])
def test_replay_budget_is_shared_across_starts_and_honoured(
    monkeypatch: pytest.MonkeyPatch, history_size: int, budget: int
) -> None:
    estimator = SysID((0.08, 0.08, 0.2), iterations=1, restarts=2, seed=1)
    observed = DenseTrajectory(t=[0], x=[0], psi=[0], xdot=[0], psidot=[0])
    record = ActionRecord((0, 0, 0, 0), PushAction(0.04, 1, DT), observed)
    calls = []

    def replay(spec: CuboidSpec, rec: ActionRecord) -> DenseTrajectory:
        calls.append(rec)
        return DenseTrajectory(t=[0], x=[spec.m], psi=[0], xdot=[0], psidot=[0])

    monkeypatch.setattr(estimator.sim, "replay", replay)
    try:
        result = estimator.update([record] * history_size, replay_budget=budget)
        assert len(calls) == budget // history_size * history_size
        assert 0 <= budget - len(calls) < history_size
        assert np.isfinite(result.point.loss)
    finally:
        estimator.close()


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_replay_budgets_are_rejected(budget: int | float) -> None:
    estimator = SysID((0.08, 0.08, 0.2))
    try:
        with pytest.raises(ValueError, match="replay_budget"):
            estimator.update([None], replay_budget=budget)
    finally:
        estimator.close()


def test_budget_requires_nonempty_history() -> None:
    estimator = SysID((0.08, 0.08, 0.2))
    try:
        with pytest.raises(ValueError, match="replay_budget"):
            estimator.update([], replay_budget=10)
    finally:
        estimator.close()


@pytest.mark.parametrize("budget", [3, 17, 45])
def test_previous_best_is_rescored_within_budget_and_can_win(
    monkeypatch: pytest.MonkeyPatch, budget: int
) -> None:
    estimator = SysID((0.08, 0.08, 0.2), seed=1)
    observed = DenseTrajectory(t=[0], x=[1], psi=[0], xdot=[0], psidot=[0])
    record = ActionRecord((0, 0, 0, 0), PushAction(0.04, 1, DT), observed)
    calls = []

    def replay(spec: CuboidSpec, rec: ActionRecord) -> DenseTrajectory:
        calls.append(spec.m)
        return DenseTrajectory(t=[0], x=[spec.m], psi=[0], xdot=[0], psidot=[0])

    monkeypatch.setattr(estimator.sim, "replay", replay)
    try:
        result = estimator.update(
            [record] * 3, replay_budget=budget, previous_best_parameters=(1, 0.3, 0.1)
        )
        assert len(calls) == budget // 3 * 3
        assert result.point.theta == pytest.approx((1, 0.3, 0.1))
        assert result.point.loss == 0
        assert calls[-3:] == [1, 1, 1]
    finally:
        estimator.close()


def test_new_search_can_replace_previous_best_on_changed_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimator = SysID((0.08, 0.08, 0.2), restarts=0, seed=1)
    records = [object(), object()]
    costs = []

    def loss(z: NDArray[np.float64], history: Sequence[object]) -> float:
        costs.append(len(history))
        return float((np.exp(z[0]) - len(history)) ** 2)

    def run(
        history: Sequence[object], seed: int, *, max_evaluations: int
    ) -> list[tuple[NDArray[np.float64], float]]:
        z = np.array([np.log(2), 0.3, 0.5])
        return [(z, estimator.loss(z, history)) for _ in range(max_evaluations)]

    monkeypatch.setattr(estimator, "loss", loss)
    monkeypatch.setattr(estimator, "_run_es", run)
    try:
        result = estimator.update(
            records, replay_budget=8, previous_best_parameters=(1, 0.3, 0.1)
        )
        assert sum(costs) == 8
        assert result.point.theta[0] == pytest.approx(2)
        assert result.point.loss == 0
    finally:
        estimator.close()
