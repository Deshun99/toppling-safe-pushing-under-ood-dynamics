"""Execution/forecast contracts shared by the delta controller and baseline."""

from dataclasses import fields, replace
from unittest.mock import Mock

import numpy as np
import pybullet as p
import pytest

from src.agents.simulator.internal_sim import InternalSim
from src.simulation.environment.env import ActionRanges, PushEnv
from src.simulation.execution.force_push import DenseTrajectory
from src.simulation.modelling.action_mapping import (
    normalized_to_physical,
    physical_to_normalized,
)
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.constants import DT
from src.simulation.modelling.records import ActionRecord
from src.simulation.modelling.object import Cuboid, CuboidSpec


@pytest.mark.parametrize("force_cap", [15.0, 50.0])
def test_action_roundtrip_preserves_configured_actuator_meaning(spec, force_cap):
    geometry = (spec.w, spec.L, spec.H)
    ranges = ActionRanges(force=(0, force_cap))
    for normalized in ([-1, -1, -1], [1, 1, 1], [0.25, -0.75, 0.4]):
        physical = normalized_to_physical(normalized, geometry, ranges)
        np.testing.assert_allclose(
            physical_to_normalized(physical, geometry, ranges), normalized, atol=1e-15
        )
    upper = normalized_to_physical([1, 1, 1], geometry, ranges)
    assert (upper.h, upper.F, upper.tau) == pytest.approx(
        (spec.H * 0.9, force_cap, 0.5)
    )


def test_custom_action_ranges_roundtrip_and_explicit_clipping(spec):
    geometry = (spec.w, spec.L, spec.H)
    ranges = ActionRanges(
        height_fraction=(0.2, 0.7), force=(0.5, 12), duration=(0.1, 0.3)
    )
    low = PushAction(h=spec.H * 0.2, F=0.5, tau=0.1)
    assert normalized_to_physical([-5, -2, -1], geometry, ranges, clip=True) == low
    np.testing.assert_array_equal(
        physical_to_normalized(low, geometry, ranges), [-1, -1, -1]
    )
    with pytest.raises(ValueError, match="within"):
        normalized_to_physical([-1.01, 0, 0], geometry, ranges)
    with pytest.raises(ValueError, match="within"):
        physical_to_normalized(PushAction(h=0.01, F=1, tau=0.2), geometry, ranges)


@pytest.mark.parametrize(
    "action", [[0, 0], [[0, 0, 0]], [0, np.nan, 0], [0, 0, np.inf]]
)
def test_action_mapping_rejects_invalid_shape_and_nonfinite(spec, action):
    with pytest.raises(ValueError, match="three finite"):
        normalized_to_physical(
            action, (spec.w, spec.L, spec.H), ActionRanges(), clip=True
        )


@pytest.mark.parametrize(
    "geometry,ranges",
    [
        ((0.08, 0.08, 0), ActionRanges()),
        ((0.08, 0.08, np.nan), ActionRanges()),
        ((0.08, 0.08), ActionRanges()),
        ((0.08, 0.08, 0.2), ActionRanges(force=(0, 0))),
        ((0.08, 0.08, 0.2), ActionRanges(duration=(0.5, 0.05))),
    ],
)
def test_action_mapping_rejects_noninvertible_bounds(geometry, ranges):
    with pytest.raises(ValueError):
        normalized_to_physical([0, 0, 0], geometry, ranges)


def test_step_exposes_actual_start_action_and_dense_record(spec: CuboidSpec) -> None:
    ranges = ActionRanges(force=(0, 15))
    env = PushEnv(fixed_spec=spec, action_ranges=ranges)
    try:
        env.reset(seed=11)
        start = env.cuboid.state()
        _, _, _, _, info = env.step([-2, -0.96, -2])
        record = info["action_record"]
        assert record.start == start
        assert (record.action.h, record.action.F, record.action.tau) == pytest.approx(
            (spec.H * 0.1, 0.3, 0.05)
        )
        assert info["executed"] == (record.action.h, record.action.F, record.action.tau)
        assert record.traj is info["trajectory"] is env.last_trajectory
        assert record.warmup_steps == 0
        assert len(record.traj.t) == round(0.05 / DT)
        assert [field.name for field in fields(record)] == [
            "start",
            "action",
            "traj",
            "warmup_steps",
            "applied_forces",
        ]
        assert record.applied_forces is None
    finally:
        env.close()


@pytest.mark.parametrize("warmup_steps", [0, 3])
def test_predict_matches_replay_without_extra_settling(spec, warmup_steps):
    sim = InternalSim()
    start = (0.3, 0.07, 0.1, -0.02)
    action = PushAction(h=0.04, F=2, tau=0.05)
    try:
        expected = sim.replay(
            spec, ActionRecord(start, action, DenseTrajectory(), warmup_steps)
        )
        actual = sim.predict(spec, start, action, warmup_steps)
        assert actual == expected
        assert len(actual.t) == warmup_steps + round(action.tau / DT)
        assert actual.t == pytest.approx(np.arange(len(actual.t)) * DT)
        assert p.getNumBodies(physicsClientId=sim.client) == 1
    finally:
        sim.close()


def test_prediction_failure_removes_body_and_preserves_other_clients(spec):
    sim = InternalSim()
    other = InternalSim()
    try:
        original_executor = sim.executor
        sim.executor = Mock()
        sim.executor.execute.side_effect = RuntimeError("execution interrupted")
        with pytest.raises(RuntimeError, match="interrupted"):
            sim.predict(spec, (0, 0, 0, 0), PushAction(0.02, 1, 0.05))
        assert p.getNumBodies(physicsClientId=sim.client) == 1
        sim.executor = original_executor
        recovered = sim.predict(spec, (0, 0, 0, 0), PushAction(0.02, 1, 0.05))
        assert len(recovered.t) == 12
        sim.close()
        sim.close()
        assert p.isConnected(other.client)
        with pytest.raises(RuntimeError, match="closed"):
            sim.predict(spec, (0, 0, 0, 0), PushAction(0.02, 1, 0.05))
    finally:
        sim.close()
        other.close()


@pytest.mark.parametrize("pitch", [-0.2, 0.0, 0.2])
@pytest.mark.parametrize("com_height", [0.04, 0.16])
def test_prediction_spawn_preserves_com_state_and_floor_contact(
    spec: CuboidSpec, pitch: float, com_height: float
) -> None:
    sim = InternalSim()
    box = Cuboid(replace(spec, H=0.2, h_c=com_height), sim.client)
    start = (0.3, pitch, 0.1, -0.02)
    try:
        sim._spawn_from_state(box, start)
        assert box.state() == pytest.approx(start)
        position, orientation = box.pose()
        rotation = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        heights = [
            (position + rotation @ np.array([x, 0.0, z]))[2]
            for x in (-box.spec.w / 2, box.spec.w / 2)
            for z in (-com_height, box.spec.H - com_height)
        ]
        assert min(heights) == pytest.approx(0.0, abs=1e-12)
    finally:
        box.remove()
        sim.close()
