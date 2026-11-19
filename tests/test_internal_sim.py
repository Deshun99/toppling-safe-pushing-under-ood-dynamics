import numpy as np
import pybullet as p
import pytest

from src.agents.simulator.internal_sim import InternalSim
from src.simulation.execution.force_push import AppliedForce, DenseTrajectory
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.constants import DT
from src.simulation.modelling.object import Cuboid, CuboidSpec
from src.simulation.modelling.records import ActionRecord


@pytest.fixture
def sim():
    instance = InternalSim()
    try:
        yield instance
    finally:
        instance.close()


@pytest.mark.parametrize("pitch", [0, 0.1, -0.1])
def test_replay_is_repeatable_and_removes_body(sim, spec, pitch):
    record = ActionRecord(
        (0.2, pitch, 0.1, 0.01),
        PushAction(h=0.04, F=4, tau=0.05),
        DenseTrajectory(),
        warmup_steps=2,
    )
    count = p.getNumBodies(physicsClientId=sim.client)
    first = sim.replay(spec, record)
    second = sim.replay(spec, record)
    assert first == second
    assert len(first.t) == 2 + round(0.05 / DT)
    assert p.getNumBodies(physicsClientId=sim.client) == count
    assert np.isfinite(first.x).all()


def test_identification_replays_actual_force_but_forecasts_requested_force(
    sim: InternalSim, spec: CuboidSpec
) -> None:
    start = (0.0, 0.0, 0.0, 0.0)
    forces = tuple(AppliedForce(force, .04) for force in [0.0] * 4 + [8.0] * 12 + [0.0] * 8)
    cuboid = Cuboid(spec, sim.client)
    try:
        sim._spawn_from_state(cuboid, start)
        measured = sim.executor.execute_forces(cuboid, forces, warmup_steps=8)
    finally:
        cuboid.remove()
    requested = PushAction(h=.04, F=15.0, tau=len(forces) * DT)
    record = ActionRecord(start, requested, measured, warmup_steps=8, applied_forces=forces)
    replayed = sim.replay(spec, record)
    forecast = sim.predict(spec, start, requested, warmup_steps=8)
    assert replayed.to_numpy() == pytest.approx(measured.to_numpy(), abs=1e-12)
    assert abs(forecast.x[-1] - replayed.x[-1]) > .01
    assert len(replayed.t) == len(forces) + 8


def test_actual_force_trace_must_match_recorded_steps() -> None:
    with pytest.raises(ValueError, match="align"):
        ActionRecord(
            (0, 0, 0, 0), PushAction(.04, 15, .1), DenseTrajectory(),
            applied_forces=(AppliedForce(3, .04),),
        )


def test_failed_actual_force_replay_removes_candidate_body(
    sim: InternalSim, spec: CuboidSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectory = DenseTrajectory()
    trajectory.append(0, 0, 0, 0, 0)
    record = ActionRecord(
        (0, 0, 0, 0), PushAction(.04, 15, .1), trajectory,
        applied_forces=(AppliedForce(3, .04),),
    )
    count = p.getNumBodies(physicsClientId=sim.client)

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("force replay failed")

    monkeypatch.setattr(sim.executor, "execute_forces", fail)
    with pytest.raises(RuntimeError, match="force replay failed"):
        sim.replay(spec, record)
    assert p.getNumBodies(physicsClientId=sim.client) == count
