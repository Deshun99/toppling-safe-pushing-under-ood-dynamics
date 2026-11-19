"""Load experiment data and count simulator replays."""

import json
from collections.abc import Sequence
from pathlib import Path

from src.agents.simulator.internal_sim import InternalSim
from src.simulation.execution.force_push import AppliedForce, DenseTrajectory
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.object import CuboidSpec
from src.simulation.modelling.records import ActionRecord


class ReplayCountingSimulator(InternalSim):
    """Count fitting replays separately from held-out evaluation replays."""

    def __init__(self) -> None:
        super().__init__()
        self.replay_count = 0

    def replay(self, spec: CuboidSpec, record: ActionRecord) -> DenseTrajectory:
        self.replay_count += 1
        return super().replay(spec, record)


def load_action_records(
    path: Path, *, accepted_only: bool = False
) -> list[ActionRecord]:
    entries = json.loads(path.read_text())
    return [
        decode_action_record(entry["record"])
        for entry in entries
        if not accepted_only or entry["accepted_for_fitting"]
    ]


def decode_action_record(value: dict) -> ActionRecord:
    forces = value.get("applied_forces")
    return ActionRecord(
        start=tuple(value["start"]),
        action=PushAction(**value["action"]),
        traj=DenseTrajectory(**value["traj"]),
        warmup_steps=value.get("warmup_steps", 0),
        applied_forces=(
            None if forces is None else tuple(AppliedForce(**force) for force in forces)
        ),
    )


def load_training_sequences(
    directory: Path, sequence_names: Sequence[str]
) -> dict[str, list[ActionRecord]]:
    return {
        name: load_action_records(
            directory / f"dataset_{name.replace(' ', '_')}.json", accepted_only=True
        )
        for name in sequence_names
    }


def load_metric_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]
