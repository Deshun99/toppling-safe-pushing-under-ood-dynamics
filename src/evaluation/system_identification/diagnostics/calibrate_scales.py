"""Calibrate LOSS_SCALES — the replay-fidelity noise floor (outline §6).

Replays a recorded history in the internal simulator using the TRUE spec and
measures the per-sample mean squared error per trajectory quantity: the
irreducible mismatch between the target sim and the internal sim even when
the physics is exactly known. These values become the 1/sigma^2 scales in
src/agents/system_identification/sysid.py, so each quantity is weighted by its own reproducibility
rather than by its SI-unit amplitude (position errors are ~1e6 larger than
tilt errors in raw units and would otherwise drown the terms that identify
h_c).

Run from the repo root:  .venv/bin/python -m src.evaluation.system_identification.diagnostics.calibrate_scales
Then paste the printed scales into LOSS_SCALES in src/agents/system_identification/sysid.py.
"""

from contextlib import ExitStack

import numpy as np

from src.agents.simulator.internal_sim import InternalSim
from src.evaluation.system_identification.shared.constants import SPEC, WARMUP
from src.evaluation.system_identification.shared.excitation import (
    get_regimes,
    settle_or_replace_object,
)
from src.simulation.environment.env import PushEnv
from src.simulation.modelling.records import ActionRecord


def collect_calibration_records(target: PushEnv) -> list[ActionRecord]:
    target.reset()
    records = []
    for action in get_regimes()["full mixed"]:
        if action.h > SPEC.h_c:
            settle_or_replace_object(target)
        records.append(target.push(action, warmup_steps=WARMUP))
    return records


def measure_replay_errors(
    simulator: InternalSim, records: list[ActionRecord]
) -> dict[str, float]:
    squared_errors = {channel: 0.0 for channel in ("x", "psi", "xdot", "psidot")}
    sample_counts = {channel: 0 for channel in squared_errors}
    for record in records:
        replay = simulator.replay(SPEC, record)
        for channel in squared_errors:
            observed = np.asarray(getattr(record.traj, channel))
            predicted = np.asarray(getattr(replay, channel))
            count = min(len(observed), len(predicted))
            squared_errors[channel] += float(
                np.sum((observed[:count] - predicted[:count]) ** 2)
            )
            sample_counts[channel] += count
    return {
        channel: total / sample_counts[channel]
        for channel, total in squared_errors.items()
    }


def main() -> None:
    with ExitStack() as resources:
        target = PushEnv(fixed_spec=SPEC)
        resources.callback(target.close)
        simulator = InternalSim()
        resources.callback(simulator.close)
        errors = measure_replay_errors(simulator, collect_calibration_records(target))
    print(
        "Per-sample replay MSE at the true theta (noise floor) and recommended scales:"
    )
    print("    LOSS_SCALES = {")
    for channel, mse in errors.items():
        print(f'        "{channel}": {1 / mse:.3e},   # mse {mse:.3e}')
    print("    }")


if __name__ == "__main__":
    main()
