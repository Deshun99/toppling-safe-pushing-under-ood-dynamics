"""Optional passive settling shared by target stepping and imagined forecasts."""

from dataclasses import dataclass
from itertools import count
from math import ceil, isfinite
import time

import numpy as np
import pybullet as p

from src.simulation.execution.force_push import DenseTrajectory, ForcePushExecutor
from src.simulation.modelling.constants import DT
from src.simulation.modelling.object import Cuboid

REST_LINEAR_SPEED = 0.01  # m/s, full 3D velocity magnitude.
REST_ANGULAR_SPEED = 0.01  # rad/s, full 3D angular velocity magnitude.
REST_HOLD_TIME = 0.05  # seconds continuously below both speed thresholds.


@dataclass(frozen=True)
class RestConfig:
    enabled: bool = False
    max_time: float = 2.0  # Zero waits until rest or toppling, without a time cap.

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("wait_for_rest must be boolean")
        if not isfinite(self.max_time) or (self.max_time != 0 and self.max_time < DT):
            raise ValueError(
                f"rest_timeout must be 0 (unlimited) or finite and at least {DT} seconds"
            )


@dataclass(frozen=True)
class RestResult:
    steps: int = 0
    rested: bool = False
    toppled: bool = False
    max_tilt: float = 0.0


def is_at_rest(cuboid: Cuboid, client_id: int) -> bool:
    linear, angular = p.getBaseVelocity(cuboid.body, physicsClientId=client_id)
    return bool(
        np.linalg.norm(linear) <= REST_LINEAR_SPEED
        and np.linalg.norm(angular) <= REST_ANGULAR_SPEED
    )


def wait_for_rest(
    cuboid: Cuboid,
    client_id: int,
    config: RestConfig,
    trajectory: DenseTrajectory | None = None,
    realtime_scale: float = 0.0,
) -> RestResult:
    """Apply no force; stop on sustained rest, toppling, or the physical time cap."""
    if not config.enabled:
        return RestResult()
    peak = abs(cuboid.state()[1])
    if peak > cuboid.spec.crit_tilt:
        return RestResult(toppled=True, max_tilt=peak)
    executor = ForcePushExecutor(client_id, realtime_scale)
    recorded = DenseTrajectory() if trajectory is None else trajectory
    interval = DT / realtime_scale if realtime_scale > 0 else 0.0
    deadline, quiet = time.perf_counter(), 0
    limit = int(config.max_time / DT)
    steps = count(1) if limit == 0 else range(1, limit + 1)
    for step in steps:
        deadline = executor.advance_and_record(
            cuboid, recorded, len(recorded.t), deadline, interval
        )
        peak = max(peak, abs(cuboid.state()[1]))
        if peak > cuboid.spec.crit_tilt:
            return RestResult(step, toppled=True, max_tilt=peak)
        quiet = quiet + 1 if is_at_rest(cuboid, client_id) else 0
        if quiet >= ceil(REST_HOLD_TIME / DT):
            return RestResult(step, rested=True, max_tilt=peak)
    return RestResult(step, max_tilt=peak)
