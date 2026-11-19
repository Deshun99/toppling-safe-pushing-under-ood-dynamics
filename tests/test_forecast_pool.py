"""Parallel forecast worker pool: serial-equivalent, order-preserving, error-propagating."""

import numpy as np
import pytest

from src.agents.simulator.internal_sim import InternalSim
from src.simulation.execution.settling import RestConfig
from src.simulation.modelling.actions import PushAction
from src.agents.simulator.forecast_pool import particle_features, ForecastPool

GEOMETRY = (0.08, 0.06, 0.20)
START = (0.0, 0.0, 0.0, 0.0)
PHYSICAL = PushAction(0.10, 4.0, 0.3)
REST = RestConfig(True, 2.0)
THETAS = [
    (1.00, 0.50, 0.10),
    (0.70, 0.40, 0.08),
    (1.30, 0.60, 0.12),
    (0.90, 0.35, 0.09),
    (1.10, 0.55, 0.11),
]


def _serial(sim, thetas):
    return [
        particle_features(sim, GEOMETRY, t, START, PHYSICAL, REST, 16, "self-attention")
        for t in thetas
    ]


def test_pool_matches_serial_forecast() -> None:
    sim = InternalSim()
    serial = _serial(sim, THETAS)
    sim.close()
    pool = ForecastPool(workers=2)
    parallel = pool.forecast_features(
        GEOMETRY, THETAS, START, PHYSICAL, REST, 16, "self-attention"
    )
    pool.close()
    assert len(parallel) == len(THETAS)
    for a, b in zip(parallel, serial):
        np.testing.assert_array_equal(a, b)


def test_pool_covers_all_particles_when_not_divisible() -> None:
    thetas = THETAS + THETAS[:2]  # 7 particles across 3 workers
    pool = ForecastPool(workers=3)
    out = pool.forecast_features(
        GEOMETRY, thetas, START, PHYSICAL, REST, 16, "self-attention"
    )
    pool.close()
    assert len(out) == 7


def test_pool_propagates_worker_errors() -> None:
    pool = ForecastPool(workers=2)
    with pytest.raises(Exception):
        # malformed geometry -> CuboidSpec arg-count error inside the worker
        pool.forecast_features(
            (0.08, 0.06), THETAS, START, PHYSICAL, REST, 16, "self-attention"
        )
    pool.close()


def test_close_is_idempotent() -> None:
    pool = ForecastPool(workers=2)
    pool.close()
    pool.close()
