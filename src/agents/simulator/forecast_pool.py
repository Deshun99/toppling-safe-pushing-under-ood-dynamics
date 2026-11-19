"""Parallel internal-simulator forecasts across CPU worker processes.

Each worker owns its own headless ``InternalSim`` (a ``p.DIRECT`` PyBullet client,
built once per worker). Forecasts are deterministic given ``(spec, start, action)``,
so distributing particles across workers reproduces the serial result exactly.
"""

from multiprocessing import get_context
from typing import Any

import numpy as np

from src.agents.rl.models.delta_actor_critic import TrajectoryEncoder
from src.agents.simulator.internal_sim import InternalSim
from src.simulation.execution.settling import RestConfig
from src.simulation.modelling.object import CuboidSpec


def particle_features(
    sim: InternalSim,
    geometry: tuple[float, float, float],
    theta: tuple[float, float, float],
    start: Any,
    physical: Any,
    rest: RestConfig,
    points: int,
    strategy: str,
) -> np.ndarray:
    """Trajectory features for one physics hypothesis under one physical action."""
    spec = CuboidSpec(*geometry, *theta)
    options = {"rest": rest} if rest.enabled else {}
    trajectory = sim.predict(spec, start, physical, **options)
    return TrajectoryEncoder.extract_trajectory_features(
        trajectory, points=points, strategy=strategy
    )


_WORKER_SIM: InternalSim | None = None


def _init_worker() -> None:
    global _WORKER_SIM
    _WORKER_SIM = InternalSim()


def _forecast_chunk(payload: tuple) -> list[np.ndarray]:
    geometry, thetas, start, physical, rest, points, strategy = payload
    return [
        particle_features(
            _WORKER_SIM, geometry, theta, start, physical, rest, points, strategy
        )
        for theta in thetas
    ]


class ForecastPool:
    """Persistent spawn-context pool; one ``InternalSim`` per worker."""

    def __init__(self, workers: int) -> None:
        if workers < 2:
            raise ValueError("ForecastPool requires at least two workers")
        self._workers = workers
        self._pool = get_context("spawn").Pool(workers, initializer=_init_worker)
        self._closed = False

    def forecast_features(
        self,
        geometry: tuple[float, float, float],
        thetas: list,
        start: Any,
        physical: Any,
        rest: RestConfig,
        points: int,
        strategy: str,
    ) -> list[np.ndarray]:
        """One feature array per theta, in input order, computed across workers."""
        if not thetas:
            return []
        chunks = [
            c
            for c in np.array_split(np.asarray(thetas, dtype=float), self._workers)
            if len(c)
        ]
        payloads = [
            (
                geometry,
                [tuple(theta) for theta in chunk],
                start,
                physical,
                rest,
                points,
                strategy,
            )
            for chunk in chunks
        ]
        results = self._pool.map(_forecast_chunk, payloads)
        return [features for chunk in results for features in chunk]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pool.terminate()
        self._pool.join()
