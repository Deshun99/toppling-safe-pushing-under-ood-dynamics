"""Sources of hidden physics used by the internal simulator (outline §10).

One provider per imagination baseline: B2 nominal, B3 point estimate,
B4 ensemble, B6 oracle. Providers configure the internal simulator only;
they never reach the policy observation.
"""

from typing import Protocol

import numpy as np

from src.agents.system_identification.sysid import SysID, SysIDResult
from src.simulation.modelling.object import Geometry, PhysicsParameters
from src.simulation.modelling.records import ActionRecord
from src.simulation.modelling.sampling import DEFAULT_OBJECT_RANGES, ObjectRanges


class PhysicsProvider(Protocol):
    def refresh(self) -> None:
        """Bring the estimate up to date with every observed push."""

    def observe(self, record: ActionRecord) -> None:
        """Receive a real push (estimators refit on it; fixed providers ignore it)."""

    def theta(self, w: float, L: float, H: float) -> PhysicsParameters:
        """Return (m, mu, h_c) for an object with the observed geometry."""


class OraclePhysics:
    """B6: the true parameters of the current object."""

    def __init__(self, theta: PhysicsParameters) -> None:
        self._theta = tuple(theta)

    def refresh(self) -> None:
        pass

    def observe(self, record: ActionRecord) -> None:
        pass

    def theta(self, w: float, L: float, H: float) -> PhysicsParameters:
        return self._theta


class NominalPhysics:
    """B2: fixed training-distribution means, h_c scaled to object height."""

    def __init__(self, ranges: ObjectRanges = DEFAULT_OBJECT_RANGES) -> None:
        self.m = (ranges.m[0] + ranges.m[1]) / 2
        self.mu = (ranges.mu[0] + ranges.mu[1]) / 2
        self.h_c_frac = (ranges.h_c_frac[0] + ranges.h_c_frac[1]) / 2

    def refresh(self) -> None:
        pass

    def observe(self, record: ActionRecord) -> None:
        pass

    def theta(self, w: float, L: float, H: float) -> PhysicsParameters:
        return (self.m, self.mu, self.h_c_frac * H)


class EstimatedPhysics:
    """B3: the SysID point estimate, refit every `refit_every` pushes.

    Falls back to nominal parameters until the first refit. One instance per
    test object; the history persists for the object's lifetime. Refits use
    a recent-history window and warm-start from the previous point estimate.
    """

    def __init__(
        self,
        geometry: Geometry,
        sysid: SysID | None = None,
        *,
        refit_every: int = 1,
        history_window: int = 20,
        replay_budget: int = 256,
        fallback: NominalPhysics | None = None,
    ) -> None:
        if refit_every < 1:
            raise ValueError("refit_every must be at least 1")
        self.geometry = geometry
        self.sysid = sysid or SysID(geometry=geometry)
        self.refit_every = refit_every
        self.history_window = history_window
        self.replay_budget = replay_budget
        self.fallback = fallback or NominalPhysics()
        self.history: list[ActionRecord] = []
        self.point: PhysicsParameters | None = None
        self._pending = 0

    def refresh(self) -> None:
        if not self.history:
            return
        result = self.sysid.update(
            self.history[-self.history_window:],
            replay_budget=self.replay_budget,
            previous_best_parameters=self.point,
        )
        self.point = result.point.theta
        self._pending = 0

    def observe(self, record: ActionRecord) -> None:
        self.history.append(record)
        self._pending += 1
        if self._pending >= self.refit_every:
            self.refresh()

    def theta(self, w: float, L: float, H: float) -> PhysicsParameters:
        return self.point if self.point is not None else self.fallback.theta(w, L, H)


class EnsemblePhysics(EstimatedPhysics):
    """B4/B5: top-K hypotheses; each rollout samples one, weighted by softmax loss."""

    def __init__(
        self,
        geometry: Geometry,
        sysid: SysID | None = None,
        *,
        refit_every: int = 1,
        history_window: int = 20,
        replay_budget: int = 256,
        fallback: NominalPhysics | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(
            geometry,
            sysid,
            refit_every=refit_every,
            history_window=history_window,
            replay_budget=replay_budget,
            fallback=fallback,
        )
        self.hypotheses: list[PhysicsParameters] = []
        self.weights: list[float] = []
        self.rng = np.random.default_rng(seed)

    def refresh(self) -> None:
        if not self.history:
            return
        result: SysIDResult = self.sysid.update(
            self.history[-self.history_window:],
            replay_budget=self.replay_budget,
            previous_best_parameters=self.point,
        )
        self.point = result.point.theta
        self.hypotheses = [h.theta for h in result.hypotheses] or [self.point]
        self.weights = [h.weight for h in result.hypotheses] or [1.0]
        self._pending = 0

    def theta(self, w: float, L: float, H: float) -> PhysicsParameters:
        if not self.hypotheses:
            return self.fallback.theta(w, L, H)
        index = self.rng.choice(len(self.hypotheses), p=self.weights)
        return self.hypotheses[index]

    def hypotheses_with_weights(
        self, w: float, L: float, H: float
    ) -> tuple[list[PhysicsParameters], list[float]]:
        """The retained set (or the nominal fallback) for pessimistic evaluation."""
        if self.hypotheses:
            return self.hypotheses, self.weights
        return [self.fallback.theta(w, L, H)], [1.0]


BASELINES = ("b2", "b3", "b4", "b6")


def make_provider(
    baseline: str,
    geometry: Geometry,
    theta_true: PhysicsParameters | None = None,
    **kwargs: object,
) -> PhysicsProvider:
    """Provider for an imagination baseline; B5 reuses the B4 ensemble."""
    if baseline == "b2":
        return NominalPhysics()
    if baseline == "b3":
        return EstimatedPhysics(geometry, **kwargs)
    if baseline in ("b4", "b5"):
        return EnsemblePhysics(geometry, **kwargs)
    if baseline == "b6":
        if theta_true is None:
            raise ValueError("b6 requires the object's true parameters")
        return OraclePhysics(theta_true)
    raise ValueError(f"baseline must be one of {BASELINES}, got {baseline!r}")
