"""B5 pessimistic filter: reject actions predicted to topple under any hypothesis.

Candidate actions come from the stochastic policy. Each candidate is
simulated under every retained hypothesis; the candidate is rejected when
any hypothesis predicts a tilt-to-critical-tilt ratio at or above the
near-topple fraction. The surviving candidate with the lowest worst-case
ratio executes. If every candidate topples, a zero-force push is returned.
"""

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.agents.rl.physics_providers import EnsemblePhysics
from src.agents.simulator.internal_sim import InternalSim
from src.simulation.environment.env import ActionRanges
from src.simulation.modelling.action_mapping import normalized_to_physical
from src.simulation.modelling.object import CuboidSpec


class Policy(Protocol):
    def predict(
        self, observations: ArrayLike, deterministic: bool = True
    ) -> NDArray[np.float32]: ...


class PessimisticFilter:
    """Shield the policy with worst-case tilt predictions over the ensemble."""

    def __init__(
        self,
        sim: InternalSim,
        provider: EnsemblePhysics,
        action_ranges: ActionRanges,
        target_distance: float,
        near_topple_frac: float = 0.8,
        candidates: int = 4,
        warmup_steps: int = 0,
    ) -> None:
        if not 0 < near_topple_frac < 1:
            raise ValueError("near_topple_frac must lie in (0, 1)")
        if candidates < 1:
            raise ValueError("candidates must be at least 1")
        self.sim = sim
        self.provider = provider
        self.action_ranges = action_ranges
        self.target_distance = target_distance
        self.near_topple_frac = near_topple_frac
        self.candidates = candidates
        self.warmup_steps = warmup_steps

    def select_action(
        self, policy: Policy, obs: ArrayLike
    ) -> NDArray[np.float32]:
        """Return a safe normalized action, overriding the policy when needed."""
        w, L, H = (float(v) for v in obs[4:])
        proposals = [
            policy.predict(obs, deterministic=False) for _ in range(self.candidates)
        ]
        best_action: NDArray[np.float32] | None = None
        best_ratio = float("inf")
        for candidate in proposals:
            ratio = self._worst_tilt_ratio(candidate, obs, w, L, H)
            if ratio >= self.near_topple_frac:
                continue
            if ratio < best_ratio:
                best_action, best_ratio = candidate, ratio
        if best_action is None:
            # All candidates topple: the safest push is no push at all.
            return np.array([0.0, -1.0, -1.0], dtype=np.float32)
        return best_action

    def _worst_tilt_ratio(
        self, action: ArrayLike, obs: ArrayLike, w: float, L: float, H: float
    ) -> float:
        """Max tilt-to-crit-tilt ratio predicted over every retained hypothesis."""
        d, psi, xdot, psidot = (float(v) for v in obs[:4])
        start = (self.target_distance - d, psi, xdot, psidot)
        worst = 0.0
        for theta, _weight in zip(*self.provider.hypotheses_with_weights(w, L, H)):
            spec = CuboidSpec(w=w, L=L, H=H, m=theta[0], mu=theta[1], h_c=theta[2])
            push = normalized_to_physical(action, (w, L, H), self.action_ranges)
            traj = self.sim.predict(spec, start, push, self.warmup_steps)
            worst = max(
                worst, max(abs(tilt) for tilt in traj.psi) / spec.crit_tilt
            )
        return worst
