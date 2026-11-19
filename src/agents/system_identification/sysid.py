"""
Online system identification (M2).

Outline §6: after each target action, candidates z = (log m, mu, h_c/H) are
scored by replaying the recorded action in the internal (headless) simulator
and comparing dense trajectories. The point estimate is the argmin of the
loss; the top-K low-loss hypotheses are retained with softmax weights
(temperature T). Estimates configure the internal simulator only — never
appended to the policy observation (§3.2).
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from src.agents.simulator.internal_sim import InternalSim
from src.agents.system_identification.scoring import get_scoring_indices
from src.simulation.execution.force_push import DenseTrajectory
from src.simulation.modelling.object import CuboidSpec, Geometry, PhysicsParameters
from src.simulation.modelling.records import ActionRecord

# 1/sigma^2 scales for the trajectory-discrepancy loss (outline §6 eq. for
# L(z)). Calibrated from the replay-fidelity noise floor at the true theta
# via src/evaluation/system_identification/diagnostics/calibrate_scales.py: each quantity is weighted by its own
# reproducibility, so tilt terms (identifying h_c) are not drowned by
# position terms in raw SI units. Recalibrate if the sim changes.
LOSS_SCALES: dict[str, float] = {
    "x": 7.899e04,  # mse 1.266e-05 m^2 per sample
    "psi": 4.534e06,  # mse 2.206e-07 rad^2
    "xdot": 2.402e03,  # mse 4.163e-04 (m/s)^2
    "psidot": 5.240e02,  # mse 1.908e-03 (rad/s)^2
}

# Search bounds for z = (log m, mu, h_c/H): wider than the training
# distribution (m 0.5-1.5, mu 0.3-0.7, h_c/H 0.35-0.60) so OOD parameters
# (m 0.2-3.0, mu 0.1-1.1, h_c/H 0.20-0.80) stay reachable (outline §6/§9.1).
Z_LO = np.array([math.log(0.2), 0.10, 0.20])
Z_HI = np.array([math.log(3.0), 1.10, 0.80])
N_PHYSICS_PARAMETERS = Z_LO.size  # log mass, friction, and relative COM height
Z_PRIOR_MEAN = np.array([math.log(0.866), 0.5, 0.475])  # training prior (§6)
Z_PRIOR_SIGMA = 0.3


def convert_z_to_cuboid_spec(
    z: Sequence[float] | NDArray[np.float64], geometry: Geometry
) -> CuboidSpec:
    """Candidate z = (log m, mu, h_c/H) -> CuboidSpec for the internal sim."""
    w, L, H = geometry
    return CuboidSpec(w=w, L=L, H=H, m=math.exp(z[0]), mu=z[1], h_c=z[2] * H)


def trajectory_loss(
    recorded: DenseTrajectory,
    simulated: DenseTrajectory,
    scales: Mapping[str, float] | None = None,
    *,
    warmup_steps: int = 0,
    sample_stride: int = 1,
) -> float:
    """Squared trajectory discrepancy, summed over aligned samples.

    Both trajectories are sampled at the same dt over the same action, so
    alignment is by index; the shorter one truncates. Warm-up samples are
    always excluded; downsampling retains the recording's original phase.
    """
    scales = scales or LOSS_SCALES
    n = min(len(recorded.x), len(simulated.x))
    if not n:
        return 0.0
    indices = get_scoring_indices(
        n,
        warmup_steps=warmup_steps,
        sample_stride=sample_stride,
    )
    window = slice(indices[0], n, sample_stride)
    total = 0.0
    for channel, weight in scales.items():
        observed = np.asarray(getattr(recorded, channel))[window]
        predicted = np.asarray(getattr(simulated, channel))[window]
        total += weight * float(np.sum((observed - predicted) ** 2))
    return total


@dataclass
class Hypothesis:
    theta: PhysicsParameters  # (m, mu, h_c)
    loss: float
    weight: float = 1.0  # softmax over the retained set, w_k ~ exp(-L/T)


@dataclass
class SysIDResult:
    point: Hypothesis  # argmin estimate
    hypotheses: list[Hypothesis] = field(default_factory=list)  # top-K, weighted


class SysID:
    """CMA-ES over z, scoring candidates by replaying the history in an
    internal simulator. One instance per test object; the history persists
    across episodes for that object (outline §8.2)."""

    def __init__(
        self,
        geometry: Geometry,
        K: int = 4,
        iterations: int = 20,
        restarts: int = 2,
        seed: int = 0,
        *,
        sample_stride: int = 1,
    ) -> None:
        import cma

        if (
            not isinstance(sample_stride, int)
            or isinstance(sample_stride, bool)
            or sample_stride < 1
        ):
            raise ValueError("sample_stride must be a positive integer")
        self._cma = cma
        self.geometry = geometry  # observable (w, L, H)
        self.K = K
        self.iterations = iterations
        self.restarts = restarts
        self.seed = seed
        self.sample_stride = sample_stride
        self.sim = InternalSim()

    def loss(
        self, z: Sequence[float] | NDArray[np.float64], history: Sequence[ActionRecord]
    ) -> float:
        spec = convert_z_to_cuboid_spec(z, self.geometry)
        return sum(
            trajectory_loss(
                rec.traj,
                self.sim.replay(spec, rec),
                warmup_steps=rec.warmup_steps,
                sample_stride=self.sample_stride,
            )
            for rec in history
        )

    def _run_es(
        self,
        history: Sequence[ActionRecord],
        seed: int,
        *,
        max_evaluations: int | None = None,
    ) -> list[tuple[NDArray[np.float64], float]]:
        """Return (z, loss) pairs; partial populations are not passed to tell()."""
        es = self._cma.CMAEvolutionStrategy(
            Z_PRIOR_MEAN,
            Z_PRIOR_SIGMA,
            {"bounds": [Z_LO, Z_HI], "seed": seed, "verbose": -9},
        )
        evaluated: list[tuple[NDArray[np.float64], float]] = []
        generations = (
            self.iterations
            if max_evaluations is None
            else math.ceil(max_evaluations / es.popsize)
        )
        for _ in range(generations):
            solutions = es.ask()
            if max_evaluations is not None:
                solutions = solutions[: max_evaluations - len(evaluated)]
            losses = [self.loss(z, history) for z in solutions]
            evaluated.extend(zip(solutions, losses))
            if len(solutions) == es.popsize:
                es.tell(solutions, losses)
        return evaluated

    def update(
        self,
        history: Sequence[ActionRecord],
        *,
        replay_budget: int | None = None,
        previous_best_parameters: PhysicsParameters | None = None,
    ) -> SysIDResult:
        """Refit the full history, optionally rescoring a previous estimate.

        The replay cap includes rescoring the previous best estimate. Search
        starts share the remaining budget; no earlier loss or state is reused.
        """
        previous = self._validate_and_convert_previous_best(
            previous_best_parameters, history
        )
        runs = self._run_searches(history, replay_budget, previous is not None)
        previous_candidate = (
            [] if previous is None else [(previous, self.loss(previous, history))]
        )
        if not runs:
            runs = [previous_candidate]
        best_run = min(runs, key=lambda evaluated: min(loss for _, loss in evaluated))
        evaluated = (
            best_run + previous_candidate
            if best_run is not previous_candidate
            else best_run
        )
        evaluated.sort(key=lambda candidate: candidate[1])
        return self._select_hypotheses(evaluated)

    def _validate_and_convert_previous_best(
        self,
        previous_best_parameters: PhysicsParameters | None,
        history: Sequence[ActionRecord],
    ) -> NDArray[np.float64] | None:
        if previous_best_parameters is None:
            return None
        values = np.asarray(previous_best_parameters, dtype=float)
        if values.shape != (3,) or not np.all(np.isfinite(values)) or values[0] <= 0:
            raise ValueError(
                "previous_best_parameters must contain finite physical parameters"
            )
        z = np.array([math.log(values[0]), values[1], values[2] / self.geometry[2]])
        if np.any(z < Z_LO) or np.any(z > Z_HI):
            raise ValueError(
                "previous_best_parameters must lie inside the physics search bounds"
            )
        if not history:
            raise ValueError("previous_best_parameters requires a nonempty history")
        return z

    def _run_searches(
        self,
        history: Sequence[ActionRecord],
        replay_budget: int | None,
        has_previous_best: bool,
    ) -> list[list[tuple[NDArray[np.float64], float]]]:
        if replay_budget is None:
            seeds = range(self.seed, self.seed + self.restarts + 1)
            return [self._run_es(history, seed) for seed in seeds]
        if (
            not isinstance(replay_budget, int)
            or isinstance(replay_budget, bool)
            or not history
            or replay_budget < len(history)
        ):
            raise ValueError(
                "replay_budget must fund at least one complete-history evaluation"
            )
        evaluations = replay_budget // len(history) - int(has_previous_best)
        starts = min(self.restarts + 1, evaluations)
        per_start, remainder = divmod(evaluations, starts) if starts else (0, 0)
        return [
            self._run_es(
                history,
                self.seed + restart,
                max_evaluations=per_start + (restart < remainder),
            )
            for restart in range(starts)
        ]

    def _select_hypotheses(
        self,
        evaluated: list[tuple[NDArray[np.float64], float]],
    ) -> SysIDResult:
        best_z, best_loss = evaluated[0]
        retained: list[tuple[NDArray[np.float64], float]] = []
        for z, loss in evaluated:
            if all(np.linalg.norm(z - other) > 1e-3 for other, _ in retained):
                retained.append((z, loss))
            if len(retained) >= self.K:
                break
        losses = np.array([loss for _, loss in retained])
        temperature = 0.05 * losses[0] + 1e-12  # Scaled to the best loss.
        weights = np.exp(-(losses - losses[0]) / temperature)
        weights = weights / weights.sum()
        point = Hypothesis(
            theta=tuple(convert_z_to_cuboid_spec(best_z, self.geometry).theta),
            loss=best_loss,
        )
        hypotheses = [
            Hypothesis(
                theta=tuple(convert_z_to_cuboid_spec(z, self.geometry).theta),
                loss=loss,
                weight=weight,
            )
            for (z, loss), weight in zip(retained, weights)
        ]
        return SysIDResult(point=point, hypotheses=hypotheses)

    def close(self) -> None:
        self.sim.close()
