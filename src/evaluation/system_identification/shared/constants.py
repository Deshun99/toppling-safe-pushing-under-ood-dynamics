"""Fixed settings for the recorded SysID experiments, separate from estimators."""

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from src.agents.system_identification.sysid import Z_HI, Z_LO
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.object import CuboidSpec

PARAMETERS = ("m", "mu", "h_c")
UPRIGHT_TILT_LIMIT = 0.15  # Observable cutoff, independent of hidden COM.

WARMUP = 8  # free steps before each push so replay contact matches (see
# ForcePushExecutor.execute docstring)

# True (hidden) parameters of the test object. mu = 0.3 leaves room for
# above-COM pushes without toppling (analytic condition, outline §4):
#   (F/mg)(h - h_c) <= w/2 - mu*h_c  ->  F <= 4.9 N at h = 0.6H.
SPEC = CuboidSpec(w=0.08, L=0.08, H=0.20, m=1.0, mu=0.3, h_c=0.10)

RETAINED_MASS = 0.999
CONVERGENCE_DISTANCE = 0.005
CONVERGENCE_LOG_EVIDENCE = 0.2
PREDICTIVE_BATCHES = 4
PREDICTIVE_BATCH_SIZE = 256


@dataclass
class Grid:
    """Cells cover the entire prior; likelihood columns are action increments."""

    low: NDArray[np.float64]
    high: NDArray[np.float64]
    log_likelihood: NDArray[np.float64]

    @property
    def points(self) -> NDArray[np.float64]:
        return (self.low + self.high) / 2

    def posterior(self, step: int) -> tuple[NDArray[np.float64], float]:
        log_prior_mass = np.sum(np.log(self.high - self.low), axis=1) - np.sum(
            np.log(Z_HI - Z_LO)
        )
        log_mass = log_prior_mass + self.log_likelihood[:, :step].sum(axis=1)
        evidence = float(logsumexp(log_mass))
        return np.exp(log_mass - evidence), evidence


EXCITATION_REGIMES = {
    "near slip threshold": [
        PushAction(h=0.3 * SPEC.H, F=F, tau=0.3) for F in (3.0, 3.5, 4.0, 5.0)
    ],
    "varied magnitudes": [
        PushAction(h=0.3 * SPEC.H, F=F, tau=0.25) for F in (3.5, 5.0, 7.0, 9.0)
    ],
    "varied heights": [
        PushAction(h=h * SPEC.H, F=5.0, tau=0.3) for h in (0.1, 0.3, 0.45)
    ],
    "varied durations": [
        PushAction(h=0.3 * SPEC.H, F=5.0, tau=tau) for tau in (0.1, 0.25, 0.4)
    ],
    "tilt boundary": [
        PushAction(h=0.6 * SPEC.H, F=4.0, tau=0.4),
        PushAction(h=0.6 * SPEC.H, F=4.5, tau=0.4),
        PushAction(h=0.7 * SPEC.H, F=2.0, tau=0.4),
        PushAction(h=0.7 * SPEC.H, F=2.3, tau=0.4),
    ],
    "passive mixed": [
        PushAction(h=0.3 * SPEC.H, F=5.0, tau=0.25),
        PushAction(h=0.3 * SPEC.H, F=0.0, tau=0.3),
        PushAction(h=0.45 * SPEC.H, F=5.0, tau=0.25),
        PushAction(h=0.3 * SPEC.H, F=0.0, tau=0.3),
        PushAction(h=0.6 * SPEC.H, F=4.5, tau=0.4),
        PushAction(h=0.7 * SPEC.H, F=2.3, tau=0.4),
    ],
    "full mixed": [
        PushAction(h=0.3 * SPEC.H, F=3.0, tau=0.3),
        PushAction(h=0.3 * SPEC.H, F=4.0, tau=0.3),
        PushAction(h=0.3 * SPEC.H, F=5.0, tau=0.3),
        PushAction(h=0.1 * SPEC.H, F=5.0, tau=0.3),
        PushAction(h=0.45 * SPEC.H, F=5.0, tau=0.3),
        PushAction(h=0.3 * SPEC.H, F=5.0, tau=0.1),
        PushAction(h=0.3 * SPEC.H, F=5.0, tau=0.4),
        PushAction(h=0.3 * SPEC.H, F=0.0, tau=0.3),
        PushAction(h=0.6 * SPEC.H, F=4.5, tau=0.4),
        PushAction(h=0.7 * SPEC.H, F=2.3, tau=0.4),
        PushAction(h=0.3 * SPEC.H, F=7.0, tau=0.25),
        PushAction(h=0.3 * SPEC.H, F=9.0, tau=0.25),
    ],
}


class EstimationErrors(TypedDict):
    m_err: float
    mu_err: float
    hc_err: float
    spread: tuple[float, float, float]


SUMMARY_FIELDS = (
    "regime",
    "seed",
    "step",
    "method",
    "fit_seconds",
    "fit_replays",
    "cumulative_fit_replays",
    "relative_mass_error",
    "com_error_fraction_height",
)
