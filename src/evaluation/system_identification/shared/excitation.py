"""Recorded excitation protocols, target settling, and audited data collection."""

from collections.abc import Sequence
from dataclasses import asdict

import numpy as np
import pybullet as p

from src.agents.system_identification.sysid import SysIDResult
from src.evaluation.system_identification.shared.constants import (
    EXCITATION_REGIMES,
    SPEC,
    UPRIGHT_TILT_LIMIT,
    WARMUP,
    EstimationErrors,
)
from src.simulation.environment.env import PushEnv
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.constants import DT
from src.simulation.modelling.records import ActionRecord


def collect_records(
    actions: Sequence[PushAction], *, independent: bool
) -> tuple[list[ActionRecord], list[dict]]:
    """Keep a rejection audit; held-out pushes reset, high training pushes settle."""
    target = PushEnv(fixed_spec=SPEC)
    records = []
    audit = []
    try:
        target.reset(seed=0)
        for action in actions:
            if independent:
                target.reset(seed=0)
            elif action.h >= 0.5 * SPEC.H:
                settle_or_replace_object(target)
            record = target.push(action, warmup_steps=WARMUP)
            accepted = (
                max((abs(value) for value in record.traj.psi)) <= UPRIGHT_TILT_LIMIT
            )
            audit.append({"accepted_for_fitting": accepted, "record": asdict(record)})
            if accepted or independent:
                records.append(record)
            else:
                target.reset(seed=0)
    finally:
        target.close()
    return (records, audit)


def settle(target: PushEnv, max_steps: int = round(1.0 / DT)) -> bool:
    """Step until flat, still, and nearly stationary.

    Near-tipping outcomes are bistable AND depend on the sliding state
    (measured: F=4.0 at 0.7H topples a resting box to 89.6 deg but tilts
    a fast-sliding box only 1.97 deg), so tilt-excitation pushes start
    from rest for reproducible replay."""
    for _ in range(max_steps):
        _, psi, vx, wy = target.cuboid.state()
        if abs(psi) < 0.001 and abs(wy) < 0.001 and (abs(vx) < 0.02):
            return True
        p.stepSimulation(physicsClientId=target.client)
    return False


def settle_or_replace_object(target: PushEnv) -> None:
    """Ensure a clean, reproducible start state before tilt-excitation
    pushes. Near-tipping replay is bistable, and mid-motion starts carry
    unrecorded residual velocities (vz/roll/yaw) that the replay zeroes
    out — a fallen object gets replaced by a fresh one at the current x."""
    if not settle(target):
        x, _, _, _ = target.cuboid.state()
        target.replace_object(x=x)
        settle(target)


def get_regimes() -> dict[str, list[PushAction]]:
    """Return independent action lists for the configured excitation regimes."""
    return {name: list(actions) for name, actions in EXCITATION_REGIMES.items()}


def compute_errors(res: SysIDResult) -> EstimationErrors:
    m_hat, mu_hat, hc_hat = res.point.theta
    m_err = abs(m_hat - SPEC.m) / SPEC.m
    mu_err = abs(mu_hat - SPEC.mu)
    hc_err = abs(hc_hat - SPEC.h_c) / SPEC.H
    spread = [float(np.std([h.theta[i] for h in res.hypotheses])) for i in range(3)]
    return {
        "m_err": m_err,
        "mu_err": mu_err,
        "hc_err": hc_err,
        "spread": (spread[0], spread[1], spread[2]),
    }
