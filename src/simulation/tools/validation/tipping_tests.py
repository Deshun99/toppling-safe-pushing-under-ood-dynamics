"""M1 scripted physics sanity tests (outline §5.3, milestone M1).

Validates the cuboid + direct-force primitive against the analytic no-tip
condition (outline §4):

    (F / (m g)) (h - h_c)  <=  w/2 - mu h_c        (while sliding)

Hard checks:
  1. a gentle low-height push slides the cuboid without toppling;
  2. a violent high-height push topples it.
A diagnostic grid compares the analytic prediction with the simulator over a
range of (h, F) — approximate agreement expected, not hard-asserted.

Run from the repo root:  python -m src.simulation.tools.validation.tipping_tests
"""

import sys

import numpy as np
import pybullet as p
import pybullet_data

from src.simulation.execution.force_push import ForcePushExecutor
from src.simulation.modelling.constants import DT
from src.simulation.modelling.object import Cuboid, CuboidSpec

G = 9.81
TAU = 0.4  # push duration for the scripted tests (s)
SETTLE_STEPS = round(0.5 / DT)  # 0.5 s settle after spawn


def no_tip_predicted(spec: CuboidSpec, h: float, F: float) -> bool:
    lhs = (F / (spec.m * G)) * (h - spec.h_c)
    rhs = spec.w / 2.0 - spec.mu * spec.h_c
    return lhs <= rhs


def run_push(
    client: int, spec: CuboidSpec, executor: ForcePushExecutor, h: float, F: float
) -> tuple[float, float]:
    """Spawn a fresh cuboid, push once, return (max |tilt|, displacement)."""
    cuboid = Cuboid(spec, client)
    cuboid.spawn(position=(0.0, 0.0, spec.H / 2.0 + 0.005))
    for _ in range(SETTLE_STEPS):
        p.stepSimulation(physicsClientId=client)
    traj = executor.execute(cuboid, h, F, TAU)
    max_tilt = max(abs(psi) for psi in traj.psi)
    dx = traj.x[-1] - traj.x[0]
    cuboid.remove()
    return max_tilt, dx


def _check_push_outcomes(
    client: int,
    spec: CuboidSpec,
    executor: ForcePushExecutor,
    failures: list[str],
) -> None:
    # 1. Gentle push, below COM, enough force to slide: no topple.
    max_tilt, dx = run_push(client, spec, executor, h=0.5 * spec.h_c, F=4.0)
    ok = dx > 0.01 and max_tilt < spec.crit_tilt
    print(
        f"[{'PASS' if ok else 'FAIL'}] gentle push: "
        f"dx={dx:.3f} m, max tilt={np.degrees(max_tilt):.1f} deg"
    )
    if not ok:
        failures.append("gentle push")

    # 2. Violent push, near the top: topples past the critical tilt.
    max_tilt, _ = run_push(client, spec, executor, h=0.9 * spec.H, F=8.0)
    ok = max_tilt > spec.crit_tilt
    print(
        f"[{'PASS' if ok else 'FAIL'}] violent push: "
        f"max tilt={np.degrees(max_tilt):.1f} deg"
    )
    if not ok:
        failures.append("violent push")


def _print_analytic_agreement(
    client: int, spec: CuboidSpec, executor: ForcePushExecutor
) -> None:
    # 3. Diagnostic grid: analytic condition vs. sim (o = sim safe, x = toppled).
    print("\nAnalytic no-tip condition vs. sim:")
    agreement = []
    for h_frac in (0.2, 0.4, 0.6, 0.8, 0.9):
        row = f"h={h_frac:.1f}H: "
        for F in (1.0, 3.0, 5.0, 7.0):
            h = h_frac * spec.H
            max_tilt, _ = run_push(client, spec, executor, h=h, F=F)
            predicted_safe = no_tip_predicted(spec, h, F)
            sim_safe = max_tilt < spec.crit_tilt
            agreement.append(predicted_safe == sim_safe)
            row += (
                ("o" if sim_safe else "x")
                + ("v" if predicted_safe == sim_safe else "!")
                + " "
            )
        print(row)
    print(
        f"\nAnalytic agreement: {np.mean(agreement) * 100:.0f}% "
        f"(approximate model — diagnostic only)"
    )


def main() -> None:
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -G, physicsClientId=client)
    p.setTimeStep(DT, physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane = p.loadURDF("plane.urdf", physicsClientId=client)
    # Floor mu = 1.0: cuboid's lateralFriction IS the effective coefficient
    # (see src/simulation/modelling/object.py friction convention).
    p.changeDynamics(plane, -1, lateralFriction=1.0, physicsClientId=client)

    spec = CuboidSpec(w=0.08, L=0.08, H=0.20, m=1.0, mu=0.3, h_c=0.10)
    executor = ForcePushExecutor(client)
    print(f"crit tilt = {np.degrees(spec.crit_tilt):.1f} deg\n")

    failures = []

    _check_push_outcomes(client, spec, executor, failures)
    _print_analytic_agreement(client, spec, executor)

    p.disconnect(physicsClientId=client)

    if failures:
        print("FAILED:", ", ".join(failures))
        sys.exit(1)
    print("All hard checks passed.")


if __name__ == "__main__":
    main()
