"""S5 feasibility check: can a 10-push gantry episode span the lane?

Scripted policy: every decision pushes at h = 0.04 m, F = 15 N, tau = 0.5 s
(the action bounds' maxima), repeated 10 times or until the block topples,
then the block coasts to rest. The final centre position is compared with
the 0.50 m lane goal band (+/- 0.05 m). This measures the kinematic ceiling
of the gantry under the episode decision budget, independent of any learned
policy. The pad-travel bound follows the admittance-controlled push-phase
speed; coasting is the only surplus.

Run from the repo root:
    python -m src.simulation.tools.validation.gantry_episode_span --objects 8
"""

import argparse

import numpy as np
import pybullet as p

from src.simulation.environment import PushEnv
from src.simulation.execution.robot import GantryPushExecutor
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.constants import DT
from src.simulation.modelling.object import CuboidSpec
from src.simulation.modelling.sampling import ObjectSampler

LANE = 0.5
GOAL_BAND_EDGE = LANE - 0.05  # inside the +/- 0.05 m goal band
PUSH_HEIGHT = 0.04  # low lever arm; keeps tall blocks upright
PUSH_FORCE = 15.0  # action-bounds maximum
PUSH_TAU = 0.5  # action-bounds maximum
PUSHES = 10  # episode decision budget
SETTLE_SECONDS = 2.0


def span(spec: CuboidSpec) -> tuple[float, bool, list[float]]:
    """Run a scripted 10-push episode; return final x, toppled, per-push x."""
    env = PushEnv(gui=False, fixed_spec=spec, target_distance=LANE)
    try:
        env.executor = GantryPushExecutor(env.client)
        env.reset(seed=0)
        xs = [env.cuboid.state()[0]]
        toppled = False
        for _ in range(PUSHES):
            env.push(PushAction(h=PUSH_HEIGHT, F=PUSH_FORCE, tau=PUSH_TAU))
            x, tilt, _, _ = env.cuboid.state()
            xs.append(x)
            if abs(tilt) >= spec.crit_tilt:
                toppled = True
                break
        for _ in range(round(SETTLE_SECONDS / DT)):
            p.stepSimulation(physicsClientId=env.client)
        return env.cuboid.state()[0], toppled, xs
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--objects", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", help="write per-object rows to this path")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for obj in range(args.objects):
        spec = ObjectSampler().sample(rng)
        final_x, toppled, xs = span(spec)
        pushes = len(xs) - 1
        rows.append(
            {
                "object": obj, "m": round(spec.m, 3), "mu": round(spec.mu, 3),
                "H": round(spec.H, 3), "final_x": round(final_x, 4),
                "toppled": toppled, "pushes_used": pushes,
                "mean_step": round((xs[-1] - xs[0]) / pushes, 4),
                "coast": round(final_x - xs[-1], 4),
            }
        )
        print(f"object {obj}: m={spec.m:.2f} mu={spec.mu:.2f} H={spec.H:.2f} "
              f"-> final x {final_x:.3f} m ({'toppled' if toppled else 'upright'})")

    upright = [r for r in rows if not r["toppled"]]
    print(f"\n{upright and len(upright)}/{len(rows)} objects upright at end")
    if upright:
        band = sum(r["final_x"] >= GOAL_BAND_EDGE for r in upright)
        goal = sum(r["final_x"] >= LANE for r in upright)
        fx = np.array([r["final_x"] for r in upright])
        steps = np.array([r["mean_step"] for r in upright])
        coast = np.array([r["coast"] for r in upright])
        print(f"reached goal band (x >= {GOAL_BAND_EDGE}): {band}/{len(upright)}")
        print(f"reached or passed x = {LANE}: {goal}/{len(upright)}")
        print(f"final x: min {fx.min():.3f}, mean {fx.mean():.3f}, max {fx.max():.3f} m")
        print(f"mean per-push increment: min {steps.min():.3f}, mean {steps.mean():.3f}, "
              f"max {steps.max():.3f} m")
        print(f"coasting surplus: mean {coast.mean():.3f} m  "
              "(pad-travel bound follows the push-phase speed)")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
