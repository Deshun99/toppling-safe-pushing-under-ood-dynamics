"""S2 validation harness: the same (h, F, tau) through both executors.

Samples objects from the default `ObjectSampler` ranges and actions from the
M3 action bounds, executes each command once with direct force and once with
the gantry executor on identical objects, and reports the gap: displacement,
peak tilt, topple outcome, and the gantry's realized-versus-commanded
execution (the G1 noise). Commanded height is clamped into the pad half-height
window because the gantry cannot contact closer than PAD_HALF_HEIGHT to
either face edge. There is no pass/fail threshold yet; the printed gap is
the deliverable.

Run from the repo root:
    python -m src.simulation.tools.validation.gantry_vs_force --objects 6 --pushes 8
"""

import argparse
import csv
from dataclasses import dataclass

import numpy as np

from src.simulation.environment import PushEnv
from src.simulation.environment.env import ActionRanges
from src.simulation.execution.robot import GantryPushExecutor, PAD_HALF_HEIGHT
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.object import CuboidSpec
from src.simulation.modelling.sampling import ObjectSampler

LANE = 0.5  # m, gantry-mode lane length (pad travel)
M3_ACTION_RANGES = ActionRanges(force=(0.0, 15.0))  # matches --f-max 15 training


@dataclass
class Outcome:
    dx: float
    peak_tilt: float
    toppled: bool
    contact: bool = True
    h_real: float | None = None
    F_real: float | None = None
    tau_real: float | None = None
    stray: int = 0


def sample_action(
    rng: np.random.Generator, spec: CuboidSpec, ranges: ActionRanges
) -> PushAction:
    lower = max(ranges.height_fraction[0] * spec.H, PAD_HALF_HEIGHT)
    upper = min(ranges.height_fraction[1] * spec.H, spec.H - PAD_HALF_HEIGHT)
    return PushAction(
        h=rng.uniform(lower, upper),
        F=rng.uniform(*ranges.force),
        tau=rng.uniform(*ranges.duration),
    )


def execute(spec: CuboidSpec, action: PushAction, robot: bool) -> Outcome:
    env = PushEnv(gui=False, fixed_spec=spec, target_distance=LANE)
    try:
        if robot:
            env.executor = GantryPushExecutor(env.client)
        env.reset(seed=0)
        x0 = env.cuboid.state()[0]
        env.push(action)
        traj = env.last_trajectory
        _, psi, _, _ = env.cuboid.state()
        out = Outcome(
            dx=traj.x[-1] - x0,
            peak_tilt=max(abs(a) for a in traj.psi),
            toppled=abs(psi) > spec.crit_tilt,
        )
        if robot:
            rep = env.executor.last_execution
            out.contact, out.h_real = rep.contact_made, rep.h_real
            out.F_real, out.tau_real = rep.F_real, rep.tau_real
            out.stray = rep.stray_contact_steps
        return out
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--objects", type=int, default=6)
    ap.add_argument("--pushes", type=int, default=8, help="pushes per object")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", help="write per-push rows to this path")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for obj in range(args.objects):
        spec = ObjectSampler().sample(rng)
        for k in range(args.pushes):
            act = sample_action(rng, spec, M3_ACTION_RANGES)
            f, r = execute(spec, act, False), execute(spec, act, True)
            rows.append(
                {
                    "object": obj, "push": k, "H": spec.H, "m": spec.m, "mu": spec.mu,
                    "h_c": spec.h_c, "h_cmd": act.h, "F_cmd": act.F, "tau_cmd": act.tau,
                    "dx_force": f.dx, "dx_robot": r.dx,
                    "tilt_force": f.peak_tilt, "tilt_robot": r.peak_tilt,
                    "topple_force": f.toppled, "topple_robot": r.toppled,
                    "contact": r.contact, "h_real": r.h_real,
                    "F_real": r.F_real, "tau_real": r.tau_real, "stray": r.stray,
                }
            )

    n = len(rows)
    ok = [x for x in rows if x["contact"]]
    print(f"{n} pushes on {args.objects} objects, lane {LANE} m, F <= "
          f"{M3_ACTION_RANGES.force[1]:g} N")
    print(f"contact failures: {n - len(ok)}/{n}")
    print(f"pushes with stray (non-pad) arm contact: {sum(x['stray'] > 0 for x in rows)}/{n}")
    if ok:
        dxf = np.array([x["dx_force"] for x in ok])
        dxr = np.array([x["dx_robot"] for x in ok])
        corr = np.corrcoef(dxf, dxr)[0, 1] if len(ok) > 2 else float("nan")
        print(f"displacement  mean force {dxf.mean():.3f} m, mean robot {dxr.mean():.3f} m, "
              f"MAE {np.abs(dxf - dxr).mean():.3f} m, corr {corr:.2f}")
        tf = np.array([x["tilt_force"] for x in ok])
        tr = np.array([x["tilt_robot"] for x in ok])
        print(f"peak tilt     MAE {np.abs(tf - tr).mean():.3f} rad")
        agree = np.mean([x["topple_force"] == x["topple_robot"] for x in ok])
        print(f"topple outcome agreement {agree:.0%}  "
              f"(force toppled {sum(x['topple_force'] for x in ok)}, "
              f"robot toppled {sum(x['topple_robot'] for x in ok)})")
        fe = [x["F_real"] / x["F_cmd"] for x in ok if x["F_real"] and x["F_cmd"] > 1.0]
        te = [x["tau_real"] / x["tau_cmd"] for x in ok if x["tau_real"] is not None]
        he = [x["h_real"] - x["h_cmd"] for x in ok if x["h_real"] is not None]
        if fe:
            print(f"realized/commanded force  mean {np.mean(fe):.2f} (std {np.std(fe):.2f})")
        if te:
            print(f"realized/commanded tau    mean {np.mean(te):.2f}")
        if he:
            print(f"height error (real-cmd)   mean {np.mean(he) * 1000:.1f} mm")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
