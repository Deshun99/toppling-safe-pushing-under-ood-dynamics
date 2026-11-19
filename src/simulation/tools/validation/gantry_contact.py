"""Measure admittance-controlled contact on sampled development objects.

Defaults produce 96 independent pushes: two seeds, six objects per seed and
eight commands per object. Each command starts with a fresh upright object;
this measures contact execution rather than sequential policy performance.
The seed choices and command sampling are implementation choices, because the
paper does not specify them. No paper result is used as an acceptance target.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.simulation.environment.env import ActionRanges, PushEnv
from src.simulation.execution.robot import PAD_HALF_HEIGHT
from src.simulation.modelling.action_mapping import physical_to_normalized
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.object import CuboidSpec
from src.simulation.modelling.sampling import ObjectSampler

CONTACT_RANGES = ActionRanges(force=(0.0, 15.0))


def sample_action(rng: np.random.Generator, spec: CuboidSpec) -> PushAction:
    """Sample within both the policy action bounds and the pad height window."""
    lower = max(CONTACT_RANGES.height_fraction[0] * spec.H, PAD_HALF_HEIGHT)
    upper = min(CONTACT_RANGES.height_fraction[1] * spec.H, spec.H - PAD_HALF_HEIGHT)
    return PushAction(float(rng.uniform(lower, upper)),
                      float(rng.uniform(*CONTACT_RANGES.force)),
                      float(rng.uniform(*CONTACT_RANGES.duration)))


def measure_push(env: PushEnv, action: PushAction, seed: int) -> dict[str, Any]:
    """Use timed-start motion and keep incomplete execution reports in the rows."""
    env.reset(seed=seed)
    geometry = (env.cuboid.spec.w, env.cuboid.spec.L, env.cuboid.spec.H)
    normalized = physical_to_normalized(action, geometry, CONTACT_RANGES)
    _, _, _, _, info = env.step(normalized)
    report, start = env.executor.last_execution, env.executor.timed_start
    trajectory = env.last_trajectory
    displacement = trajectory.x[-1] - start[0] if start and trajectory.x else None
    return dict(
        **asdict(env.cuboid.spec), h_cmd=action.h, F_cmd=action.F, tau_cmd=action.tau,
        displacement_m=displacement, toppled=bool(info["toppled"]),
        execution_status=info["execution_status"],
        contact_made=bool(report and report.contact_made),
        h_real=report.h_real if report else None,
        F_real=report.F_real if report else None,
        tau_real=report.tau_real if report else 0.0,
        normal_impulse_ns=report.impulse_real if report else 0.0,
        stray_contact_steps=report.stray_contact_steps if report else 0,
    )


def _moments(values: list[float]) -> dict[str, float | int | None]:
    """Return null for unavailable statistics, preserving strict JSON output."""
    return dict(count=len(values), mean=float(np.mean(values)) if values else None,
                std=float(np.std(values)) if values else None)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Normal forces average only contact timesteps, matching paper Eq. (6)."""
    contact = [row for row in rows if row["tau_real"] > 0]
    force = [row["F_real"] / row["F_cmd"] for row in contact
             if row["F_real"] is not None and row["F_cmd"] > 0]
    height = [row["h_real"] - row["h_cmd"] for row in contact
              if row["h_real"] is not None]
    return dict(
        push_count=len(rows),
        contact_failures=sum(row["tau_real"] <= 0 for row in rows),
        execution_failures=sum(row["execution_status"] != "completed" for row in rows),
        pushes_with_stray_contact=sum(row["stray_contact_steps"] > 0 for row in rows),
        toppled_pushes=sum(row["toppled"] for row in rows),
        measured_requested_normal_force_ratio=_moments(force),
        contact_commanded_duration_ratio=_moments(
            [row["tau_real"] / row["tau_cmd"] for row in contact]),
        signed_pad_height_error_m=_moments(height),
        timed_displacement_m=_moments(
            [row["displacement_m"] for row in rows if row["displacement_m"] is not None]),
    )


def run(seeds: list[int], objects: int, pushes: int) -> dict[str, Any]:
    """Save commands and sampled objects so each measured trial is reviewable."""
    if not seeds or any(seed < 0 for seed in seeds) or objects < 1 or pushes < 1:
        raise ValueError("provide nonnegative seeds and positive object/push counts")
    rows, sampler = [], ObjectSampler()
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for object_index in range(objects):
            spec = sampler.sample(rng)
            env = PushEnv(robot=True, fixed_spec=spec, target_distance=0.5,
                          action_ranges=CONTACT_RANGES)
            try:
                for push_index in range(pushes):
                    action = sample_action(rng, spec)
                    row = measure_push(env, action, seed)
                    rows.append(dict(seed=seed, object=object_index,
                                     push=push_index, **row))
            finally:
                env.close()
    settings = dict(seeds=seeds, objects_per_seed=objects, pushes_per_object=pushes,
                    target_distance_m=0.5, reset_before_each_push=True,
                    object_ranges=asdict(sampler.ranges), action_ranges=asdict(CONTACT_RANGES))
    return dict(settings=settings, summary=summarize(rows), pushes=rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--objects", type=int, default=6)
    parser.add_argument("--pushes", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("artifacts/gantry_contact.json"))
    args = parser.parse_args(argv)
    try:
        result = run(args.seeds, args.objects, args.pushes)
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2, allow_nan=False))
    print(f"Saved measurements and settings to {args.output}")


if __name__ == "__main__":
    main()
