"""B0 evaluation: conservative controller plus online SysID on the target.

Runs the fixed in-distribution set through the B0 protocol (outline §10):
per object, N_ep episodes of at most T_max decisions; after every push the
CMA-ES SysID refits the point estimate from the full history under a replay
budget, warm-started from the previous best estimate. Reports success,
toppling, execution failures, and the final parameter error per object,
on the direct-force or gantry target.

Run from the repo root:
    python -m src.evaluation.baselines.evaluate_conservative --robot
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p

from src.agents.baselines.conservative_controller import ConservativeController
from src.agents.system_identification.sysid import SysID, Z_PRIOR_MEAN, convert_z_to_cuboid_spec
from src.evaluation.rl.constants import (
    DEFAULT_EVALUATION_SEED,
    DEFAULT_EVALUATION_SET_PATH,
    OBJECT_SEED_STRIDE,
    SMOKE_OBJECT_COUNT,
)
from src.evaluation.rl.evaluation_sets import EvaluationSet
from src.simulation.environment.env import ActionRanges, PushEnv
from src.simulation.execution.robot import ExecutionReport
from src.simulation.modelling.action_mapping import physical_to_normalized
from src.simulation.modelling.object import CuboidSpec, Geometry, PhysicsParameters
from src.simulation.modelling.records import ActionRecord
from src.simulation.tools.demos.arm_demo import mark_goal

DEFAULT_EPISODES = 20  # adaptation episodes per object (outline N_ep)
DEFAULT_REPLAY_BUDGET = 256  # SysID replays per refit, warm-started
DEFAULT_HISTORY_WINDOW = 20  # recent records fed to SysID (outline risk table)
DEFAULT_TARGET_DISTANCE = 0.5
DEFAULT_F_MAX = 15.0
SMOKE_EPISODES = 2
SMOKE_REPLAY_BUDGET = 64


@dataclass
class EpisodeRow:
    episode: int
    outcome: str
    reward: float
    push_count: int
    final_position_error_m: float


def prior_theta(geometry: Geometry) -> PhysicsParameters:
    """The SysID training prior (§6), used before any target data exists."""
    return convert_z_to_cuboid_spec(Z_PRIOR_MEAN, geometry).theta


def _outcome(terminated: bool, truncated: bool, info: dict[str, Any]) -> str:
    if info.get("toppled"):
        return "toppled"
    if info.get("success"):
        return "success"
    if truncated:
        return "timeout"
    if info.get("execution_failed"):
        return "execution_failed"
    raise ValueError("Unrecognized episode termination")


def _format_realized(report: ExecutionReport | None) -> str:
    """Console rendering of realized execution; None fields mean no contact."""
    if report is None:
        return "realized n/a"
    h_real = f"{report.h_real:.3f}" if report.h_real is not None else "n/a"
    F_real = f"{report.F_real:.1f}" if report.F_real is not None else "n/a"
    return f"realized (h={h_real}, F={F_real}, tau={report.tau_real:.2f})"


def _adapt_episode(
    env: PushEnv,
    geometry: Geometry,
    controller: ConservativeController,
    ranges: ActionRanges,
    sysid: SysID,
    history: list[ActionRecord],
    point: PhysicsParameters,
    previous_best: PhysicsParameters | None,
    seed: int | None,
    *,
    history_window: int,
    replay_budget: int,
    console: bool = False,
) -> tuple[EpisodeRow, PhysicsParameters, PhysicsParameters | None]:
    """Run one adaptation episode, appending to the shared SysID history."""
    observation, _ = env.reset(seed=seed)
    reward_total = 0.0
    outcome = "timeout"
    push_count = 0
    for decision in range(env.T_max):
        action = controller.choose_action(
            geometry, point, float(observation[0]),
            env.T_max - decision, ranges,
        )
        observation, reward, terminated, truncated, info = env.step(
            physical_to_normalized(action, geometry, ranges)
        )
        reward_total += float(reward)
        push_count += 1
        record = info.get("action_record")
        if record is not None:
            history.append(record)
            result = sysid.update(
                history[-history_window:],
                replay_budget=replay_budget,
                previous_best_parameters=previous_best,
            )
            point = result.point.theta
            previous_best = point
        if console:
            print(
                f"  push {push_count}/{env.T_max}: cmd (h={action.h:.3f}, "
                f"F={action.F:.1f}, tau={action.tau:.2f}) | "
                f"{_format_realized(env.last_execution)} | "
                f"d={float(observation[0]):+.3f} | est (m={point[0]:.2f}, "
                f"mu={point[1]:.2f}, h_c={point[2]:.3f})"
            )
        if terminated or truncated:
            outcome = _outcome(terminated, truncated, info)
            break
    row = EpisodeRow(
        episode=0,
        outcome=outcome,
        reward=round(reward_total, 4),
        push_count=push_count,
        final_position_error_m=round(abs(float(observation[0])), 4),
    )
    return row, point, previous_best


def run_object(
    spec: CuboidSpec,
    episodes: int,
    seed: int,
    args: argparse.Namespace,
    controller: ConservativeController,
    ranges: ActionRanges,
) -> tuple[list[EpisodeRow], PhysicsParameters]:
    """Run the B0 protocol on one object; SysID history persists across episodes."""
    geometry = (spec.w, spec.L, spec.H)
    env = PushEnv(
        robot=args.robot,
        fixed_spec=spec,
        target_distance=args.target_distance,
        action_ranges=ranges,
    )
    rows: list[EpisodeRow] = []
    try:
        sysid = SysID(geometry=geometry, seed=seed)
        history: list[ActionRecord] = []
        point = prior_theta(geometry)
        previous_best: PhysicsParameters | None = None
        for episode in range(episodes):
            row, point, previous_best = _adapt_episode(
                env, geometry, controller, ranges, sysid, history,
                point, previous_best, seed if episode == 0 else None,
                history_window=args.history_window,
                replay_budget=args.replay_budget,
            )
            row.episode = episode
            rows.append(row)
        return rows, point
    finally:
        env.close()


def demo_b0(
    spec: CuboidSpec, args: argparse.Namespace,
    controller: ConservativeController, ranges: ActionRanges,
) -> None:
    """M6 demonstration: B0 warm-up headless, then one live GUI episode."""
    geometry = (spec.w, spec.L, spec.H)
    print(f"object: (m, mu, h_c) = {spec.theta} | warm-up episodes: "
          f"{args.demo_warmup}")
    sysid = SysID(geometry=geometry, seed=args.seed)
    history: list[ActionRecord] = []
    point = prior_theta(geometry)
    previous_best: PhysicsParameters | None = None
    headless = PushEnv(
        robot=not args.force_executor,
        fixed_spec=spec, target_distance=args.target_distance,
        action_ranges=ranges,
    )
    try:
        for episode in range(args.demo_warmup):
            row, point, previous_best = _adapt_episode(
                headless, geometry, controller, ranges, sysid, history,
                point, previous_best, args.seed if episode == 0 else None,
                history_window=args.history_window,
                replay_budget=args.replay_budget,
            )
            print(f"  warm-up episode {episode}: {row.outcome} in "
                  f"{row.push_count} pushes | est (m={point[0]:.2f}, "
                  f"mu={point[1]:.2f}, h_c={point[2]:.3f})")
    finally:
        headless.close()

    gui = PushEnv(
        robot=not args.force_executor, gui=True, realtime_scale=1.0,
        fixed_spec=spec,
        target_distance=args.target_distance, action_ranges=ranges,
    )
    try:
        p.resetDebugVisualizerCamera(
            1.5, 45, -20, (0.1, 0, 0.45), physicsClientId=gui.client)
        mark_goal(gui.client, gui.target_distance, gui.epsilon)
        print(f"\nlive episode (target {gui.target_distance} m +/- "
              f"{gui.epsilon} m):")
        row, point, _ = _adapt_episode(
            gui, geometry, controller, ranges, sysid, history,
            point, previous_best, None,
            history_window=args.history_window,
            replay_budget=args.replay_budget, console=True,
        )
        print(f"\nresult: {row.outcome} in {row.push_count} pushes | "
              f"true theta {spec.theta} | final est ({point[0]:.2f}, "
              f"{point[1]:.2f}, {point[2]:.3f})")
        if args.linger > 0:
            time.sleep(args.linger)
    finally:
        gui.close()


def demo_b1(
    spec: CuboidSpec, args: argparse.Namespace,
) -> None:
    """M6 demonstration: the frozen B1 policy, one live GUI episode."""
    from src.agents.rl.models.sac import SAC
    from src.simulation.environment.config import (
        load_action_ranges,
        load_checkpoint_metadata,
    )

    model = SAC.load(args.model, inference_only=True)
    metadata = load_checkpoint_metadata(args.model, model.training_metadata)
    env = PushEnv(
        robot=not args.force_executor, gui=True, realtime_scale=1.0,
        fixed_spec=spec,
        target_distance=args.target_distance,
        action_ranges=load_action_ranges(args.model, metadata),
    )
    try:
        p.resetDebugVisualizerCamera(
            1.5, 45, -20, (0.1, 0, 0.45), physicsClientId=env.client)
        mark_goal(env.client, env.target_distance, env.epsilon)
        print(f"\nB1 frozen live episode (object {spec.theta}):")
        observation, _ = env.reset(seed=args.seed)
        for decision in range(env.T_max):
            action = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            print(f"  push {decision + 1}/{env.T_max}: "
                  f"reward={reward:+.2f} d={float(observation[0]):+.3f} "
                  f"tilt={abs(float(observation[1])):.3f}")
            if terminated or truncated:
                break
        print(f"result: {_outcome(terminated, truncated, info)}")
        if args.linger > 0:
            time.sleep(args.linger)
    finally:
        env.close()


def summarize(rows: list[EpisodeRow]) -> dict[str, float]:
    count = len(rows)
    return {
        "episodes": count,
        "success_rate": sum(r.outcome == "success" for r in rows) / count,
        "topple_rate": sum(r.outcome == "toppled" for r in rows) / count,
        "timeout_rate": sum(r.outcome == "timeout" for r in rows) / count,
        "execution_failure_rate": (
            sum(r.outcome == "execution_failed" for r in rows) / count
        ),
        "mean_reward": float(np.mean([r.reward for r in rows])),
        "mean_push_count": float(np.mean([r.push_count for r in rows])),
        "mean_final_position_error_m": float(
            np.mean([r.final_position_error_m for r in rows])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--robot", action="store_true", help="execute with the gantry instead of direct force"
    )
    parser.add_argument("--target-distance", type=float, default=DEFAULT_TARGET_DISTANCE)
    parser.add_argument("--f-max", type=float, default=DEFAULT_F_MAX)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--replay-budget", type=int, default=DEFAULT_REPLAY_BUDGET)
    parser.add_argument("--history-window", type=int, default=DEFAULT_HISTORY_WINDOW,
                        help="most recent records fed to each SysID refit")
    parser.add_argument("--seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--objects", type=int, default=8,
                        help="number of in-distribution objects (default: 8)")
    parser.add_argument("--split", type=str, default="in_distribution",
                        help="evaluation-set split to run")
    parser.add_argument(
        "--demo", action="store_true",
        help="M6 demonstration: warm up headless, then play one live GUI episode",
    )
    parser.add_argument("--method", choices=("b0", "b1"), default="b0",
                        help="demonstrated method (default: b0)")
    parser.add_argument(
        "--force-executor", action="store_true",
        help="run the demo on direct force instead of the gantry "
        "(the trained policy topples there)",
    )
    parser.add_argument("--object", type=int, default=0,
                        help="in-distribution object index for the demo (default: 0)")
    parser.add_argument("--demo-warmup", type=int, default=5,
                        help="headless B0 warm-up episodes before the live one")
    parser.add_argument("--model", type=Path,
                        default=Path("artifacts/sac_gantry_lane05/best_model/best_model.pt"),
                        help="B1 checkpoint for the --method b1 demo")
    parser.add_argument("--linger", type=float, default=3.0,
                        help="seconds to keep the final state on screen")
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/conservative_b0/results.json"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.objects, args.episodes, args.replay_budget = (
            SMOKE_OBJECT_COUNT, SMOKE_EPISODES, SMOKE_REPLAY_BUDGET,
        )

    evaluation_set = (
        EvaluationSet.load(DEFAULT_EVALUATION_SET_PATH, args.split)
        .limit(args.objects)
    )
    ranges = ActionRanges(force=(0.0, args.f_max))
    controller = ConservativeController()
    if args.demo:
        case = evaluation_set.cases[args.object]
        if args.method == "b0":
            demo_b0(case.spec, args, controller, ranges)
        else:
            demo_b1(case.spec, args)
        return
    report: dict[str, Any] = {
        "settings": dict(robot=args.robot,
                         target_distance=args.target_distance, f_max=args.f_max,
                         episodes=args.episodes, replay_budget=args.replay_budget,
                         history_window=args.history_window, split=args.split,
                         seed=args.seed, controller=asdict(controller)),
        "per_object": [],
    }
    all_rows: list[EpisodeRow] = []
    for case in evaluation_set.cases:
        print(f"--- {case.split} object {case.seed_index}: "
              f"(m, mu, h_c) = {case.spec.theta} ---")
        seed = args.seed + OBJECT_SEED_STRIDE * case.seed_index
        rows, theta_final = run_object(
            case.spec, args.episodes, seed, args, controller, ranges)
        all_rows.extend(rows)
        report["per_object"].append(
            {
                "index": case.seed_index,
                "theta_true": list(case.spec.theta),
                "theta_final": list(theta_final),
                "summary": summarize(rows),
                "episodes": [asdict(row) for row in rows],
            }
        )
        print(f"  {summarize(rows)}")

    means = summarize(all_rows)
    report["summary"] = means
    print(f"overall: {means}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
