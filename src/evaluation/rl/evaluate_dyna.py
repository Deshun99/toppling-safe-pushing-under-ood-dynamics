"""B2-B6 evaluation: imagination-assisted adaptation on the held-out set.

Per object, N_ep episodes of at most T_max decisions. Each real transition
goes into the SAC replay, the provider observes it (SysID refits on its
schedule), the imagination rolls the current policy out from real replay
states under the provider's physics, and the policy updates on the mixed
buffer. B5 additionally filters every action through worst-case ensemble
tilt predictions. All baselines start from the same pretrained checkpoint
and differ only in adaptation.

Run from the repo root:
    python -m src.evaluation.rl.evaluate_dyna \
        --model artifacts/sac_gantry_lane05/best_model/best_model.pt \
        --baseline b3 --robot
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.agents.rl.filter import PessimisticFilter
from src.agents.rl.imagination import Imagination
from src.agents.rl.models.sac import SAC
from src.agents.rl.physics_providers import make_provider
from src.agents.simulator.internal_sim import InternalSim
from src.evaluation.rl.constants import (
    DEFAULT_EVALUATION_SEED,
    DEFAULT_EVALUATION_SET_PATH,
    DEFAULT_FINE_TUNING_EPISODES,
    DEFAULT_FINE_TUNING_LR_SCALE,
    EpisodeResult,
    FINE_TUNING_REPLAY_CAPACITY,
    OBJECT_SEED_STRIDE,
    SMOKE_OBJECT_COUNT,
)
from src.evaluation.rl.evaluation_sets import EvaluationSet
from src.evaluation.rl.metrics import (
    EpisodeRecorder,
    EvaluationMetrics,
)
from src.simulation.environment import PushEnv
from src.simulation.environment.config import (
    load_action_ranges,
    load_checkpoint_metadata,
)
from src.simulation.modelling.object import PhysicsParameters

SMOKE_EPISODES = 2
SMOKE_ROLLOUTS = 2
SMOKE_REPLAY_BUDGET = 64
SMOKE_FILTER_CANDIDATES = 2


def _environment_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Environment kwargs overriding the checkpoint's recorded ones, if any."""
    overrides: dict[str, Any] = {}
    if args.robot is not None:
        overrides["robot"] = args.robot
    if args.target_distance is not None:
        overrides["target_distance"] = args.target_distance
    return overrides


def parse_parameter_errors(specs: list[str] | None) -> dict[str, float]:
    """Parse PARAM:FACTOR pairs, e.g. mass:1.25, into per-parameter multipliers."""
    errors: dict[str, float] = {}
    for spec in specs or []:
        name, _, value = spec.partition(":")
        if name not in ("mass", "mu", "h_c") or not value:
            raise ValueError(
                f"parameter error must be mass|mu|h_c:factor, got {spec!r}"
            )
        factor = float(value)
        if factor <= 0 or not np.isfinite(factor):
            raise ValueError("parameter error factor must be positive and finite")
        errors[name] = factor
    return errors


def perturb_theta(theta: PhysicsParameters, errors: dict[str, float]) -> PhysicsParameters:
    """Multiply the named parameters of theta by their error factors."""
    factors = [errors.get(name, 1.0) for name in ("mass", "mu", "h_c")]
    return tuple(value * factor for value, factor in zip(theta, factors))


def run_case(
    model_path: Path,
    case: Any,
    episodes: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], tuple[EpisodeResult, ...]]:
    """Run one baseline's adaptation protocol on one object."""
    seed = args.seed + OBJECT_SEED_STRIDE * case.seed_index
    model = SAC.load(
        model_path,
        lr_scale=DEFAULT_FINE_TUNING_LR_SCALE,
        buffer_size=FINE_TUNING_REPLAY_CAPACITY,
        seed=seed,
        inference_only=False,
    )
    metadata = load_checkpoint_metadata(model_path, model.training_metadata)
    env_kwargs = dict(metadata.get("environment", {}))
    env_kwargs.update(_environment_overrides(args))
    env = PushEnv(
        fixed_spec=case.spec,
        action_ranges=load_action_ranges(model_path, metadata),
        **env_kwargs,
    )
    sim = InternalSim()
    try:
        geometry = (case.spec.w, case.spec.L, case.spec.H)
        provider = make_provider(
            args.baseline,
            geometry,
            theta_true=(
                perturb_theta(case.spec.theta, args.parameter_errors)
                if args.baseline == "b6"
                else None
            ),
            refit_every=args.refit_every,
            history_window=args.history_window,
            replay_budget=args.replay_budget,
        )
        imagination = Imagination(
            sim,
            provider,
            env.reward,
            env.action_ranges,
            env.target_distance,
            env.epsilon,
            env.psi_succ_tol,
            env.v_succ_tol,
            horizon=args.horizon,
            seed=seed,
        )
        safety_filter = (
            PessimisticFilter(
                sim,
                provider,
                env.action_ranges,
                env.target_distance,
                near_topple_frac=env.near_topple_frac,
                candidates=args.filter_candidates,
            )
            if args.baseline == "b5"
            else None
        )

        seeds = [seed] + [None] * (episodes - 1)
        episodes = []
        for episode_seed in seeds:
            observation, _ = env.reset(seed=episode_seed)
            recorder = EpisodeRecorder(episode_seed, asdict(env.cuboid.spec))
            while True:
                action = model.predict(observation, deterministic=False)
                if safety_filter is not None:
                    action = safety_filter.select_action(model, observation)
                next_observation, reward, terminated, truncated, info = env.step(
                    action
                )
                assert model.replay is not None
                model.replay.add(
                    observation, action, reward, next_observation, terminated
                )
                model.num_timesteps += 1
                model.update()
                record = info.get("action_record")
                if record is not None:
                    imagination.observe(record)
                imagination.record_state(observation)
                imagination.generate(model, args.rollouts)
                observation = next_observation
                recorder.record_step(reward, info)
                if terminated or truncated:
                    episodes.append(
                        recorder.finish(observation, terminated, truncated, info)
                    )
                    break
        episode_tuple = tuple(episodes)
        return (
            {
                "index": case.seed_index,
                "theta_true": list(case.spec.theta),
                "summary": asdict(EvaluationMetrics(episode_tuple).summarize()),
                "episodes": [asdict(episode) for episode in episode_tuple],
            },
            episode_tuple,
        )
    finally:
        env.close()
        sim.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True,
                        help="pretrained SAC checkpoint shared by all baselines")
    parser.add_argument("--baseline", type=str, required=True,
                        choices=("b2", "b3", "b4", "b5", "b6"))
    parser.add_argument(
        "--robot", action=argparse.BooleanOptionalAction, default=None,
        help="override the checkpoint's recorded environment: --robot evaluates "
        "on the gantry, --no-robot on direct force, absent keeps the recorded one",
    )
    parser.add_argument("--target-distance", type=float, default=None)
    parser.add_argument("--episodes", type=int, default=DEFAULT_FINE_TUNING_EPISODES)
    parser.add_argument("--rollouts", type=int, default=8,
                        help="imagined rollouts generated after each real push")
    parser.add_argument("--horizon", type=int, default=2,
                        help="imagined rollout length in decisions")
    parser.add_argument("--filter-candidates", type=int, default=4,
                        help="candidate actions the B5 filter evaluates per decision")
    parser.add_argument("--refit-every", type=int, default=1,
                        help="SysID refit cadence in pushes")
    parser.add_argument("--history-window", type=int, default=20)
    parser.add_argument("--replay-budget", type=int, default=256)
    parser.add_argument("--seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--objects", type=int, default=8)
    parser.add_argument("--split", type=str, default="in_distribution",
                        help="evaluation-set split to run")
    parser.add_argument(
        "--parameter-error",
        action="append",
        metavar="PARAM:FACTOR",
        help="E2: multiply one oracle parameter (mass, mu, or h_c) by FACTOR; "
        "repeatable, b6 only",
    )
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/dyna_baselines/results.json"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.objects, args.episodes = SMOKE_OBJECT_COUNT, SMOKE_EPISODES
        args.rollouts, args.replay_budget = SMOKE_ROLLOUTS, SMOKE_REPLAY_BUDGET
        args.filter_candidates = SMOKE_FILTER_CANDIDATES
    if args.episodes < 1 or args.rollouts < 1:
        parser.error("--episodes and --rollouts must be positive")
    if args.parameter_error and args.baseline != "b6":
        parser.error("--parameter-error applies to the b6 oracle only")
    args.parameter_errors = parse_parameter_errors(args.parameter_error)

    evaluation_set = (
        EvaluationSet.load(DEFAULT_EVALUATION_SET_PATH, args.split)
        .limit(args.objects)
    )
    report: dict[str, Any] = {
        "settings": {
            "baseline": args.baseline,
            "model": str(args.model),
            "robot": args.robot,
            "target_distance": args.target_distance,
            "split": args.split,
            "parameter_errors": args.parameter_errors,
            "episodes": args.episodes,
            "rollouts": args.rollouts,
            "horizon": args.horizon,
            "filter_candidates": args.filter_candidates,
            "refit_every": args.refit_every,
            "history_window": args.history_window,
            "replay_budget": args.replay_budget,
            "seed": args.seed,
        },
        "per_object": [],
    }
    all_episodes: list[EpisodeResult] = []
    for case in evaluation_set.cases:
        print(f"--- {case.split} object {case.seed_index}: "
              f"(m, mu, h_c) = {case.spec.theta} ---")
        row, episodes = run_case(args.model, case, args.episodes, args)
        report["per_object"].append(row)
        all_episodes.extend(episodes)
        summary = row["summary"]
        print(f"  success {summary['success_rate']:.0%}, "
              f"topple {summary['topple_rate']:.0%}, "
              f"mean reward {summary['mean_reward']:.2f}")

    report["summary"] = asdict(
        EvaluationMetrics(all_episodes).summarize()
    )
    print(f"overall: {report['summary']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
