"""Compare frozen critic rankings with actual one-push outcomes from cloned states."""

import argparse
from copy import deepcopy
from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p
import torch

from src.agents.rl.models.delta_actor_critic import DeltaActorCritic
from src.agents.rl.models.delta_controller import DeltaController
from src.agents.rl.training.delta_utils import compute_delta_bounds
from src.agents.rl.training.train_delta import _open_object, collect_actions
from src.evaluation.rl.evaluation_sets import EvaluationSet
from src.evaluation.rl.constants import EvaluationCase
from src.simulation.environment import PushEnv
from src.simulation.execution.settling import wait_for_rest
from src.simulation.modelling.action_mapping import normalized_to_physical


def measure_candidate_toppling(env: PushEnv, actions: np.ndarray) -> list[bool]:
    """Execute each action from the same full Bullet state, then restore it."""
    state = p.saveState(physicsClientId=env.client)
    outcomes = []
    try:
        for action in actions:
            p.restoreState(stateId=state, physicsClientId=env.client)
            spec = env.cuboid.spec
            physical = normalized_to_physical(
                action, (spec.w, spec.L, spec.H), env.action_ranges
            )
            env.executor.execute(
                env.cuboid, h=physical.h, F=physical.F, tau=physical.tau
            )
            wait_for_rest(env.cuboid, env.client, env.rest_config)
            outcomes.append(abs(env.cuboid.state()[1]) > spec.crit_tilt)
    finally:
        p.restoreState(stateId=state, physicsClientId=env.client)
        p.removeState(stateUniqueId=state, physicsClientId=env.client)
    return outcomes


def summarize_ranking(scores: np.ndarray, toppled: list[bool]) -> dict[str, Any]:
    """Pair accuracy is defined only when both outcome classes are present."""
    outcomes = np.asarray(toppled, dtype=bool)
    safe, falling = scores[~outcomes], scores[outcomes]
    accuracy = None
    if len(safe) and len(falling):
        margins = safe[:, None] - falling[None, :]
        accuracy = float(((margins > 0) + 0.5 * (margins == 0)).mean())
    return dict(
        candidate_count=len(scores),
        topple_count=int(outcomes.sum()),
        best_candidate_toppled=bool(outcomes[np.argmax(scores)]),
        safe_over_topple_pair_accuracy=accuracy,
        scores=scores.tolist(),
        toppled=toppled,
    )


def inspect_context(
    model: DeltaActorCritic,
    controller: DeltaController,
    env: PushEnv,
    observation: np.ndarray,
    candidates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    context = controller.build_context(observation)
    reference = torch.as_tensor(np.array(context.reference_action, copy=True))
    lower, upper = compute_delta_bounds(reference, model.actor.delta_scale.cpu())
    deltas = rng.uniform(lower.numpy(), upper.numpy(), size=(candidates, 3))
    forecasts = [controller._forecast(context, delta) for delta in deltas]
    scores = model.score(
        observation, context.real_trajectory, deltas, forecasts, context.weights
    )
    actions = context.reference_action + deltas
    result = summarize_ranking(scores, measure_candidate_toppling(env, actions))
    return dict(observation=observation.tolist(), actions=actions.tolist(), **result)


def diagnose(args: argparse.Namespace) -> list[dict[str, Any]]:
    model = DeltaActorCritic.load(args.checkpoint, device="cpu")
    metadata = deepcopy(model.training_metadata)
    if metadata is None:
        raise ValueError("Ranking diagnostics require checkpoint training settings")
    metadata["arguments"]["gui"] = False
    cases = EvaluationSet.load(
        Path(metadata["distribution_config_path"]),
        sampled=True,
        count=args.num_objects,
        seed=args.seed,
    ).cases
    records = []
    with args.out_file.open("w") as stream:
        for index, case in enumerate(cases):
            for record in diagnose_object(model, case, index, metadata, args):
                records.append(record)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                summary = {
                    key: value
                    for key, value in record.items()
                    if key not in ("scores", "toppled", "actions", "observation")
                }
                print(summary, flush=True)
    return records


def diagnose_object(
    model: DeltaActorCritic,
    case: EvaluationCase,
    index: int,
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> Iterator[dict[str, Any]]:
    seed = args.seed + index
    rng = np.random.default_rng(seed)
    with _open_object(model, case, seed, metadata) as (controller, env):
        for trace, _ in collect_actions(env, controller, model, seed):
            if trace["terminated"] or trace["truncated"]:
                continue
            result = inspect_context(
                model, controller, env, trace["next_observation"], args.candidates, rng
            )
            yield dict(object_index=index, **result)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--num-objects", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-file", type=Path)
    args = parser.parse_args(argv)
    if args.candidates < 1 or args.num_objects < 1 or args.seed < 0:
        parser.error("candidate/object counts must be positive and seed nonnegative")
    args.out_file = args.out_file or args.checkpoint.with_suffix(".ranking.jsonl")
    torch.set_num_threads(1)
    diagnose(args)


if __name__ == "__main__":
    main()
