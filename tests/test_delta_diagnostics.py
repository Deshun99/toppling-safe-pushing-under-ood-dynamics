"""Live training records, usable checkpoints, and controlled ranking measurements."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.agents.rl.models.delta_actor_critic import DeltaActorCritic
from src.agents.rl.training.delta_utils import parse_args
from src.agents.rl.training.train_delta import run
from src.evaluation.rl.diagnose_delta_ranking import (
    diagnose,
    measure_candidate_toppling,
    summarize_ranking,
)
from src.simulation.environment import PushEnv


def test_candidate_outcomes_restore_target_state() -> None:
    env = PushEnv(gui=False)
    try:
        env.reset(seed=42)
        state, steps = env.cuboid.state(), env.steps
        actions = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
        first = measure_candidate_toppling(env, actions)
        second = measure_candidate_toppling(env, actions)
        assert first == second
        assert env.cuboid.state() == state
        assert env.steps == steps
    finally:
        env.close()


def test_pair_accuracy_requires_both_outcome_classes() -> None:
    outcomes = [False, True]
    assert (
        summarize_ranking(np.array([2.0, 1.0]), outcomes)[
            "safe_over_topple_pair_accuracy"
        ]
        == 1
    )
    assert (
        summarize_ranking(np.array([1.0, 2.0]), outcomes)[
            "safe_over_topple_pair_accuracy"
        ]
        == 0
    )
    assert (
        summarize_ranking(np.ones(2), outcomes)["safe_over_topple_pair_accuracy"] == 0.5
    )
    assert (
        summarize_ranking(np.ones(2), [False, False])["safe_over_topple_pair_accuracy"]
        is None
    )


def test_training_diagnostics_and_checkpoint_ranking(tmp_path: Path) -> None:
    settings = "--smoke --backend known --log-every 1 --checkpoint-every 2".split()
    args = parse_args(settings + ["--out-dir", str(tmp_path / "run")])
    result = run(args)
    updates = [
        json.loads(line)
        for line in (args.out_dir / "updates.jsonl").read_text().splitlines()
    ]
    assert len(updates) == result["summary"]["gradient_steps"]
    assert {
        "critic_q_mean",
        "target_mean",
        "normalized_entropy",
        "actor_saturation",
        "actor_std_force_median",
    } <= updates[0].keys()
    checkpoint = args.out_dir / "delta_latest.pt"
    model = DeltaActorCritic.load(checkpoint)
    assert model.num_timesteps >= 2 and model.training_metadata is not None
    report = tmp_path / "ranking.jsonl"
    records = diagnose(
        SimpleNamespace(
            checkpoint=checkpoint, num_objects=1, seed=42, candidates=4, out_file=report
        )
    )
    assert records and len(records[0]["scores"]) == 4
    assert len(report.read_text().splitlines()) == len(records)
    assert "Checkpoint saved" in (args.out_dir / "training.log").read_text()


@pytest.mark.parametrize(
    "option", ["--log-every", "--checkpoint-every", "--train-freq"]
)
def test_diagnostic_intervals_must_be_positive(option: str) -> None:
    with pytest.raises(SystemExit):
        parse_args([option, "0"])


@pytest.mark.parametrize("frequency,gradient_steps", [(1, 1), (2, 1), (2, 3), (8, 1)])
def test_update_frequency_keeps_replay_and_uses_post_warmup_pushes(
    tmp_path: Path,
    frequency: int,
    gradient_steps: int,
) -> None:
    settings = "--smoke --backend known --checkpoint-every 2".split()
    settings += f"--train-freq {frequency} --gradient-steps {gradient_steps}".split()
    args = parse_args(settings + ["--out-dir", str(tmp_path / "run")])
    result = run(args)
    path = args.out_dir / "updates.jsonl"
    updates = (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )
    eligible = [
        push for push in range(2, 7) if (push - args.learning_starts) % frequency == 0
    ]
    assert [row["push_count"] for row in updates] == [
        p for p in eligible for _ in range(gradient_steps)
    ]
    assert result["summary"]["real_interactions"] == 6
    assert result["summary"]["transitions"] == 4
    assert result["summary"]["gradient_steps"] == len(eligible) * gradient_steps
    assert result["metadata"]["arguments"]["train_freq"] == frequency
    assert all(row["update_seconds"] > 0 for row in updates)
    assert (args.out_dir / "delta_latest.pt").exists()


def test_update_frequency_restarts_for_each_adaptation_object(tmp_path: Path) -> None:
    source_args = parse_args(
        "--smoke --backend known".split() + ["--out-dir", str(tmp_path / "source")]
    )
    run(source_args)
    checkpoint = source_args.out_dir / "delta_final.pt"
    settings = "--smoke --mode adapt --backend known --train-freq 2".split()
    args = parse_args(
        settings
        + ["--checkpoint", str(checkpoint), "--out-dir", str(tmp_path / "adapt")]
    )
    args.num_objects, args.episodes_per_object = 2, 1
    result = run(args)
    updates = [
        json.loads(line)
        for line in (args.out_dir / "updates.jsonl").read_text().splitlines()
    ]
    assert [row["push_count"] for row in updates] == [3, 6]
    assert result["summary"]["gradient_steps"] == 2
