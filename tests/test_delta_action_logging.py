"""Physical action histories with true object settings and episode joins."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.agents.rl.training.delta_utils import parse_args
from src.agents.rl.training.train_delta import run
from src.simulation.environment import ActionRanges
from src.simulation.modelling.action_mapping import normalized_to_physical


@pytest.fixture
def logged_run(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    settings = "--smoke --backend known --f-max 15 --train-freq 2".split()
    args = parse_args(settings + ["--out-dir", str(tmp_path / "train")])
    args.num_objects, args.total_steps = 2, 12
    return args.out_dir, run(args)


def test_action_log_records_each_push_and_matches_object_settings(
    logged_run: tuple[Path, dict[str, Any]],
) -> None:
    directory, result = logged_run
    actions = [
        json.loads(line)
        for line in (directory / "actions.jsonl").read_text().splitlines()
    ]
    assert len(actions) == result["summary"]["real_interactions"] == 12
    assert [row["push_count"] for row in actions] == list(range(1, 13))
    assert len({row["object_index"] for row in actions}) >= 2
    bounds = ActionRanges(**result["metadata"]["action_ranges"])
    for row in actions:
        spec = result["objects"][row["object_index"]]["spec"]
        assert row["object_parameters"] == spec
        assert set(spec) == {"w", "L", "H", "m", "mu", "h_c"}
        physical = normalized_to_physical(
            np.array(row["normalized_action"]),
            (spec["w"], spec["L"], spec["H"]),
            bounds,
        )
        assert row["action"] == pytest.approx(
            dict(height_m=physical.h, force_n=physical.F, duration_s=physical.tau)
        )
        assert row["is_probe"] == (row["episode_push"] == 1)


def test_actions_join_episode_records(logged_run: tuple[Path, dict[str, Any]]) -> None:
    directory, _ = logged_run
    actions = [
        json.loads(line)
        for line in (directory / "actions.jsonl").read_text().splitlines()
    ]
    episodes = [
        json.loads(line)
        for line in (directory / "episodes.jsonl").read_text().splitlines()
    ]
    for episode in episodes:
        group = [
            r
            for r in actions
            if (r["object_index"], r["episode_index"])
            == (episode["object_index"], episode["episode_index"])
        ]
        assert [r["episode_push"] for r in group] == list(range(1, len(group) + 1))
        assert sum(r["reward"] for r in group) == pytest.approx(
            episode["episode_reward"]
        )
        assert {r["seed"] for r in group} == {episode["seed"]}


def test_frozen_evaluation_also_records_probe_and_policy_actions(
    tmp_path: Path,
) -> None:
    settings = "--smoke --backend known".split()
    args = parse_args(settings + ["--out-dir", str(tmp_path / "train")])
    run(args)
    checkpoint = args.out_dir / "delta_final.pt"
    evaluate = parse_args(
        "--smoke --mode evaluate --backend known".split()
        + ["--checkpoint", str(checkpoint), "--out-dir", str(tmp_path / "evaluate")]
    )
    result = run(evaluate)
    actions = [
        json.loads(line)
        for line in (evaluate.out_dir / "actions.jsonl").read_text().splitlines()
    ]
    assert len(actions) == result["summary"]["real_interactions"]
    assert result["summary"]["gradient_steps"] == 0
    assert sum(row["is_probe"] for row in actions) == result["summary"]["episode_count"]
    assert any(not row["is_probe"] for row in actions)
