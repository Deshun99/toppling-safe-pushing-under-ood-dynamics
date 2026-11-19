"""Check evaluation set identity, selection settings, and paired evaluation reports."""

import json
import sys
from argparse import Namespace
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import pytest
import torch

from src.agents.rl.models.sac import SAC, SACConfig
from src.agents.rl.training.constants import DEFAULT_DISTRIBUTION_PATH
from src.evaluation.rl import EpisodeResult, EvaluationMetrics, EvaluationResult
from src.evaluation.rl.constants import (
    DEFAULT_EVALUATION_SET_PATH,
    OBJECT_SEED_STRIDE,
    EvaluationCase,
    EvaluationCaseResult,
)
from src.evaluation.rl.evaluate import SACEvaluation
from src.evaluation.rl.evaluation_sets import EvaluationSet
from src.simulation.environment import ACTION_RANGES
from src.simulation.modelling.sampling import ObjectRanges, load_object_sampler

SAMPLER_EXAMPLES = DEFAULT_EVALUATION_SET_PATH.parent
EVALUATION_SPLITS = EvaluationSet.load().get_groups()


@pytest.fixture
def model() -> SAC:
    torch.set_num_threads(1)
    model = SAC(SACConfig(7, 3, hidden_dim=16, buffer_size=32, batch_size=8))
    model.training_metadata = {
        "action_ranges": asdict(ACTION_RANGES),
        "training_distribution": asdict(load_object_sampler(DEFAULT_DISTRIBUTION_PATH)),
    }
    return model


@pytest.fixture
def run_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: SAC,
) -> Callable[[list[str]], dict[str, Any]]:
    def run(arguments: list[str]) -> dict[str, Any]:
        path = tmp_path / "model.pt"
        output = tmp_path / "results.json"
        model.save(path)
        argv = ["evaluate", "--model", str(path), "--out", str(output)]
        argv.extend(["--frozen-episodes", "1", "--fine-tuning-episodes", "1"])
        monkeypatch.setattr(sys, "argv", argv + arguments)
        args = SACEvaluation.parse_args()
        requested_settings = vars(args).copy()
        evaluation = SACEvaluation(args)
        report = evaluation.run()
        assert vars(args) == requested_settings
        assert json.loads(output.read_text()) == json.loads(json.dumps(report))
        return report

    return run


def test_evaluation_set_splits_preserve_existing_cases_and_stable_seeds() -> None:
    evaluation_set = EvaluationSet.load()
    ranges = ObjectRanges(**evaluation_set.reference_ranges)
    assert len(evaluation_set.select("in_distribution").cases) == 8
    assert [case.seed_index for case in evaluation_set.cases] == list(range(24))
    for split in EVALUATION_SPLITS:
        selected = EvaluationSet.load(selected=split)
        assert selected.cases == evaluation_set.select(split).cases
    ood = evaluation_set.select(EVALUATION_SPLITS[1:]).cases
    assert len(ood) == 16
    assert all(ranges.violated_parameters(case.spec) for case in ood)


@pytest.mark.parametrize("invalid", ["nonphysical", "nonfinite"])
def test_invalid_evaluation_set_objects_fail_before_physics(
    tmp_path: Path,
    invalid: str,
) -> None:
    config = json.loads(DEFAULT_EVALUATION_SET_PATH.read_text())
    spec = config["splits"]["mass_ood"][0]
    spec["m"] = -1 if invalid == "nonphysical" else float("nan")
    path = tmp_path / "evaluation_set.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        EvaluationSet.load(path, "all")


def test_all_splits_record_checkpoint_membership_and_paired_outcomes(
    model: SAC,
    run_cli: Callable[[list[str]], dict[str, Any]],
    tmp_path: Path,
) -> None:
    ranges = dict(EvaluationSet.load().reference_ranges, m=[0.1, 3.5])
    model.training_metadata["training_distribution"] = {"ranges": ranges}
    (tmp_path / "config.json").write_text(json.dumps({"f_max": 7}))
    report = run_cli(["--split", "all"])
    assert set(report["summary_by_split"]) == set(EVALUATION_SPLITS)
    assert len(report["objects"]) == 24
    assert report["checkpoint_training_metadata"] == model.training_metadata
    assert len(report["model_sha256"]) == 64
    for result in report["objects"]:
        assert result["frozen_policy"]["summary"]["episode_count"] == 1
        assert result["fine_tuning"]["summary"]["episode_count"] == 1
        if result["split"] == "mass_ood":
            assert result["outside_checkpoint_ranges"] == []


def test_arbitrary_fixed_groups_are_labels_not_distribution_claims(
    tmp_path: Path,
) -> None:
    original = json.loads(DEFAULT_EVALUATION_SET_PATH.read_text())["splits"]
    path = tmp_path / "custom.json"
    path.write_text(json.dumps({"splits": {"heavy_validation": original["mass_ood"]}}))
    evaluation_set = EvaluationSet.load(path, "heavy_validation")
    assert len(evaluation_set.cases) == 4
    assert evaluation_set.reference_ranges is None
    assert all(case.split == "heavy_validation" for case in evaluation_set.cases)
    with pytest.raises(ValueError, match="Unknown evaluation set groups"):
        EvaluationSet.load(path, "missing")


def test_sampling_is_reproducible_and_independent_of_group_selection() -> None:
    path = SAMPLER_EXAMPLES / "sampled_mass_extrapolation.json"
    evaluation_set = EvaluationSet.load(path, sampled=True, count=10, seed=42)
    assert (
        evaluation_set.cases
        == EvaluationSet.load(path, sampled=True, count=10, seed=42).cases
    )
    assert (
        evaluation_set.cases
        != EvaluationSet.load(path, sampled=True, count=10, seed=43).cases
    )
    heavy = EvaluationSet.load(path, "heavy", sampled=True, count=10, seed=42)
    assert heavy.cases == evaluation_set.select("heavy").cases
    assert len(evaluation_set.cases) == 20
    assert [case.seed_index for case in heavy.cases] == list(range(10, 20))
    for case in evaluation_set.cases:
        low, high = (0.2, 0.4) if case.split == "light" else (2, 3)
        assert low <= case.spec.m <= high


def test_reserved_regions_match_training_exclusions() -> None:
    training = load_object_sampler(DEFAULT_DISTRIBUTION_PATH)
    evaluation_set = EvaluationSet.load(
        SAMPLER_EXAMPLES / "sampled_reserved_regions.json", sampled=True, count=10
    )
    assert len(evaluation_set.cases) == 80
    for case in evaluation_set.cases:
        region_index = int(case.split.removeprefix("reserved_region_")) - 1
        assert training.exclusions[region_index].contains(case.spec)
        membership = case.classify({"training_distribution": asdict(training)})
        assert membership["membership"] == "excluded_region"


def test_membership_checks_geometry_exclusions_and_unknown_metadata() -> None:
    training = load_object_sampler(DEFAULT_DISTRIBUTION_PATH)
    metadata = {"training_distribution": asdict(training)}
    case = EvaluationSet.load(selected="in_distribution").cases[0]
    assert case.classify(metadata)["membership"] == "excluded_region"
    allowed = replace(case, spec=replace(case.spec, m=0.8))
    assert allowed.classify(metadata)["membership"] == "training_support"
    outside = replace(case, spec=replace(case.spec, w=0.2))
    assert outside.classify(metadata) == {
        "membership": "outside_ranges",
        "outside_checkpoint_ranges": ["w"],
    }
    assert case.classify({})["membership"] == "unknown"


@pytest.mark.parametrize("count,seed", [(0, 42), (2, -1)])
def test_invalid_sample_budgets_are_rejected(count: int, seed: int) -> None:
    with pytest.raises(ValueError):
        EvaluationSet.load(
            SAMPLER_EXAMPLES / "sampled_mass_extrapolation.json",
            sampled=True,
            count=count,
            seed=seed,
        )


def test_sampler_cli_runs_paired_fixed_objects_without_sidecar(
    run_cli: Callable[[list[str]], dict[str, Any]],
) -> None:
    config_path = SAMPLER_EXAMPLES / "sampled_mass_extrapolation.json"
    report = run_cli(["--sampler-config", str(config_path), "--num-objects", "1"])
    assert report["schema_version"] == 2
    assert report["object_source"] == "sampled"
    assert report["object_config"] == json.loads(config_path.read_text())
    assert len(report["objects"]) == 2
    expected = EvaluationSet(report["object_config"], sampled=True, count=1, seed=42)
    for result, case in zip(report["objects"], expected.cases, strict=True):
        assert result["spec"] == asdict(case.spec)
        assert result["membership"] == "outside_ranges"
        for condition in ("frozen_policy", "fine_tuning"):
            assert result[condition]["episodes"][0]["object_spec"] == result["spec"]
            assert (
                result[condition]["metadata"]["initial_seed"]
                == 42 + OBJECT_SEED_STRIDE * case.seed_index
            )


def test_fixed_evaluation_set_uses_one_config_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads = []
    original = Path.read_text

    def read_once(path: Path, *args: Any, **kwargs: Any) -> str:
        reads.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_once)
    evaluation_set = EvaluationSet.load()
    assert reads == [DEFAULT_EVALUATION_SET_PATH]
    assert len(evaluation_set.cases) == 24
    assert (
        asdict(evaluation_set.cases[0].spec)
        == evaluation_set.config["splits"]["in_distribution"][0]
    )


def test_checkpoint_report_reads_state_without_constructing_sac(
    tmp_path: Path, model: SAC
) -> None:
    model.num_timesteps = 123
    path = tmp_path / "model.pt"
    model.save(path)
    with patch.object(
        SAC, "load", side_effect=AssertionError("Should not construct SAC")
    ):
        with patch.object(torch, "load", wraps=torch.load) as read:
            metadata = SACEvaluation._load_checkpoint_metadata(path)
    assert read.call_count == 1
    assert metadata["checkpoint_training_steps"] == 123
    assert metadata["checkpoint_training_metadata"] == model.training_metadata


@pytest.mark.parametrize("count", [1, 20])
def test_smoke_report_separates_requested_and_evaluated_objects(
    run_cli: Callable[[list[str]], dict[str, Any]],
    count: int,
) -> None:
    report = run_cli(
        [
            "--sampler-config",
            str(SAMPLER_EXAMPLES / "sampled_mass_extrapolation.json"),
            "--num-objects",
            str(count),
            "--smoke",
        ]
    )
    actual = min(count, 2)
    assert report["smoke"] is True
    assert report["num_objects_per_group"] == actual
    assert report["requested_num_objects_per_group"] == count
    assert report["evaluated_objects_by_group"] == {"light": actual, "heavy": actual}
    assert report["fine_tuning_episodes"] == 3
    assert report["frozen_episodes"] == 1
    heavy = [result for result in report["objects"] if result["split"] == "heavy"]
    assert [result["seed_index"] for result in heavy] == list(
        range(count, count + actual)
    )


def test_limiting_preserves_objects_and_does_not_modify_source() -> None:
    evaluation_set = EvaluationSet.load(
        SAMPLER_EXAMPLES / "sampled_mass_extrapolation.json", sampled=True, count=20
    )
    limited = evaluation_set.limit(2)
    assert len(evaluation_set.cases) == 40
    assert limited.cases == tuple(
        case for case in evaluation_set.cases if case.seed_index in (0, 1, 20, 21)
    )
    assert limited.config is evaluation_set.config
    with pytest.raises(ValueError, match="positive"):
        evaluation_set.limit(0)


def make_evaluation_set_result(
    case: EvaluationCase, successes: int, topples: int
) -> EvaluationCaseResult:
    outcomes: list[Literal["success", "toppled"]] = ["success"] * successes
    outcomes.extend(["toppled"] * topples)
    episodes = tuple(
        EpisodeResult(42, None, outcome, 0.0, 1, 0.1, 0.1, None) for outcome in outcomes
    )
    metrics = EvaluationMetrics(episodes)
    result = EvaluationResult(episodes, metrics.summarize(), {})
    progress = metrics.summarize_progress()
    return EvaluationCaseResult(case, result, result, progress, progress)


@pytest.fixture
def synthetic_evaluation(tmp_path: Path) -> SACEvaluation:
    specs = EvaluationSet.load().cases[:2]
    config = {"splits": {"a": [asdict(specs[0].spec)], "b": [asdict(specs[1].spec)]}}
    path = tmp_path / "evaluation_set.json"
    path.write_text(json.dumps(config))
    args = Namespace(
        sampler_config=None,
        evaluation_config=path,
        split=None,
        num_objects=20,
        seed=42,
        smoke=False,
        frozen_episodes=1,
        fine_tuning_episodes=3,
        lr_scale=0.5,
    )
    evaluation = SACEvaluation(args)
    a, b = evaluation.evaluation_set.cases
    evaluation.results = (
        make_evaluation_set_result(b, 0, 1),
        make_evaluation_set_result(a, 2, 1),
    )
    evaluation.checkpoint_metadata = {"checkpoint_training_metadata": {}}
    return evaluation


def test_report_groups_attached_cases_and_preserves_episode_weighting(
    synthetic_evaluation: SACEvaluation,
) -> None:
    b = synthetic_evaluation.results[0].case
    report = synthetic_evaluation._build_report()
    assert [row["split"] for row in report["objects"]] == ["b", "a"]
    assert report["objects"][0]["spec"] == asdict(b.spec)
    assert report["summary_by_split"]["a"][
        "frozen_policy_success_rate"
    ] == pytest.approx(2 / 3)
    assert report["summary_by_split"]["b"]["frozen_policy_success_rate"] == 0
    assert report["means"]["frozen_policy_success_rate"] == 0.5
    assert report["means"]["frozen_policy_topples"] == 1
    assert report["means"]["frozen_policy_object_success_fraction"] == 0.5
