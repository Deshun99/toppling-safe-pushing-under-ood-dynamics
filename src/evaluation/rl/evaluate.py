"""
Evaluate SAC on held-out objects with frozen weights and while fine-tuning.

B1 (baseline 1) is pretrained SAC fine-tuned using only transitions collected
from the current test object in the target environment. It uses no system
identification, internal-simulator rollouts, or safety shield. This provides
the model-free comparison for methods that add simulator-generated experience
or a safety filter. Here, the target environment is also a PyBullet simulation.

Object selection:
- For fixed evaluation sets (default or --evaluation-config), omitting --split
  selects only the in_distribution group if present; otherwise, all groups run.
  The default config's in_distribution group contains the eight original objects.
- Use --split all to evaluate every configured group, or --split GROUP [GROUP ...]
  to select specific groups explicitly.
- Use --sampler-config to draw a seeded set of test objects once. All sampled
  groups run by default; --split can restrict which groups are sampled.
Objects remain fixed across frozen evaluation and adaptation.

Frozen-policy evaluation performs no learning; fine-tuning updates SAC after
every push. Fine-tuning scores measure performance during learning.
Each object loads identical pretrained weights and frozen observation moments,
with empty replay. Fine-tuning uses stochastic policy actions from the first
step and learns after every transition, including the last episode's last step.
Updates persist across episodes for one object; the next object starts again
from the original pretrained checkpoint. There is no separate frozen-policy
evaluation of the final fine-tuned model.

Run: python -m src.evaluation.rl.evaluate --model artifacts/sac_pretrain/sac_final.pt

Command-line arguments (paths are relative to the working directory unless
absolute; the built-in evaluation config is resolved relative to this module):
    --model PATH
        Required native SAC checkpoint, for example
        artifacts/sac_pretrain/sac_final.pt. Each object starts from this checkpoint.
    --fine-tuning-episodes N
        Positive number of learning episodes per object (default: 20).
    --frozen-episodes N
        Positive number of frozen-policy episodes per object (default: 1).
        Identical resets with deterministic actions do not need repeated rollouts.
    --lr-scale SCALE
        Positive, finite multiplier of the checkpoint's pretrained learning rate
        for fine-tuning (default: 0.5).
    --evaluation-config PATH, --benchmark-config PATH
        Aliases for a JSON file containing fixed object groups under "splits".
        Default: configs/rl/evaluation_sets/fixed_cases.json in this repository.
        Mutually exclusive with --sampler-config.
    --sampler-config PATH
        JSON file containing one sampler or named samplers under "groups".
        Draws objects once for use in both evaluation conditions. Mutually
        exclusive with --evaluation-config / --benchmark-config; absent by default.
    --split GROUP [GROUP ...]
        Select configured group names, or pass --split all for every group.
        When omitted, fixed sets use in_distribution if present, otherwise all
        groups; sampled sets use all groups. Names refer to config labels, not
        inferred membership in the checkpoint's training distribution.
    --num-objects N
        Positive number of objects sampled per selected group (default: 20).
        Has no effect on fixed evaluation sets.
    --seed SEED
        Nonnegative base seed (default: 42), used for object sampling and
        evaluation. Object i uses SEED + 1000 * i for both evaluation conditions;
        i is its global config index, preserved when selecting groups.
    --out PATH
        Results JSON destination. By default, writes under
        artifacts/held_out_evaluation/: results.json for fixed in_distribution,
        otherwise results[_sampled][_<selected_groups>].json. Multiple groups
        share one report; selected group names form the filename suffix.
        Parent directories are created and an existing destination is overwritten.
    --smoke
        Quick run (disabled by default): keep the first two objects per selected
        group and override episode counts to three fine-tuning and one frozen
        episode per object. Sampling still occurs before this limit is applied.
    -h, --help
        Print usage and argument descriptions, then exit.

Examples (append to the Run command above):
    --split all                         Evaluate all fixed groups.
    --split in_distribution --smoke     Quickly check the default fixed group.
    --sampler-config PATH --num-objects 10 --seed 7
                                        Evaluate ten sampled objects per group.
"""

import argparse
import json
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.evaluation.rl import EvaluationMetrics, Evaluator
from src.evaluation.rl.constants import (
    DEFAULT_EVALUATION_SEED,
    DEFAULT_EVALUATION_SET_PATH,
    DEFAULT_FINE_TUNING_EPISODES,
    DEFAULT_FINE_TUNING_LR_SCALE,
    DEFAULT_FROZEN_EPISODES,
    DEFAULT_RESULTS_PATH,
    DEFAULT_SAMPLED_OBJECT_COUNT,
    OBJECT_SEED_STRIDE,
    RL_EVALUATION_SCHEMA_VERSION,
    SMOKE_FINE_TUNING_EPISODES,
    SMOKE_FROZEN_EPISODES,
    SMOKE_OBJECT_COUNT,
    EvaluationCase,
    EvaluationCaseResult,
)
from src.evaluation.rl.evaluation_sets import EvaluationSet
from src.simulation.environment.config import load_checkpoint_metadata


class SACEvaluation:
    """Own evaluation set execution and reporting for paired frozen and learning runs."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.frozen_episodes = (
            SMOKE_FROZEN_EPISODES if args.smoke else args.frozen_episodes
        )
        self.fine_tuning_episodes = (
            SMOKE_FINE_TUNING_EPISODES if args.smoke else args.fine_tuning_episodes
        )
        self.evaluation_set = self._load_evaluation_set()
        self.results: tuple[EvaluationCaseResult, ...] = ()
        self.checkpoint_metadata: dict[str, Any] = {}

    def run(self) -> dict[str, Any]:
        torch.set_num_threads(1)
        self.checkpoint_metadata = self._load_checkpoint_metadata(self.args.model)
        self.results = tuple(
            self._evaluate_case(case) for case in self.evaluation_set.cases
        )
        report = self._build_report()
        self._print_results(report["means"])
        self._write_report(report)
        return report

    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        cls._add_evaluation_arguments(parser)
        cls._add_source_arguments(parser)
        cls._add_output_arguments(parser)
        args = parser.parse_args()
        cls._validate_args(args, parser)
        return args

    @staticmethod
    def _validate_args(
        args: argparse.Namespace, parser: argparse.ArgumentParser
    ) -> None:
        """Report invalid evaluation settings through argparse before loading objects."""
        if args.fine_tuning_episodes < 1 or args.frozen_episodes < 1:
            parser.error("episode counts must be positive")
        if not np.isfinite(args.lr_scale) or args.lr_scale <= 0:
            parser.error("--lr-scale must be positive and finite")
        if args.num_objects < 1 or args.seed < 0:
            parser.error("--num-objects must be positive and --seed nonnegative")

    @staticmethod
    def _add_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model",
            type=Path,
            required=True,
            help="native SAC checkpoint (sac_final.pt)",
        )
        parser.add_argument(
            "--fine-tuning-episodes",
            type=int,
            default=DEFAULT_FINE_TUNING_EPISODES,
            help="episodes while fine-tuning per object",
        )
        parser.add_argument(
            "--frozen-episodes",
            type=int,
            default=DEFAULT_FROZEN_EPISODES,
            help="frozen-policy episodes per object; deterministic identical resets need only one",
        )
        parser.add_argument(
            "--lr-scale",
            type=float,
            default=DEFAULT_FINE_TUNING_LR_SCALE,
            help="fine-tuning lr = pretrain lr * scale (conservative, outline section 13)",
        )

    @staticmethod
    def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
        source = parser.add_mutually_exclusive_group()
        source.add_argument(
            "--evaluation-config",
            "--benchmark-config",
            type=Path,
            help="fixed object groups JSON",
        )
        source.add_argument(
            "--sampler-config",
            type=Path,
            help="one sampler or named sampler groups JSON",
        )
        parser.add_argument(
            "--split",
            nargs="+",
            help="configured group names, or all; default: original fixed group or all sampled groups",
        )
        parser.add_argument(
            "--num-objects",
            type=int,
            default=DEFAULT_SAMPLED_OBJECT_COUNT,
            help="objects sampled per group",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=DEFAULT_EVALUATION_SEED,
            help=f"base seed; object i uses seed+{OBJECT_SEED_STRIDE}*i",
        )

    @staticmethod
    def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--out",
            type=Path,
            default=None,
            help="results JSON; defaults to a separate file per selected split",
        )
        parser.add_argument(
            "--smoke",
            action="store_true",
            help=f"smoke mode: first {SMOKE_OBJECT_COUNT} specs per split, "
            f"fine-tuning-episodes={SMOKE_FINE_TUNING_EPISODES}, "
            f"frozen-episodes={SMOKE_FROZEN_EPISODES}",
        )

    @staticmethod
    def _load_checkpoint_metadata(model_path: Path) -> dict[str, Any]:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        metadata = load_checkpoint_metadata(
            model_path, checkpoint.get("training_metadata")
        )
        ranges = metadata.get("training_distribution", {}).get(
            "ranges", metadata.get("train_ranges")
        )
        return {
            "model": str(model_path.resolve()),
            "model_sha256": sha256(model_path.read_bytes()).hexdigest(),
            "checkpoint_training_steps": checkpoint["num_timesteps"],
            "checkpoint_training_metadata": metadata,
            "checkpoint_training_ranges": ranges,
        }

    def _load_evaluation_set(self) -> EvaluationSet:
        args = self.args
        path = (
            args.sampler_config or args.evaluation_config or DEFAULT_EVALUATION_SET_PATH
        )
        evaluation_set = EvaluationSet.load(
            path,
            args.split,
            sampled=bool(args.sampler_config),
            count=args.num_objects,
            seed=args.seed,
        )
        if not args.sampler_config and args.split is None:
            if "in_distribution" in evaluation_set.get_groups():
                evaluation_set = evaluation_set.select("in_distribution")
        return (
            evaluation_set.limit(SMOKE_OBJECT_COUNT) if args.smoke else evaluation_set
        )

    def _evaluate_case(self, case: EvaluationCase) -> EvaluationCaseResult:
        seed = self.args.seed + OBJECT_SEED_STRIDE * case.seed_index
        print(
            f"--- {case.split} object {case.seed_index}: (m, mu, h_c) = {case.spec.theta} ---"
        )
        frozen = Evaluator.evaluate_checkpoint(
            self.args.model, case.spec, self.frozen_episodes, seed
        )
        tuned = Evaluator.evaluate_checkpoint(
            self.args.model,
            case.spec,
            self.fine_tuning_episodes,
            seed,
            fine_tune=True,
            lr_scale=self.args.lr_scale,
        )
        return EvaluationCaseResult(
            case,
            frozen,
            tuned,
            EvaluationMetrics(frozen.episodes).summarize_progress(),
            EvaluationMetrics(tuned.episodes).summarize_progress(),
        )

    @staticmethod
    def _summarize_results(results: Sequence[EvaluationCaseResult]) -> dict[str, float]:
        if not results:
            raise ValueError("At least one evaluation set result is required")
        summary = {}
        for name in ("frozen_policy", "fine_tuning"):
            outcomes = [getattr(result, name).summary for result in results]
            episode_count = sum(outcome.episode_count for outcome in outcomes)
            topples = sum(outcome.topple_count for outcome in outcomes)
            successes = sum(outcome.success_count for outcome in outcomes)
            summary.update(
                {
                    f"{name}_topples": topples / len(results),
                    f"{name}_success_rate": successes / episode_count,
                    f"{name}_topple_rate": topples / episode_count,
                    f"{name}_object_success_fraction": sum(
                        outcome.success_count > 0 for outcome in outcomes
                    )
                    / len(results),
                }
            )
        return summary

    def _build_report(self) -> dict[str, Any]:
        metadata = self.checkpoint_metadata["checkpoint_training_metadata"]
        return {
            "schema_version": RL_EVALUATION_SCHEMA_VERSION,
            **self.evaluation_set.get_metadata(),
            **self.checkpoint_metadata,
            "summary_by_split": {
                group: self._summarize_results(
                    [result for result in self.results if result.case.split == group]
                )
                for group in self.evaluation_set.get_groups()
            },
            "seed": self.args.seed,
            "fine_tuning_episodes": self.fine_tuning_episodes,
            "frozen_episodes": self.frozen_episodes,
            "lr_scale": self.args.lr_scale,
            "smoke": self.args.smoke,
            "means": self._summarize_results(self.results),
            "objects": [result.get_report(metadata) for result in self.results],
        }

    def _print_results(self, means: dict[str, float]) -> None:
        print("\n=== Held-out SAC evaluation (B1): frozen policy vs fine-tuning ===")
        print(
            f"{'m':>5} {'mu':>4} {'h_c':>5} | "
            f"{'frozen_topple':>14} {'frozen_success':>14} {'frozen_first':>14} | "
            f"{'tuning_topple':>14} {'tuning_success':>14} {'tuning_first':>14}"
        )
        for result in self.results:
            spec = result.case.spec
            frozen, tuned = result.frozen_policy.summary, result.fine_tuning.summary
            frozen_first = result.frozen_progress.interactions_to_first_success
            tuned_first = result.fine_tuning_progress.interactions_to_first_success
            print(
                f"{spec.m:>5.2f} {spec.mu:>4.2f} {spec.h_c:>5.3f} | "
                f"{frozen.topple_count:>14} {frozen.success_count:>14} {str(frozen_first):>14} | "
                f"{tuned.topple_count:>14} {tuned.success_count:>14} {str(tuned_first):>14}"
            )
        print(
            f"\nmeans over {len(self.results)} objects: frozen policy topples "
            f"{means['frozen_policy_topples']:.2f}, frozen policy success "
            f"{means['frozen_policy_success_rate']:.2f}, fine-tuning topples "
            f"{means['fine_tuning_topples']:.2f}, fine-tuning success "
            f"{means['fine_tuning_success_rate']:.2f}"
        )

    def _write_report(self, report: dict[str, Any]) -> None:
        suffix = "_sampled" if self.args.sampler_config else ""
        if report["groups"] != ["in_distribution"]:
            suffix += "_" + "_".join(report["groups"])
        # Group names are labels, not paths.
        suffix = "".join(
            c if c.isascii() and (c.isalnum() or c in "_-") else "_" for c in suffix
        )
        out = self.args.out or DEFAULT_RESULTS_PATH.with_name(f"results{suffix}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        for name, summary in report["summary_by_split"].items():
            print(
                f"{name}: frozen success={summary['frozen_policy_success_rate']:.1%}, "
                f"fine-tuning success={summary['fine_tuning_success_rate']:.1%}"
            )
        out.write_text(json.dumps(report, indent=2))
        print(f"results written to {out}")


if __name__ == "__main__":
    SACEvaluation(SACEvaluation.parse_args()).run()
