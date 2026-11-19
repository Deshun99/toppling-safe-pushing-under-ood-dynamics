"""RL evaluation settings, records, summaries, and their units."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal

from src.simulation.modelling.object import CuboidSpec
from src.simulation.modelling.sampling import ObjectRanges

EVALUATION_SCHEMA_VERSION: Final = 1
FINE_TUNING_REPLAY_CAPACITY: Final = 10_000


RL_EVALUATION_SCHEMA_VERSION: Final = 2
DEFAULT_EVALUATION_SEED: Final = 42
# Separate each object's random stream while preserving paired method seeds.
OBJECT_SEED_STRIDE: Final = 1_000
DEFAULT_FINE_TUNING_EPISODES: Final = 20
# Fixed objects, identical resets and deterministic actions need only one rollout.
DEFAULT_FROZEN_EPISODES: Final = 1
DEFAULT_FINE_TUNING_LR_SCALE: Final = 0.5
DEFAULT_RESULTS_PATH: Final = Path("artifacts/held_out_evaluation/results.json")
SMOKE_OBJECT_COUNT: Final = 2
SMOKE_FINE_TUNING_EPISODES: Final = 3
SMOKE_FROZEN_EPISODES: Final = DEFAULT_FROZEN_EPISODES
DEFAULT_MODEL_PATH: Final = Path("artifacts/sac_pretrain/sac_final.pt")
DEFAULT_VISUALIZATION_EPISODES: Final = 5
DEFAULT_VISUALIZATION_SEED: Final = 0
DEFAULT_PLAYBACK_SPEED: Final = 1.0
DEFAULT_EPISODE_LINGER_S: Final = 2.0
DEFAULT_SAMPLED_OBJECT_COUNT: Final = 20

DEFAULT_EVALUATION_SET_PATH: Final = (
    Path(__file__).resolve().parents[3] / "configs/rl/evaluation_sets/fixed_cases.json"
)


@dataclass(frozen=True)
class EpisodeResult:
    seed: int | None
    object_spec: dict[str, float] | None
    outcome: Literal["success", "toppled", "timeout"]
    total_reward: float
    push_count: int
    push_time_s: float | None
    final_position_error_m: float
    max_abs_tilt_rad: float | None


@dataclass(frozen=True)
class EvaluationSummary:
    episode_count: int
    success_count: int
    topple_count: int
    timeout_count: int
    mean_reward: float
    std_reward: float
    success_rate: float
    topple_rate: float
    timeout_rate: float
    mean_final_position_error_m: float
    mean_successful_push_count: float | None
    mean_successful_push_time_s: float | None

    def get_scalars(self) -> dict[str, float | int]:
        """Omit unavailable values rather than logging misleading zeros."""
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class EvaluationResult:
    episodes: tuple[EpisodeResult, ...]
    summary: EvaluationSummary
    metadata: dict[str, Any]
    schema_version: int = EVALUATION_SCHEMA_VERSION


@dataclass(frozen=True)
class ProgressSummary:
    """
    One object's chronological episode history, including failed episodes.

    First-success time includes all prior failures. None denotes no success
    within the recorded budget (or unavailable duration instrumentation).
    Aggregate histories across objects separately; never concatenate objects
    when calculating first success.
    """

    interactions_to_first_success: int | None
    push_time_to_first_success_s: float | None
    cumulative_target_interactions: tuple[int, ...]
    cumulative_topples: tuple[int, ...]
    cumulative_successes: tuple[int, ...]


@dataclass(frozen=True)
class EvaluationCase:
    split: str
    seed_index: int
    spec: CuboidSpec

    def classify(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Check checkpoint support, independently of the evaluation set's group label."""
        distribution = metadata.get("training_distribution", {})
        ranges = distribution.get("ranges", metadata.get("train_ranges"))
        if ranges is None:
            return {"membership": "unknown", "outside_checkpoint_ranges": None}
        outside = ObjectRanges(**ranges).violated_parameters(self.spec)
        excluded = any(
            ObjectRanges(**region).contains(self.spec)
            for region in distribution.get("exclusions", [])
        )
        return {
            "membership": "outside_ranges"
            if outside
            else ("excluded_region" if excluded else "training_support"),
            "outside_checkpoint_ranges": outside,
        }


@dataclass(frozen=True)
class EvaluationCaseResult:
    """Keep object identity and both conditions together, with prepared progress."""

    case: EvaluationCase
    frozen_policy: EvaluationResult
    fine_tuning: EvaluationResult
    frozen_progress: ProgressSummary
    fine_tuning_progress: ProgressSummary

    def get_report(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Preserve the external report fields independently of internal records."""
        return {
            "split": self.case.split,
            "seed_index": self.case.seed_index,
            **self.case.classify(metadata),
            "spec": asdict(self.case.spec),
            "frozen_policy": asdict(self.frozen_policy),
            "fine_tuning": asdict(self.fine_tuning),
            "adaptation_progress": asdict(self.fine_tuning_progress),
        }
