"""Episode measurements and summaries for frozen evaluation and fine-tuning.

Returns are undiscounted. Pushing time counts commanded force durations only;
it excludes reset settling, robot approach/retraction, and wall-clock time.
"""

from collections.abc import Sequence
from itertools import accumulate
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.evaluation.rl.constants import (
    EpisodeResult,
    EvaluationSummary,
    ProgressSummary,
)


class EvaluationMetrics:
    """Summarize recorded episodes without executing or updating a policy."""

    def __init__(self, episodes: Sequence[EpisodeResult]) -> None:
        if not episodes:
            raise ValueError("At least one episode is required")
        self.episodes = tuple(episodes)

    def summarize(self) -> EvaluationSummary:
        """Compute equally weighted batch statistics; episode order does not matter."""
        count = len(self.episodes)
        successful = [e for e in self.episodes if e.outcome == "success"]
        topples = sum(e.outcome == "toppled" for e in self.episodes)
        timeouts = sum(e.outcome == "timeout" for e in self.episodes)
        rewards = [e.total_reward for e in self.episodes]
        errors = [e.final_position_error_m for e in self.episodes]
        return EvaluationSummary(
            episode_count=count,
            success_count=len(successful),
            topple_count=topples,
            timeout_count=timeouts,
            mean_reward=float(np.mean(rewards)),
            std_reward=float(np.std(rewards)),
            success_rate=len(successful) / count,
            topple_rate=topples / count,
            timeout_rate=timeouts / count,
            mean_final_position_error_m=float(np.mean(errors)),
            mean_successful_push_count=(
                float(np.mean([e.push_count for e in successful]))
                if successful
                else None
            ),
            mean_successful_push_time_s=self._mean_push_time(successful),
        )

    def summarize_progress(self) -> ProgressSummary:
        """Track one object's chronological history, including prior failures.

        Works for frozen policies too. Never pool objects for first-success cost.
        """
        pushes = [episode.push_count for episode in self.episodes]
        topples = [episode.outcome == "toppled" for episode in self.episodes]
        successes = [episode.outcome == "success" for episode in self.episodes]
        first_interactions, first_time = self._get_first_success_cost()
        return ProgressSummary(
            interactions_to_first_success=first_interactions,
            push_time_to_first_success_s=first_time,
            cumulative_target_interactions=tuple(accumulate(pushes)),
            cumulative_topples=tuple(accumulate(int(value) for value in topples)),
            cumulative_successes=tuple(accumulate(int(value) for value in successes)),
        )

    def _get_first_success_cost(self) -> tuple[int | None, float | None]:
        """Count prior failures; missing durations make first-success time unknown."""
        interactions = 0
        duration: float | None = 0.0
        for episode in self.episodes:
            interactions += episode.push_count
            duration = (
                duration + episode.push_time_s
                if duration is not None and episode.push_time_s is not None
                else None
            )
            if episode.outcome == "success":
                return interactions, duration
        return None, None

    @staticmethod
    def _mean_push_time(episodes: Sequence[EpisodeResult]) -> float | None:
        """Require a duration for every success to preserve the denominator."""
        durations = [episode.push_time_s for episode in episodes]
        if not durations or any(value is None for value in durations):
            return None
        return float(np.mean(durations))


class EpisodeRecorder:
    """Collect measurements without controlling policy updates or resets."""

    def __init__(self, seed: int | None, object_spec: dict[str, float] | None) -> None:
        self.seed = seed
        self.object_spec = object_spec
        self.total_reward = 0.0
        self.push_count = 0
        self.push_time_s: float | None = 0.0
        self.max_abs_tilt_rad: float | None = None

    def record_step(self, reward: float, info: dict[str, Any]) -> None:
        self.total_reward += float(reward)
        self.push_count += 1
        executed = info.get("executed")
        if executed is None:
            self.push_time_s = None
        elif self.push_time_s is not None:
            self.push_time_s += float(executed[2])
        if "max_tilt" in info:
            peak = abs(float(info["max_tilt"]))
            self.max_abs_tilt_rad = max(self.max_abs_tilt_rad or 0.0, peak)

    def finish(
        self,
        observation: NDArray[np.float32] | Sequence[float],
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> EpisodeResult:
        """Finish a terminated episode, giving toppling precedence over success."""
        if not (terminated or truncated):
            raise ValueError("Cannot finish an ongoing episode")
        # Safety takes precedence if an environment reports both flags.
        if info["toppled"]:
            outcome = "toppled"
        elif info["success"]:
            outcome = "success"
        elif truncated:
            outcome = "timeout"
        else:
            raise ValueError("Unrecognized episode termination")
        return EpisodeResult(
            self.seed,
            self.object_spec,
            outcome,
            self.total_reward,
            self.push_count,
            self.push_time_s,
            abs(float(observation[0])),
            self.max_abs_tilt_rad,
        )
