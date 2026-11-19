"""Selectable progress rewards with shared pushing costs and terminal outcomes."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal, Protocol

RewardStrategyName = Literal["linear", "goal_proximity"]


@dataclass
class RewardConfig:
    """Provisional reward settings; tune on training objects, then freeze."""

    alpha: float = 10.0  # Linear progress gain per metre.
    c_time: float = 1.0  # Per second of pushing plus optional passive settling.
    c_step: float = 0.1  # Per decision.
    R_succ: float = 100.0
    R_topple: float = 100.0
    strategy: RewardStrategyName = "linear"
    proximity_bonus: float = 100.0  # Bounds the added closeness score.
    proximity_scale: float = 0.25  # Metres; half the bonus at this distance.
    no_progress_penalty: float = 0.0  # Opt-in, goal_proximity only.
    no_progress_threshold: float = 0.001  # Net goal-distance reduction in metres.

    def __post_init__(self) -> None:
        if self.strategy not in ("linear", "goal_proximity"):
            raise ValueError("reward strategy must be linear or goal_proximity")
        for name in (
            "alpha",
            "c_time",
            "c_step",
            "R_succ",
            "R_topple",
            "proximity_bonus",
            "no_progress_penalty",
            "no_progress_threshold",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isfinite(self.proximity_scale) or self.proximity_scale <= 0:
            raise ValueError("proximity_scale must be finite and positive")


class ProgressRewardStrategy(Protocol):
    def compute_progress(self, previous_distance: float, distance: float) -> float:
        """Reward the change in unsigned distance, independently of push costs."""
        ...


@dataclass
class LinearProgressReward:
    config: RewardConfig

    def compute_progress(self, previous_distance: float, distance: float) -> float:
        return self.config.alpha * (abs(previous_distance) - abs(distance))


@dataclass
class GoalProximityReward:
    config: RewardConfig

    def compute_progress(self, previous_distance: float, distance: float) -> float:
        return self._compute_potential(distance) - self._compute_potential(
            previous_distance
        )

    def _compute_potential(self, distance: float) -> float:
        remaining = abs(distance)
        closeness = self.config.proximity_scale / (
            remaining + self.config.proximity_scale
        )
        return -self.config.alpha * remaining + self.config.proximity_bonus * closeness


class RewardModel:
    """Delegate progress to a strategy and apply common costs and terminal rewards."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config if config is not None else RewardConfig()
        self.strategy: ProgressRewardStrategy
        if self.config.strategy == "goal_proximity":
            self.strategy = GoalProximityReward(self.config)
        else:
            self.strategy = LinearProgressReward(self.config)

    def compute_reward(
        self,
        *,
        previous_distance: float,
        distance: float,
        duration: float,
        success: bool,
        toppled: bool,
        goal_tolerance: float = 0.05,
    ) -> float:
        """Compute reward using signed target-minus-position distances in metres.

        Both strategies use absolute distance so overshooting is penalized when
        it moves farther from the goal. Duration includes optional settling. An
        optional no-progress penalty excludes terminal outcomes and positions
        inside goal_tolerance. Proximity uses an undiscounted score difference,
        changing the discounted objective with no policy-invariance guarantee.
        """
        return (
            self.strategy.compute_progress(previous_distance, distance)
            - self._compute_no_progress_penalty(
                previous_distance, distance, success, toppled, goal_tolerance
            )
            - self.config.c_time * duration
            - self.config.c_step
            + (self.config.R_succ if success else 0.0)
            - (self.config.R_topple if toppled else 0.0)
        )

    def _compute_no_progress_penalty(
        self,
        previous_distance: float,
        distance: float,
        success: bool,
        toppled: bool,
        goal_tolerance: float,
    ) -> float:
        if self.config.strategy != "goal_proximity" or success or toppled:
            return 0.0
        if abs(distance) <= goal_tolerance:
            return 0.0
        progress = abs(previous_distance) - abs(distance)
        if progress <= self.config.no_progress_threshold:
            return self.config.no_progress_penalty
        return 0.0
