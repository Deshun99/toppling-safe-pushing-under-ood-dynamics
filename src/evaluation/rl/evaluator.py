"""Shared episode execution for frozen-policy evaluation and SAC fine-tuning."""

from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike

from src.agents.rl.models.sac import SAC, SACConfig
from src.evaluation.rl.constants import (
    FINE_TUNING_REPLAY_CAPACITY,
    EpisodeResult,
    EvaluationResult,
)
from src.evaluation.rl.metrics import EpisodeRecorder, EvaluationMetrics
from src.simulation.environment import PushEnv
from src.simulation.environment.config import (
    load_action_ranges,
    load_checkpoint_metadata,
)
from src.simulation.modelling.object import CuboidSpec


class Policy(Protocol):
    def predict(
        self, observations: ArrayLike, deterministic: bool = True
    ) -> ArrayLike: ...


class Evaluator:
    """Run frozen evaluation or stochastic SAC fine-tuning on a dedicated env.

    The caller owns and closes the env. Explicit episode seeds repeat resets;
    None continues the env's RNG stream. Observation normalization stays frozen.
    Hidden object properties are recorded as metrics, never passed to the policy.
    """

    def __init__(self, env: PushEnv, episode_seeds: Sequence[int | None]) -> None:
        if not episode_seeds:
            raise ValueError("At least one episode seed is required")
        self.env = env
        self.episode_seeds = tuple(episode_seeds)

    def evaluate(
        self,
        policy: Policy,
        *,
        fine_tune: bool = False,
        on_episode_end: Callable[[EpisodeResult], None] | None = None,
    ) -> EvaluationResult:
        """Run all episodes, optionally reporting each result before the next reset."""
        self._validate_fine_tuning(policy, fine_tune)
        episodes = []
        for seed in self.episode_seeds:
            episode = self._run_episode(policy, seed, should_fine_tune=fine_tune)
            episodes.append(episode)
            if on_episode_end is not None:
                on_episode_end(episode)
        return EvaluationResult(
            episodes=tuple(episodes),
            summary=EvaluationMetrics(episodes).summarize(),
            metadata=self._get_metadata(fine_tune),
        )

    def _run_episode(
        self,
        policy: Policy,
        seed: int | None,
        should_fine_tune: bool,
    ) -> EpisodeResult:
        """Execute one rollout, recording terminal transitions before finishing."""
        observation, _ = self.env.reset(seed=seed)
        recorder = EpisodeRecorder(seed, asdict(self.env.cuboid.spec))
        while True:
            action = policy.predict(observation, deterministic=not should_fine_tune)
            next_observation, reward, terminated, truncated, info = self.env.step(
                action
            )
            if should_fine_tune:
                self._fine_tune_policy(
                    policy,
                    observation,
                    action,
                    reward,
                    next_observation,
                    terminated,
                )
            observation = next_observation
            recorder.record_step(reward, info)
            if terminated or truncated:
                return recorder.finish(observation, terminated, truncated, info)

    @staticmethod
    def _fine_tune_policy(
        policy: SAC,
        observation: ArrayLike,
        action: ArrayLike,
        reward: float,
        next_observation: ArrayLike,
        terminated: bool,
    ) -> None:
        """Record a transition and update the validated trainable SAC policy."""
        assert policy.replay is not None
        policy.replay.add(observation, action, reward, next_observation, terminated)
        policy.num_timesteps += 1
        policy.update()

    @staticmethod
    def _validate_fine_tuning(policy: Policy, fine_tune: bool) -> None:
        """Reject invalid fine-tuning before interacting with the environment."""
        if not fine_tune:
            return
        if not isinstance(policy, SAC):
            raise TypeError("Fine-tuning requires a native SAC policy")
        if policy.replay is None:
            raise RuntimeError(
                "Cannot fine-tune an inference-only SAC model; reload with inference_only=False."
            )

    def _get_metadata(self, fine_tune: bool) -> dict[str, Any]:
        return {
            "episode_seeds": list(self.episode_seeds),
            "deterministic": not fine_tune,
            "fine_tuning_enabled": fine_tune,
            "action_ranges": asdict(self.env.action_ranges),
            "reward": asdict(self.env.reward.config),
            "target_distance_m": self.env.target_distance,
            "position_tolerance_m": self.env.epsilon,
            "tilt_tolerance_rad": self.env.psi_succ_tol,
            "speed_tolerance_m_s": self.env.v_succ_tol,
            "max_push_count": self.env.T_max,
        }

    @classmethod
    def evaluate_checkpoint(
        cls,
        model_path: str | Path,
        spec: CuboidSpec,
        n_episodes: int,
        seed: int = SACConfig.seed,
        *,
        fine_tune: bool = False,
        lr_scale: float = 1.0,
    ) -> EvaluationResult:
        """Evaluate an independent SAC copy, then close its fixed-object env.

        Resets continue the seeded RNG stream; scores include ongoing fine-tuning.
        """
        if n_episodes < 1:
            raise ValueError("At least one episode is required")
        if not np.isfinite(lr_scale) or lr_scale <= 0:
            raise ValueError("Learning-rate scale must be positive and finite")
        model, env = cls._load_model_and_create_env(
            model_path, spec, seed, fine_tune, lr_scale
        )
        try:
            seeds = [seed] + [None] * (n_episodes - 1)
            result = cls(env, seeds).evaluate(model, fine_tune=fine_tune)
            result.metadata["initial_seed"] = seed
            return result
        finally:
            env.close()

    @staticmethod
    def _load_model_and_create_env(
        model_path: str | Path,
        spec: CuboidSpec,
        seed: int,
        fine_tune: bool,
        lr_scale: float,
    ) -> tuple[SAC, PushEnv]:
        model = SAC.load(
            model_path,
            lr_scale=lr_scale,
            buffer_size=FINE_TUNING_REPLAY_CAPACITY if fine_tune else None,
            seed=seed,
            inference_only=not fine_tune,
        )
        metadata = load_checkpoint_metadata(model_path, model.training_metadata)
        env = PushEnv(
            fixed_spec=spec,
            action_ranges=load_action_ranges(model_path, metadata),
            **metadata.get("environment", {}),
        )
        return model, env
