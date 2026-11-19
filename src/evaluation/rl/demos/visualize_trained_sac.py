"""Watch a frozen SAC policy and report the shared evaluation metrics.

Run: python -m src.evaluation.rl.demos.visualize_trained_sac
"""

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch

from src.agents.rl.models.sac import SAC
from src.evaluation.rl import EpisodeResult, Evaluator
from src.evaluation.rl.constants import (
    DEFAULT_EPISODE_LINGER_S,
    DEFAULT_MODEL_PATH,
    DEFAULT_PLAYBACK_SPEED,
    DEFAULT_VISUALIZATION_EPISODES,
    DEFAULT_VISUALIZATION_SEED,
)
from src.simulation.environment import PushEnv
from src.simulation.environment.config import (
    load_action_ranges,
    load_checkpoint_metadata,
)


class SACVisualization:
    """Watch evaluation episodes and pause before each subsequent reset."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    @staticmethod
    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
        parser.add_argument(
            "--episodes", type=int, default=DEFAULT_VISUALIZATION_EPISODES
        )
        parser.add_argument("--seed", type=int, default=DEFAULT_VISUALIZATION_SEED)
        parser.add_argument(
            "--speed",
            type=float,
            default=DEFAULT_PLAYBACK_SPEED,
            help="1 = real time, 0 = unpaced",
        )
        parser.add_argument(
            "--linger",
            type=float,
            default=DEFAULT_EPISODE_LINGER_S,
            help="Pause after each episode (s)",
        )
        args = parser.parse_args()
        if args.episodes < 1:
            parser.error("--episodes must be positive")
        if not (0 <= args.speed < float("inf")) or not (
            0 <= args.linger < float("inf")
        ):
            parser.error("--speed and --linger must be finite and nonnegative")
        return args

    def run(self) -> None:
        args = self.args
        torch.set_num_threads(1)
        model = SAC.load(args.model, inference_only=True)
        metadata = load_checkpoint_metadata(args.model, model.training_metadata)
        env = PushEnv(
            gui=True,
            realtime_scale=args.speed,
            action_ranges=load_action_ranges(args.model, metadata),
            **metadata.get("environment", {}),
        )
        try:
            evaluator = Evaluator(env, range(args.seed, args.seed + args.episodes))
            result = evaluator.evaluate(model, on_episode_end=self._report_episode)
            print("Evaluation summary:")
            for key, value in asdict(result.summary).items():
                print(f"  {key}: {value}")
        finally:
            env.close()

    def _report_episode(self, episode: EpisodeResult) -> None:
        print(
            f"seed={episode.seed} reward={episode.total_reward:.2f} "
            f"outcome={episode.outcome} pushes={episode.push_count} "
            f"error={episode.final_position_error_m:.3f} m"
        )
        if self.args.linger:
            time.sleep(self.args.linger)


if __name__ == "__main__":
    SACVisualization(SACVisualization.parse_args()).run()
