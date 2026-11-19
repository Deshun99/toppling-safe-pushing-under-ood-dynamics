"""Fixed training defaults and smoke-test budgets."""

from pathlib import Path
from typing import Final

TRAIN_MODE: Final = "train"
ADAPT_MODE: Final = "adapt"
EVALUATE_MODE: Final = "evaluate"

DEFAULT_CURRICULUM_MAX_PUSHES: Final = 10_000
CURRICULUM_MIN_PUSHES: Final = 2_000
CURRICULUM_SUCCESS_WINDOW: Final = 30
CURRICULUM_SUCCESS_THRESHOLD: Final = 0.9

DEFAULT_TOTAL_STEPS: Final = 800_000
DEFAULT_N_ENVS: Final = 8
DEFAULT_LEARNING_STARTS: Final = 10_000
DEFAULT_EVAL_FREQ: Final = 20_000
DEFAULT_N_EVAL_EPISODES: Final = 20
# Keep checkpoint-validation cases separate from training reset seeds.
EVALUATION_SEED_OFFSET: Final = 10_000

SMOKE_ARGUMENT_OVERRIDES: Final = {
    "total_steps": 512,
    "eval_freq": 256,
    "learning_starts": 64,
    "n_eval_episodes": 4,
    "buffer_size": 10_000,
}

DEFAULT_DISTRIBUTION_PATH: Final = (
    Path(__file__).resolve().parents[4]
    / "configs/rl/training_distributions/uniform_excluding_evaluation_regions.json"
)
