"""Evaluate frozen delta weights with object-specific online identification."""

from src.agents.rl.training.constants import EVALUATE_MODE
from src.agents.rl.training.train_delta import main


if __name__ == "__main__":
    main(default_mode=EVALUATE_MODE)
