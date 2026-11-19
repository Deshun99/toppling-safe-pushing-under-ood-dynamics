"""Checkpoint environment configuration, with support for legacy sidecars."""

import json
from pathlib import Path
from typing import Any

import torch

from src.simulation.environment.env import ActionRanges


def load_checkpoint_metadata(
    model_path: str | Path, embedded: dict[str, Any] | None
) -> dict[str, Any]:
    """Resolve already-read metadata; consult config.json only for legacy weights.

    An embedded configuration is authoritative. Do not fill missing fields from
    a possibly unrelated sidecar or silently substitute environment defaults.
    """
    path = Path(model_path)
    if embedded is not None:
        return embedded
    config_path = path.with_name("config.json")
    if not config_path.is_file():
        raise FileNotFoundError(
            f"No embedded checkpoint metadata or legacy sidecar {config_path}"
        )
    return json.loads(config_path.read_text())


def load_action_ranges(
    model_path: str | Path, metadata: dict[str, Any] | None = None
) -> ActionRanges:
    """Restore physical action semantics; missing bounds are an error."""
    if metadata is None:
        path = Path(model_path)
        state = (
            torch.load(path, map_location="cpu", weights_only=True)
            if path.is_file()
            else {}
        )
        metadata = load_checkpoint_metadata(path, state.get("training_metadata"))
    config = metadata
    if "action_ranges" in config:
        return ActionRanges(
            **{key: tuple(value) for key, value in config["action_ranges"].items()}
        )
    if "f_max" in config:
        return ActionRanges(force=(0.0, float(config["f_max"])))
    raise ValueError("Checkpoint metadata contains no action_ranges or legacy f_max")
