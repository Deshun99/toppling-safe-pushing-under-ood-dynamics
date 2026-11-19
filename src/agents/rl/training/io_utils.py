"""Output directories and JSON conversion shared by the training commands."""

from dataclasses import asdict, is_dataclass
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, TextIO
from uuid import uuid4

import numpy as np


class _TeeOutput:
    def __init__(self, console: TextIO, log: TextIO) -> None:
        self.console = console
        self.log = log

    def write(self, message: str) -> int:
        self.console.write(message)
        self.log.write(message)
        self.log.flush()
        return len(message)

    def flush(self) -> None:
        self.console.flush()
        self.log.flush()


@contextmanager
def log_console_output(path: Path) -> Iterator[None]:
    """Mirror Python stdout and stderr to a flushed log while retaining the console."""
    with path.open("a", encoding="utf-8") as log:
        with (
            redirect_stdout(_TeeOutput(sys.stdout, log)),
            redirect_stderr(_TeeOutput(sys.stderr, log)),
        ):
            yield


def create_output_directory(
    output_directory: Path | None,
    default_root: Path,
) -> tuple[Path, str]:
    """Reserve an empty output directory and return it with a unique run ID."""
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    )
    directory = (
        Path(output_directory)
        if output_directory is not None
        else default_root / run_id
    )
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory, run_id


def to_json_value(value: Any) -> Any:
    """Convert experiment settings and results to JSON-compatible values."""
    if is_dataclass(value):
        return to_json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
