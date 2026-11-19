"""Physical action types shared by environments, executors, and agents."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PushAction:
    """Physical push: height h (m), force F (N), and duration tau (s)."""

    h: float
    F: float
    tau: float
