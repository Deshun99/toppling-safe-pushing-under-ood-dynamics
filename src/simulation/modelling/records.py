"""Recorded physical actions shared by target environments and replay."""

from dataclasses import dataclass

from src.simulation.execution.force_push import AppliedForce, DenseTrajectory
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.object import ObjectState


@dataclass
class ActionRecord:
    """Measured start, requested action, trajectory and optional actual forces.

    Direct-force records start before their optional recorded warm-up. Gantry
    records start after approach, at the beginning of the timed push, and have
    no recorded warm-up. Actual forces drive identification only; forecasts
    continue to apply the requested action directly.
    """

    start: ObjectState
    action: PushAction
    traj: DenseTrajectory
    warmup_steps: int = 0
    applied_forces: tuple[AppliedForce, ...] | None = None

    def __post_init__(self) -> None:
        if self.applied_forces is not None and (
            len(self.applied_forces) + self.warmup_steps != len(self.traj.t)
        ):
            raise ValueError("actual forces must align with timed trajectory samples")
