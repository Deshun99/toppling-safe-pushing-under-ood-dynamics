"""Shared action coordinates for policies, target execution, and forecasting.

Mappings depend only on observable geometry and configured actuator bounds.
Controllers reject invalid candidates by default; the target environment uses
``clip=True`` to retain its existing normalized-policy clipping convention.
"""

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.object import Geometry


class ActionBounds(Protocol):
    height_fraction: tuple[float, float]
    force: tuple[float, float]
    duration: tuple[float, float]


def _physical_bounds(
    geometry: Geometry, ranges: ActionBounds
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    dimensions = np.asarray(geometry, dtype=np.float64)
    if dimensions.shape != (3,) or not np.all(np.isfinite(dimensions)):
        raise ValueError("geometry must contain three finite dimensions (w, L, H)")
    if np.any(dimensions <= 0):
        raise ValueError("geometry dimensions must be positive")
    bounds = np.asarray(
        [ranges.height_fraction, ranges.force, ranges.duration], dtype=np.float64
    )
    if bounds.shape != (3, 2) or not np.all(np.isfinite(bounds)):
        raise ValueError("action ranges must contain three finite lower/upper pairs")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("each action range must have an increasing lower/upper pair")
    bounds[0] *= dimensions[2]
    return bounds[:, 0], bounds[:, 1]


def normalized_to_physical(
    action: ArrayLike,
    geometry: Geometry,
    ranges: ActionBounds,
    *,
    clip: bool = False,
) -> PushAction:
    """Map normalized ``(height, force, duration)`` to physical units.

    Reject out-of-bounds candidates unless clipping is requested explicitly.
    Finite vectors of exactly three coordinates are required in either mode.
    """
    values = np.asarray(action, dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("normalized action must contain three finite coordinates")
    if clip:
        values = np.clip(values, -1.0, 1.0)
    elif np.any((values < -1.0) | (values > 1.0)):
        raise ValueError("normalized action must lie within [-1, 1]")
    low, high = _physical_bounds(geometry, ranges)
    h, force, duration = low + (values + 1.0) / 2.0 * (high - low)
    return PushAction(h=float(h), F=float(force), tau=float(duration))


def physical_to_normalized(
    action: PushAction, geometry: Geometry, ranges: ActionBounds
) -> NDArray[np.float64]:
    """Invert the action map without silently clipping a physical command.

    Only roundoff at the endpoints (a few floating-point ulps) is tolerated.
    """
    if not isinstance(action, PushAction):
        raise TypeError("physical action must be a PushAction")
    values = np.asarray([action.h, action.F, action.tau], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("physical action must contain finite coordinates")
    low, high = _physical_bounds(geometry, ranges)
    normalized = 2.0 * (values - low) / (high - low) - 1.0
    tolerance = 8 * np.finfo(np.float64).eps
    if np.any((normalized < -1.0 - tolerance) | (normalized > 1.0 + tolerance)):
        raise ValueError("physical action must lie within the configured bounds")
    return np.clip(normalized, -1.0, 1.0)
