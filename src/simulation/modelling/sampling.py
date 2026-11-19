"""Configured cuboid sampling shared by environments and experiments."""

import json
from dataclasses import dataclass, fields
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from src.simulation.modelling.object import CuboidSpec

MAX_SAMPLING_ATTEMPTS = 10_000


@dataclass(frozen=True)
class ObjectRanges:
    """Closed object-parameter box; None leaves a parameter unconstrained.

    Lengths are in metres, mass in kg, and h_c_frac denotes h_c/H.
    Sampling requires all six intervals; exclusions may constrain any subset.
    """

    w: tuple[float, float] | None = None
    L: tuple[float, float] | None = None
    H: tuple[float, float] | None = None
    m: tuple[float, float] | None = None
    mu: tuple[float, float] | None = None
    h_c_frac: tuple[float, float] | None = None

    def violated_parameters(self, spec: CuboidSpec) -> list[str]:
        """List violated bounds, using inclusive endpoints and relative COM height."""
        violations = []
        for field in fields(self):
            interval = getattr(self, field.name)
            if interval is None:
                continue
            low, high = interval
            if field.name == "h_c_frac":
                # Compare in metres to retain inclusive boundaries without division.
                low, high = low * spec.H, high * spec.H
                value = spec.h_c
            else:
                value = getattr(spec, field.name)
            if not low <= value <= high:
                violations.append(field.name)
        return violations

    def contains(self, spec: CuboidSpec) -> bool:
        """Return whether the object satisfies every constrained parameter."""
        return not self.violated_parameters(spec)


DEFAULT_OBJECT_RANGES = ObjectRanges(
    w=(0.05, 0.10),
    L=(0.05, 0.10),
    H=(0.12, 0.25),
    m=(0.5, 1.5),
    mu=(0.3, 0.7),
    h_c_frac=(0.35, 0.60),
)


@dataclass(frozen=True)
class ObjectSampler:
    """Uniform cuboid ranges conditioned on avoiding closed joint exclusions."""

    ranges: ObjectRanges = DEFAULT_OBJECT_RANGES
    name: str = "default"
    exclusions: tuple[ObjectRanges, ...] = ()

    def __post_init__(self) -> None:
        if any(
            getattr(self.ranges, field.name) is None for field in fields(ObjectRanges)
        ):
            raise ValueError("Sampling requires all six object range intervals")

    def sample(self, rng: np.random.Generator) -> CuboidSpec:
        for _ in range(MAX_SAMPLING_ATTEMPTS):
            ranges = self.ranges
            H = rng.uniform(*ranges.H)
            spec = CuboidSpec(
                w=rng.uniform(*ranges.w),
                L=rng.uniform(*ranges.L),
                H=H,
                m=rng.uniform(*ranges.m),
                mu=rng.uniform(*ranges.mu),
                h_c=H * rng.uniform(*ranges.h_c_frac),
            )
            if not any(region.contains(spec) for region in self.exclusions):
                return spec
        raise RuntimeError("Exclusions leave too little probability to sample")


def load_intervals(
    values: dict[str, object], names: tuple[str, ...]
) -> dict[str, tuple[float, float]]:
    """Reject misspelled fields and invalid bounds before creating physics clients."""
    if not isinstance(values, dict) or set(values) != set(names):
        raise ValueError(f"Expected exactly these range fields: {names}")
    result: dict[str, tuple[float, float]] = {}
    for name in names:
        interval = values[name]
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or any(type(v) not in (int, float) or not np.isfinite(v) for v in interval)
            or interval[0] >= interval[1]
        ):
            raise ValueError(f"{name} must contain two finite increasing bounds")
        low, high = interval
        if (name != "mu" and low <= 0) or (name == "mu" and low < 0):
            raise ValueError(f"{name} has an invalid lower bound")
        if name == "h_c_frac" and high >= 1:
            raise ValueError("h_c_frac must lie strictly between zero and one")
        result[name] = (float(low), float(high))
    return result


def _parse_exclusions(values: list[dict[str, Any]]) -> tuple[ObjectRanges, ...]:
    """Validate partial boxes; omitted or null axes remain unconstrained."""
    if not isinstance(values, list):
        raise ValueError("exclusions must be a list, empty to preserve the full box")
    names = {field.name for field in fields(ObjectRanges)}
    exclusions = []
    for region in values:
        if not isinstance(region, dict) or set(region) - names:
            raise ValueError(
                f"Exclusions may contain only these range fields: {sorted(names)}"
            )
        constrained = {key: value for key, value in region.items() if value is not None}
        if not constrained:
            raise ValueError("An exclusion must constrain at least one parameter")
        exclusions.append(
            ObjectRanges(**load_intervals(constrained, tuple(constrained)))
        )
    return tuple(exclusions)


def _partition_midpoints(
    ranges: ObjectRanges,
    exclusions: tuple[ObjectRanges, ...],
    name: str,
) -> list[float]:
    """Partition one axis at exclusion boundaries inside the sampling support."""
    low, high = getattr(ranges, name)
    cuts = sorted(
        {low, high}.union(
            bound
            for region in exclusions
            if getattr(region, name) is not None
            for bound in getattr(region, name)
            if low < bound < high
        )
    )
    return [(left + right) / 2 for left, right in zip(cuts, cuts[1:])]


def _validate_sampling_support(
    ranges: ObjectRanges,
    exclusions: tuple[ObjectRanges, ...],
) -> None:
    """Reject unions of boxes covering all positive-volume sampling support."""
    if not exclusions:
        return
    names = [
        field.name
        for field in fields(ObjectRanges)
        if any(getattr(region, field.name) is not None for region in exclusions)
    ]
    axes = [_partition_midpoints(ranges, exclusions, name) for name in names]
    if not any(
        not any(
            all(
                getattr(region, name) is None
                or getattr(region, name)[0] <= value <= getattr(region, name)[1]
                for name, value in zip(names, point)
            )
            for region in exclusions
        )
        for point in product(*axes)
    ):
        raise ValueError("Exclusions cover the entire sampling distribution")


def sampler_from_config(config: dict[str, Any]) -> ObjectSampler:
    """Build a named sampler after validating bounds, exclusions, and support."""
    if not isinstance(config, dict) or set(config) != {"name", "ranges", "exclusions"}:
        raise ValueError("Distribution config requires name, ranges and exclusions")
    name = config["name"]
    if (
        not isinstance(name, str)
        or not name
        or any(not (c.isascii() and (c.isalnum() or c in "_-")) for c in name)
    ):
        raise ValueError(
            "Distribution name must use letters, digits, underscores or hyphens"
        )
    ranges = ObjectRanges(
        **load_intervals(
            config["ranges"], tuple(field.name for field in fields(ObjectRanges))
        )
    )
    exclusions = _parse_exclusions(config["exclusions"])
    _validate_sampling_support(ranges, exclusions)
    return ObjectSampler(ranges=ranges, name=name, exclusions=exclusions)


def load_object_sampler(path: str | Path) -> ObjectSampler:
    """Load the same sampler format for training or evaluation."""
    return sampler_from_config(json.loads(Path(path).read_text()))
