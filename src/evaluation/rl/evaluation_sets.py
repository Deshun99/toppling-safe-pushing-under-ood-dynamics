"""Resolve fixed or sampled evaluation set objects and record how they were selected."""

import json
from collections import Counter
from copy import copy
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.rl.constants import (
    DEFAULT_EVALUATION_SEED,
    DEFAULT_EVALUATION_SET_PATH,
    DEFAULT_SAMPLED_OBJECT_COUNT,
    EvaluationCase,
)
from src.simulation.modelling.object import CuboidSpec
from src.simulation.modelling.sampling import sampler_from_config


class EvaluationSet:
    """Build evaluation cases once; filtering preserves global object seed indices."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        sampled: bool = False,
        count: int = DEFAULT_SAMPLED_OBJECT_COUNT,
        seed: int = DEFAULT_EVALUATION_SEED,
        selected: list[str] | str | None = None,
        config_path: Path | None = None,
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.sampled = sampled
        self.requested_count = count if sampled else None
        self.cases = tuple(
            self._sample_cases(config, count, seed, selected)
            if sampled
            else self._parse_fixed_cases(config, selected)
        )
        self.reference_ranges = None if sampled else config.get("reference_ranges")

    @classmethod
    def load(
        cls,
        path: str | Path = DEFAULT_EVALUATION_SET_PATH,
        selected: list[str] | str | None = None,
        *,
        sampled: bool = False,
        count: int = DEFAULT_SAMPLED_OBJECT_COUNT,
        seed: int = DEFAULT_EVALUATION_SEED,
    ) -> "EvaluationSet":
        """Read one configuration snapshot for both execution and reporting."""
        path = Path(path)
        return cls(
            json.loads(path.read_text()),
            sampled=sampled,
            count=count,
            seed=seed,
            selected=selected,
            config_path=path,
        )

    def select(self, selected: list[str] | str) -> "EvaluationSet":
        names = self._select_groups(self.get_groups(), selected)
        evaluation_set = copy(self)
        evaluation_set.cases = tuple(case for case in self.cases if case.split in names)
        return evaluation_set

    def limit(self, count: int) -> "EvaluationSet":
        """Keep the first count cases per group without redrawing sampled objects."""
        if count < 1:
            raise ValueError("Object limit must be positive")
        counts: Counter[str] = Counter()
        cases = []
        for case in self.cases:
            if counts[case.split] < count:
                cases.append(case)
                counts[case.split] += 1
        evaluation_set = copy(self)
        evaluation_set.cases = tuple(cases)
        return evaluation_set

    def get_groups(self) -> list[str]:
        return list(dict.fromkeys(case.split for case in self.cases))

    def get_metadata(self) -> dict[str, Any]:
        counts = dict(Counter(case.split for case in self.cases))
        common_counts = set(counts.values())
        return {
            "groups": self.get_groups(),
            "object_source": "sampled" if self.sampled else "fixed",
            "object_config_path": str(self.config_path.resolve())
            if self.config_path
            else None,
            "object_config": self.config,
            "reference_ranges": self.reference_ranges,
            "num_objects_per_group": next(iter(common_counts))
            if self.sampled and len(common_counts) == 1
            else None,
            "requested_num_objects_per_group": self.requested_count,
            "evaluated_objects_by_group": counts,
        }

    @staticmethod
    def _select_groups(names: list[str], selected: list[str] | str | None) -> set[str]:
        if (
            "all" in names
            or not names
            or any(not isinstance(n, str) or not n for n in names)
        ):
            raise ValueError(
                "EvaluationSet requires nonempty group names; 'all' is reserved"
            )
        if isinstance(selected, str):
            selected = [selected]
        if selected is None or selected == ["all"]:
            return set(names)
        if not selected or any(name not in names for name in selected):
            raise ValueError(
                f"Unknown evaluation set groups {selected}; available: {names}"
            )
        return set(selected)

    @classmethod
    def _parse_fixed_cases(
        cls,
        config: dict[str, Any],
        selected: list[str] | str | None,
    ) -> list[EvaluationCase]:
        if (
            not isinstance(config, dict)
            or "splits" not in config
            or set(config) - {"splits", "reference_ranges"}
            or not isinstance(config["splits"], dict)
        ):
            raise ValueError(
                "EvaluationSet config requires a splits mapping of fixed objects"
            )
        names = cls._select_groups(list(config["splits"]), selected)
        cases = []
        index = 0
        for name, specifications in config["splits"].items():
            if not isinstance(specifications, list) or not specifications:
                raise ValueError(
                    f"EvaluationSet group {name} requires object specifications"
                )
            for values in specifications:
                spec = cls._parse_fixed_object(values, name)
                if name in names:
                    cases.append(EvaluationCase(name, index, spec))
                index += 1
        return cases

    @staticmethod
    def _parse_fixed_object(values: dict[str, Any], group: str) -> CuboidSpec:
        """Validate fixed physics before creating any simulation clients."""
        expected = {field.name for field in fields(CuboidSpec)}
        if (
            not isinstance(values, dict)
            or set(values) != expected
            or any(
                type(v) not in (int, float) or not np.isfinite(v)
                for v in values.values()
            )
        ):
            raise ValueError(f"Invalid object specification in {group}")
        spec = CuboidSpec(**values)
        if (
            min(spec.w, spec.L, spec.H, spec.m, spec.h_c) <= 0
            or spec.mu < 0
            or spec.h_c >= spec.H
        ):
            raise ValueError(f"Nonphysical object specification in {group}")
        return spec

    @classmethod
    def _sample_cases(
        cls,
        config: dict[str, Any],
        count: int,
        seed: int,
        selected: list[str] | str | None,
    ) -> list[EvaluationCase]:
        if count < 1 or seed < 0:
            raise ValueError("Object count must be positive and seed nonnegative")
        if not isinstance(config, dict):
            raise ValueError("Sampler config must be a mapping")
        groups = (
            config["groups"]
            if set(config) == {"groups"}
            else {config.get("name", "sampled"): config}
        )
        if not isinstance(groups, dict):
            raise ValueError("groups must map labels to sampler configurations")
        names = cls._select_groups(list(groups), selected)
        cases = []
        for index, (name, values) in enumerate(groups.items()):
            sampler = sampler_from_config(values)
            if name not in names:
                continue
            rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
            cases.extend(
                EvaluationCase(name, index * count + offset, sampler.sample(rng))
                for offset in range(count)
            )
        return cases
