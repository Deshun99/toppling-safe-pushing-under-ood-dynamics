import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from src.agents.rl.training.constants import DEFAULT_DISTRIBUTION_PATH
from src.evaluation.rl.evaluation_sets import (
    DEFAULT_EVALUATION_SET_PATH,
    EvaluationSet,
)
from src.simulation.environment import PushEnv
from src.simulation.modelling.object import CuboidSpec
from src.simulation.modelling.sampling import (
    ObjectRanges,
    ObjectSampler,
    load_object_sampler,
)

HELD_OUT_OBJECTS = tuple(
    case.spec
    for case in EvaluationSet.load(DEFAULT_EVALUATION_SET_PATH, "in_distribution").cases
)

UNRESTRICTED_DISTRIBUTION_PATH = DEFAULT_DISTRIBUTION_PATH.with_name(
    "uniform_without_exclusions.json"
)
DEFAULT_DISTRIBUTION = load_object_sampler(DEFAULT_DISTRIBUTION_PATH)


@pytest.fixture
def distribution() -> ObjectSampler:
    return DEFAULT_DISTRIBUTION


@pytest.mark.parametrize(
    "spec,region", zip(HELD_OUT_OBJECTS, DEFAULT_DISTRIBUTION.exclusions)
)
def test_test_regions_reserve_hidden_physics_at_any_geometry(
    spec: CuboidSpec, region: ObjectRanges
) -> None:
    assert region.contains(spec)
    assert region.contains(
        replace(spec, w=0.08, L=0.07, H=spec.H / 2, h_c=spec.h_c / 2)
    )


@pytest.mark.parametrize("field", ["m", "mu", "h_c"])
def test_test_region_boundaries_are_excluded_but_individual_parameters_are_allowed(
    distribution: ObjectSampler,
    field: str,
) -> None:
    spec = HELD_OUT_OBJECTS[0]
    region = distribution.exclusions[0]
    low, high = (
        (region.h_c_frac[0] * spec.H, region.h_c_frac[1] * spec.H)
        if field == "h_c"
        else getattr(region, field)
    )
    assert region.contains(replace(spec, **{field: low}))
    assert region.contains(replace(spec, **{field: high}))
    assert not region.contains(replace(spec, **{field: np.nextafter(low, -np.inf)}))
    assert not region.contains(replace(spec, **{field: np.nextafter(high, np.inf)}))


@pytest.mark.parametrize("spec", HELD_OUT_OBJECTS)
def test_sampling_rejects_each_test_object_before_returning_allowed_physics(
    env: PushEnv, spec: CuboidSpec, distribution: ObjectSampler
) -> None:
    env._object_sampler = distribution
    # Force a forbidden first draw, followed by physics outside all eight boxes.
    env._rng = Mock()
    env._rng.uniform.side_effect = [
        spec.H,
        spec.w,
        spec.L,
        spec.m,
        spec.mu,
        spec.h_c / spec.H,
        0.2,
        0.08,
        0.08,
        0.8,
        0.68,
        0.36,
    ]
    sampled = env._sample_spec()
    assert sampled.m == 0.8
    assert sampled.mu == 0.68
    assert sampled.h_c / sampled.H == pytest.approx(0.36)
    assert env._rng.uniform.call_count == 12


def test_sampling_respects_configured_ranges_exclusions_and_seed(
    distribution: ObjectSampler,
) -> None:
    first_rng = np.random.default_rng(7)
    second_rng = np.random.default_rng(7)
    for _ in range(5_000):
        spec = distribution.sample(first_rng)
        assert spec == distribution.sample(second_rng)
        assert not any(region.contains(spec) for region in distribution.exclusions)
        for field in ("w", "L", "H", "m", "mu"):
            low, high = getattr(distribution.ranges, field)
            assert low <= getattr(spec, field) <= high
        low, high = distribution.ranges.h_c_frac
        assert low <= spec.h_c / spec.H <= high


def test_plain_environment_has_no_evaluation_set_exclusions() -> None:
    spec = HELD_OUT_OBJECTS[0]
    sampler = Mock(spec=ObjectSampler)
    sampler.sample.return_value = spec
    env = PushEnv(object_sampler=sampler)
    try:
        env.reset(seed=7)
        assert env.cuboid.spec is spec
        assert sampler.sample.call_count == 1
        sampler.reset_mock()
        env._fixed_spec = HELD_OUT_OBJECTS[1]
        env.reset(seed=7)
        sampler.sample.assert_not_called()
        assert env.cuboid.spec is HELD_OUT_OBJECTS[1]
    finally:
        env.close()


def test_configured_exclusions_can_be_disabled() -> None:
    distribution = load_object_sampler(UNRESTRICTED_DISTRIBUTION_PATH)
    assert distribution.exclusions == ()
    spec = HELD_OUT_OBJECTS[0]
    rng = Mock()
    rng.uniform.side_effect = [
        spec.H,
        spec.w,
        spec.L,
        spec.m,
        spec.mu,
        spec.h_c / spec.H,
    ]
    assert distribution.sample(rng).theta == pytest.approx(spec.theta)
    assert rng.uniform.call_count == 6


@pytest.mark.parametrize(
    "invalid",
    [
        "missing",
        "unknown",
        "nonfinite",
        "reversed",
        "nonphysical",
        "covered",
        "jointly_covered",
    ],
)
def test_invalid_config_is_rejected_before_sampling(
    tmp_path: Path, invalid: str
) -> None:
    config = json.loads(UNRESTRICTED_DISTRIBUTION_PATH.read_text())
    if invalid == "missing":
        del config["exclusions"]
    elif invalid == "unknown":
        config["ranges"]["mass"] = config["ranges"].pop("m")
    elif invalid == "nonfinite":
        config["ranges"]["m"] = [0.5, float("nan")]
    elif invalid == "reversed":
        config["ranges"]["m"] = [1.5, 0.5]
    else:
        region = {key: config["ranges"][key] for key in ("m", "mu", "h_c_frac")}
        config["exclusions"] = [region]
        if invalid == "nonphysical":
            region["m"] = [-0.1, 1.0]
        elif invalid == "jointly_covered":
            config["exclusions"] = [
                dict(region, m=[0.5, 1.0]),
                dict(region, m=[1.0, 1.5]),
            ]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_object_sampler(path)


def test_custom_distribution_reaches_spawn_workers(tmp_path: Path) -> None:
    from src.agents.rl.training.train_sac import make_train_vec

    config = json.loads(UNRESTRICTED_DISTRIBUTION_PATH.read_text())
    config["ranges"]["m"] = [1.9, 2.0]
    config["ranges"]["mu"] = [0.8, 0.9]
    path = tmp_path / "custom.json"
    path.write_text(json.dumps(config))
    distribution = load_object_sampler(path)
    vec = make_train_vec(2, seed=7, distribution=distribution)
    try:
        vec.reset(seed=7)
        for spec in vec.call("_sample_spec"):
            assert 1.9 <= spec.m <= 2.0
            assert 0.8 <= spec.mu <= 0.9
    finally:
        vec.close()


@pytest.mark.parametrize(
    "axis,outside", [("m", [2.0, 3.0]), ("mu", [0.85, 1.1]), ("h_c_frac", [0.65, 0.8])]
)
def test_fully_ood_exclusions_leave_training_samples_unchanged(
    tmp_path: Path, axis: str, outside: list[float]
) -> None:
    config = json.loads(UNRESTRICTED_DISTRIBUTION_PATH.read_text())
    region = {name: config["ranges"][name] for name in ("m", "mu", "h_c_frac")}
    region[axis] = outside
    config["exclusions"] = [region]
    path = tmp_path / "ood_exclusion.json"
    path.write_text(json.dumps(config))
    distribution = load_object_sampler(path)
    baseline = load_object_sampler(UNRESTRICTED_DISTRIBUTION_PATH)
    left, right = np.random.default_rng(11), np.random.default_rng(11)
    for _ in range(100):
        assert distribution.sample(left) == baseline.sample(right)
    assert list(getattr(distribution.exclusions[0], axis)) == outside


def test_partially_outside_exclusion_rejects_only_its_joint_overlap(
    tmp_path: Path,
) -> None:
    config = json.loads(UNRESTRICTED_DISTRIBUTION_PATH.read_text())
    config["exclusions"] = [{"m": [1.4, 2.0], "mu": [0.4, 0.6], "h_c_frac": [0.4, 0.6]}]
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(config))
    distribution = load_object_sampler(path)
    rng = Mock()
    # First draw intersects the box. Second shares its mass, but has allowed friction.
    rng.uniform.side_effect = [
        0.2,
        0.08,
        0.08,
        1.45,
        0.5,
        0.5,
        0.2,
        0.08,
        0.08,
        1.45,
        0.65,
        0.5,
    ]
    sampled = distribution.sample(rng)
    assert sampled.m == 1.45
    assert sampled.mu == 0.65
    assert rng.uniform.call_count == 12
    assert distribution.exclusions[0].m == (1.4, 2.0)


def test_extrapolated_exclusions_cannot_mask_full_training_coverage(
    tmp_path: Path,
) -> None:
    config = json.loads(UNRESTRICTED_DISTRIBUTION_PATH.read_text())
    # Each box extends outside training. Their union still covers every training sample.
    config["exclusions"] = [
        {"m": [0.1, 1.0], "mu": [0.1, 1.1], "h_c_frac": [0.1, 0.9]},
        {"m": [1.0, 3.0], "mu": [0.1, 1.1], "h_c_frac": [0.1, 0.9]},
    ]
    path = tmp_path / "covered.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="entire sampling distribution"):
        load_object_sampler(path)


def test_shared_ranges_leave_unspecified_parameters_unconstrained() -> None:
    spec = HELD_OUT_OBJECTS[0]
    region = ObjectRanges(m=(0.65, 0.75))
    assert region.contains(spec)
    assert region.contains(replace(spec, w=0.2, H=0.4, h_c=0.3, mu=1.1))
    assert not region.contains(replace(spec, m=0.8))
    with pytest.raises(ValueError, match="all six"):
        ObjectSampler(ranges=region)


def test_geometry_exclusions_use_the_same_range_type() -> None:
    from dataclasses import asdict

    from src.simulation.modelling.sampling import sampler_from_config

    config = json.loads(UNRESTRICTED_DISTRIBUTION_PATH.read_text())
    config["exclusions"] = [{"w": [0.05, 0.075]}]
    sampler = sampler_from_config(config)
    assert isinstance(sampler.exclusions[0], ObjectRanges)
    rng = np.random.default_rng(7)
    for _ in range(100):
        assert sampler.sample(rng).w > 0.075
    # Resolved metadata includes None for unconstrained axes and remains loadable.
    assert sampler_from_config(json.loads(json.dumps(asdict(sampler)))) == sampler
    config["exclusions"] = [{"w": [0.04, 0.11]}]
    with pytest.raises(ValueError, match="entire sampling distribution"):
        sampler_from_config(config)


@pytest.mark.parametrize(
    "name", ["uniform_excluding_evaluation_regions", "uniform_without_exclusions"]
)
def test_sampler_preserves_configured_name(name: str) -> None:
    sampler = load_object_sampler(DEFAULT_DISTRIBUTION_PATH.with_name(f"{name}.json"))
    assert sampler.name == name


def test_range_membership_and_violation_names_share_inclusive_com_boundaries() -> None:
    spec = HELD_OUT_OBJECTS[0]
    region = ObjectRanges(h_c_frac=(0.4, 0.5))
    boundary = replace(spec, h_c=0.4 * spec.H)
    assert region.contains(boundary)
    assert region.violated_parameters(boundary) == []
    outside = replace(boundary, h_c=np.nextafter(boundary.h_c, -np.inf))
    assert not region.contains(outside)
    assert region.violated_parameters(outside) == ["h_c_frac"]
