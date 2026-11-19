"""Contact metrics use the paper's contact-window and timed-motion definitions."""

import json

import numpy as np
import pytest

from src.simulation.environment.env import PushEnv
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.object import CuboidSpec
from src.simulation.tools.validation.gantry_contact import (
    CONTACT_RANGES, measure_push, run, sample_action, summarize,
)


def test_sampled_height_fits_pad_and_policy_bounds() -> None:
    rng = np.random.default_rng(0)
    spec = CuboidSpec(w=.08, L=.08, H=.12, m=1., mu=.5, h_c=.06)
    for _ in range(50):
        action = sample_action(rng, spec)
        assert .02 <= action.h <= .10
        assert 0 <= action.F <= 15
        assert .05 <= action.tau <= .5


def test_summary_keeps_failures_and_uses_contact_window_force() -> None:
    rows = [dict(tau_real=.2, tau_cmd=.4, F_real=6., F_cmd=10., h_real=.039,
                 h_cmd=.04, displacement_m=.03, toppled=False,
                 execution_status="completed", stray_contact_steps=0),
            dict(tau_real=0., tau_cmd=.2, F_real=None, F_cmd=5., h_real=None,
                 h_cmd=.04, displacement_m=None, toppled=True,
                 execution_status="block_toppled", stray_contact_steps=2)]
    summary = summarize(rows)
    assert summary["push_count"] == 2
    assert summary["contact_failures"] == summary["execution_failures"] == 1
    assert summary["pushes_with_stray_contact"] == summary["toppled_pushes"] == 1
    assert summary["measured_requested_normal_force_ratio"]["mean"] == pytest.approx(.6)
    assert summary["contact_commanded_duration_ratio"]["mean"] == pytest.approx(.5)
    assert summary["signed_pad_height_error_m"]["mean"] == pytest.approx(-.001)
    assert summary["timed_displacement_m"]["count"] == 1
    json.dumps(summary, allow_nan=False)


def test_one_headless_contact_trial_saves_settings_and_measurements() -> None:
    result = run([0], objects=1, pushes=1)
    assert result["settings"]["reset_before_each_push"] is True
    assert result["summary"]["push_count"] == 1
    row = result["pushes"][0]
    assert row["execution_status"] == "completed"
    assert row["contact_made"]
    assert row["F_real"] > 0
    assert 0 < row["tau_real"] <= row["tau_cmd"] + .005
    assert row["displacement_m"] is not None
    json.dumps(result, allow_nan=False)


def test_failed_approach_produces_measurement_row(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = CuboidSpec(w=.08, L=.08, H=.2, m=1., mu=.5, h_c=.1)
    env = PushEnv(robot=True, fixed_spec=spec, action_ranges=CONTACT_RANGES)
    try:
        def fail_approach(*args: object) -> None:
            raise TimeoutError("test contact timeout")

        monkeypatch.setattr(env.executor, "_approach", fail_approach)
        row = measure_push(env, PushAction(h=.04, F=5., tau=.2), seed=0)
        assert row["execution_status"] == "contact_timeout"
        assert row["displacement_m"] is None
        assert row["F_real"] is None
        assert row["tau_real"] == 0
        json.dumps(row, allow_nan=False)
    finally:
        env.close()


@pytest.mark.parametrize("seeds,objects,pushes", [([], 1, 1), ([-1], 1, 1),
                                               ([0], 0, 1), ([0], 1, 0)])
def test_invalid_counts_fail_before_creating_world(
    seeds: list[int], objects: int, pushes: int,
) -> None:
    with pytest.raises(ValueError):
        run(seeds, objects, pushes)
