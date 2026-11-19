"""Unit tests for the B0 conservative analytic controller."""
import pytest

from src.agents.baselines.conservative_controller import ConservativeController
from src.simulation.environment.env import ActionRanges

GEOMETRY = (0.08, 0.08, 0.20)
THETA = (1.0, 0.5, 0.08)
RANGES = ActionRanges(force=(0.0, 15.0))
CONTROLLER = ConservativeController()


def choose(distance, decisions_left=10, theta=THETA, geometry=GEOMETRY,
           ranges=RANGES, controller=CONTROLLER):
    return controller.choose_action(geometry, theta, distance, decisions_left, ranges)


def test_stops_inside_goal_band_and_past_it():
    for distance in (0.05, 0.0, -0.5):
        action = choose(distance)
        assert action.F == 0.0


def test_force_grows_with_remaining_distance():
    near = choose(0.1)
    far = choose(0.5)
    assert far.F > near.F > 0


def test_velocity_cap_binds_for_large_distances():
    action = choose(10.0, decisions_left=1)
    expected = THETA[0] * (
        CONTROLLER.velocity_cap / RANGES.duration[1] + THETA[1] * 9.81
    )
    assert action.F == pytest.approx(expected, rel=1e-9)


def test_tipping_bound_caps_force_when_estimate_places_com_under_the_pad():
    controller = ConservativeController(h_fraction=0.5)  # h = 0.10 > h_c = 0.02
    action = controller.choose_action(
        GEOMETRY, (1.0, 0.5, 0.02), 1.0, 10, RANGES
    )
    tipping_force = 1.0 * 9.81 * (0.04 - 0.5 * 0.02) / (0.10 - 0.02)
    assert action.F == pytest.approx(0.5 * tipping_force, rel=1e-9)


def test_force_is_capped_by_the_action_range():
    ranges = ActionRanges(force=(0.0, 2.0))
    action = choose(10.0, decisions_left=1, ranges=ranges)
    assert action.F == 2.0


def test_implausible_estimates_stay_safe():
    action = choose(0.5, theta=(-1.0, 0.5, 0.08))
    assert action.F == 0.0
    action = choose(0.5, theta=(0.0, 0.5, 0.08))
    assert action.F == 0.0


def test_decisions_left_never_divides_by_zero():
    assert choose(0.3, decisions_left=0) == choose(0.3, decisions_left=1)


def test_uses_max_duration_and_lowest_height():
    action = choose(0.5)
    assert action.h == 0.1 * GEOMETRY[2]
    assert action.tau == RANGES.duration[1]


def test_format_realized_handles_missing_contact():
    """Zero-force pushes have no contact: realized fields are None, not crash."""
    from src.evaluation.baselines.evaluate_conservative import _format_realized
    from src.simulation.execution.robot import ExecutionReport

    assert _format_realized(None) == "realized n/a"
    no_contact = ExecutionReport(h_cmd=.02, F_cmd=0.0, tau_cmd=.05)
    assert _format_realized(no_contact) == "realized (h=n/a, F=n/a, tau=0.00)"
    full = ExecutionReport(h_cmd=.02, F_cmd=10, tau_cmd=.5, contact_made=True,
                           h_real=.02, F_real=3.1, tau_real=.5)
    assert _format_realized(full) == "realized (h=0.020, F=3.1, tau=0.50)"


def test_adapt_episode_ends_cleanly_on_execution_failure():
    """A failed push carries no action_record; the episode must end, not crash."""
    from src.agents.system_identification.sysid import SysID
    from src.evaluation.baselines.evaluate_conservative import _adapt_episode
    from src.simulation.environment.env import ActionRanges, PushEnv
    from src.simulation.execution.robot import TravelLimitError
    from src.simulation.modelling.object import CuboidSpec

    spec = CuboidSpec(w=.08, L=.08, H=.20, m=1, mu=.5, h_c=.08)
    env = PushEnv(robot=True, fixed_spec=spec, target_distance=0.5)
    try:

        def fail(*args, **kwargs):
            raise TravelLimitError("travel limit (test)")

        env.executor._check_forward_travel = fail
        geometry = (spec.w, spec.L, spec.H)
        sysid = SysID(geometry=geometry)
        row, _, _ = _adapt_episode(
            env, geometry, ConservativeController(),
            ActionRanges(force=(0.0, 15.0)), sysid, [], (1.0, 0.5, 0.095),
            None, seed=0, history_window=20, replay_budget=64,
        )
        assert row.outcome == "execution_failed"
        assert row.push_count == 1
    finally:
        env.close()
