"""Smoke checks for the force-versus-gantry validation harness."""
import numpy as np

from src.simulation.environment.env import ActionRanges
from src.simulation.execution.robot import PAD_HALF_HEIGHT
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.object import CuboidSpec
from src.simulation.tools.validation.gantry_vs_force import execute, sample_action


def test_sample_action_clamps_height_into_pad_window():
    spec = CuboidSpec(w=.08, L=.08, H=.12, m=1, mu=.5, h_c=.06)  # H < 0.2 forces clamp
    ranges = ActionRanges(force=(0.0, 15.0))
    rng = np.random.default_rng(0)
    for _ in range(50):
        act = sample_action(rng, spec, ranges)
        assert PAD_HALF_HEIGHT <= act.h <= spec.H - PAD_HALF_HEIGHT


def test_gantry_and_force_outcomes_on_one_push():
    spec = CuboidSpec(w=.08, L=.08, H=.20, m=1, mu=.5, h_c=.08)
    act = PushAction(h=.04, F=10, tau=.3)
    force = execute(spec, act, robot=False)
    robot = execute(spec, act, robot=True)
    assert force.dx > 0
    assert robot.contact
    assert robot.dx > 0
    assert robot.F_real > 0
    assert 0 < robot.tau_real <= act.tau + 1e-9
