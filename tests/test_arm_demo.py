"""Goal termination checks using deterministic motion, independent of physics."""
import unittest
from types import SimpleNamespace
from src.simulation.tools.demos.arm_demo import push_forward


class GoalDemoTests(unittest.TestCase):
    def run_demo(self, step=.048, start=0, tilt=0, pushes=20, duration=.6):
        state = [start, tilt, 0, 0]
        block = SimpleNamespace(state=lambda: state, spec=SimpleNamespace(w=.08, crit_tilt=.4))
        calls = []
        def execute(block, height, force, duration):
            calls.append(duration)
            state[0] += step
        executor = SimpleNamespace(pad_ik=lambda x, z: (x, z), execute=execute)
        args = SimpleNamespace(goal_x=.5, height=.04, force=10, duration=duration, pushes=pushes)
        push_forward(executor, block, args)
        return calls, state

    def test_continues_beyond_three_pushes_until_goal_band(self):
        calls, state = self.run_demo()
        self.assertEqual(len(calls), 10)
        self.assertLessEqual(abs(state[0] - .5), .05)

    def test_budget_limits_stalled_block(self):
        calls, _ = self.run_demo(step=0, pushes=4)
        self.assertEqual(len(calls), 4)

    def test_no_push_when_already_at_goal_toppled_or_overshot(self):
        for start, tilt in [(.5, 0), (0, .5), (.6, 0)]:
            calls, _ = self.run_demo(start=start, tilt=tilt)
            self.assertEqual(calls, [])

    def test_shortens_large_duration_near_goal(self):
        calls, _ = self.run_demo(start=.44, duration=10, pushes=1)
        self.assertAlmostEqual(calls[0], .75)
