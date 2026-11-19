"""SMC² example: infer hidden physics from pushes and plot each posterior.

Run: .venv/bin/python -m src.evaluation.system_identification.demos.smc2_demo --n-theta 64 --n-state 8
The true physics configures only the target; it is not passed to the estimator.
"""

import argparse
from contextlib import ExitStack

import matplotlib.pyplot as plt
import numpy as np

from src.agents.system_identification.smc2 import SMC2SysID
from src.agents.system_identification.sysid import SysIDResult
from src.evaluation.system_identification.shared.constants import (
    UPRIGHT_TILT_LIMIT,
    WARMUP,
)
from src.evaluation.system_identification.shared.metrics import (
    convert_to_physical_parameters,
)
from src.simulation.environment.env import PushEnv
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.object import CuboidSpec
from src.simulation.modelling.records import ActionRecord


class SMC2Demo:
    """Show how accepted pushes update the physics posterior."""

    def run(self) -> None:
        args = self.parse_args()
        self.spec = CuboidSpec(w=0.08, L=0.08, H=0.2, m=args.m, mu=args.mu, h_c=args.hc)
        with ExitStack() as resources:
            self.target = PushEnv(fixed_spec=self.spec)
            resources.callback(self.target.close)
            self.estimator = resources.enter_context(
                SMC2SysID(
                    (self.spec.w, self.spec.L, self.spec.H),
                    n_theta=args.n_theta,
                    n_state=args.n_state,
                    seed=args.seed,
                    proposal=args.proposal,
                )
            )
            self.history: list[ActionRecord] = []
            self._run_pushes(args)
        plt.show()

    @staticmethod
    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--n-theta", type=int, default=128)
        parser.add_argument("--n-state", type=int, default=32)
        parser.add_argument(
            "--proposal", choices=("fixed", "covariance"), default="covariance"
        )
        parser.add_argument("--seed", type=int, default=7)
        parser.add_argument("--m", type=float, default=1.0)
        parser.add_argument("--mu", type=float, default=0.3)
        parser.add_argument("--hc", type=float, default=0.1)
        parser.add_argument(
            "--push-forces",
            type=lambda s: [float(x) for x in s.split(",")],
            default="3.0,4.0,5.0,7.0",
        )
        return parser.parse_args()

    def _run_pushes(self, args: argparse.Namespace) -> None:
        for push_number, force in enumerate(args.push_forces, start=1):
            self.target.reset(seed=args.seed)
            record = self.target.push(
                PushAction(h=0.02, F=force, tau=0.1), warmup_steps=WARMUP
            )
            if max(abs(tilt) for tilt in record.traj.psi) > UPRIGHT_TILT_LIMIT:
                print(f"Skipping F={force}: trajectory outside upright regime")
                continue
            self.history.append(record)
            result = self.estimator.update(self.history)
            self._report_posterior(result, push_number, force)
        print(
            f"Resampling events: {self.estimator.resampling_count}; PMMH accepted: {self.estimator.acceptances}/{self.estimator.proposals}"
        )
        if self.history:
            print("Retained hypotheses:", result.hypotheses)
        print("Evaluation-only true theta:", self.spec.theta)

    def _report_posterior(
        self, result: SysIDResult, push_number: int, force: float
    ) -> None:
        parameters = convert_to_physical_parameters(
            self.estimator.particles, self.spec.H
        )
        mean = self.estimator.weights @ parameters
        std = np.sqrt(self.estimator.weights @ (parameters - mean) ** 2)
        print(
            f"F={force:.1f}: mean(m, mu, hc)={mean.round(4)}, std={std.round(4)}, pre-resample ESS={self.estimator.last_ess:.1f}, Highest posterior-mass theta(m, mu, hc)={np.asarray(result.point.theta).round(4)}"
        )
        figure = self.estimator.plot_particle_distribution(ground_truth=self.spec.theta)
        figure.suptitle(
            f"Physics posterior after push {push_number} · F={force:.1f} N · {self.estimator.n_theta} particles"
        )
        plt.show(block=False)
        plt.pause(0.001)


if __name__ == "__main__":
    SMC2Demo().run()
