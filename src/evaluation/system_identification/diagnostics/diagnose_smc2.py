"""
Diagnose particle depletion using an existing benchmark training dataset.

No held-out data are used to select settings. Zero start uncertainty is an
ablation appropriate to noise-free simulated state measurements, not a fix
for real sensors. All variants retain the same observation likelihood scales.
Estimator probes and ablations exclude warm-up; they do not reproduce the
historical include-warm-up runs. The COM profile compares both windows locally.
"""

import argparse
import json
import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from src.agents.system_identification.smc2 import CHANNELS, SMC2SysID
from src.evaluation.system_identification.shared.common import (
    ReplayCountingSimulator,
    load_action_records,
)
from src.evaluation.system_identification.shared.metrics import summarize_particles
from src.simulation.execution.force_push import DenseTrajectory
from src.simulation.modelling.object import CuboidSpec
from src.simulation.modelling.records import ActionRecord


def measure_likelihood_variation(
    spec: CuboidSpec,
    record: ActionRecord,
    sim: ReplayCountingSimulator,
    repeats: int,
) -> list[dict]:
    """Measure likelihood variation from sampled starts at fixed true physics."""
    configurations = [
        ("default_8", {"n_state": 8}),
        ("default_64", {"n_state": 64}),
        ("smaller_start_8", {"n_state": 8, "start_std": (0.001, 0.0002, 0.001, 0.002)}),
        ("exact_start", {"n_state": 1, "start_std": (0, 0, 0, 0)}),
    ]
    results = []
    for name, settings in configurations:
        row = _measure_likelihood_variant(spec, record, sim, repeats, settings)
        row["variant"] = name
        results.append(row)
        print(
            f"Likelihood {name}: std(log L)={row['log_likelihood_std']:.2f}, mean inner ESS={row['mean_inner_ess']:.2f}",
            flush=True,
        )
    return results


def _measure_likelihood_variant(
    spec: CuboidSpec,
    record: ActionRecord,
    sim: ReplayCountingSimulator,
    repeats: int,
    settings: dict,
) -> dict:
    estimator = SMC2SysID(
        (spec.w, spec.L, spec.H), n_theta=1, proposal="fixed", sim=sim, **settings
    )
    truth = np.array([np.log(spec.m), spec.mu, spec.h_c / spec.H])
    estimates, inner_ess = [], []
    for seed in range(1, repeats + 1):
        estimator.rng = np.random.default_rng(seed)
        inner, increment = estimator._evaluate_record(truth, record)
        estimates.append(increment)
        inner_ess.append(float(1 / np.sum(inner.weights**2)))
    return {
        "settings": settings,
        "log_likelihood_estimates": estimates,
        "log_likelihood_std": float(np.std(estimates, ddof=1)),
        "mean_inner_ess": float(np.mean(inner_ess)),
    }


def measure_com_likelihood(
    spec: CuboidSpec,
    records: list[ActionRecord],
    sim: ReplayCountingSimulator,
) -> list[dict]:
    """Compare warm-up and pushing scores at fixed true mass and friction.

    This conditional COM profile is not a marginal posterior.
    """
    estimator = SMC2SysID(
        (spec.w, spec.L, spec.H), n_theta=1, sim=sim, proposal="fixed"
    )
    return [
        _measure_com_height(estimator, records, replace(spec, h_c=height))
        for height in np.linspace(0.2 * spec.H, 0.8 * spec.H, 25)
    ]


def _measure_com_height(
    estimator: SMC2SysID,
    records: list[ActionRecord],
    candidate: CuboidSpec,
) -> dict:
    full, pushing = 0.0, 0.0
    errors = {channel: {"warmup": 0.0, "pushing": 0.0} for channel in CHANNELS}
    for record in records:
        predicted = estimator.sim.replay(candidate, record)
        observed = estimator._trajectory_array(record.traj)[:: estimator.sample_stride]
        record_full, record_pushing = _accumulate_scoring_errors(
            estimator, record, predicted, observed, errors
        )
        full += record_full
        pushing += record_pushing
    return {
        "h_c": float(candidate.h_c),
        "full_log_likelihood": full,
        "pushing_log_likelihood": pushing,
        "standardized_squared_errors": errors,
    }


def _accumulate_scoring_errors(
    estimator: SMC2SysID,
    record: ActionRecord,
    predicted: DenseTrajectory,
    observed: np.ndarray,
    errors: dict,
) -> tuple[float, float]:
    predicted_states = estimator._trajectory_array(predicted)[
        :: estimator.sample_stride
    ]
    indices = np.arange(len(record.traj.t))[:: estimator.sample_stride]
    pushing_mask = indices >= record.warmup_steps
    full = estimator._compute_log_likelihood(observed, predicted_states)
    pushing = estimator._compute_log_likelihood(
        observed[pushing_mask], predicted_states[pushing_mask]
    )
    squared_errors = ((observed - predicted_states) / estimator.observation_std) ** 2
    for index, channel in enumerate(CHANNELS):
        errors[channel]["warmup"] += float(np.sum(squared_errors[~pushing_mask, index]))
        errors[channel]["pushing"] += float(np.sum(squared_errors[pushing_mask, index]))
    return full, pushing


def run_ablation_trial(
    name: str,
    settings: dict,
    seed: int,
    spec: CuboidSpec,
    records: list[ActionRecord],
    sim: ReplayCountingSimulator,
) -> list[dict]:
    estimator = SMC2SysID(
        (spec.w, spec.L, spec.H), seed=seed, sim=sim, proposal="fixed", **settings
    )
    true_z = np.array([np.log(spec.m), spec.mu, spec.h_c / spec.H])
    initial = LikelihoodDiagnostics.measure_initial_collapse(
        estimator, records[0], true_z
    )
    rows = []
    for push_count in range(1, len(records) + 1):
        row = _update_and_measure_ablation(estimator, records[:push_count], spec)
        row.update(
            variant=name, settings=settings, seed=seed, step=push_count, initial=initial
        )
        rows.append(row)
        acceptance = row["acceptance_rate"]
        acceptance_text = "n/a" if acceptance is None else f"{acceptance:.3f}"
        print(
            f"{name} seed={seed} step={push_count}: ESS={estimator.last_ess:.1f}, unique={row['unique_particles']}, accept={acceptance_text}, theta={np.asarray(row['theta']).round(4)}",
            flush=True,
        )
    return rows


def _update_and_measure_ablation(
    estimator: SMC2SysID,
    history: list[ActionRecord],
    spec: CuboidSpec,
) -> dict:
    before = estimator.sim.replay_count
    start = time.perf_counter()
    result = estimator.update(history)
    elapsed = time.perf_counter() - start
    return {
        "theta": list(result.point.theta),
        "absolute_error": np.abs(np.asarray(result.point.theta) - spec.theta).tolist(),
        "posterior": summarize_particles(estimator, spec),
        "pre_resample_ess": estimator.last_ess,
        "unique_particles": len(np.unique(estimator.particles, axis=0)),
        "acceptance_rate": estimator.acceptances / estimator.proposals
        if estimator.proposals
        else None,
        "fit_replays": estimator.sim.replay_count - before,
        "fit_seconds": elapsed,
    }


class LikelihoodDiagnostics:
    def run(self) -> None:
        self._parse_args()
        self._prepare_inputs()
        self._run_diagnostics()
        self._finish()

    def _parse_args(self) -> None:
        self.parser = argparse.ArgumentParser(description=__doc__)
        self.parser.add_argument(
            "--dataset",
            type=Path,
            default=Path(
                "artifacts/sysid_benchmark_initial/dataset_varied_magnitudes.json"
            ),
        )
        self.parser.add_argument(
            "--config",
            type=Path,
            default=Path("artifacts/sysid_benchmark_initial/config.json"),
        )
        self.parser.add_argument(
            "--out", type=Path, default=Path("artifacts/smc2_diagnosis")
        )
        self.parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
        self.parser.add_argument("--repeats", type=int, default=32)
        self.args = self.parser.parse_args()

    def _prepare_inputs(self) -> None:
        if self.args.repeats < 2:
            self.parser.error("--repeats must be at least 2")
        self.spec = CuboidSpec(**json.loads(self.args.config.read_text())["object"])
        self.records = load_action_records(self.args.dataset, accepted_only=True)
        if not self.records:
            self.parser.error("Dataset has no accepted records")
        self.args.out.mkdir(parents=True, exist_ok=False)
        (self.args.out / "config.json").write_text(
            json.dumps(
                {
                    "dataset": str(self.args.dataset),
                    "object": json.loads(self.args.config.read_text())["object"],
                    "seeds": self.args.seeds,
                    "repeats": self.args.repeats,
                    "note": __doc__,
                },
                indent=2,
            )
        )

    def _run_diagnostics(self) -> None:
        with ExitStack() as stack:
            sim = ReplayCountingSimulator()
            stack.callback(sim.close)
            self._write_oracle_residuals(sim)
            variants = self._get_ablation_variants()
            with (self.args.out / "ablations.jsonl").open("w") as stream:
                for name, settings in variants.items():
                    for seed in self.args.seeds:
                        for row in run_ablation_trial(
                            name,
                            settings,
                            seed,
                            self.spec,
                            self.records,
                            sim,
                        ):
                            stream.write(json.dumps(row) + "\n")
                        stream.flush()

    def _finish(self) -> None:
        print(f"Saved diagnosis to {self.args.out}")

    def _write_probes_and_measure_residuals(
        self, sim: ReplayCountingSimulator
    ) -> dict[str, list[float]]:
        probes = measure_likelihood_variation(
            self.spec, self.records[0], sim, self.args.repeats
        )
        (self.args.out / "likelihood_probe.json").write_text(
            json.dumps(probes, indent=2)
        )
        profiles = measure_com_likelihood(self.spec, self.records, sim)
        (self.args.out / "com_profiles.json").write_text(json.dumps(profiles, indent=2))
        residuals = {channel: [] for channel in CHANNELS}
        for record in self.records:
            prediction = sim.replay(self.spec, record)
            for channel in CHANNELS:
                residuals[channel].extend(
                    np.asarray(getattr(record.traj, channel))
                    - np.asarray(getattr(prediction, channel))
                )
        return residuals

    def _write_oracle_residuals(self, sim: ReplayCountingSimulator) -> None:
        residuals = self._write_probes_and_measure_residuals(sim)
        (self.args.out / "oracle_residuals.json").write_text(
            json.dumps(
                {
                    channel: {
                        "rmse": float(np.sqrt(np.mean(np.asarray(values) ** 2))),
                        "mean": float(np.mean(values)),
                    }
                    for channel, values in residuals.items()
                },
                indent=2,
            )
        )

    @staticmethod
    def _get_ablation_variants() -> dict[str, dict]:
        variants = {
            "baseline": {"n_theta": 64, "n_state": 8},
            "more_inner": {"n_theta": 64, "n_state": 64},
            "exact_start": {"n_theta": 64, "n_state": 1, "start_std": (0, 0, 0, 0)},
            "exact_start_more_outer": {
                "n_theta": 256,
                "n_state": 1,
                "start_std": (0, 0, 0, 0),
            },
            "exact_start_more_moves": {
                "n_theta": 64,
                "n_state": 1,
                "start_std": (0, 0, 0, 0),
                "mcmc_steps": 20,
            },
        }
        return variants

    @staticmethod
    def measure_initial_collapse(
        estimator: SMC2SysID, record: ActionRecord, true_z: np.ndarray
    ) -> dict:
        """Measure first-update ESS before resampling without changing run RNG."""
        from copy import deepcopy

        state = deepcopy(estimator.rng.bit_generator.state)
        initial = estimator.particles
        increments = [estimator._evaluate_record(z, record)[1] for z in initial]
        estimator.rng.bit_generator.state = state
        weights = np.exp(increments - logsumexp(increments))
        return {
            "first_update_ess": float(1 / np.sum(weights**2)),
            "max_initial_weight": float(np.max(weights)),
            "first_log_likelihood_range": float(np.ptp(increments)),
            "nearest_prior_scaled_distance": float(
                np.min(
                    np.linalg.norm(
                        (initial - true_z) / np.array([np.log(3 / 0.2), 1, 0.6]), axis=1
                    )
                )
            ),
        }


if __name__ == "__main__":
    LikelihoodDiagnostics().run()
