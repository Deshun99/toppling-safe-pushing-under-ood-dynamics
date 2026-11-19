"""Compare SMC² and CMA-ES on shared excitation data and held-out pushes.

Run with --smoke to check the pipeline before spending a larger replay budget.
Seeds vary inference randomness; target data are deterministic, noise-free
simulator measurements shared across seeds. Coverage is therefore conditional
on this object and dataset, not a population calibration guarantee.
"""

import argparse
import csv
import json
import time
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

from src.evaluation.system_identification.shared.excitation import (
    collect_records,
    get_regimes,
)
from src.evaluation.system_identification.shared.metrics import (
    evaluate_predictions,
    summarize_particles,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.agents.simulator.internal_sim import InternalSim
from src.agents.system_identification.smc2 import SMC2SysID
from src.agents.system_identification.sysid import SysID
from src.evaluation.system_identification.shared.common import ReplayCountingSimulator
from src.evaluation.system_identification.shared.constants import (
    PARAMETERS,
    SPEC,
    SUMMARY_FIELDS,
    UPRIGHT_TILT_LIMIT,
)
from src.simulation.modelling.actions import PushAction
from src.simulation.modelling.records import ActionRecord


class EstimatorTrial:
    """Compare two estimators on successive prefixes of one recorded sequence."""

    def __init__(
        self,
        args: argparse.Namespace,
        sequence_name: str,
        seed: int,
        training_records: list[ActionRecord],
        evaluation_records: list[ActionRecord],
        output: Path,
    ) -> None:
        self.args = args
        self.sequence_name = sequence_name
        self.seed = seed
        self.training_records = training_records
        self.evaluation_records = evaluation_records
        self.output = output

    def run(self) -> list[dict]:
        rows = []
        with ExitStack() as resources:
            smc, cma, prediction_simulator = self._create_estimators(resources)
            for push_count in range(1, len(self.training_records) + 1):
                for method, estimator in (("smc2", smc), ("cma_es", cma)):
                    row = self._update_estimator(
                        estimator, prediction_simulator, method, push_count
                    )
                    if method == "smc2":
                        row.update(self._summarize_smc(smc, push_count))
                        self._save_posterior(smc, push_count)
                    rows.append(row)
                    print(
                        f"{self.sequence_name} seed={self.seed} step={push_count} {method}: "
                        f"theta={np.asarray(row['theta']).round(4)} replays={row['fit_replays']}",
                        flush=True,
                    )
        return rows

    def _update_estimator(
        self,
        estimator: SMC2SysID | SysID,
        prediction_simulator: InternalSim,
        method: str,
        push_count: int,
    ) -> dict:
        replays_before = estimator.sim.replay_count
        start = time.perf_counter()
        result = estimator.update(self.training_records[:push_count])
        elapsed = time.perf_counter() - start
        parameters = np.asarray(result.point.theta)
        mass, friction, com_height = parameters
        estimated_object = replace(SPEC, m=mass, mu=friction, h_c=com_height)
        return {
            "regime": self.sequence_name,
            "seed": self.seed,
            "step": push_count,
            "method": method,
            "theta": parameters.tolist(),
            "absolute_error": np.abs(parameters - SPEC.theta).tolist(),
            "relative_mass_error": float(abs(mass - SPEC.m) / SPEC.m),
            "com_error_fraction_height": float(abs(com_height - SPEC.h_c) / SPEC.H),
            "fit_seconds": elapsed,
            "fit_replays": estimator.sim.replay_count - replays_before,
            "cumulative_fit_replays": estimator.sim.replay_count,
            "held_out": evaluate_predictions(
                prediction_simulator, estimated_object, self.evaluation_records
            ),
        }

    def _create_estimators(
        self,
        resources: ExitStack,
    ) -> tuple[SMC2SysID, SysID, InternalSim]:
        smc_simulator = ReplayCountingSimulator()
        resources.callback(smc_simulator.close)
        cma = self._create_cma_estimator(resources)
        prediction_simulator = InternalSim()
        resources.callback(prediction_simulator.close)
        smc = SMC2SysID(
            (SPEC.w, SPEC.L, SPEC.H),
            n_theta=self.args.n_theta,
            n_state=self.args.n_state,
            seed=self.seed,
            sim=smc_simulator,
            sample_stride=self.args.sample_stride,
            **({"start_std": (0, 0, 0, 0)} if self.args.known_start else {}),
            temper_updates=self.args.temper_updates,
            proposal=self.args.proposal,
        )
        return smc, cma, prediction_simulator

    def _create_cma_estimator(self, resources: ExitStack) -> SysID:
        estimator = SysID(
            (SPEC.w, SPEC.L, SPEC.H),
            iterations=self.args.cma_iterations,
            restarts=self.args.cma_restarts,
            seed=self.seed,
            sample_stride=self.args.sample_stride,
        )
        resources.callback(estimator.close)
        estimator.sim.close()
        estimator.sim = ReplayCountingSimulator()
        return estimator

    def _save_posterior(self, estimator: SMC2SysID, push_count: int) -> None:
        stem = (
            self.output
            / f"{self.sequence_name.replace(' ', '_')}_seed{self.seed}_step{push_count}"
        )
        np.savez_compressed(
            stem.with_suffix(".npz"), z=estimator.particles, weights=estimator.weights
        )
        figure = estimator.plot_particle_distribution(ground_truth=SPEC.theta)
        figure.suptitle(
            f"{self.sequence_name} · seed {self.seed} · {push_count} accepted pushes"
        )
        figure.savefig(stem.with_suffix(".png"), dpi=100)
        plt.close(figure)

    @staticmethod
    def _summarize_smc(estimator: SMC2SysID, push_count: int) -> dict:
        return {
            "posterior": summarize_particles(estimator),
            "pre_resample_ess": estimator.last_ess,
            "unique_particles": len(np.unique(estimator.particles, axis=0)),
            "resampling_count": estimator.resampling_count,
            "pmmh_acceptances": estimator.acceptances,
            "pmmh_proposals": estimator.proposals,
            "tempering_stages": [
                event
                for event in estimator.tempering_history
                if event["step"] == push_count
            ],
        }


def write_comparison_report(rows: list[dict], output: Path) -> None:
    """Save tabular learning curves and final-step summaries across seeds."""
    table = []
    for row in rows:
        table.append(
            {key: row[key] for key in SUMMARY_FIELDS}
            | row["held_out"]
            | {
                f"absolute_error_{name}": error
                for name, error in zip(PARAMETERS, row["absolute_error"])
            }
        )
    if not table:
        return
    fields = sorted(set().union(*(row.keys() for row in table)))
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    summary = _comparison_report_summarize_final_updates(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))


def _comparison_report_summarize_final_trials(
    final: dict, sequence_name: str, method: str
) -> tuple[list[dict], dict]:
    trials = [
        row for (r, m, _), row in final.items() if (r, m) == (sequence_name, method)
    ]
    group = {
        "regime": sequence_name,
        "method": method,
        "trials": len(trials),
        "mean_absolute_error": np.mean(
            [row["absolute_error"] for row in trials], axis=0
        ).tolist(),
        "mean_held_out": {
            key: float(np.mean([row["held_out"][key] for row in trials]))
            for key in trials[0]["held_out"]
        },
    }
    return trials, group


def _comparison_report_summarize_final_updates(rows: list[dict]) -> list[dict]:
    final = {}
    for row in rows:
        final[row["regime"], row["method"], row["seed"]] = row
    summary = []
    groups = sorted({(row["regime"], row["method"]) for row in rows})
    for sequence_name, method in groups:
        trials, group = _comparison_report_summarize_final_trials(
            final, sequence_name, method
        )
        if method == "smc2":
            group["conditional_coverage_95"] = {
                name: float(
                    np.mean([row["posterior"][name]["covers_truth"] for row in trials])
                )
                for name in PARAMETERS
            }
        summary.append(group)
    return summary


class EstimatorComparison:
    def run(self) -> None:
        self._parse_args()
        self._validate_options()
        self._prepare_output()
        evaluation_records = self._collect_evaluation_records()
        all_rows = self._compare_push_sequences(evaluation_records)
        self._write_report(all_rows)

    def _parse_args(self) -> None:
        self.parser = argparse.ArgumentParser(description=__doc__)
        self.parser.add_argument(
            "--out", type=Path, default=Path("artifacts/sysid_benchmark")
        )
        self.parser.add_argument(
            "--regimes",
            nargs="+",
            choices=list(get_regimes()),
            default=["varied magnitudes", "varied heights", "tilt boundary"],
        )
        self._add_inference_arguments()
        self.parser.add_argument("--cma-iterations", type=int, default=20)
        self.parser.add_argument("--cma-restarts", type=int, default=2)
        self.parser.add_argument("--smoke", action="store_true")
        self.parser.add_argument(
            "--temper-updates",
            action="store_true",
            help="Gradually introduce each push; requires --known-start",
        )
        self.args = self.parser.parse_args()

    def _validate_options(self) -> None:
        if self.args.smoke:
            self.args.regimes = ["varied magnitudes", "varied heights"]
            self.args.seeds = [1, 2]
            self.args.n_theta, self.args.n_state = (8, 2)
            self.args.cma_iterations, self.args.cma_restarts = (2, 0)
        if self.args.known_start:
            self.args.n_state = 1
        if self.args.temper_updates and (not self.args.known_start):
            self.parser.error("--temper-updates requires --known-start")
        if (
            min(
                self.args.n_theta,
                self.args.n_state,
                self.args.cma_iterations,
                self.args.sample_stride,
            )
            < 1
            or self.args.cma_restarts < 0
        ):
            self.parser.error(
                "Particle counts and iterations must be positive; restarts nonnegative"
            )

    def _prepare_output(self) -> None:
        self.args.out.mkdir(parents=True, exist_ok=False)
        config = vars(self.args) | {
            "out": str(self.args.out),
            "object": asdict(SPEC),
            "upright_tilt_limit": UPRIGHT_TILT_LIMIT,
            "note": __doc__,
        }
        (self.args.out / "config.json").write_text(json.dumps(config, indent=2))

    def _collect_evaluation_records(self) -> list[ActionRecord]:
        evaluation_actions = (
            [
                PushAction(h=0.25 * SPEC.H, F=4.25, tau=0.18),
                PushAction(h=0.55 * SPEC.H, F=4.75, tau=0.22),
                PushAction(h=0.7 * SPEC.H, F=4.25, tau=0.25),
            ]
            if self.args.development
            else [
                PushAction(h=0.2 * SPEC.H, F=4.5, tau=0.2),
                PushAction(h=0.4 * SPEC.H, F=6.0, tau=0.15),
                PushAction(h=0.75 * SPEC.H, F=12.0, tau=0.3),
            ]
        )
        evaluation_records, audit = collect_records(
            evaluation_actions, independent=True
        )
        (self.args.out / "held_out.json").write_text(json.dumps(audit, indent=2))
        oracle_sim = InternalSim()
        try:
            oracle = evaluate_predictions(oracle_sim, SPEC, evaluation_records)
        finally:
            oracle_sim.close()
        (self.args.out / "oracle.json").write_text(json.dumps(oracle, indent=2))
        return evaluation_records

    def _compare_push_sequences(
        self, evaluation_records: list[ActionRecord]
    ) -> list[dict]:
        all_rows = []
        with (self.args.out / "metrics.jsonl").open("w") as stream:
            for sequence_name in self.args.regimes:
                training_records = self._collect_training_records(sequence_name)
                if not training_records:
                    print(
                        f"Skipping {sequence_name}: no upright training records",
                        flush=True,
                    )
                    continue
                for seed in self.args.seeds:
                    rows = EstimatorTrial(
                        self.args,
                        sequence_name,
                        seed,
                        training_records,
                        evaluation_records,
                        self.args.out,
                    ).run()
                    all_rows.extend(rows)
                    for row in rows:
                        stream.write(json.dumps(row) + "\n")
                    stream.flush()
        return all_rows

    def _write_report(self, all_rows: list[dict]) -> None:
        write_comparison_report(all_rows, self.args.out)
        print(f"Saved benchmark to {self.args.out}")

    def _add_inference_arguments(self) -> None:
        self.parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
        self.parser.add_argument("--n-theta", type=int, default=128)
        self.parser.add_argument("--n-state", type=int, default=32)
        self.parser.add_argument(
            "--proposal", choices=("fixed", "covariance"), default="covariance"
        )
        self.parser.add_argument(
            "--known-start",
            action="store_true",
            help="Noise-free reference: zero start uncertainty and one inner particle",
        )
        self.parser.add_argument(
            "--sample-stride",
            type=int,
            default=12,
            help="Same observation indices for SMC² and CMA-ES",
        )
        self.parser.add_argument(
            "--development",
            action="store_true",
            help="Use separate development evaluation actions, not the original held-out set",
        )

    def _collect_training_records(self, sequence_name: str) -> list[ActionRecord]:
        actions = get_regimes()[sequence_name]
        if self.args.smoke:
            actions = actions[:2]
        training_records, audit = collect_records(actions, independent=False)
        name = sequence_name.replace(" ", "_")
        (self.args.out / f"dataset_{name}.json").write_text(json.dumps(audit, indent=2))
        oracle_sim = InternalSim()
        try:
            training_oracle = (
                evaluate_predictions(oracle_sim, SPEC, training_records)
                if training_records
                else {}
            )
        finally:
            oracle_sim.close()
        (self.args.out / f"oracle_{name}.json").write_text(
            json.dumps(training_oracle, indent=2)
        )
        return training_records


if __name__ == "__main__":
    EstimatorComparison().run()
