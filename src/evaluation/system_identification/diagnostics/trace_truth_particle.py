"""Oracle diagnostic: insert exact physics and trace resampling and moves.

This deliberately privileged initialization is not a performance benchmark.
Paired controls retain the original random particle. Diagnostic tracing adds
no inference RNG draws and does not modify the production estimator.
"""

import argparse
import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

from src.agents.system_identification.sysid import SysIDResult
from src.evaluation.system_identification.shared.common import load_training_sequences
from src.evaluation.system_identification.shared.grid import (
    integrate_marginal,
    load_checked_grid,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

from src.agents.system_identification.smc2 import SMC2SysID
from src.agents.system_identification.sysid import Z_HI, Z_LO
from src.evaluation.system_identification.shared.common import ReplayCountingSimulator
from src.simulation.modelling.object import CuboidSpec
from src.simulation.modelling.records import ActionRecord

PRIOR_SCALED_RADIUS = 0.05
COM_BAND_HALF_WIDTH = 0.005  # Five millimetres on either side of true COM.


class TracedSMC2(SMC2SysID):
    """Observe production resampling/move methods without changing their RNG."""

    def __init__(
        self,
        spec: CuboidSpec,
        truth_scores: np.ndarray,
        *,
        inject_truth: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__((spec.w, spec.L, spec.H), **kwargs)
        self.truth = np.array([np.log(spec.m), spec.mu, spec.h_c / spec.H])
        self.truth_scores = truth_scores
        self.height = spec.H
        self.inject_truth = inject_truth
        self.roots = np.arange(self.n_theta)
        if inject_truth:
            self._z[0] = self.truth
        self.step = 0
        self.beta = 1.0
        self.targets = np.zeros(self.n_theta)
        self.events: list[dict] = []

    def summarize(self) -> dict:
        exact = np.all(self._z == self.truth, axis=1)
        descendants = (
            self.roots == 0 if self.inject_truth else np.zeros(self.n_theta, dtype=bool)
        )
        distance = np.linalg.norm((self._z - self.truth) / (Z_HI - Z_LO), axis=1)
        com = self._z[:, 2] * self.height
        near_com = abs(com - self.truth[2] * self.height) <= COM_BAND_HALF_WIDTH
        truth_target = (
            self.truth_scores[: self.step - 1].sum()
            + self.beta * self.truth_scores[self.step - 1]
            if self.step
            else 0.0
        )
        full = np.array([state.log_likelihood for state in self._filters])
        return {
            **self._get_rank_metrics(truth_target, full),
            **self._get_population_metrics(exact, descendants, distance, near_com),
        }

    def _select_tempering_beta(self, beta: float, increments: np.ndarray) -> float:
        next_beta = super()._select_tempering_beta(beta, increments)
        self.beta = next_beta
        full = np.array([state.log_likelihood for state in self._filters])
        self.targets = full - increments + next_beta * increments
        return next_beta

    def _resample_particles(self) -> np.ndarray:
        before = self.summarize()
        ancestors = super()._resample_particles()
        self.roots = self.roots[ancestors]
        self.targets = self.targets[ancestors]
        self.events.append(
            {
                "kind": "resample",
                "before": before,
                "after": self.summarize(),
                "ancestors": ancestors.tolist(),
            }
        )
        return ancestors

    def record_moves(
        self, before: np.ndarray, proposals: int, acceptances: int
    ) -> None:
        moved = np.flatnonzero(np.any(self._z != before, axis=1))
        self.events.append(
            {
                "kind": "move",
                "step": self.step,
                "beta": self.beta,
                "proposals": self.proposals - proposals,
                "accepted": self.acceptances - acceptances,
                "moved_indices": moved.tolist(),
                "after": self.summarize(),
            }
        )

    def _resample_move(self) -> None:
        self.beta = 1.0
        self.targets = np.array([state.log_likelihood for state in self._filters])
        proposals, acceptances = (self.proposals, self.acceptances)
        original = self._z.copy()
        super()._resample_move()
        ancestors = np.asarray(self.events[-1]["ancestors"])
        self.targets = np.array([state.log_likelihood for state in self._filters])
        self.record_moves(original[ancestors], proposals, acceptances)

    def _move_tempered_particles(
        self,
        record: ActionRecord,
        beta: float,
        past: np.ndarray,
        increments: np.ndarray,
    ) -> None:
        before = self._z.copy()
        proposals, acceptances = (self.proposals, self.acceptances)
        super()._move_tempered_particles(record, beta, past, increments)
        self.targets = past + beta * increments
        self.record_moves(before, proposals, acceptances)

    def update(self, history: Iterable[ActionRecord]) -> SysIDResult:
        records = list(history)
        self.step = len(records)
        result = super().update(records)
        self.beta = 1.0
        self.targets = np.array([state.log_likelihood for state in self._filters])
        self.events.append({"kind": "update", "after": self.summarize()})
        return result

    def _get_rank_metrics(self, truth_target: float, full: np.ndarray) -> dict:
        return {
            "step": self.step,
            "beta": self.beta,
            "truth_increment": float(self.truth_scores[self.step - 1])
            if self.step
            else None,
            "truth_cumulative_log_likelihood": float(
                self.truth_scores[: self.step].sum()
            ),
            "truth_target_rank": int(
                1 + np.count_nonzero(self.targets > truth_target + 1e-09)
            ),
            "truth_full_likelihood_rank": int(
                1
                + np.count_nonzero(full > self.truth_scores[: self.step].sum() + 1e-09)
            ),
            "best_minus_truth_log_target": float(self.targets.max() - truth_target),
        }

    def _get_population_metrics(
        self,
        exact: np.ndarray,
        descendants: np.ndarray,
        distance: np.ndarray,
        near_com: np.ndarray,
    ) -> dict:
        return {
            "exact_truth_count": int(exact.sum()),
            "exact_truth_weight": float(self.weights[exact].sum()),
            "truth_descendant_count": int(descendants.sum()),
            "truth_descendant_weight": float(self.weights[descendants].sum()),
            "near_truth_count": int((distance <= PRIOR_SCALED_RADIUS).sum()),
            "near_truth_weight": float(
                self.weights[distance <= PRIOR_SCALED_RADIUS].sum()
            ),
            "minimum_scaled_distance": float(distance.min()),
            "com_band_weight": float(self.weights[near_com].sum()),
            "unique_particles": len(np.unique(self._z, axis=0)),
            "unique_roots": len(np.unique(self.roots)),
            "ess": float(1 / np.sum(self.weights**2)),
        }


def plot_truth_survival(rows: list[dict], output: Path) -> None:
    regimes = list(dict.fromkeys(row["regime"] for row in rows))
    figure, axes = plt.subplots(
        2, len(regimes), figsize=(12, 6), squeeze=False, layout="constrained"
    )
    for column, regime in enumerate(regimes):
        for trial in rows:
            if trial["regime"] == regime and trial["injected"]:
                _plot_trial(axes[:, column], trial)
        for axis, label in zip(
            axes[:, column],
            ("Weight on exact truth copies", "Weight on truth descendants"),
        ):
            axis.set(
                xlabel="Push number (fraction = evidence introduced)",
                ylabel=label,
                ylim=(-0.02, 1.02),
            )
            axis.grid(alpha=0.2)
        axes[0, column].set_title(regime)
        axes[0, column].legend(fontsize=8)
    figure.suptitle("Injected ground truth: resampling and accepted moves")
    figure.savefig(output / "truth_survival.png", dpi=150)
    plt.close(figure)


def _plot_trial(axes: np.ndarray, trial: dict) -> None:
    # A short run may never resample; its initial population is still known.
    summaries = [trial["initial_summary"]]
    for event in trial["events"]:
        if "before" in event:
            summaries.append(event["before"])
        summaries.append(event["after"])
    push_numbers = [summary["step"] - 1 + summary["beta"] for summary in summaries]
    for axis, key in zip(axes, ("exact_truth_weight", "truth_descendant_weight")):
        axis.plot(
            push_numbers,
            [summary[key] for summary in summaries],
            color=f"C{trial['seed'] - 1}",
            linestyle="-" if trial["mode"] == "full" else "--",
            drawstyle="steps-post",
            label=f"{trial['mode']}, seed {trial['seed']}",
        )


def write_truth_report(rows: list[dict], output: Path) -> None:
    lines = _build_truth_report(rows)
    lines += [
        "",
        "## First resampling with injected truth",
        "",
        "| Sequence | Mode | Seed | beta | Truth target rank | Truth weight before | Copies after resampling |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if not row["injected"]:
            continue
        events = [event for event in row["events"] if event["kind"] == "resample"]
        if events:
            event = events[0]
            summary = event["before"]
            lines.append(
                f"| {row['regime']} | {row['mode']} | {row['seed']} | {summary['beta']:.4f} | {summary['truth_target_rank']} | {summary['exact_truth_weight']:.5f} | {event['after']['exact_truth_count']} |"
            )
    lines += [
        "",
        "## Trace files and interpretation",
        "",
        "Each trial JSON records pre/post resampling summaries and ancestry indices, followed by proposal/acceptance counts and indices that moved. Snapshot NPZ files store particles, weights and root ancestry after each push. Accepted descendants are not pinned at truth or protected from resampling. Counterfactual likelihood probes do not enter inference weights or consume inference RNG draws.",
        "",
        "These results are conditional on the saved object, datasets, and inference seeds. Full updates and gradual updates have unequal compute. Controls replace only the injected slot, but later resampling and proposal paths can diverge. Losing an exact atom is not evidence of incorrect inference for a continuous posterior. Grid comparison is available only for varied durations and is an empirical numerical check, not ground truth.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def _summarize_truth_survival(rows: list[dict]) -> list[str]:
    injected = [row for row in rows if row["injected"]]
    first_resamples = [
        next((event for event in row["events"] if event["kind"] == "resample"), None)
        for row in injected
    ]
    first_resamples = [event for event in first_resamples if event is not None]
    full = [row for row in injected if row["mode"] == "full"]
    gradual = [row for row in injected if row["mode"] == "gradual"]
    full_weights = [row["summary"]["exact_truth_weight"] for row in full]
    findings = [
        "## Main findings",
        "",
        f"At first resampling, truth has the best target likelihood (rank 1, allowing ties) in {sum((event['before']['truth_target_rank'] == 1 for event in first_resamples))}/{len(first_resamples)} injected runs. Runs without resampling are excluded from this count.",
        "",
        f"Full updates retain exact truth copies in {sum((row['summary']['exact_truth_count'] > 0 for row in full))}/{len(full)} final populations. Final exact-truth weight ranges from {min(full_weights, default=0.0):.1%} to {max(full_weights, default=0.0):.1%}.",
        "",
        f"Gradual updates retain exact truth in {sum((row['summary']['exact_truth_count'] > 0 for row in gradual))}/{len(gradual)} final populations and any truth descendants in {sum((row['summary']['truth_descendant_count'] > 0 for row in gradual))}/{len(gradual)}. The move traces distinguish accepted moves away from the exact point from later removal of descendants by resampling. Moves at weak intermediate likelihoods may legitimately leave a high-likelihood point; a surviving neighborhood matters more than preserving its exact coordinates.",
        "",
        "Injecting truth does not establish correct uncertainty: an artificially inserted high-density point can dominate a poorly explored population, while the integrated posterior remains broad. Inspect the grid distances below rather than interpreting truth-copy survival as sampler correctness. The original control populations can omit highly competitive physics, and their fixed random-walk proposals may fail to discover it. These findings concern the tested dataset, not all observation models.",
        "",
        "![Weight on exact copies and their descendants](truth_survival.png)",
        "",
    ]
    return findings


def _build_truth_report(rows: list[dict]) -> list[str]:
    lines = [
        "# Ground-truth-particle diagnostic",
        "",
        "One initial random particle is replaced by the exact known physics, without increasing its weight. Paired controls use the original population and identical RNG seeds. Both use exact starts, one inner state, corrected scoring, and two proposal attempts per particle per resampling stage. Oracle initialization is diagnostic only, not a valid performance baseline.",
        "",
        "Exact copies, descendants, and nearby hypotheses are different: a legitimate accepted move removes an exact copy but retains its ancestry. Losing all descendants does not prove a posterior region was lost. The nearby region uses Euclidean distance <=0.05 in prior-scaled z coordinates; the COM band is truth ±5 mm and does not constrain mass/friction.",
        "",
        "Ranks compare the separately evaluated true likelihood with the current population (ties share rank). Rank 1 is best. During tempering, target ranks use full old evidence plus beta times new evidence; full-likelihood ranks are reported separately. Weights before resampling include the population's inherited weights. A good likelihood rank alone does not guarantee dominant weight.",
        "",
        "| Sequence | Mode | Seed | Injected | Final exact copies | Final truth descendants | Distinct particles | Truth full-likelihood rank | COM band weight | COM distance to grid (mm) |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    findings = _summarize_truth_survival(rows)
    lines[2:2] = findings
    for row in rows:
        summary = row["summary"]
        distance = row.get("com_distance_to_grid_mm")
        lines.append(
            f"| {row['regime']} | {row['mode']} | {row['seed']} | {row['injected']} | {summary['exact_truth_count']} | {summary['truth_descendant_count']} | {summary['unique_particles']} | {summary['truth_full_likelihood_rank']} | {summary['com_band_weight']:.3f} | {('—' if distance is None else f'{distance:.2f}')} |"
        )
    return lines


class TruthParticleDiagnostic:
    def run(self) -> None:
        self._parse_args()
        self._load_baseline()
        self._prepare_output()
        self._load_reference()
        rows = self._run_trials()
        self._finish(rows)

    def _parse_args(self) -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--baseline",
            type=Path,
            default=Path("artifacts/sysid_development_tempered_256"),
        )
        parser.add_argument(
            "--out", type=Path, default=Path("artifacts/sysid_truth_particle_trace")
        )
        self.args = parser.parse_args()

    def _load_baseline(self) -> None:
        self.config = json.loads((self.args.baseline / "config.json").read_text())
        self.spec = CuboidSpec(**self.config["object"])
        self.training_sequences = load_training_sequences(
            self.args.baseline, self.config["regimes"]
        )

    def _prepare_output(self) -> None:
        self.args.out.mkdir(parents=True, exist_ok=False)
        self.output_config = {
            "object": asdict(self.spec),
            "n_theta": self.config["n_theta"],
            "n_state": 1,
            "proposal": "fixed",
            "start_std": [0] * 4,
            "sample_stride": self.config["sample_stride"],
            "seeds": self.config["seeds"],
            "modes": ["full", "gradual"],
            "injected_slot": 0,
            "initial_weight": 1 / self.config["n_theta"],
            "diagnostic_only": True,
        }

    def _load_reference(self) -> None:
        grid_path = Path("artifacts/sysid_reference_posterior/grid24_level5.npz")
        self.grid = None
        if grid_path.exists() and "varied durations" in self.training_sequences:
            self.grid, difference = load_checked_grid(
                grid_path,
                self.spec,
                self.training_sequences["varied durations"],
                self.config["sample_stride"],
            )
            self.output_config["grid_likelihood_check_max_difference"] = difference
        (self.args.out / "config.json").write_text(
            json.dumps(self.output_config, indent=2) + "\n"
        )

    def _run_trials(self) -> list[dict]:
        rows = []
        for regime, records in self.training_sequences.items():
            truth_scores = self.evaluate_truth(
                self.spec, records, self.config["sample_stride"]
            )
            for mode in ("full", "gradual"):
                for seed in self.config["seeds"]:
                    for injected in (False, True):
                        rows.append(
                            self._run_trial(
                                regime=regime,
                                records=records,
                                truth_scores=truth_scores,
                                mode=mode,
                                seed=seed,
                                injected=injected,
                            )
                        )
        return rows

    def _finish(self, rows: list[dict]) -> None:
        (self.args.out / "summary.json").write_text(
            json.dumps(
                [
                    {key: value for key, value in row.items() if key != "events"}
                    for row in rows
                ],
                indent=2,
            )
            + "\n"
        )
        write_truth_report(rows, self.args.out)
        plot_truth_survival(rows, self.args.out)
        print(f"Saved diagnostic to {self.args.out}")

    def _trace_updates(
        self,
        estimator: TracedSMC2,
        records: list[ActionRecord],
        stem: str,
    ) -> dict:
        initial_summary = estimator.summarize()
        steps = []
        for push_count in range(1, len(records) + 1):
            estimator.update(records[:push_count])
            steps.append(estimator.summarize())
            np.savez_compressed(
                self.args.out / f"{stem}_step{push_count}.npz",
                z=estimator.particles,
                weights=estimator.weights,
                roots=estimator.roots,
            )
        return {
            "initial_summary": initial_summary,
            "steps": steps,
            "fit_replays": estimator.sim.replay_count,
            "summary": estimator.summarize(),
            "events": estimator.events,
            "proposals": estimator.proposals,
            "acceptances": estimator.acceptances,
            "initial_weight": 1 / estimator.n_theta,
        }

    def _create_estimator(
        self,
        truth_scores: np.ndarray,
        mode: str,
        seed: int,
        injected: bool,
        simulator: ReplayCountingSimulator,
    ) -> TracedSMC2:
        return TracedSMC2(
            self.spec,
            truth_scores,
            inject_truth=injected,
            n_theta=self.config["n_theta"],
            proposal="fixed",
            n_state=1,
            start_std=(0, 0, 0, 0),
            sample_stride=self.config["sample_stride"],
            seed=seed,
            sim=simulator,
            temper_updates=mode == "gradual",
        )

    def _run_trial(
        self,
        regime: str,
        records: list[ActionRecord],
        truth_scores: np.ndarray,
        mode: str,
        seed: int,
        injected: bool,
    ) -> dict:
        stem = f"{regime.replace(' ', '_')}_{mode}_seed{seed}_{'injected' if injected else 'control'}"
        simulator = ReplayCountingSimulator()
        try:
            estimator = self._create_estimator(
                truth_scores, mode, seed, injected, simulator
            )
            row = {"regime": regime, "mode": mode, "seed": seed, "injected": injected}
            row.update(self._trace_updates(estimator, records, stem))
            self._save_trial(estimator, records, row, stem)
            return row
        finally:
            simulator.close()

    def _save_trial(
        self,
        estimator: TracedSMC2,
        records: list[ActionRecord],
        row: dict,
        stem: str,
    ) -> None:
        if self.grid is not None and row["regime"] == "varied durations":
            values, weights = integrate_marginal(
                self.grid, len(records), 2, self.spec.H
            )
            row["com_distance_to_grid_mm"] = 1000 * wasserstein_distance(
                values,
                estimator.particles[:, 2] * self.spec.H,
                weights,
                estimator.weights,
            )
        (self.args.out / f"{stem}.json").write_text(json.dumps(row, indent=2) + "\n")
        summary = row["summary"]
        print(
            f"{stem}: exact={summary['exact_truth_count']}, descendants={summary['truth_descendant_count']}, unique={summary['unique_particles']}",
            flush=True,
        )

    @staticmethod
    def evaluate_truth(
        spec: CuboidSpec, records: list[ActionRecord], stride: int
    ) -> np.ndarray:
        with SMC2SysID(
            (spec.w, spec.L, spec.H),
            n_theta=1,
            n_state=1,
            start_std=(0, 0, 0, 0),
            sample_stride=stride,
        ) as probe:
            truth = np.array([np.log(spec.m), spec.mu, spec.h_c / spec.H])
            return np.array(
                [probe._evaluate_record(truth, record)[1] for record in records]
            )


if __name__ == "__main__":
    TruthParticleDiagnostic().run()
