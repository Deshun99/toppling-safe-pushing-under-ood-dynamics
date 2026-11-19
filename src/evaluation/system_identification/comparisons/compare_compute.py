"""Match CMA-ES cumulative fitting replays to a saved known-start SMC² run.

Reuse observations and seeds rather than collecting new target trajectories.
CMA-ES refits each prefix with three independent starts by default. Earlier
fits count against every cumulative allowance. Complete-history evaluations
may leave fewer than one history's worth of replays unused at each update.
"""

import argparse
import json
import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.agents.simulator.internal_sim import InternalSim
from src.agents.system_identification.sysid import SysID
from src.evaluation.system_identification.shared.common import (
    ReplayCountingSimulator,
    load_action_records,
    load_metric_rows,
    load_training_sequences,
)
from src.evaluation.system_identification.shared.metrics import evaluate_predictions
from src.simulation.modelling.object import CuboidSpec, PhysicsParameters
from src.simulation.modelling.records import ActionRecord


def write_replay_budget_report(
    rows: list[dict], output: Path, starts: int = 3, retain_best: bool = False
) -> None:
    lines, methods, summary = _prepare_report(rows, starts, retain_best)
    _append_comparison_rows(rows, lines, methods, summary)
    for regime in sorted({row["regime"] for row in rows}):
        selected = [row for row in rows if row["regime"] == regime]
        last_step = max((row["step"] for row in selected))
        for row in selected:
            if row["step"] != last_step:
                continue
            smc, cma = (row["smc2"], row["cma_es_matched"])
            lines.append(
                f"| {regime} | {row['seed']} | {1000 * smc['held_out']['upright_rmse_x']:.3f} | {1000 * cma['held_out']['upright_rmse_x']:.3f} | {1000 * smc['absolute_error'][2]:.2f} / {1000 * cma['absolute_error'][2]:.2f} |"
            )
    lines += [
        "",
        "## Limits",
        "",
        "This comparison uses the saved object, datasets, inference seeds, and estimator settings described in the baseline config.json. The matched comparison measures these implementations and this restart policy, not universal algorithm superiority. Replay counts match simulation calls, not physics timesteps or optimizer CPU work. Previous history fits are charged to CMA-ES; SMC² updates sequentially. Search initialization differs between methods as in the existing implementation. No posterior-calibration claims are made for CMA-ES's retained candidates.",
        "",
        "`metrics.jsonl` records each update's allowance, actual cumulative cost, selected parameters, and development prediction errors. `config.json` records the experiment settings and baseline directory.",
    ]
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "report.md").write_text("\n".join(lines) + "\n")


def _prepare_report(
    rows: list[dict], starts: int, retain_best: bool
) -> tuple[list[str], list[str], list[dict]]:
    lines = [
        "# CMA-ES versus SMC² with matched cumulative replay limits",
        "",
        f"CMA-ES fits the saved datasets with the same seeds and scoring stride. Warmup is replayed but excluded from fitting. Each CMA-ES update refits the full history with up to {starts} independent starts; the iteration limit is replaced by a replay cap. All earlier CMA-ES fits count toward the cumulative SMC² allowance. A final partial population is evaluated but not passed to the optimizer's tell step. Incomplete history evaluations are never performed.",
        "",
        "Position errors use selected point hypotheses on upright development evaluation pushes, averaged over inference seeds. They are not posterior predictive mixture errors. Parameter columns are mean absolute errors. Costs below sum all fitting updates across seeds; evaluation replays are excluded.",
        "",
        "| Sequence | Method | Position RMSE (mm) | Mass error (kg) | Friction error | COM error (mm) | Topple accuracy | Fitting replays |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    if retain_best:
        lines.insert(
            2,
            "The previous selected physics is rescored on the full current history and retained unless the new search has lower loss. Its replays are deducted before allocating the search budget. No previous loss or optimizer state is reused. This guarantees no worse current training loss than the previous best estimate, not better held-out predictions.",
        )
    methods = ["smc2", "cma_es_small"]
    if rows and "cma_es_refit" in rows[0]:
        methods.append("cma_es_refit")
    methods.append("cma_es_matched")
    summary = []
    return lines, methods, summary


def _append_method_summary(
    regime: str,
    method: str,
    trials: list[dict],
    position: float,
    error: np.ndarray,
    summary: list[dict],
    lines: list[str],
) -> None:
    topple = float(np.mean([trial["held_out"]["topple_accuracy"] for trial in trials]))
    cost = sum((trial["cumulative_fit_replays"] for trial in trials))
    summary.append(
        {
            "regime": regime,
            "method": method,
            "position_rmse_mm": position * 1000,
            "absolute_error": error.tolist(),
            "topple_accuracy": topple,
            "fit_replays": cost,
        }
    )
    lines.append(
        f"| {regime} | {method} | {1000 * position:.3f} | {error[0]:.4f} | {error[1]:.4f} | {1000 * error[2]:.2f} | {topple:.3f} | {cost:,} |"
    )


def _append_comparison_rows(
    rows: list[dict], lines: list[str], methods: list[str], summary: list[dict]
) -> None:
    for regime in sorted({row["regime"] for row in rows}):
        selected = [row for row in rows if row["regime"] == regime]
        last_step = max((row["step"] for row in selected))
        final = [row for row in selected if row["step"] == last_step]
        for method in methods:
            trials = [row[method] for row in final]
            error = np.mean([trial["absolute_error"] for trial in trials], axis=0)
            position = float(
                np.mean([trial["held_out"]["upright_rmse_x"] for trial in trials])
            )
            _append_method_summary(
                regime, method, trials, position, error, summary, lines
            )
    lines += [
        "",
        "## Per-seed final errors",
        "",
        "| Sequence | Seed | SMC² position RMSE (mm) | Matched CMA-ES position RMSE (mm) | SMC² / CMA-ES COM error (mm) |",
        "|---|---:|---:|---:|---|",
    ]


class ReplayBudgetComparison:
    def run(self) -> None:
        self._parse_args()
        self._validate_baseline()
        self._load_records()
        self._save_config()
        rows = self._run_trials()
        self._finish(rows)

    def _parse_args(self) -> None:
        self.parser = argparse.ArgumentParser(description=__doc__)
        self.parser.add_argument(
            "--baseline",
            type=Path,
            default=Path("artifacts/sysid_development_tempered_256"),
        )
        self.parser.add_argument(
            "--out", type=Path, default=Path("artifacts/sysid_compute_matched")
        )
        self.parser.add_argument("--restarts", type=int, default=2)
        self.parser.add_argument(
            "--retain-best",
            action="store_true",
            help="Rescore the previous best physics within each update's replay budget",
        )
        self.parser.add_argument(
            "--refit-comparison",
            type=Path,
            help="Saved matched-budget fresh-refit run to include in the results",
        )
        self.args = self.parser.parse_args()

    def _validate_baseline(self) -> None:
        self.config = json.loads((self.args.baseline / "config.json").read_text())
        if self.args.restarts < 0:
            self.parser.error("restarts must be nonnegative")
        if not self.config["known_start"] or self.config.get("include_warmup", False):
            self.parser.error("Baseline must use known starts and exclude warmup")
        self.previous_refits = {}
        if self.args.refit_comparison:
            self._validate_refit_comparison()
            self.previous_refits = {
                (row["regime"], row["seed"], row["step"]): row
                for row in (
                    json.loads(line)
                    for line in (self.args.refit_comparison / "metrics.jsonl")
                    .read_text()
                    .splitlines()
                )
            }

    def _load_records(self) -> None:
        self.spec = CuboidSpec(**self.config["object"])
        saved = load_metric_rows(self.args.baseline / "metrics.jsonl")
        self.indexed = {
            (row["regime"], row["seed"], row["step"], row["method"]): row
            for row in saved
        }
        self.training = load_training_sequences(
            self.args.baseline, self.config["regimes"]
        )
        self.held_out = load_action_records(self.args.baseline / "held_out.json")

    def _save_config(self) -> None:
        self.args.out.mkdir(parents=True, exist_ok=False)
        output_config = {
            "baseline": str(self.args.baseline),
            "restarts": self.args.restarts,
            "retain_best": self.args.retain_best,
            "refit_comparison": str(self.args.refit_comparison)
            if self.args.refit_comparison
            else None,
            "iterations": "determined by per-update replay cap",
            "sample_stride": self.config["sample_stride"],
            "object": self.config["object"],
            "seeds": self.config["seeds"],
            "regimes": self.config["regimes"],
        }
        (self.args.out / "config.json").write_text(
            json.dumps(output_config, indent=2) + "\n"
        )

    def _run_trials(self) -> list[dict]:
        rows = []
        with (self.args.out / "metrics.jsonl").open("w") as stream:
            for regime, records in self.training.items():
                for seed in self.config["seeds"]:
                    trial = self._run_trial(regime, seed, records)
                    rows.extend(trial)
                    for row in trial:
                        stream.write(json.dumps(row) + "\n")
                    stream.flush()
        return rows

    def _finish(self, rows: list[dict]) -> None:
        write_replay_budget_report(
            rows,
            self.args.out,
            starts=self.args.restarts + 1,
            retain_best=self.args.retain_best,
        )
        print(f"Saved comparison to {self.args.out}")

    def _validate_refit_comparison(self) -> None:
        refit_config = json.loads(
            (self.args.refit_comparison / "config.json").read_text()
        )
        for key in ("object", "seeds", "regimes", "sample_stride"):
            if refit_config[key] != self.config[key]:
                self.parser.error(f"Refit comparison differs in {key}")
        if refit_config["restarts"] != self.args.restarts or refit_config.get(
            "retain_best", False
        ):
            self.parser.error(
                "Refit comparison must use the same restarts without retention"
            )

    def _run_trial(
        self, regime: str, seed: int, records: list[ActionRecord]
    ) -> list[dict]:
        rows = []
        previous_best_parameters = None
        with ExitStack() as resources:
            estimator, evaluation_sim = self._create_estimators(resources, seed)
            for push_count in range(1, len(records) + 1):
                smc = self.indexed[regime, seed, push_count, "smc2"]
                budget = smc["cumulative_fit_replays"] - estimator.sim.replay_count
                matched = self._fit_with_budget(
                    estimator, records[:push_count], budget, previous_best_parameters
                )
                self._add_prediction_metrics(matched, evaluation_sim)
                assert (
                    0
                    <= smc["cumulative_fit_replays"] - estimator.sim.replay_count
                    < push_count
                )
                row = self._build_comparison_row(
                    regime, seed, push_count, budget, matched
                )
                rows.append(row)
                if self.args.retain_best:
                    previous_best_parameters = tuple(matched["theta"])
                print(
                    f"{regime} seed={seed} step={push_count}: replays={estimator.sim.replay_count}/{smc['cumulative_fit_replays']}, position={1000 * matched['held_out']['upright_rmse_x']:.3f} mm",
                    flush=True,
                )
        return rows

    def _create_estimators(
        self, resources: ExitStack, seed: int
    ) -> tuple[SysID, InternalSim]:
        estimator = SysID(
            (self.spec.w, self.spec.L, self.spec.H),
            seed=seed,
            restarts=self.args.restarts,
            sample_stride=self.config["sample_stride"],
        )
        resources.callback(estimator.close)
        estimator.sim.close()
        estimator.sim = ReplayCountingSimulator()
        evaluation_sim = InternalSim()
        resources.callback(evaluation_sim.close)
        return estimator, evaluation_sim

    def _fit_with_budget(
        self,
        estimator: SysID,
        history: list[ActionRecord],
        budget: int,
        previous_best_parameters: PhysicsParameters | None,
    ) -> dict:
        before, start = estimator.sim.replay_count, time.perf_counter()
        result = estimator.update(
            history,
            replay_budget=budget,
            previous_best_parameters=previous_best_parameters,
        )
        elapsed = time.perf_counter() - start
        theta = np.asarray(result.point.theta)
        has_previous_best = previous_best_parameters is not None
        retained = has_previous_best and np.allclose(
            theta, previous_best_parameters, rtol=1e-12, atol=1e-12
        )
        return {
            "theta": theta.tolist(),
            "loss": result.point.loss,
            "absolute_error": np.abs(theta - self.spec.theta).tolist(),
            "fit_replays": estimator.sim.replay_count - before,
            "cumulative_fit_replays": estimator.sim.replay_count,
            "fit_seconds": elapsed,
            "previous_best_parameters": previous_best_parameters,
            "previous_best_replays": len(history) if has_previous_best else 0,
            "retained_previous_point": bool(retained),
        }

    def _add_prediction_metrics(self, matched: dict, simulator: InternalSim) -> None:
        mass, friction, com_height = matched["theta"]
        candidate = replace(self.spec, m=mass, mu=friction, h_c=com_height)
        matched["held_out"] = evaluate_predictions(simulator, candidate, self.held_out)

    def _build_comparison_row(
        self,
        regime: str,
        seed: int,
        push_count: int,
        budget: int,
        matched: dict,
    ) -> dict:
        smc = self.indexed[regime, seed, push_count, "smc2"]
        row = {
            "regime": regime,
            "seed": seed,
            "step": push_count,
            "replay_allowance": budget,
            "smc2": smc,
            "cma_es_small": self.indexed[regime, seed, push_count, "cma_es"],
            "cma_es_matched": matched,
        }
        if self.previous_refits:
            previous = self.previous_refits[regime, seed, push_count]
            if previous["smc2"] != smc:
                raise ValueError(
                    "Refit comparison uses different SMC² baseline results"
                )
            row["cma_es_refit"] = previous["cma_es_matched"]
        return row


if __name__ == "__main__":
    ReplayBudgetComparison().run()
