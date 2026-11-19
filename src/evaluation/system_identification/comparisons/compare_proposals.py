"""Compare fixed and covariance-shaped SMC² proposals on saved development data.

Use identical random initial populations, known starts and gradual updates.
Capped arms share the saved fixed-kernel cumulative replay ceilings. An
uncapped covariance arm isolates the effect of truncating move stages. No
truth particle is injected and no evaluation data influence proposals.
"""

import argparse
import json
from collections.abc import Iterable, Iterator
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

from src.agents.system_identification.smc2 import InnerFilter
from src.evaluation.system_identification.shared.common import (
    load_action_records,
    load_metric_rows,
    load_training_sequences,
)
from src.evaluation.system_identification.shared.constants import Grid
from src.evaluation.system_identification.shared.grid import (
    integrate_marginal,
    load_checked_grid,
)
from src.evaluation.system_identification.shared.metrics import (
    evaluate_predictions,
    summarize_particles,
)
from src.simulation.modelling.records import ActionRecord

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

from src.agents.simulator.internal_sim import InternalSim
from src.agents.system_identification.smc2 import SMC2SysID
from src.agents.system_identification.sysid import Z_HI, Z_LO
from src.evaluation.system_identification.shared.common import ReplayCountingSimulator
from src.simulation.modelling.object import CuboidSpec


class MoveBudgetExhausted(RuntimeError):
    """No complete candidate likelihood fits the remaining replay allowance."""


class ReplayCappedSMC(SMC2SysID):
    """Study-only cap: finish evidence assimilation but stop unaffordable moves.

    Candidate likelihoods are never evaluated partially. Budget checks occur
    before their first replay. Caps affect rejuvenation, not the final beta.
    This study is restricted to deterministic one-state likelihoods.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.n_state != 1 or np.any(self.start_std != 0) or not self.temper_updates:
            raise ValueError("Replay cap study requires deterministic gradual updates")
        self.replay_limit: int | None = None
        self.blocked_stages = 0
        self.move_stages: list[dict] = []
        self._inside_move = False

    def _run_filter(
        self, z: np.ndarray, history: Iterable[ActionRecord]
    ) -> InnerFilter | None:
        if self._inside_move and self.replay_limit is not None:
            # A proposal replays all past records plus the current record.
            required = len(self._history) + 1
            if self.sim.replay_count + required > self.replay_limit:
                raise MoveBudgetExhausted
        return super()._run_filter(z, history)

    def _move_tempered_particles(
        self,
        record: ActionRecord,
        beta: float,
        past: np.ndarray,
        increments: np.ndarray,
    ) -> None:
        before = self.sim.replay_count
        proposals, acceptances = self.proposals, self.acceptances
        blocked = False
        self._inside_move = True
        try:
            super()._move_tempered_particles(record, beta, past, increments)
        except MoveBudgetExhausted:
            self.blocked_stages += 1
            blocked = True
        finally:
            self._inside_move = False
        self._record_move_stage(beta, before, proposals, acceptances, blocked)

    def _record_move_stage(
        self,
        beta: float,
        before: int,
        proposals: int,
        acceptances: int,
        blocked: bool,
    ) -> None:
        transform = self._proposal_transform
        self.move_stages.append(
            {
                "step": len(self._history) + 1,
                "beta": beta,
                "replays": self.sim.replay_count - before,
                "proposals": self.proposals - proposals,
                "acceptances": self.acceptances - acceptances,
                "budget_blocked": blocked,
                "transform_scaled": None
                if transform is None
                else (transform / (Z_HI - Z_LO)[:, None]).tolist(),
            }
        )


def plot_proposal_comparison(
    rows: list[dict], grid: Grid, spec: CuboidSpec, output: Path
) -> None:
    step, seeds, x, cdf, figure, axes = _prepare_com_axes(rows, grid)
    for axis, seed in zip(axes[0], seeds):
        axis.plot(
            x * spec.H * 1000,
            cdf,
            color="black",
            linewidth=2,
            label="Grid calculation",
        )
        for color, arm in enumerate(
            ("fixed_capped", "covariance_capped", "covariance_uncapped")
        ):
            _plot_particle_cdf(spec, output, axis, seed, arm, step, x, color)
        axis.axvline(spec.h_c * 1000, color="crimson", linestyle="--", label="True COM")
        axis.set(
            title=f"Seed {seed}",
            xlabel="COM height (mm)",
            ylabel="Cumulative probability",
            ylim=(0, 1),
        )
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Final COM distribution after varied-duration pushes")
    figure.savefig(output / "com_distribution_comparison.png", dpi=150)
    plt.close(figure)


def _prepare_com_axes(
    rows: list[dict], grid: Grid
) -> tuple[int, list[int], np.ndarray, np.ndarray, plt.Figure, np.ndarray]:
    """Compare empirical COM CDFs without smoothing away particle duplicates."""
    selected = [row for row in rows if row["regime"] == "varied durations"]
    step = max((row["step"] for row in selected))
    seeds = list(dict.fromkeys((row["seed"] for row in selected)))
    x = np.linspace(Z_LO[2], Z_HI[2], 500)
    weights, _ = grid.posterior(step)
    low, high = (grid.low[:, 2], grid.high[:, 2])
    cdf = np.array(
        [weights @ np.clip((value - low) / (high - low), 0, 1) for value in x]
    )
    figure, axes = plt.subplots(
        1, len(seeds), figsize=(13, 4), squeeze=False, layout="constrained"
    )
    return step, seeds, x, cdf, figure, axes


def _plot_particle_cdf(
    spec: CuboidSpec,
    output: Path,
    axis: plt.Axes,
    seed: int,
    arm: str,
    step: int,
    x: np.ndarray,
    color: int,
) -> None:
    saved = np.load(output / f"varied_durations_seed{seed}_{arm}_step{step}.npz")
    com, mass = (
        saved["z"][:, 2] * spec.H * 1000,
        saved["weights"],
    )
    order = np.argsort(com)
    axis.step(
        np.r_[
            x[0] * spec.H * 1000,
            com[order],
            x[-1] * spec.H * 1000,
        ],
        np.r_[0, np.cumsum(mass[order]), 1],
        where="post",
        color=f"C{color}",
        label=arm.replace("_", " "),
    )


def write_proposal_report(rows: list[dict], output: Path) -> None:
    lines, summary = _prepare_report()
    for regime in dict.fromkeys((row["regime"] for row in rows)):
        selected = [row for row in rows if row["regime"] == regime]
        last = max((row["step"] for row in selected))
        for arm in dict.fromkeys((row["arm"] for row in selected)):
            _append_arm_summary(regime, arm, selected, last, summary, lines)
    lines += [
        "",
        "## Final COM intervals",
        "",
        "| Sequence | Arm | Seed | 95% COM interval (mm) | Truth included | Budget-blocked stages |",
        "|---|---|---:|---|---|---:|",
    ]
    for row in rows:
        if row["step"] != max(
            (r["step"] for r in rows if r["regime"] == row["regime"])
        ):
            continue
        com = row["posterior"]["h_c"]
        lines.append(
            f"| {row['regime']} | {row['arm']} | {row['seed']} | [{1000 * com['low_95']:.1f}, {1000 * com['high_95']:.1f}] | {com['covers_truth']} | {row['blocked_stages']} |"
        )
    _write_report_files(output, lines, summary)


def _prepare_report() -> tuple[list[str], list[dict]]:
    lines = [
        "# Covariance-shaped proposal experiment",
        "",
        "The saved data, inference seeds, and particle count from the baseline, exact starts, gradual updates, at most two proposal attempts per particle per stage, and proposal scale 0.08. No true-physics particle is injected. The new option uses weighted particle covariance in prior-scaled z coordinates, measured before resampling and frozen during each move stage. Its trace is normalized so both kernels have the same expected squared jump length. A small diagonal floor preserves mobility; a collapsed cloud falls back to the fixed shape. There is no acceptance-based scale adaptation.",
        "",
        "Capped arms use the saved fixed-kernel cumulative replay ceilings at every prefix. New-push likelihoods are always fully assimilated (beta reaches one); only proposal moves can be stopped. The check reserves a complete candidate history before its first replay, and never redraws an out-of-bounds proposal. Actual costs can fall below the ceilings. The uncapped covariance arm uses the original two-moves-per-stage schedule to expose effects of the cap. Replay-based stopping is a practical adaptive compute policy, not a proof of exact finite-population sampling.",
        "",
        "Point prediction errors are for upright development evaluation pushes. COM truth inclusion is conditional on this object/dataset, not population calibration. COM grid distances compare full distributions and exist only for varied durations; lower is closer.",
        "",
        "| Sequence | Arm | Position RMSE (mm) | COM intervals include truth | Distinct final particles, seeds 1/2/3 | Move acceptance | Fitting replays | Mean COM distance to grid (mm) |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    summary = []
    return lines, summary


def _write_report_files(output: Path, lines: list[str], summary: list[dict]) -> None:
    lines += [
        "",
        "## Scope and sources",
        "",
        "One noise-free object and two development sequences; no final-test generalization claim. Covariance adaptation is a standard SMC technique ([Zhou, Johansen & Aston, section 3.3.3](https://arxiv.org/html/1303.3123v2#S3.SS3.SSS3)), but the fixed-trace normalization here is a deliberate experimental choice to isolate shape. This is not a claim to reproduce the paper's complete proposal-adaptation scheme. Narrow or curved regions may still be represented badly by one global covariance. High acceptance alone is not evidence of adequate exploration.",
        "",
        "Per-update costs, proposal counts, posterior intervals and marginal grid distances are in metrics.jsonl. Stage matrices and cap events are in separate move-trace JSON files; NPZ files retain particles and weights. Experiment settings and the baseline directory are recorded in config.json. Output directories must be new.",
    ]
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "report.md").write_text("\n".join(lines) + "\n")


def _summarize_accuracy(regime: str, arm: str, final: list[dict]) -> dict:
    item = {}
    item.update(
        {
            "regime": regime,
            "arm": arm,
            "position_rmse_mm": 1000
            * float(np.mean([row["held_out"]["upright_rmse_x"] for row in final])),
            "com_truth_count": sum(
                (row["posterior"]["h_c"]["covers_truth"] for row in final)
            ),
        }
    )
    return item


def _summarize_sampling(regime: str, arm: str, final: list[dict]) -> dict:
    proposals = sum((row["proposals"] for row in final))
    item = _summarize_accuracy(regime, arm, final)
    item.update(
        {
            "unique_particles": [row["unique_particles"] for row in final],
            "acceptance_fraction": sum((row["acceptances"] for row in final))
            / proposals
            if proposals
            else None,
            "fit_replays": sum((row["cumulative_fit_replays"] for row in final)),
            "blocked_stages": sum((row["blocked_stages"] for row in final)),
        }
    )
    return item


def _append_arm_summary(
    regime: str,
    arm: str,
    selected: list[dict],
    last: int,
    summary: list[dict],
    lines: list[str],
) -> None:
    final = [row for row in selected if row["arm"] == arm and row["step"] == last]
    item = _summarize_sampling(regime, arm, final)
    item.update(
        {
            "com_distance_to_grid_mm": float(
                np.mean([row["com_distance_to_grid_mm"] for row in final])
            )
            if "com_distance_to_grid_mm" in final[0]
            else None
        }
    )
    summary.append(item)
    distance = item["com_distance_to_grid_mm"]
    lines.append(
        f"| {regime} | {arm} | {item['position_rmse_mm']:.3f} | {item['com_truth_count']}/{len(final)} | {item['unique_particles']} | {item['acceptance_fraction']:.1%} | {item['fit_replays']:,} | {('—' if distance is None else f'{distance:.2f}')} |"
    )


class ProposalComparison:
    def run(self) -> None:
        self._parse_args()
        self._load_baseline()
        self._prepare_output()
        self._load_reference()
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
            "--out", type=Path, default=Path("artifacts/sysid_covariance_proposals")
        )
        self.args = self.parser.parse_args()

    def _load_baseline(self) -> None:
        self.config = json.loads((self.args.baseline / "config.json").read_text())
        if (
            not self.config["known_start"]
            or not self.config["temper_updates"]
            or self.config.get("include_warmup", False)
        ):
            self.parser.error(
                "Baseline must use deterministic gradual updates excluding warmup"
            )
        self.spec = CuboidSpec(**self.config["object"])
        self.geometry = (self.spec.w, self.spec.L, self.spec.H)
        self.training_sequences = load_training_sequences(
            self.args.baseline, self.config["regimes"]
        )
        saved = load_metric_rows(self.args.baseline / "metrics.jsonl")
        self.reference = {
            (row["regime"], row["seed"], row["step"]): row
            for row in saved
            if row["method"] == "smc2"
        }
        self.evaluation = load_action_records(self.args.baseline / "held_out.json")

    def _prepare_output(self) -> None:
        self.args.out.mkdir(parents=True, exist_ok=False)
        self.output_config = {
            "baseline": str(self.args.baseline),
            "object": self.config["object"],
            "n_theta": self.config["n_theta"],
            "n_state": 1,
            "start_std": [0] * 4,
            "seeds": self.config["seeds"],
            "sample_stride": self.config["sample_stride"],
            "arms": ["fixed_capped", "covariance_capped", "covariance_uncapped"],
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
            self.output_config["grid_max_likelihood_difference"] = difference
        (self.args.out / "config.json").write_text(
            json.dumps(self.output_config, indent=2) + "\n"
        )

    def _run_trials(self) -> list[dict]:
        rows = []
        with (self.args.out / "metrics.jsonl").open("w") as stream:
            for regime, records in self.training_sequences.items():
                for seed in self.config["seeds"]:
                    for arm in self.output_config["arms"]:
                        for row in self._run_trial(regime, records, seed, arm):
                            rows.append(row)
                            stream.write(json.dumps(row) + "\n")
                            stream.flush()
        return rows

    def _finish(self, rows: list[dict]) -> None:
        write_proposal_report(rows, self.args.out)
        if self.grid is not None:
            plot_proposal_comparison(rows, self.grid, self.spec, self.args.out)
        print(f"Saved comparison to {self.args.out}")

    def _create_estimators(
        self,
        resources: ExitStack,
        seed: int,
        arm: str,
    ) -> tuple[ReplayCappedSMC, InternalSim]:
        simulator = ReplayCountingSimulator()
        resources.callback(simulator.close)
        evaluation_sim = InternalSim()
        resources.callback(evaluation_sim.close)
        estimator = ReplayCappedSMC(
            self.geometry,
            n_theta=self.config["n_theta"],
            n_state=1,
            start_std=(0, 0, 0, 0),
            seed=seed,
            sample_stride=self.config["sample_stride"],
            temper_updates=True,
            proposal="fixed" if arm == "fixed_capped" else "covariance",
            sim=simulator,
        )
        return estimator, evaluation_sim

    def _run_trial(
        self,
        regime: str,
        records: list[ActionRecord],
        seed: int,
        arm: str,
    ) -> Iterator[dict]:
        stem = f"{regime.replace(' ', '_')}_seed{seed}_{arm}"
        with ExitStack() as resources:
            estimator, evaluation_sim = self._create_estimators(resources, seed, arm)
            for push_count in range(1, len(records) + 1):
                cap = self.reference[regime, seed, push_count]["cumulative_fit_replays"]
                estimator.replay_limit = None if arm == "covariance_uncapped" else cap
                row = {"regime": regime, "seed": seed, "arm": arm, "step": push_count}
                row.update(self._update_estimator(estimator, records[:push_count]))
                mass, friction, com_height = row["theta"]
                candidate = replace(self.spec, m=mass, mu=friction, h_c=com_height)
                row["held_out"] = evaluate_predictions(
                    evaluation_sim, candidate, self.evaluation
                )
                self._save_update(estimator, row, stem)
                yield row
            (self.args.out / f"{stem}_moves.json").write_text(
                json.dumps(estimator.move_stages, indent=2) + "\n"
            )

    def _update_estimator(
        self, estimator: ReplayCappedSMC, history: list[ActionRecord]
    ) -> dict:
        before = estimator.sim.replay_count
        result = estimator.update(history)
        if estimator.replay_limit is not None:
            assert estimator.sim.replay_count <= estimator.replay_limit
        return {
            "theta": list(result.point.theta),
            "replay_ceiling": estimator.replay_limit,
            "fit_replays": estimator.sim.replay_count - before,
            "cumulative_fit_replays": estimator.sim.replay_count,
            "posterior": summarize_particles(estimator, self.spec),
            "unique_particles": len(np.unique(estimator.particles, axis=0)),
            "proposals": estimator.proposals,
            "acceptances": estimator.acceptances,
            "blocked_stages": estimator.blocked_stages,
        }

    def _save_update(self, estimator: ReplayCappedSMC, row: dict, stem: str) -> None:
        push_count = row["step"]
        cap = self.reference[row["regime"], row["seed"], push_count][
            "cumulative_fit_replays"
        ]
        if row["arm"] == "fixed_capped":
            baseline = f"{row['regime'].replace(' ', '_')}_seed{row['seed']}_step{push_count}.npz"
            with np.load(self.args.baseline / baseline) as original:
                np.testing.assert_array_equal(original["z"], estimator.particles)
                np.testing.assert_array_equal(original["weights"], estimator.weights)
            assert estimator.sim.replay_count == cap
        if self.grid is not None and row["regime"] == "varied durations":
            row["marginal_distances_to_grid"] = self._measure_grid_distances(
                estimator, push_count
            )
            row["com_distance_to_grid_mm"] = 1000 * row["marginal_distances_to_grid"][2]
        np.savez_compressed(
            self.args.out / f"{stem}_step{push_count}.npz",
            z=estimator.particles,
            weights=estimator.weights,
        )
        print(
            f"{stem} step={push_count}: replays={estimator.sim.replay_count}/{cap}, unique={row['unique_particles']}, blocked={estimator.blocked_stages}",
            flush=True,
        )

    def _measure_grid_distances(
        self, estimator: SMC2SysID, push_count: int
    ) -> list[float]:
        distances = []
        for column in range(len(Z_LO)):
            values, weights = integrate_marginal(
                self.grid, push_count, column, self.spec.H
            )
            particles = estimator.particles[:, column]
            if column == 0:
                particles = np.exp(particles)
            if column == 2:
                particles = particles * self.spec.H
            distances.append(
                float(
                    wasserstein_distance(values, particles, weights, estimator.weights)
                )
            )
        return distances


if __name__ == "__main__":
    ProposalComparison().run()
