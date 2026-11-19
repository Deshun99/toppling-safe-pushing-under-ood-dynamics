"""Volume-weighted grid reference for the known-start replay posterior.

Uniform initial grids cover the full z prior. Adaptive midpoint quadrature
subdivides cells carrying most mass at ANY history prefix, retaining all other
cells. Compare independently initialized grids before treating this as a
reference. Refinement stability is empirical, not a bound on quadrature error.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

from src.evaluation.system_identification.shared.common import (
    load_action_records,
)
from src.evaluation.system_identification.shared.constants import (
    CONVERGENCE_DISTANCE,
    CONVERGENCE_LOG_EVIDENCE,
    PARAMETERS,
    RETAINED_MASS,
    Grid,
)
from src.evaluation.system_identification.shared.grid import (
    compare_grids,
    create_uniform_cells,
    evaluate_cells,
    integrate_marginal,
    load_checked_grid,
    refine_grid,
    summarize_grid,
)
from src.evaluation.system_identification.shared.metrics import (
    convert_to_physical_parameters,
    summarize_distribution,
)
from src.evaluation.system_identification.shared.predictive import (
    evaluate_predictive_mixture,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.agents.system_identification.smc2 import SMC2SysID
from src.agents.system_identification.sysid import Z_HI, Z_LO
from src.evaluation.system_identification.shared.common import ReplayCountingSimulator
from src.simulation.modelling.object import CuboidSpec


def plot_reference_comparison(
    reference: Grid, comparisons: list[dict], output: Path, spec: CuboidSpec
) -> None:
    figure, axes, limits, labels = _prepare_reference_axes(comparisons, spec)
    for row, comparison in enumerate(comparisons):
        step = comparison["step"]
        for col, axis in enumerate(axes[row]):
            bins = np.linspace(*limits[col], 41)
            values, weights = integrate_marginal(reference, step, col, spec.H)
            axis.hist(
                values,
                bins=bins,
                weights=weights,
                density=True,
                histtype="step",
                color="black",
                linewidth=2,
                label="Grid reference",
            )
            _plot_saved_particles(
                output, spec, axis, col, comparison, step, bins, row, labels
            )
    figure.savefig(output / "marginals.png", dpi=130)
    plt.close(figure)


def _prepare_reference_axes(
    comparisons: list[dict], spec: CuboidSpec
) -> tuple[plt.Figure, np.ndarray, tuple, tuple[str, ...]]:
    figure, axes = plt.subplots(
        len(comparisons),
        len(PARAMETERS),
        figsize=(12, 3 * len(comparisons)),
        squeeze=False,
        layout="constrained",
    )
    limits = (
        (np.exp(Z_LO[0]), np.exp(Z_HI[0])),
        (Z_LO[1], Z_HI[1]),
        (Z_LO[2] * spec.H, Z_HI[2] * spec.H),
    )
    labels = ("Mass (kg)", "Friction μ", "COM height (m)")
    return figure, axes, limits, labels


def _label_reference_axis(
    spec: CuboidSpec,
    axis: plt.Axes,
    col: int,
    row: int,
    comparison: dict,
    step: int,
    labels: tuple[str, ...],
) -> None:
    axis.axvline(
        spec.theta[col],
        color="crimson",
        linestyle="--",
        label="Ground truth",
    )
    status = (
        "cross-grid stable"
        if comparison["cross_grid"]["stable"]
        else "grid NOT converged"
    )
    axis.set(
        xlabel=labels[col],
        ylabel="Density",
        title=f"Push {step} · {status}",
    )
    if row == 0 and col == 0:
        axis.legend(fontsize=8)


def _plot_saved_particles(
    output: Path,
    spec: CuboidSpec,
    axis: plt.Axes,
    col: int,
    comparison: dict,
    step: int,
    bins: np.ndarray,
    row: int,
    labels: tuple[str, ...],
) -> None:
    for trial in comparison["smc2"]:
        saved = np.load(output / f"smc_seed{trial['seed']}_step{step}.npz")
        theta = convert_to_physical_parameters(saved["z"], spec.H)
        axis.hist(
            theta[:, col],
            bins=bins,
            weights=saved["weights"],
            density=True,
            histtype="step",
            alpha=0.8,
            label=f"SMC² seed {trial['seed']}",
        )
    _label_reference_axis(spec, axis, col, row, comparison, step, labels)


def write_reference_report(comparisons: list[dict], output: Path) -> None:
    lines = _build_interval_table(comparisons)
    lines += [
        "",
        "All intervals are marginal 95% intervals, not joint regions.",
        "",
        "| Push | SMC² seed | Unique particles | COM interval (m) | Covers true COM |",
        "|---|---|---|---|---|",
    ]
    for comparison in comparisons:
        for trial in comparison["smc2"]:
            low, high = trial["distribution"]["intervals_95"]["h_c"]
            covered = trial["distribution"]["covers_truth"]["h_c"]
            lines.append(
                f"| {comparison['step']} | {trial['seed']} | {trial['unique_particles']} | [{low:.4f}, {high:.4f}] | {covered} |"
            )
    _finish_reference_report(comparisons, output, lines)


def _build_interval_table(comparisons: list[dict]) -> list[str]:
    lines = [
        "# Known-start posterior reference",
        "",
        "The reference uses the same uniform-in-z prior, observation scales, scoring window, and recorded pushes as SMC². Cells cover the full prior; their volumes enter the quadrature weights. Local refinement changes resolution, not prior probability.",
        "",
        "Convergence checks compare independently initialized grids using normalized marginal Wasserstein distance (<0.005 of the prior range) and log-evidence change (<0.2). These are empirical stability checks, not rigorous integration-error bounds. Broad cells are integrated within each marginal; intervals invert their piecewise-uniform CDF.",
        "",
        "| Push | Cross-grid stable | Reference mass interval (kg) | Friction interval | COM interval (m) |",
        "|---|---|---|---|---|",
    ]
    for comparison in comparisons:
        intervals = comparison["reference"]["intervals_95"]
        formatted = [
            f"[{intervals[name][0]:.4f}, {intervals[name][1]:.4f}]"
            for name in PARAMETERS
        ]
        lines.append(
            f"| {comparison['step']} | {comparison['cross_grid']['stable']} | "
            + " | ".join(formatted)
            + " |"
        )
    return lines


def _append_interpretation(comparisons: list[dict], lines: list[str]) -> None:
    final = comparisons[-1]
    misses = sum(
        (not trial["distribution"]["covers_truth"]["h_c"] for trial in final["smc2"])
    )
    interpretation = (
        "This directly demonstrates loss of plausible posterior mass in the particle approximation for this dataset; the corrected likelihood does not itself require those narrow COM intervals."
        if misses
        else "No final COM intervals exclude truth in these runs. Truth inclusion alone does not establish agreement with the grid distribution; inspect the complete marginals."
    )
    lines += [
        "",
        "## Finding",
        "",
        f"The stable reference intervals contain all three true parameters after every push. At the final push, {misses}/{len(final['smc2'])} SMC² seeds exclude the true COM height. {interpretation}",
        "",
        "The next justified experiment is improved particle exploration (tempering/rejuvenation) against this deterministic reference. This result does not establish that the observation model is correct on other objects or actions.",
    ]


def _finish_reference_report(
    comparisons: list[dict], output: Path, lines: list[str]
) -> None:
    lines += [
        "",
        "See marginals.png for the distributions, comparison.json for means, correlations and predictive distributions, and refinements.json for convergence history. Raw grids and SMC² particle snapshots are saved as NPZ files.",
        "",
        "Reference predictive distributions use four independent batches of 256 posterior draws, with locations sampled inside grid cells. Predictive Monte Carlo errors are recorded separately. SMC² predictions use its weighted particle mixture, retaining at least 99.9% of mass. Development predictions do not influence fitting or grid refinement.",
        "",
        "Interpretation: where the reference is stable and broad but SMC² is narrow or misses plausible regions, the difference is a sampling-approximation failure. Where a stable reference also excludes truth, investigate the likelihood model. Unconverged steps cannot adjudicate that distinction.",
        "",
        "This is a single-object development experiment, not population calibration. No production estimator settings were changed by this runner.",
    ]
    if all(
        (
            row["cross_grid"]["stable"]
            and all(row["reference"]["covers_truth"].values())
            for row in comparisons
        )
    ):
        _append_interpretation(comparisons, lines)
    (output / "report.md").write_text("\n".join(lines) + "\n")


class PosteriorReferenceComparison:
    def run(self) -> None:
        self._parse_args()
        self._load_inputs()
        self._prepare_output()
        self._run_comparisons()
        self._finish()

    def _parse_args(self) -> None:
        self._create_parser()
        self.parser.add_argument("--refinements", type=int, default=5)
        self.parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
        self.parser.add_argument("--sample-stride", type=int, default=12)
        self.parser.add_argument("--max-new-cells", type=int, default=20000)
        self.parser.add_argument(
            "--resume", action="store_true", help="Reuse matching saved grids in --out"
        )
        self.args = self.parser.parse_args()

    def _load_inputs(self) -> None:
        if (
            len(set(self.args.grids)) < 2
            or min(self.args.grids) < 2
            or self.args.refinements < 1
            or (self.args.max_new_cells < 8)
            or (self.args.sample_stride < 1)
        ):
            self.parser.error(
                "Use at least two initial grids (size >=2) and at least one refinement"
            )
        self.spec = CuboidSpec(**json.loads(self.args.config.read_text())["object"])
        self.records = load_action_records(self.args.dataset, accepted_only=True)
        if not self.records:
            self.parser.error("No accepted training records")
        self.evaluation = load_action_records(self.args.evaluation)

    def _prepare_output(self) -> None:
        self.args.out.mkdir(parents=True, exist_ok=self.args.resume)
        (self.args.out / "config.json").write_text(
            json.dumps(
                {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(self.args).items()
                }
                | {
                    "object": json.loads(self.args.config.read_text())["object"],
                    "retained_mass": RETAINED_MASS,
                    "convergence_distance": CONVERGENCE_DISTANCE,
                    "convergence_log_evidence": CONVERGENCE_LOG_EVIDENCE,
                },
                indent=2,
            )
        )

    def _run_comparisons(self) -> None:
        sim = ReplayCountingSimulator()
        try:
            estimator = SMC2SysID(
                (self.spec.w, self.spec.L, self.spec.H),
                n_theta=1,
                n_state=1,
                start_std=(0, 0, 0, 0),
                sample_stride=self.args.sample_stride,
                sim=sim,
            )
            grids = self._refine_initial_grids(estimator)
            self._compare_particle_runs(grids, estimator, sim)
        finally:
            sim.close()

    def _finish(self) -> None:
        print(f"Saved reference comparison to {self.args.out}")

    def _create_parser(self) -> None:
        self.parser = argparse.ArgumentParser(description=__doc__)
        self.parser.add_argument(
            "--dataset",
            type=Path,
            default=Path(
                "artifacts/sysid_development_known_after/dataset_varied_durations.json"
            ),
        )
        self.parser.add_argument(
            "--config",
            type=Path,
            default=Path("artifacts/sysid_development_known_after/config.json"),
        )
        self.parser.add_argument(
            "--evaluation",
            type=Path,
            default=Path("artifacts/sysid_development_known_after/held_out.json"),
        )
        self.parser.add_argument(
            "--out", type=Path, default=Path("artifacts/sysid_reference_posterior")
        )
        self.parser.add_argument("--grids", type=int, nargs="+", default=[12, 24])

    def _compare_grids(
        self, grids: list[Grid], estimator: SMC2SysID
    ) -> tuple[list[dict], Grid]:
        reference = grids[-1]
        comparisons = []
        for step in range(1, len(self.records) + 1):
            weights, evidence = reference.posterior(step)
            comparison = {
                "step": step,
                "cross_grid": compare_grids(grids[-2], reference, step, self.spec),
                "reference": summarize_grid(reference, step, self.spec),
                "log_evidence": evidence,
                "reference_predictive": evaluate_predictive_mixture(
                    reference.points,
                    weights,
                    estimator,
                    self.evaluation,
                    self.spec,
                    grid=reference,
                ),
                "smc2": [],
            }
            comparisons.append(comparison)
        return comparisons, reference

    def _save_refinement(
        self,
        size: int,
        level: int,
        grid: Grid,
        changes: list[dict] | None,
        start: float,
        refinements: list[dict],
    ) -> None:
        print(
            f"Grid {size}, refinement {level}: {len(grid.low)} cells; stable={(None if changes is None else [item['stable'] for item in changes])}",
            flush=True,
        )
        refinements.append(
            {
                "initial_size": size,
                "level": level,
                "cells": len(grid.low),
                "seconds_so_far": time.perf_counter() - start,
                "changes": changes,
            }
        )
        np.savez_compressed(
            self.args.out / f"grid{size}_level{level}.npz",
            low=grid.low,
            high=grid.high,
            log_likelihood=grid.log_likelihood,
        )
        report_path = self.args.out / "refinements.json"
        report_path.write_text(json.dumps(refinements, indent=2))

    def _record_particle_comparison(
        self,
        seed: int,
        step: int,
        theta: np.ndarray,
        weights: np.ndarray,
        smc: SMC2SysID,
        estimator: SMC2SysID,
        comparison: dict,
    ) -> None:
        np.savez_compressed(
            self.args.out / f"smc_seed{seed}_step{step}.npz",
            z=theta,
            weights=weights,
        )
        comparison["smc2"].append(
            {
                "seed": seed,
                "distribution": summarize_distribution(theta, weights, self.spec),
                "unique_particles": len(np.unique(theta, axis=0)),
                "ess": smc.last_ess,
                "predictive": evaluate_predictive_mixture(
                    theta, weights, estimator, self.evaluation, self.spec
                ),
            }
        )

    def _compare_particle_runs(
        self, grids: list[Grid], estimator: SMC2SysID, sim: ReplayCountingSimulator
    ) -> None:
        comparisons, reference = self._compare_grids(grids, estimator)
        for seed in self.args.seeds:
            smc = SMC2SysID(
                (self.spec.w, self.spec.L, self.spec.H),
                n_theta=64,
                proposal="fixed",
                n_state=1,
                start_std=(0, 0, 0, 0),
                seed=seed,
                sample_stride=self.args.sample_stride,
                sim=sim,
            )
            for step, comparison in enumerate(comparisons, 1):
                smc.update(self.records[:step])
                theta, weights = (smc.particles, smc.weights)
                self._record_particle_comparison(
                    seed, step, theta, weights, smc, estimator, comparison
                )
        (self.args.out / "comparison.json").write_text(
            json.dumps(comparisons, indent=2)
        )
        plot_reference_comparison(reference, comparisons, self.args.out, self.spec)
        write_reference_report(comparisons, self.args.out)

    def _load_or_refine_grid(
        self, size: int, level: int, grid: Grid | None, estimator: SMC2SysID
    ) -> Grid:
        path = self.args.out / f"grid{size}_level{level}.npz"
        if self.args.resume and path.exists():
            next_grid, _ = load_checked_grid(
                path, self.spec, self.records, self.args.sample_stride
            )
        elif level:
            assert grid is not None
            next_grid = refine_grid(
                grid, estimator, self.records, self.args.max_new_cells
            )
        else:
            next_grid = evaluate_cells(
                *create_uniform_cells(size),
                estimator,
                self.records,
            )
        return next_grid

    def _refine_initial_grids(self, estimator: SMC2SysID) -> list[Grid]:
        grids, refinements = [], []
        for size in self.args.grids:
            start = time.perf_counter()
            print(f"Initial grid {size}³", flush=True)
            grid: Grid | None = None
            for level in range(self.args.refinements + 1):
                next_grid = self._load_or_refine_grid(size, level, grid, estimator)
                if level:
                    assert grid is not None
                    previous, grid = (grid, next_grid)
                    changes = [
                        compare_grids(previous, grid, step, self.spec)
                        for step in range(1, len(self.records) + 1)
                    ]
                else:
                    grid = next_grid
                    changes = None
                self._save_refinement(size, level, grid, changes, start, refinements)
            grids.append(grid)
        return grids


if __name__ == "__main__":
    PosteriorReferenceComparison().run()
