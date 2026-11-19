# System identification

This package owns SysID experiments, comparison runners, and diagnostics.
Production estimators remain in `src/agents/system_identification/`; none of
these workflows is required to call either estimator.

## Layout

```text
system_identification/
├── shared/        # Reusable data loading, protocols, metrics, and grids
├── comparisons/   # Estimator accuracy, replay budgets, proposal comparisons
├── diagnostics/   # Identifiability, calibration, likelihood, and posterior checks
└── demos/         # Examples of estimator API usage
```

## Runtime estimators

| File | Purpose |
|---|---|
| [`sysid.py`](../../agents/system_identification/sysid.py) | CMA-ES estimator, physics parameter transforms, hypothesis/result dataclasses, and trajectory loss. |
| [`smc2.py`](../../agents/system_identification/smc2.py) | SMC² estimator with covariance-shaped proposals by default, optional gradual evidence updates, diagnostics, and posterior plotting. |
| [`scoring.py`](../../agents/system_identification/scoring.py) | Shared scoring indices; warmup is replayed but excluded from fitting. |

## Research entry points

Run modules from the repository root using `.venv/bin/python -m` followed by
the module name below. Output artifacts are separate from source code.

| Module suffix after `src.evaluation.system_identification.` | Role | When to use |
|---|---|---|
| `demos.smc2_demo` | Demonstration | Inspect the estimator API and posterior plots after each push. Also the VS Code debug entry point. |
| `diagnostics.identifiability` | Identifiability diagnostic | Study how push forces, heights, durations, and passive motion reveal physical parameters. |
| `diagnostics.calibrate_scales` | Calibration | Measure replay discrepancy at known physics before explicitly revising loss scales. Prints results without rewriting estimator constants. |
| `comparisons.benchmark_smc2` | Main comparison | Collect shared datasets and compare CMA-ES and SMC² errors, particle diversity, and replay costs. |
| `comparisons.compare_compute` | Budget comparison | Give CMA-ES the saved SMC² cumulative replay limits; optionally retain its previous best estimate with `--retain-best`. |
| `comparisons.compare_proposals` | Sampler comparison | Compare fixed and covariance-shaped proposals, grid agreement, and costs under shared replay ceilings, with an uncapped control. |
| `diagnostics.diagnose_smc2` | Failure investigation | Probe likelihood variance, collapse, residuals, COM likelihood profiles, and controlled ablations. |
| `diagnostics.reference_posterior` | Numerical posterior check | Integrate the same prior/likelihood on refined grids to distinguish sampling failure from model mismatch. Can be expensive. |
| `diagnostics.trace_truth_particle` | Oracle diagnostic | Insert exact physics at equal weight and track scores, resampling ancestry and accepted moves, alongside paired controls. Not a performance benchmark. |

The SMC² diagnostic tools were introduced while debugging SMC². They are retained to
reproduce the investigation and check future sampler changes. They are not
part of normal inference. The grid tool is particularly useful when a sampler
reports suspiciously narrow uncertainty. Their module paths moved under
`src.evaluation.system_identification.diagnostics`; older saved commands may use the former paths.

Shared code has explicit owners:

- `shared/common.py` loads saved action records and counts simulator replays.
- `shared/metrics.py` measures predictions and weighted particles.
- `shared/excitation.py` defines push sequences, settling, and data collection.
- `shared/grid.py` integrates posterior grids and checks cached likelihoods.
- `shared/predictive.py` evaluates posterior predictive mixtures.
- `shared/constants.py` holds fixed settings and shared records.

Runners separate argument parsing, data loading, execution, and reporting. Shared
helpers do not depend on a CLI runner. New output directories are required;
historical evidence and saved results are not rewritten. Grid reuse checks the
number of records and recomputes likelihoods at 32 evenly
spaced cell indices under the current object, records, and scoring stride;
a maximum absolute difference above `1e-8` rejects the cache. This checks a
sample of cells, not every saved value. It does not compare source hashes,
Git state, or installed dependency versions.

## Where to start

For a walkthrough, run `demos.smc2_demo`. For measured comparisons, run
`comparisons.benchmark_smc2 --smoke` first. Diagnostic tools are not prerequisites.

The [project README](../../../README.md) has commands and experiment options.
The [findings handoff](../../../docs/findings/smc2_development_handoff.md)
records the evidence, limitations, and next experiments. Saved historical
artifacts remain unchanged.

## Calling shared helpers

Helpers are module functions rather than classes used only to group functions.
Import them from the owning module:

| Module | Useful entry points |
|---|---|
| `shared.common` | `load_action_records`, `decode_action_record`, `load_training_sequences`, `load_metric_rows`, `ReplayCountingSimulator` |
| `shared.excitation` | `collect_records`, `get_regimes`, `settle`, `settle_or_replace_object` |
| `shared.metrics` | `evaluate_predictions`, `summarize_particles`, `convert_to_physical_parameters` |
| `shared.grid` | `load_checked_grid`, `create_uniform_cells`, `refine_grid`, `integrate_marginal` |
| `shared.predictive` | `evaluate_predictive_mixture` |

`load_action_records(path, accepted_only=True)` selects the records accepted
for fitting. Omit that option to load all records, including rejected training
pushes or held-out topples. `load_training_sequences(directory, sequence_names)`
loads accepted records from each named dataset. `ReplayCountingSimulator`
extends the simulator to count replays; it does not own experiment settings or
report generation. Helpers receive their calculation inputs explicitly.

## Existing scripts and saved results

Current runners save experiment settings and results without querying Git,
hashing source files, or collecting dependency inventories. Older saved results
may contain those fields; they describe the older run and are not required by
the current loaders. RL checkpoint identification and embedded training settings
are separate and remain supported.

The cleanup changed these interfaces. Update external scripts when reading new
results; historical files have not been rewritten:

| Earlier name | Current name or call |
|---|---|
| `SysID.update(..., incumbent=...)` | `SysID.update(..., previous_best_parameters=...)` |
| `CountingSim` | `ReplayCountingSimulator` |
| Record-loader wrapper classes | Functions in `shared.common` listed above |
| Metric, grid, excitation, and prediction wrapper classes | Functions in their owning modules listed above |
| Compute-comparison fields under `cma_es_matched`: `incumbent`, `incumbent_replays` | `previous_best_parameters`, `previous_best_replays` |
| Truth-trace fields `truth_lineage_count`, `truth_lineage_weight` | `truth_descendant_count`, `truth_descendant_weight` |

Truth-trial JSON now includes `initial_summary` in addition to per-update
summaries and resampling/move events. Plots handle runs without resampling;
first-resampling rank counts exclude those runs. The diagnostic observes the
sampler and does not protect injected truth particles from moves or removal.
