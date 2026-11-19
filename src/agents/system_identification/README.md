# System identification estimators

This package contains runtime inference: `sysid.py` (CMA-ES), `smc2.py`
(SMC²), and `scoring.py` (shared replay scoring indices).

Experiments, comparisons, calibration, and diagnostics live in
[`src/evaluation/system_identification/`](../../evaluation/system_identification/README.md).
They are optional workflows, separate from estimator execution.

## Runtime API

Both estimators accept an `ActionRecord` history and return a `SysIDResult`
containing a point estimate and weighted hypotheses. Records contain physical
`PushAction` values, the measured start state, warmup count, and dense trajectory.
Geometry is supplied to the estimator; hidden target parameters are not.

- `SysID.update(history)` refits the complete history with CMA-ES. Optional
  `replay_budget` caps this call's simulator replays. Pass
  `previous_best_parameters=previous_result.point.theta` to rescore the previous
  estimate on the current history and let it compete with the new search.
  Its replays count toward the same budget; earlier losses are not reused.
  Close the estimator with `close()` when finished.
- `SMC2SysID.update(history)` processes only appended records for the same
  object. Edited or shortened histories are rejected. Use its context manager
  or `close()` to release an internally created simulator; a supplied simulator
  remains caller-owned. `particles` and `weights` expose the full population;
  `result.hypotheses` contains only the retained unique particles.
- Both exclude warmup observations from scoring while replaying the warmup.
  `sample_stride` preserves the original recording's sampling phase.

CMA-ES hypothesis weights are computed from retained trajectory losses, with
scale `0.05 * best_loss + 1e-12`. There is no constructor `temperature` argument;
the removed argument did not affect these weights. SMC² uses likelihood-based
weights instead, so the two methods' loss and uncertainty summaries are not
interchangeable.

The [project README](../../../README.md)
describes proposal choices, gradual evidence updates, and runnable examples.
