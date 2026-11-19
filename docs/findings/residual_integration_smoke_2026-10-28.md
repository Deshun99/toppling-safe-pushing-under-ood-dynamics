# Residual integration smoke findings (2026-10-28)

These checks exercise the proposed architecture on draft PR #19. They validate
the implementation path, not a useful or safe learned policy. The checkpoint
has only two optimizer updates; no comparison against trained baselines or
multi-seed OOD evaluation is established.

## Reproduce the small integration run

Use a fresh output directory for each run:

```bash
.venv/bin/python -m src.agents.rl.training.train_delta \
    --smoke --check-gradients --out-dir artifacts/delta/integration_smoke
.venv/bin/python -m src.evaluation.rl.evaluate_delta \
    --checkpoint artifacts/delta/integration_smoke/delta_final.pt \
    --benchmark-config configs/rl/evaluation_sets/fixed_cases.json --split mass_ood \
    --smoke --out-dir artifacts/delta/fixed_ood_smoke
```

The commands above use the current module/config paths after the main-branch
merge; the measurements below remain the historical 2026-10-28 results.

The first command uses seed 42, the reserved-region training distribution
(originally named `with_holes`), a
50 N actuator bound, two episodes on one sampled object, three decisions per
episode, four full SMC² physics particles and no rejuvenation for smoke timing.
These reduced budgets are not recommended inference/training settings.

## Observed collection and learning

| Quantity | SMC² training smoke |
| --- | ---: |
| Real target interactions | 6 |
| Probes / actor-generated actions | 2 / 4 |
| Complete-return examples / optimizer updates | 4 / 2 |
| Successes / topples / actual timeouts / gate-censored stops | 0 / 0 / 2 / 0 |
| Proposal candidates / invalid final actions / unsafe predictions / passing candidates | 64 / 42 / 4 / 18 |
| Candidate particle simulations | 76 |
| Actor parameter change, L2 norm | 0.0189750 |

The actor had nonzero gradients and changed parameters. This does not show that
those changes improved task performance. Candidate simulation counts exclude
SysID fitting replays and learning/diagnostic forecast regeneration.

## Direct gradient versus recomputed forecasts

The diagnostic perturbs normalized residual coordinates and reruns PyBullet.
Coordinate order is height, force, duration. The after-training example produced:

| Derivative | Height | Force | Duration |
| --- | ---: | ---: | ---: |
| Direct critic delta gradient | 0.0857080 | 0.00678532 | -0.0758612 |
| Rerun finite difference, epsilon 0.001 | 0.0963882 | -0.00632554 | -0.0758618 |
| Rerun finite difference, epsilon 0.003 | 0.0966613 | -0.00661860 | -0.0647356 |

Relative vector errors were 0.13768 and 0.17659; cosine similarities were
0.99221 and 0.98428. **The force coordinate has the opposite sign**, despite the
high aggregate cosine. Do not summarize this as validation of the actor gradient.
The missing simulator derivative matters. Duration discretization and contact
transitions can also affect finite differences; broader gradient and target-return
checks remain open.

## OOD limitation already observed

The frozen evaluation smoke selected the first configured mass-OOD object
(0.2 kg) and ran two short episodes. Both toppled, over five real pushes total,
despite the model-based gate. This is a failure to record and investigate, not
evidence that the gate is safe on the bounded object domain. Hypothesis coverage,
contact reconstruction, inference calibration and the near-untrained proposer
all require evaluation; this run does not isolate the cause.

CMA-ES controller and training/evaluation smoke paths also run. Independent
per-object adaptation writes separate checkpoints; model and replay do not
transfer between adaptation objects. These are functional checks only.

## Validation and next experiments

- Final automated suite: **269 passed**, 13 existing matplotlib/pyparsing
  deprecation warnings, 91% aggregate coverage under the repository's coverage
  configuration. Command: `.venv/bin/python -m pytest`.
- Ruff lint/format checks pass for changed Python files; `git diff --check`
  passes. Runtime: Python 3.12.14, CPU PyTorch 2.3.1 and PyBullet 3.2.7.
- Automated coverage exercises actual-record alignment, fresh historical
  forecasts, input exclusions, reference updates, both SysID backends, rule-gate
  behavior, censored episodes, checkpoint loading and benchmark membership.
- Physics sanity checks pass with 100% agreement on the existing diagnostic grid.
- Controlled model-mismatch experiments, calibrated uncertainty, reliable useful
  actor gradients, matched-budget baseline comparisons and multi-seed OOD
  performance remain unvalidated.
- Keep this implementation in draft while investigating the force-gradient
  disagreement and model-approved OOD toppling. Do not tune on a held-out smoke
  result and then report that same object as an untouched test.
