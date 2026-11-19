# Short curriculum training check

The physics policy completed three curriculum stages at 0.10, 0.15 and 0.20 m,
with **exactly 500 target-environment decisions per stage**. The run completed
1500 decisions, 107 episodes and 187 gradient updates; all logged update metrics
were finite. Training and checkpoint evaluation ran end to end.

The strongest observed improvement is at 0.10 m: frozen held-out success rose
from 12/24 to 21/24. Success also increased at the longer goals, but those
stages remain unreliable and mechanical failures increased. This short check
supports a learning signal, not reproduction of the paper's full benchmark.

## Settings and training corrections

The run used seed 42, the paper's gantry executor, 32 SMC² hypotheses with two
MCMC moves, self-attention with hidden width 256, force bound 15 N, 40 decisions
per episode and 0.08 m goal tolerance. Physics uncertainty reward weight was
50. The user selected minibatches of eight to shorten this validation run;
the paper preset uses 64 and 10000 decisions per stage.

Training warm-started from the diagnostic checkpoint at interaction 201,
before any optimizer updates. Its actor and critic matched the untrained
baseline. Replay and optimizers were fresh, with `learning_starts=0`; the
old warm-up replay was not reused. Earlier interrupted training runs are
diagnostic only and are excluded from the reported training results.

The initial diagnostic runs exposed two implementation bugs, corrected before
the reported run:

- Terminal execution failures were dropped from learning. Their terminal
  reward now trains the critic, without inventing a completed trajectory or
  adding a failed execution to SysID history.
- Curriculum collection could overshoot a stage budget by finishing an
  object's episodes. Collection now stops at the exact decision budget.
  Counts include probes and failed execution attempts.

## Frozen held-out checks

Each row uses the same eight fixed in-distribution objects, three episodes
per object and seed 42 at the checkpoint's native goal. Neural weights stay
frozen: every evaluation made **zero additional gradient updates**. Physics
identification continues across an object's three episodes and resets for the
next object. Untrained baselines use identical initial neural weights with
only the goal changed; trained checkpoints accumulate earlier stages.

| Goal | Policy | Success | Toppled episodes | Timeouts | Other execution failures |
| --- | --- | --- | --- | --- | --- |
| 0.10 m | Untrained | 12/24 (50.0%) | 8 | 1 | 3 |
| 0.10 m | Trained (500 decisions) | 21/24 (87.5%) | 2 | 1 | 0 |
| 0.15 m | Untrained | 5/24 (20.8%) | 14 | 4 | 1 |
| 0.15 m | Trained (1000 decisions) | 8/24 (33.3%) | 9 | 3 | 4 |
| 0.20 m | Untrained | 2/24 (8.3%) | 12 | 7 | 3 |
| 0.20 m | Trained (1500 decisions) | 5/24 (20.8%) | 10 | 2 | 7 |

Other execution failures count contact timeouts, travel limits and sideways
plane failures; pre-push toppling is included in the toppling outcome.
At 0.20 m, the trained policy still had seven mechanical execution failures
and ten topples. Mean final absolute goal error also increased at 0.15 and
0.20 m despite their higher success counts, so improved success does not imply
that every motion metric improved. Overshoot remains a concern.

## Evidence and scope

[The saved summary](evidence/paper_curriculum/summary.json) contains settings,
actual stage budgets, training-quarter summaries, loss/gradient/timing
statistics, and the frozen evaluation counts. Raw logs and checkpoints stay
under `artifacts/paper_curriculum_fixed`; frozen results are under
`artifacts/paper_curriculum_eval_stage0`, `stage1` and `stage2`. Baseline results
are under `artifacts/paper_curriculum_baseline` and
`artifacts/paper_curriculum_eval_baseline_d0.15` / `d0.20`.

These are one training seed and short-goal in-distribution checks. The 0.08 m
acceptance tolerance makes the first goal comparatively easy. The 13-stage
curriculum to 1.0 m, the paper's 10000-decision stage budgets, its five-episode
held-out protocol, OOD performance and additional training seeds remain
unvalidated. Training-quarter comparisons use changing objects and are
descriptive rather than controlled learning comparisons.
