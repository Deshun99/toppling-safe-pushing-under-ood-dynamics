# Reverse-curriculum training for delta toppling-safe pushing (2026-11-03)

Reproduction commands for the two conditions compared here: the physics
simulator plus SysID, and the learned world-model ablation. Both ran as a single
curriculum to 1.0 m.

Physics (simulator + SysID), curriculum to 1.0 m at horizon 40:

```bash
.venv/bin/python -m src.agents.rl.training.train_delta \
  --distribution-config configs/rl/training_distributions/uniform_excluding_evaluation_regions.json \
  --f-max 15 --batch-size 64 --backend smc2 --particles 32 --forecast-workers 8 \
  --candidates 4 --max-batches 2 --trajectory-feature-strategy self-attention \
  --learning-starts 200 --reward-strategy goal_proximity --info-gain-beta 50 \
  --epsilon 0.08 --topple-penalty 10 --no-progress-penalty 1.5 \
  --curriculum-distances 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --curriculum-steps 10000 --max-decisions 40 \
  --episodes-per-object 10 --train-freq 8 --gradient-steps 1 \
  --out-dir artifacts/delta/curriculum/physics
```

Learned world-model ablation (`--world-model learned`, single run to 1.0 m;
`--particles` and `--forecast-workers` are inert in learned mode):

```bash
.venv/bin/python -m src.agents.rl.training.train_delta \
  --distribution-config configs/rl/training_distributions/uniform_excluding_evaluation_regions.json \
  --batch-size 64 --backend smc2 --particles 64 --forecast-workers 1 \
  --candidates 4 --max-batches 2 --trajectory-feature-strategy interpolate \
  --learning-starts 50 --reward-strategy goal_proximity \
  --epsilon 0.08 --topple-penalty 10 --no-progress-penalty 1.5 \
  --curriculum-distances 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --curriculum-steps 10000 --max-decisions 40 \
  --world-model learned \
  --out-dir artifacts/delta/curriculum/learned
```

Branch: `residual-actor-simulator-integration-4` at `f71dabf`.

The delta agent trained directly at the full `0.25 m` target distance never
learned the task: success stayed at 0-5% across long runs and the policy
collapsed. A reverse curriculum over target distance, warm-started between
stages with the new `--init-from` flag, produced a working agent. It first
cleared the `0.25 m` target at ~74% success versus 0% for direct training, and a
later single-command curriculum with a distance-scaled action horizon carried the
same approach out to the full `1.0 m` target at ~81% success (see "Extension to
1.0 m"). These are single-seed exploratory runs, not multi-seed validation; treat
the numbers as directional evidence, not final metrics. A separate ablation (see
"Ablation: imagination fidelity") compares the physics simulator against a learned
world model and a model-free agent, and finds topple-safety tracks the fidelity of
the forward model.

Common settings across the runs below unless noted: `--backend smc2
--particles 32 --forecast-workers 8 --candidates 4 --max-batches 2
--trajectory-feature-strategy self-attention --reward-strategy goal_proximity
--info-gain-beta 50 --epsilon 0.08 --max-decisions 10`.

## Direct training at 0.25 collapses

With the original reward (`R_topple 50`, no no-progress penalty), a ~430-object
run drove force to literally zero: the last 10% of non-probe pushes had mean,
median and max force all `0.00 N`. Success fell from a ~5.5% random-warmup
baseline to 0% and reward slid to about -4. The critic correctly learned that
`Q(push) < Q(do nothing)` because success was rare (traverse 0.25 m and stop in
an 8 cm window without toppling) while toppling (-50) was common and standing
still was nearly free. A better optimiser of that objective finds the
do-nothing optimum faster, which is why the learned policy underperformed random.

Reward shaping alone only moved the failure around. Dropping `R_topple` to 10
and adding `--no-progress-penalty 4` broke the do-nothing collapse (force
recovered, the object moved on 58% of pushes) but swapped it for reckless
over-pushing: overshoot climbed to 76% and reward kept sliding. Softening to
`--no-progress-penalty 1.5` sat between timid and reckless. The task is a
precise, irreversible ballistic-landing problem, and single-step reward tuning
could not reliably cross it.

## Overshoot is the dominant failure, and it is irreversible

Pushing is forward-only (fixed direction, 1D translation), so the object's
position is monotonically non-decreasing and an overshoot cannot be undone.
`remaining_distance_after_m` in `actions.jsonl` is unsigned (`|d|`) and cannot
distinguish overshoot from undershoot; the signed distance `d = target -
position` is element `[0]` of `final_observation` in `episodes.jsonl`
(`d < 0` = past the target).

In the reckless run, 51% of episodes ended past the target (`d < -0.08`) versus
32% short and 17% in the window, and 110/226 episodes entered the 8 cm window
and were then pushed back out. Success also requires `|v| < 0.05`, so the object
must *coast to rest* in the window, not pass through it.

## SysID: mass is weak; tip-margin matters early but decouples at competence

Mass is poorly identified from sliding motion. Under a near-random policy with 8
and then 32 particles, mass relative error ran 110-230% (median ~120%, only ~8%
of episodes within 20%), while friction and COM height were weak-to-mediocre.
This is identifiability, not particle budget: sliding constrains only
`a = F/m - mu*g` (one equation, two unknowns), and the mass prior is wide
(`0.2-3.0 kg`, 15x). The 0.5 N probe is also sub-threshold (`mu*m*g` is ~5-30 N
for typical objects), so the first push often carries no information.

To separate "needs accurate mass" from "needs the tipping boundary", per-episode
logging now records the mass-free tip-stability margin `w/2 - mu*h_c` (the
hidden-physics core of the topple condition `(F/mg)(h - h_c) <= w/2 - mu*h_c`),
estimated vs true, plus the raw posterior estimate. See
`sysid_accuracy_record` in `train_delta.py`.

- **Near-random policy:** success tracked tip-margin accuracy far more than mass.
  Successes had `tip_margin_abs_err` 0.011 vs 0.047 for failures (4x), while mass
  error barely separated. But this was largely object-difficulty confounding:
  low-COM objects are both easier to succeed on and easier to identify.
- **Trained policy (stage 3):** success decoupled from both.
  `tip_margin_abs_err` was 0.022 (success) vs 0.025 (failure), and mass error
  0.45 vs 0.42 (no separation). A competent policy succeeds across ~72% of
  objects even when its physics estimate is wrong.

The stronger reading of the imagination hypothesis is the second one: once the
agent is skilled, success does not require an accurate physics estimate. Mass
error also dropped from ~130% to ~45% in the trained runs; whether that is the
`--info-gain-beta 50` shaping finally driving informative pushes is unconfirmed
and worth a controlled check.

## Reverse curriculum over target distance

Start where success is easy and dense (goal one push away), accumulate
successes, then push the target distance outward. This manufactures the
successful examples the critic was starved of, instead of hoping to find them in
sparse data (which upsampling cannot create). Each stage warm-starts from the
previous stage's `delta_latest.pt` via `--init-from`, which loads weights and
architecture from the checkpoint while taking all reward and task settings from
the new command line. Stages used `--topple-penalty 10 --no-progress-penalty
1.5`; stage 1 was fresh with `--learning-starts 200`, later stages used
`--init-from` with `--learning-starts 50` (no uniform-random warmup).

| stage | target (m) | outcome (success / overshoot / topple) | notes |
|-------|-----------|-----------------------------------------|-------|
| 1 | 0.10 | ~74% / ~15% / ~11% | fresh; plateaued over ~141 objects |
| 2 | 0.15 | ~86% / ~11% / ~5% | opened at 57% (warm-start carryover) |
| 3 | 0.20 | ~73% / ~12% / ~8% | struggled (overshoot peaked 46%) for ~35 objects, then cracked and held over ~86 objects |
| 4 | 0.25 | ~74% / ~13% / ~4% (last quarter) | climbed 39->48->62->74% over ~85 objects; the distance where direct training gave 0% |

Each stage opened higher than the last started, because the warm-started policy
walked into the larger distance already competent. Stage 3 showed the
characteristic shape clearly: overshoot sat at ~40% while success sagged, then
overshoot collapsed to ~10% and success locked into the low-70s and stayed there.
Residual stage-3 failures (final half): overshoot 12%, topple 8%, undershoot 6%.

## Commands (reproduction)

This section records the earlier manual, one-stage-per-run approach to 0.25 m
(before the single-command curriculum). The authoritative reproduction commands
for the physics and learned conditions are at the top of this document; the block
below is kept for history.

Run from the repo root. Stage 1 is a fresh run; stages 2-4 warm-start from the
previous stage's `delta_latest.pt` via `--init-from`, changing only
`--target-distance`, the `--init-from` path and `--out-dir`. The warm-start
stages omit `--f-max` and `--trajectory-feature-strategy` because architecture
and action bounds are inherited from the checkpoint (passing a conflicting
`--f-max` is rejected). Each run was launched in the background with its stdout
redirected to a log; that redirect is not part of the command.

Stage 1 (fresh, target 0.10):

```bash
.venv/bin/python -m src.agents.rl.training.train_delta \
  --distribution-config configs/rl/training_distributions/uniform_excluding_evaluation_regions.json \
  --f-max 15 --batch-size 64 --backend smc2 --particles 32 --forecast-workers 8 \
  --candidates 4 --max-batches 2 --trajectory-feature-strategy self-attention \
  --learning-starts 200 --reward-strategy goal_proximity --info-gain-beta 50 \
  --target-distance 0.10 --epsilon 0.08 \
  --topple-penalty 10 --no-progress-penalty 1.5 \
  --episodes-per-object 10 --train-freq 8 --gradient-steps 1 --max-decisions 10 \
  --out-dir artifacts/delta/curriculum/stage1_d010
```

Stage 2 (warm-start, target 0.15):

```bash
.venv/bin/python -m src.agents.rl.training.train_delta \
  --distribution-config configs/rl/training_distributions/uniform_excluding_evaluation_regions.json \
  --batch-size 64 --backend smc2 --particles 32 --forecast-workers 8 \
  --candidates 4 --max-batches 2 \
  --init-from artifacts/delta/curriculum/stage1_d010/delta_latest.pt \
  --learning-starts 50 --reward-strategy goal_proximity --info-gain-beta 50 \
  --target-distance 0.15 --epsilon 0.08 \
  --topple-penalty 10 --no-progress-penalty 1.5 \
  --episodes-per-object 10 --train-freq 8 --gradient-steps 1 --max-decisions 10 \
  --out-dir artifacts/delta/curriculum/stage2_d015
```

Stage 3 (target 0.20) and stage 4 (target 0.25) are identical to stage 2 with
these three substitutions:

| stage | `--init-from` | `--target-distance` | `--out-dir` |
|-------|---------------|---------------------|-------------|
| 3 | `artifacts/delta/curriculum/stage2_d015/delta_latest.pt` | `0.20` | `artifacts/delta/curriculum/stage3_d020` |
| 4 | `artifacts/delta/curriculum/stage3_d020/delta_latest.pt` | `0.25` | `artifacts/delta/curriculum/stage4_d025` |

## Extension to 1.0 m (single-command curriculum)

The four manual stages were later automated into one command, a `--curriculum-distances`
list trained for a fixed `--curriculum-steps` push budget each, warm-starting the
model in place between stages (no checkpoint juggling). Every episode and action
row is tagged with its stage and target distance. This let the schedule push out
to the full `1.0 m` target. This feature is uncommitted on the branch as of this
writing.

**The action horizon has to scale with distance.** An earlier curriculum at
`max_decisions 20` succeeded through ~0.4 m but then collapsed: success fell off from ~0.45 m and by
0.60 m every episode hit the 20-decision cap and none reached the goal (100%
timed-out-short, 0% success, 0% topple). The cause is that the agent learns a
topple-safe style of many small pushes, so pushes-per-episode grows with distance
(from ~10 at 0.2 m to 20-capped at 0.6 m) rather than staying flat. The physics curriculum uses
`max_decisions 40`: at 1.0 m only ~6% of episodes touch the cap, 0% time
out short, and the average is ~15 pushes, so 40 is comfortable. The binding
constraint at long range is therefore force and topple, not decisions.

**0.1-m steps are enough past 0.4 m.** With 0.1-m steps beyond 0.4 m (0.5, 0.6,
... 1.0) and horizon 40, each stage opened competent and recovered; no stage
stalled. The fine 0.05-m steps only earned their keep early, where precise
landing is learned. Final per-stage results:

| dist (m) | success% | overshoot% | undershoot% | topple% | avg pushes |
|----------|----------|------------|-------------|---------|------------|
| 0.50 | 79 | 13 | 6 | 10 | 15.9 |
| 0.60 | 62 | 15 | 19 | 13 | 17.0 |
| 0.70 | 70 | 13 | 12 | 13 | 14.4 |
| 0.80 | 77 | 5 | 16 | 16 | 13.6 |
| 0.90 | 80 | 5 | 14 | 15 | 14.1 |
| 1.00 | 81 | 5 | 12 | 14 | 15.3 |

0.60 m was the weakest stage (62%), an early 0.1-m jump; it recovered on its own
but would take a 0.05-m intermediate if tightened. Topple is the dominant far-distance failure (~14% at 1.0 m), the
long-range analog of overshoot: covering a meter needs harder pushes, which trade
against staying upright. Within the 1.0 m stage topple declined from ~18% to ~10%
as it matured, and success held ~80%, so an early-stage topple spike (31% over a
35-episode window) was a hard batch, not a drift.

### SysID at distance

The 1.0 m run logs SysID accuracy on every episode, and it is far better than the
early near-random runs (which had ~100-130% mass error):

| parameter | median rel. error | within 20% | within 50% |
|-----------|-------------------|------------|------------|
| mass m | 0.25 | 43% | 73% |
| friction mu | 0.25 | 41% | 80% |
| COM h_c | 0.40 | 23% | 67% |
| tip-margin (w/2 - mu*h_c) | 0.020 m | - | - |

Accuracy improves with distance: mass median error falls from 0.35 at 0.5 m to
0.19 at 1.0 m. Longer pushes slide the object much further, which is rich
excitation for identifying mass and friction, so the far-distance task does better
SysID than the short-distance one could. COM height stays the weakest (median
0.40, only 23% within 20%): it is only observable in near-tipping motion, and this
topple-averse agent rarely produces it. The tip-margin survives at ~2 cm because
friction is well identified and the geometry w is observed.

Unlike the 0.25 m regime above, SysID accuracy here correlates with success:
successful episodes average 0.38 mass error versus 0.63 for failures, and 0.025 m
tip-margin error versus 0.034 m. The likely reason is that over a 1 m trajectory
of many pushes small physics errors compound, so an accurate estimate matters more
for landing than it did over a short hop. Caveat: this is partly object-difficulty
confounding, since the hard high-COM and heavy objects are both harder to identify
and harder to land, so some of the correlation is "hard objects fail at both"
rather than bad SysID causing failure.

## Ablation: imagination fidelity (physics vs learned vs model-free)

The study compares three ways for the agent to anticipate a push, all under the
same reverse curriculum and reward:

- **Physics** (`--world-model physics`, `artifacts/delta/curriculum/physics/`):
  InternalSim rollouts across the SysID particle ensemble (the main method above).
- **Learned** (`--world-model learned`, `artifacts/delta/curriculum/learned/`): an
  MLP (`src/agents/rl/models/latent_world_model.py`) predicts the next trajectory
  from the observation, action reference, candidate delta, and previous real
  trajectory. No particles, no belief over physics, no PyBullet; trained online by
  MSE against the actual next trajectory. With no posterior, info-gain shaping is
  inert here by construction.
- **Model-free** (`--world-model none`, `artifacts/delta/curriculum/baseline/`):
  the controller supplies zero forecasts, so the actor and critic get no imagined
  rollout at all.

Matched-distance success / topple (%), same curriculum and reward for all three:

| dist (m) | physics | learned | model-free |
|----------|---------|---------|------------|
| 0.10 | 31 / 33 | 57 / 35 | 22 / 51 |
| 0.15 | 40 / 24 | 70 / 28 | 1 / 80 |
| 0.20 | 65 / 23 | 68 / 29 | 0 / 98 |
| 0.25 | 62 / 17 | 65 / 32 | 0 / 100 |
| 0.30 | 78 / 9 | 73 / 22 | 0 / 100 |
| 0.35 | 71 / 10 | 70 / 24 | 0 / 100 |
| 0.50 | 79 / 10 | 58 / 36 | - |
| 0.60 | 62 / 13 | 58 / 36 | - |
| 0.70 | 70 / 13 | 49 / 47 | - |
| 0.80 | 77 / 16 | 41 / 55 | - |
| 0.90 | 80 / 15 | 44 / 50 | - |
| 1.00 | 81 / 14 | 47 / 43 | - |

A monotonic ladder of imagination fidelity. **Physics** holds ~74-81% success and
~10-16% topple all the way to 1.0 m. **Learned** reaches 1.0 m but is 2-3x less
safe: topple ~22-55%, success degrading to ~47% at 1.0 m, and the gap widens with
distance. **Model-free** collapses almost immediately: by 0.2 m it topples ~98% of
episodes and essentially never succeeds, so the run was stopped at 0.35 m (0%
success, 100% topple from 0.25 m) rather than continued up a harder curriculum. (The
0.1-0.15 m rows understate physics: those were the first stages it trained, while
still naive.)

Two mechanisms behind the ladder:

- **Learned is accurate on average but blind at the tip.** `world_model_loss` fell
  from 0.033 to 0.011 and kept improving, so by its own training metric the world
  model looks accurate, yet it topples 43% of the time at 1.0 m. The aggregate
  trajectory MSE is dominated by the position channel, so it reads low while tilt
  prediction, the safety-critical channel, stays poor. A low loss coexists with
  catastrophic toppling. A per-episode `wm_tilt_peak_abs_err` (predicted vs actual
  peak tilt) was added to the logging to measure this directly, but postdates this
  run.
- **Model-free has nothing to anticipate with.** With zero forecasts the critic
  cannot foresee a topple at all, so the agent can only learn safety from the raw
  terminal penalty, which is far too weak: it flails (undershoot) at short distance
  and tips on essentially every push beyond 0.15 m.

Read together: the forward model is not a refinement, it is what makes the task
learnable, and its fidelity sets the safety ceiling. The explicit physics simulator
plus SysID does the topple-safety work; an approximate learned model loses the tip
dynamics and is unsafe at range; no model is unusable past 0.2 m.

Caveats: single seed, and the schedules are not a perfectly controlled A/B. The
learned run started at a degenerate 0.05 m stage (inside the `epsilon 0.08`
window, so trivially solved in one push, 10,000 wasted pushes) and used finer
early steps. But the topple gap is large and consistent across every far distance,
so it is robust. The learned model did reach 1.0 m, so it learns something; it is
just unsafe.

## Tooling added (committed in `f71dabf`)

- **`--init-from PATH`** (train mode only): warm-start training from a saved
  delta checkpoint. Weights and action bounds load from the checkpoint; reward,
  environment, schedule and distribution come from the new args; the loaded
  policy drives from the first push with no uniform-random warmup. Mutually
  exclusive with `--checkpoint`. This is what makes staged curricula a sequence
  of runs with no code changes.
- **Tip-margin SysID logging**: per-episode `sysid_est_{m,mu,h_c}`,
  `tip_margin_{est,true}` and `tip_margin_abs_err`, so success can be correlated
  with topple-boundary accuracy rather than raw mass error.

## Open questions and caveats

- Single-seed, exploratory. Needs multi-seed repeats and held-out OOD evaluation
  before any claim is load-bearing.
- All stages ran at a loosened `epsilon 0.08`, not the stricter default `0.05`.
  The success numbers are at that tolerance; tightening epsilon toward the real
  tolerance (warm-started from a final checkpoint) is untested.
- Success plateaued around ~74-81% across distances from 0.2 to 1.0 m, which
  looks like the ceiling of this setup rather than a per-distance artifact. The
  tail is overshoot at mid distances, shifting to topple and undershoot at long
  distances.
- The action horizon was hand-tuned (20 then 40). It must scale with target
  distance, and the right value is coupled to how hard the agent is willing to
  push; a schedule that grows the horizon with distance was not built.
- Overshoot remains the largest residual failure at every distance. The force
  action granularity is coarse (`delta_scale` force 0.4 -> 6 N against
  `f_max 15`), which may cap landing precision; finer resolution is untested.
- Fixed distance per stage risks a distance-specialised policy. The observation
  carries signed `d`, so the policy can generalise, but a range-sampling
  schedule (sample target distance from `[small, current_max]`, grow the max)
  would be smoother than discrete stages. Not yet built.
- The success-vs-SysID decoupling and the mass-error drop to ~45% both deserve a
  controlled check (e.g. info-gain ablation) before being treated as results.
