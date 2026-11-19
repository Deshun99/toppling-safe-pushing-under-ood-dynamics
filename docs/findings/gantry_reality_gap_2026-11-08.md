# Gantry reality gap: measured force-vs-gantry execution mismatch (2026-11-08)

First measured commanded-versus-realized execution statistics for the gantry
embodiment (`feat/gantry-reality-gap`, off `feat/2-joined-robot-arm`). This is
the measurement half of S4 from `robot_integration_plan.md`: before the
"G-robot" row can be added to the reality-gap table, the executor's noise
channels must be quantified. The embodiment is the X/Z sliding gantry, not
the plan's KUKA IIWA — the row below is drafted accordingly.

## Setup

- Harness: `src/simulation/tools/validation/gantry_vs_force.py` — each
  `(h, F, τ)` command executed once by `ForcePushExecutor` and once by
  `GantryPushExecutor` on identical fresh environments.
- 6 objects sampled from the default ranges (H 0.12–0.24 m, m 0.52–1.36 kg,
  μ 0.40–0.69) × 8 commands each, seed 0, lane 0.5 m, F ≤ 15 N,
  τ ∈ (0.05, 0.50) s, DT = 1/240 s.
- Commanded height is clamped into `[0.02, H − 0.02]` m (the pad half-height
  window) before sampling — an embodiment constraint the force executor does
  not share. Both executors receive identical actions within that subset.
- Realized values come from `GantryPushExecutor.last_execution`
  (`ExecutionReport`): pad-contact normal force summed over pad–block contact
  points, contact time inside the push window, pad height at first contact,
  approach impulse, stray-contact steps, and phase timing.
- Evidence: development run at 0.08 m/s (seed 0,
  [`rows.csv`](evidence/gantry_vs_force_2026-11-08/rows.csv),
  [`summary.json`](evidence/gantry_vs_force_2026-11-08/summary.json)) and the
  validation re-measurement at 0.12 m/s (seeds 303 and 404,
  [`rows_012_seed303.csv`](evidence/gantry_vs_force_2026-11-08/rows_012_seed303.csv),
  [`rows_012_seed404.csv`](evidence/gantry_vs_force_2026-11-08/rows_012_seed404.csv),
  [`summary_012.json`](evidence/gantry_vs_force_2026-11-08/summary_012.json)).

## Results

| Quantity | 0.08 m/s (seed 0) | 0.12 m/s (seeds 303, 404) |
|---|---:|---:|
| Pushes | 48 | 96 |
| Contact failures | 0 | 0 |
| Stray (non-pad) arm contact | 0 | 0 |
| Displacement, mean (force vs gantry) | 0.266 vs 0.017 m | 0.154 vs 0.017 m |
| Displacement MAE / correlation | 0.249 m / 0.80 | 0.137 m / 0.88 |
| Peak tilt MAE | 0.536 rad | 0.350 rad |
| Topple outcomes (force vs gantry) | 20 vs 2 (62%) | 30 vs 6 (75%) |
| F realized / commanded | 0.58 ± 0.29 | 0.70 ± 0.29 |
| τ realized / commanded | 0.986 | 0.985 |
| Height error | −0.01 mm | −0.01 mm |

## What the numbers show

1. **Contact establishment is solved.** Zero failures and zero stray contacts
   across the sampled geometry confirm the recessed-bracket fix holds; the
   pad is the only link that ever touches the block.
2. **The height channel is essentially exact** (−0.01 mm). Unlike the plan's
   KUKA framing, where IK error contributes to G1, the gantry's direct slide
   gives ĥ ≈ h. Height is not a mismatch channel for this embodiment.
3. **Force is the G1 channel, and it is wide.** Realized contact force
   averages 0.58 (0.08 m/s) to 0.70 (0.12 m/s) of the commanded ceiling with
   std ~0.29 and a near-zero minimum: against light blocks the pad meets
   almost no resistance, so commanded `F` is a ceiling the object only
   occasionally realizes. The realized fraction rises with speed — the noise
   is speed-dependent. F̂ is also the quantity the imagination cannot
   predict from `(h, F, τ)` alone — the target's own dynamics determine it.
   This is simultaneously real G1 noise and a G2-style structural dependence.
4. **τ fidelity is high but not complete** (mean 0.985, min 0.55 at 0.12):
   contact breaks when the block outruns the pad or a weak force ceiling
   cannot sustain contact. Contact-time deficit is where impulse is lost,
   mirroring the KUKA impulse-control finding.
5. **Displacement is regime-capped, not noise-corrupted.** Robot dx stays
   ~0.017 m at both speeds for the sampled action mix, because for short and
   weak pushes the force ceiling — not the speed — is the binding constraint;
   the force executor's mean 0.154–0.266 m comes from free acceleration
   after the impulse. The displacement MAE (0.137–0.249 m) and tilt MAE
   (0.350–0.536 rad) are dominated by this scale gap, not by execution
   error — correlation 0.88–0.90 at 0.12 shows the ordering of commands
   survives.
6. **Safety asymmetry repeats the KUKA result.** Direct force topples
   30/96 pushes vs 6/96 on the gantry at 0.12 m/s (seed spread 20/6 and
   10/0): the same policy is literally gentler on the robot. Cross-executor
   success and topple rates are not directly comparable; evaluations must
   report them separately (outline M6 already requires this).

## Feasibility verdict for S5 (measured 2026-11-08)

`src/simulation/tools/validation/gantry_episode_span.py` ran a scripted
episode — 10 maximum pushes (h = 0.04 m, F = 15 N, τ = 0.5 s), then settling
— on 8 sampled objects per speed (evidence: `episode_span*.csv`):

| PUSH_SPEED | final x | in goal band | notes |
|---|---:|---:|---|
| 0.08 (previous default) | 0.40 m (0/8) | 0/8 | pad-travel bound; **infeasible** |
| 0.10 | 0.501–0.503 m | 8/8 | zero margin: bound equals the goal |
| 0.12 | 0.601–0.604 m | 0/8 | overshoots past 0.55; τ braking leaves headroom |
| 0.15 | — | — | X travel limit raises mid-episode |

Block displacement tracks the pad exactly at every speed: per-push increment
is speed × 0.5 s for all sampled objects (0.52–1.48 kg, μ 0.39–0.65), coasting
is ~0, and 15 N sustains every object. No topples at h = 0.04 even at
0.12 m/s. Conclusions:

1. The 10-decision budget at the previous default (0.08 m/s) **cannot span
   the 0.50 m lane** — no policy, however clever, can reach the goal. The
   default is now 0.12 m/s in `GantryPushExecutor` and `PushEnv(robot=True)`.
2. Feasibility needs PUSH_SPEED ≥ ~0.10 m/s; above ~0.13 m/s the X travel
   limit truncates the episode.
3. **0.11–0.12 m/s is the feasible window**: enough span with margin, inside
   travel, and it forces the policy to use τ for braking (which exercises
   the action space rather than a fixed full-duration script).

## The G-robot row (draft for `project_outline.md` §5.2)

| Level | Mismatch | Example implementation |
|---|---|---|
| **G-robot: Robot-executed target** | real measured execution noise in F̂ (ĥ ≈ exact, τ̂ ≈ commanded) plus unmodelled executor dynamics: velocity cap, pad compliance, approach impact | gantry executor beneath `(h, F, τ)`; imagination stays direct-force; realized `(ĥ, F̂, τ̂)` recorded per push |

Supporting sentences to add under the table:

> G-robot subsumes G1 for the force channel (measured, not injected) and adds
> a G2-style structural mismatch (velocity cap) that the direct-force model
> class cannot represent even with oracle parameters. The internal simulator
> remains direct-force by design, so robot records are not interchangeable
> with replay records.

Open decision: the default level. Options — (a) raise `PUSH_SPEED` to shrink
the regime gap, (b) keep 0.08 m/s and define G-robot as measured here,
(c) re-scale `F` per executor. The feasibility verdict above decides this:
option (b) leaves the lane unreachable, so S5 requires (a) with
`PUSH_SPEED ≈ 0.11–0.12 m/s`. The 0.12 m/s column above is the re-measurement
at that speed on fresh seeds (303, 404), so the row's F̂ and τ̂ ratios are
now speed-matched to the S5 setting; re-run `gantry_vs_force.py --push-speed`
only if the speed changes again. Option (c) weakens the "measured, not
injected" story and stays rejected.

## S3 env integration (applied 2026-11-08)

`PushEnv(robot=True)` executes episodes through the gantry on the shared
physics client: reset homes the arm (raise-then-retract), executor failures
(contact timeout, travel limit, lateral drift, pre-push topple) terminate
the episode with an `execution_status` in info, realized execution is
reported per step, and heights outside the pad window clamp with the clip
recorded as `h_cmd` vs `h_real`. `train_sac` records
robot/push_speed/target_distance in checkpoint metadata; `evaluate`
reconstructs it, and `--robot`/`--no-robot` override it for cross-executor
runs. The default push speed is 0.12 m/s everywhere. The full
train-to-evaluate round trip was validated end to end on a smoke run.

## Caveats

- The 0.08 m/s column is a single-seed development measurement; the
  0.12 m/s column is the two-seed validation (303, 404). Both are sampled
  from the default object ranges and are not held-out evaluation sets.
- F̂ is PyBullet's engine-reported contact normal force summed over contact
  points, not a force sensor; it is valid for within-engine comparisons only.
- Approach time and approach impulse are excluded from τ̂ and the trajectory;
  the block's residual motion after the push window is included in dx for
  both executors.
- The height clamp restricts the comparison to the gantry-feasible action
  subset; the force executor's full height range is not exercised.
- Gantry records remain non-interchangeable with direct-force SysID replay
  records (approach outside the record, force-ceiling semantics).
