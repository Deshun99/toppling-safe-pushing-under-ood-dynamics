# B0 conservative controller: direct-force failure mode (2026-11-08)

The B0 baseline (conservative analytic controller + online SysID) fails
catastrophically on the direct-force target while succeeding on the gantry.
The cause is not estimation quality but a control-law mismatch: the force
law assumes velocity-tracking contact, which the gantry provides natively
and direct force does not. Recorded as a finding by decision; the
controller is intentionally left unchanged.

## Setup

- Controller (`src/agents/baselines/conservative_controller.py`): lowest
  height `h = 0.1H`, `tau = 0.5 s`, force `F = m̂(v/τ + μ̂g)` with target
  speed `v ≤ 0.12 m/s`, capped by half the tipping inequality when
  `h > ĥ_c`.
- Adaptation: CMA-ES SysID refits after every push from the 20 most recent
  records, replay budget 256, warm-started from the previous best; prior
  `(m̂, μ̂, ĥ_c) = (0.866, 0.5, 0.475H)`.
- Target: lane 0.5 m, `F_max = 15 N`; trace object `(m, μ, h_c) =
  (0.7, 0.4, 0.064)`, `(w, L, H) = (0.06, 0.09, 0.16)`.
- Evidence: smoke runs `artifacts/conservative_b0/smoke_force.json` and
  `smoke_robot.json`; the trace below.

## Results

Full run (8 objects × 20 episodes per executor):

| Executor | Success | Topple | Timeout | Mean final error |
|---|---:|---:|---:|---:|
| Direct force | 0% | 14% | 86% | 0.40 m |
| Gantry | **88%** | 0% | 12% | 0.017 m |

On the gantry, 7/8 objects reach 100% success with zero topples; the one
failure is the light, low-friction, high-COM object `(0.5, 0.3, 0.132)`.
Per-object summaries and SysID final estimates are in
`artifacts/conservative_b0/results_force.json` and `results_robot.json`.

Single force-target episode trace (object above):

| Step | d before | F (N) | x (m) | v (m/s) | θ̂ after (m, μ, h_c) |
|---|---:|---:|---:|---:|---|
| 0 | +0.500 | 4.42 | 0.00 → 0.298 | 0.00 → **1.17** | (0.94, 0.23, 0.038) |
| 1 | +0.202 | 2.21 | 0.298 → 0.778 | 1.17 → 0.75 | (0.71, 0.38, 0.036) |
| 2 | **−0.278** | 0.00 | 0.778 → 0.810 | 0.75 → 0.56 | — |
| 3–9 | ≤ −0.31 | 0.00 | → 0.849 (rest) | → 0.00 | — |

## What the numbers show

1. **The force law is a velocity-tracking law, and direct force is
   open-loop acceleration.** The intended Δv = v_target holds only when
   `(m̂, μ̂) = (m, μ)`; otherwise
   `Δv = v_target + [(m̂/m)·μ̂ − μ]·g·τ`. Estimation error converts
   one-to-one into speed error, at ~0.5 m/s per 0.1 of friction error.
2. **The first push already loses the episode.** The prior μ̂ = 0.5 against
   true μ = 0.4 delivers 1.17 m/s instead of the intended 0.12 m/s and
   0.298 m of displacement (10× the intended ~3 cm). The block overshoots
   the ±0.05 m band in two pushes, and a forward-only controller can never
   pull it back — steps 3–9 are F = 0 with the block at rest 0.35 m past
   the goal.
3. **Adaptation is slower than the failure.** SysID reaches μ̂ = 0.38 after
   the first push, but the overshoot has already happened; the estimate
   converges on a lost episode.
4. **The gantry inverts the mechanism into success.** Its 0.12 m/s motor
   cap enforces the target speed regardless of F, so estimation error only
   changes whether the pad sustains contact, never the block speed. The
   embodiment supplies the safety the controller assumes. This is the
   G-robot structural difference (velocity cap) acting as a **safety
   mechanism rather than noise** — the same cap that halves the force
   executor's displacement in the reality-gap measurement.
5. **SysID degrades on the robot target, exactly as the G-robot row
   predicts.** Final mass errors are 6–55% on the force target but 2–227%
   on the gantry: robot-recorded trajectories are fitted with direct-force
   replays, so the optimizer compensates the executor's velocity cap and
   force-ceiling semantics with wrong parameters (m̂ mostly inflated). Yet
   B0 still succeeds at 88% — the velocity cap makes the force law tolerant
   of exactly the parameter error it induces. The imagination's model-class
   mismatch is visible in the estimates even where it does not hurt the
   policy.
6. **Implication for the baseline family.** On the direct-force target the
   controller needs an estimate-error-safe force law (pure friction balance
   `F = m̂μ̂g`, or a safety fraction on the acceleration term). Left as-is
   by decision; the failure mode is the recorded result.

## Caveats

- The trace uses the per-push-refit protocol with a 20-record history
  window, not the full history.
- B0 pushes are weakly exciting (lowest height, bounded force), so SysID
  accuracy here understates what richer excitation could achieve; the
  force-vs-gantry estimate gap is the point, not the absolute accuracy.
