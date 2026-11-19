# B2-B6: imagination baselines on the gantry target (2026-11-08)

The five imagination baselines (Dyna variants from outline §10) were run
through the shared adaptation protocol on the 8-object in-distribution
set, on both executors. Result: **every imagination variant degrades
adaptation on the gantry target**, including the oracle that imagines with
the true parameters. The harm is not parameter error — it is mixing
structurally mismatched (direct-force) simulated experience into a
competent gantry-trained policy.

## Setup

- Starting checkpoint: `artifacts/sac_gantry_lane05/best_model/best_model.pt`
  (B1, 75% frozen success) — all baselines share it and differ only in
  adaptation.
- Protocol: 8 objects × 20 episodes × ≤10 decisions; after every real push
  the imagination rolls the current policy out from recorded real target
  states (8 rollouts, horizon 2) under the provider's physics into the SAC
  replay; SAC updates on the mixed buffer. SysID refits per push (window 20,
  budget 256) for B3/B4/B5. B5 additionally filters every action through
  worst-case ensemble tilt predictions (near-topple fraction 0.8, zero-force
  fallback).
- Providers: B2 nominal training-mean parameters; B3 point estimate; B4
  top-K weighted ensemble; B5 B4 + filter; B6 true parameters.
- Evidence: `artifacts/dyna_baselines/results_{b2..b6}_{robot,force}.json`.

## Results

Rollout start states come from recorded real target states only (the
corrected implementation). The previous-implementation numbers are kept in
`artifacts/dyna_baselines/`; the corrected rerun lives in
`artifacts/dyna_baselines_v2/` and changed cells by noise-scale amounts
without changing any conclusion.

| Baseline | Gantry success | Gantry topple | Gantry timeout | Force success | Force topple |
|---|---:|---:|---:|---:|---:|
| B1 frozen (reference) | 75.0% | 0.00 | — | 0.0% | 0.75 |
| B1 fine-tuning | 41.9% | 0.00 | — | 1.2% | 12.75 |
| B2 nominal Dyna | 15.0% | 7.5% | 76.3% | 0.6% | 66.2% |
| B3 point-estimate Dyna | 15.0% | 6.9% | 78.1% | 0.0% | 59.4% |
| B4 ensemble Dyna | **19.4%** | 5.6% | 75.0% | 0.0% | 61.9% |
| B5 ensemble + filter | 3.8% | 4.4% | 91.9% | 1.2% | 14.4% |
| B6 oracle Dyna | 15.0% | 8.1% | 76.9% | 0.0% | 65.0% |

Gantry per-object success (B1 frozen: 75% aggregate):

| Baseline | Per-object success |
|---|---|
| B2 | 30, 10, 0, 0, 25, 15, 10, 30% |
| B3 | 20, 5, 0, 5, 25, 35, 5, 25% |
| B4 | 20, 10, 0, 35, 30, 20, 5, 35% |
| B5 | 5, 0, 0, 10, 0, 0, 0, 15% |
| B6 | 20, 5, 0, 5, 30, 15, 10, 35% |

## What the numbers show

1. **Imagination hurts adaptation from a strong start, on every variant.**
   The best imagination method (B4, 19.4%) loses to plain fine-tuning
   (41.9%) and far below the frozen policy (75%). The ranking B4 > B3 > B6
   > B2 ≈ B3 ≈ B6 > B5 survives the correction,
   but the entire family is below the model-free baselines.
2. **The oracle result isolates the cause.** B6 imagines with the *true*
   parameters, so parameter error explains none of its collapse to 15%.
   The harm is the model-class mismatch: direct-force forecasts of
   gantry-calibrated actions predict large impulses the real executor never
   delivers, so imagined gradients push the policy away from the actions
   that work. This is the G-robot mismatch acting inside the training
   signal, not in the estimates.
3. **The filter (B5) is the shield story at its extreme.** Worst-case
   direct-force predictions veto nearly every action (3.8% success, 91.9%
   timeouts — the policy is almost motionless). Yet on the force target B5
   is the only Dyna variant that reduces toppling (14.4% vs ~60%): the
   shield does its job exactly where its simulator model is right, and
   fails closed where it is wrong.
4. **The force target keeps the cross-executor story.** All Dyna variants
   collapse to ~0% with ~60% toppling under direct force, matching B1's
   transfer failure; imagination provides no transfer either.
5. **Against the baseline ladder, B0 remains the bar.** B0's 88% (no
   imagination, no learning) dominates every learned method on the robot
   target; B1 frozen (75%) is second. The original Dyna hypotheses — that
   estimated simulated experience helps adaptation — is decisively negative
   on this embodiment, which is itself the study's H3/H4 answer for
   G-robot.

## Caveats

- Single starting checkpoint and one seed per object; the qualitative
  ordering (all imagination variants below model-free adaptation) is large
  enough to survive seed noise, but individual baseline gaps (B3 vs B4)
  may not.
- Rollout settings (8 rollouts, horizon 2, direct-into-replay mixing)
  follow the original imagination-baselines design; the ρ and horizon
  ablations from the outline remain open and could shift the picture.
- The B5 filter design (worst-case over hypotheses, zero-force fallback)
  is one faithful reading of "pessimistic"; a fail-open variant would
  produce different numbers.
