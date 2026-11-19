# E2: parameter-error sensitivity on the well-specified target (2026-11-08)

The E2 sensitivity sweep (outline §9.3): the oracle-Dyna method with known
error injected into one internal-simulator parameter at a time, on the
well-specified (G0, direct-force) target. Result: **performance is flat
across ±50% parameter error in every direction** — the whole sweep sits at
the frozen/fine-tuning level. Parameter accuracy is not the binding
constraint; on G0 the imagination is neutral, unlike its destructive
effect on the gantry target.

## Setup

- Starting policy: a force-target SAC pretrained at lane 0.5, f-max 15,
  400k steps (`artifacts/sac_force_lane05/best_model`; 50% in-training
  validation).
- Method: B6 oracle Dyna — 8 rollouts × horizon 2 per real push into the
  SAC replay, 8 objects × 20 episodes on the direct-force target, with the
  oracle's true parameters multiplied by the error factor on one named
  parameter.
- Anchor: the same checkpoint evaluated frozen (1 episode) and with plain
  fine-tuning (20 episodes), same protocol.
- Factors per the outline: {0, ±10%, ±25%, ±50%} (factors 0.5–1.5).
- Evidence: `artifacts/e2/results_*.json` (19 conditions) and
  `results_b1_anchor.json`.

## Results

Rollout start states come from recorded real target states only (the
corrected implementation). The previous-implementation numbers are kept in
`artifacts/e2/`; the corrected rerun lives in `artifacts/e2_v2/` and changed
cells by noise-scale amounts without changing any conclusion.

Anchor (no imagination):

| Condition | Success | Topple |
|---|---:|---:|
| Frozen | 12.5% | 0.12 |
| Fine-tuning | 11.2% | 3.75 |
| Oracle Dyna, ε = 0 | 11.9% | 16.9% |

Sensitivity matrix (success, topples in parentheses):

| Factor | mass | μ | h_c |
|---|---:|---:|---:|
| ×0.50 | 11.2% (13.8) | 10.6% (20.6) | 11.9% (16.2) |
| ×0.75 | 12.5% (17.5) | 10.0% (16.2) | 12.5% (16.9) |
| ×0.90 | 13.8% (15.0) | 11.2% (14.4) | 11.2% (17.5) |
| ×1.00 | 11.9% (16.9) | 11.9% (16.9) | 11.9% (16.9) |
| ×1.10 | 10.6% (15.6) | 14.4% (16.2) | 13.1% (16.2) |
| ×1.25 | 13.8% (15.0) | 17.5% (18.1) | 16.9% (18.1) |
| ×1.50 | 12.5% (18.1) | 15.6% (19.4) | 14.4% (13.8) |

## What the numbers show

1. **No parameter dominates.** The full matrix spans only 10.0–17.5%
   success with no monotone trend in any row: ±50% error moves outcomes by
   a few points, comparable to the run-to-run noise between cells. To the
   study's E2 question — which latent parameter must be estimated
   accurately for imagination to remain useful — the measured answer on G0
   is: none of them, within this range.
2. **Imagination is neutral on the well-specified target.** Oracle Dyna
   (11.9%) sits at the frozen (12.5%) and fine-tuning (11.2%) levels:
   mixing correct-model simulated data neither helps nor hurts a weak
   policy on a matching target. This is the mirror image of the gantry
   result, where the same method collapsed a strong 75% policy to
   11.9–33.1%. Together: imagination's effect is dominated by the
   starting-policy strength and the model-class mismatch, not by estimate
   accuracy.
3. **Topples shift slightly with imagination.** All Dyna cells topple
   more than frozen (16–21% vs 0.12 per object) while fine-tuning topples
   3.75 — imagined rollouts mildly encourage riskier pushes even when the
   model is right, consistent with the mixed-buffer distribution shift.
4. **The weak starting policy limits resolution.** The force checkpoint
   scores 12.5% held-out (50% in-training validation), matching the
   historical force-baseline level. With most episodes failing regardless,
   the sweep can only resolve large sensitivity effects — and it found
   none. A stronger G0 policy could in principle reveal finer structure,
   but the qualitative conclusion (flat, sub-dominant) is unlikely to
   change.

## Caveats

- Single seed per cell; the flatness claim rests on the absence of
  systematic trends across 19 conditions, which seed noise cannot create.
- E2 uses the 8-object in-distribution set on G0 per the outline; OOD
  sensitivity is not covered here.
- The rollout ratio (8 rollouts × horizon 2 per real push) is fixed; the
  outline's ρ ablation remains open and could interact with sensitivity.
