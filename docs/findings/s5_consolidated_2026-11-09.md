# S5 consolidated: the full baseline ladder on the gantry target (2026-11-09)

Consolidation of all S5 runs on `feat/gantry-reality-gap`: the baseline
ladder B0–B6 and experiments E1 (in-distribution + OOD) and E2 on the
gantry (G-robot) and direct-force (G0) targets. The study's central
question — does an internal simulator help an agent adapt to objects with
unknown dynamics — gets a decisive answer on this embodiment: **imagination
does not help; the best methods are the conservative controller (B0) and
the frozen pretrained policy (B1), and every imagination variant makes
adaptation worse.**

## Setup

- Target: gantry executor (0.12 m/s, lane 0.5 m, f-max 15) and the
  direct-force executor as the well-specified contrast.
- Baselines share the B1 pretrained checkpoint (800k steps, gantry) for
  B1–B6; B0 is the analytic controller with per-push SysID. Protocol:
  8 in-distribution or 4 OOD objects × 20 episodes (adaptation) or 1
  episode (frozen).
- Evidence: `artifacts/dyna_baselines_v2/`, `artifacts/e2_v2/`,
  `artifacts/ood_v2/`, `artifacts/conservative_b0/`,
  `artifacts/sac_gantry_lane05/`, plus the per-part findings:
  [`b0_conservative_controller_2026-11-08.md`](b0_conservative_controller_2026-11-08.md),
  [`b1_gantry_sac_2026-11-08.md`](b1_gantry_sac_2026-11-08.md),
  [`b2_b6_imagination_gantry_2026-11-08.md`](b2_b6_imagination_gantry_2026-11-08.md),
  [`e2_parameter_error_sensitivity_2026-11-08.md`](e2_parameter_error_sensitivity_2026-11-08.md),
  [`gantry_reality_gap_2026-11-08.md`](gantry_reality_gap_2026-11-08.md).

## Results: the baseline ladder (gantry target)

B2–B6 and E2 cells come from the corrected rollout implementation
(real-state-only starts); B0 and B1 are unaffected. In-distribution
(8 objects) and OOD splits (4 objects each), frozen and adapted success:

| Method | in-dist | mass OOD | friction OOD | COM OOD | joint OOD |
|---|---:|---:|---:|---:|---:|
| B0 conservative + SysID | **88%** | **49%** | **74%** | **99%** | **48%** |
| B1 SAC, frozen | 75% | 75% | 75% | **100%** | 50% |
| B1 SAC, fine-tuned | 42% | 28% | 34% | 72% | 20% |
| B2 nominal Dyna | 15% | 18% | 25% | 22% | 14% |
| B3 point-estimate Dyna | 15% | 12% | 26% | 29% | 20% |
| B4 ensemble Dyna | 19% | 15% | 26% | 22% | 20% |
| B5 ensemble + filter | 4% | 1% | 1% | 5% | 11% |
| B6 oracle Dyna | 15% | 11% | 21% | 20% | 10% |

Direct-force target (G0): B0 0%; B1 frozen 0% (immediate toppling);
B2–B6 0–1%; B6 on OOD topples 54–100% of episodes.

## Results: E2 sensitivity (G0 target)

With the force-target starting policy (12.5% held-out frozen), the
19-condition oracle-Dyna sweep (0 to ±50% error in one parameter) is flat:
10.0–17.5% success with no parameter dominating, and the ε=0 oracle (11.9%)
sits at the frozen (12.5%) and fine-tuning (11.2%) levels.

## Conclusions

1. **Imagination hurts adaptation on the robot target — everywhere.**
   Every Dyna variant (B2–B6) underperforms both B1-frozen and B1
   fine-tuning on every in-distribution and OOD split, and B5's filter
   nearly immobilizes the agent (1–11%). The oracle collapse (B6: 15% in
   distribution, 10–21% OOD, 54–100% toppling on force OOD) isolates the
   cause: the harm is the direct-force model-class mismatch leaking into
   the training signal, not parameter error — B6 imagines with the true
   parameters.
2. **Fine-tuning is destructive from this start.** On the gantry,
   20 episodes of fine-tuning cut frozen success on every split
   (75→42% in-dist; 75→28, 75→34, 100→72, 50→20% OOD). Progress-seeking
   updates from the strong frozen policy walk it into worse pushes,
   repeating the KUKA-study pattern.
3. **The embodiment decides the winners.** B0 and B1-frozen, the methods
   that never adapt with simulated data, are the only strong performers,
   and only on the gantry: B0 48–99% OOD, B1-frozen 50–100% OOD with zero
   topples. The gantry's velocity cap is the safety mechanism that makes
   both possible (B0's force law assumes exactly the tracking the cap
   provides), while the same methods collapse on direct force.
4. **Parameter accuracy does not bind.** E2's flat sweep shows estimate
   error within ±50% changes nothing on G0; combined with B6, the
   imagination's failure is structural (model class), not parametric.
5. **The study's answer, in one sentence.** On the gantry target, an
   internal simulator helps neither as data nor as a shield — the
   conservative analytic controller (88%) and the frozen pretrained
   policy (75%) are the strongest methods, and imagination in every
   tested form makes adaptation worse.

## Remaining work

E3-G2 (synthetic structural mismatch levels) and the ρ/horizon ablations
remain open; the M6 robot-policy demonstration and the documentation sweep
are the closing tasks.
