# PR record: gantry reality gap and B0–B6 baselines

- **This repository:** PR #2
- **Status:** merged into `main` as a record of the reality-gap and baseline contribution

## Code in this PR

- `src/agents/baselines/conservative_controller.py` — B0: conservative analytic
  pushing from SysID point estimates.
- `src/agents/rl/physics_providers.py` — B2/B3/B4/B6 physics sources for the
  internal simulator (nominal, point estimate, ensemble, oracle).
- `src/agents/rl/imagination.py` — Dyna-style short policy rollouts in the
  internal simulator, starting only from recorded real target-environment
  states; scoring uses the target environment's current reward rules.
- `src/agents/rl/filter.py` — B5 pessimistic filter over the ensemble's
  worst-case tilt predictions.
- `src/evaluation/baselines/evaluate_conservative.py` — B0 evaluation protocol
  (online SysID adaptation) and M6 GUI demos.
- `src/evaluation/rl/evaluate_dyna.py` — B2–B6 adaptation evaluation on the
  held-out set, with its own fine-tuning episode loop.
- `src/simulation/tools/validation/gantry_vs_force.py` and
  `gantry_episode_span.py` — S2/S5 harnesses comparing the gantry executor
  with direct force and measuring the episode-span feasibility ceiling.
- `tests/test_conservative_controller.py`, `tests/test_gantry_vs_force.py`,
  `tests/test_imagination_baselines.py` — regression coverage for the above.

All modules were ported to the current executor and environment APIs
(admittance-controlled `GantryPushExecutor.execute(cuboid, h, F, tau)`,
`PushEnv` action records, and the current `RewardModel` scoring rules).

## What it contributed

- **Target-environment integration (S3):** `PushEnv(robot=True)` runs the
  gantry on the environment's shared physics client, homes it per episode,
  and reports realized execution per step. Contact timeouts, travel limits,
  lateral drift, and pre-push toppling are handled as reported episode
  outcomes. Checkpoint metadata preserves the embodiment.
- **Reality-gap measurement (S4):** commanded-versus-realized execution
  reports (pad-contact normal force, contact duration, pad-centre height at
  first contact, approach-contact measurements, stray contact, execution
  timing). At 0.12 m/s across 96 sampled pushes, the realized-to-commanded
  force ratio was 0.70 ± 0.29 with displacement MAE 0.137 m and correlation
  0.88. Gantry contact and actuator dynamics introduce a measured
  target-environment mismatch (G-robot).
- **B0–B6 baseline comparison (S5):** frozen evaluation and adaptation
  results for the direct-force and gantry embodiments across in-distribution
  and OOD object sets, with rollout starts sampled only from recorded real
  target-environment states.

## Findings and evidence

- `docs/findings/gantry_reality_gap_2026-11-08.md` — the S4 reality-gap study.
- `docs/findings/b0_conservative_controller_2026-11-08.md`,
  `b1_gantry_sac_2026-11-08.md`, `b2_b6_imagination_gantry_2026-11-08.md`,
  `e2_parameter_error_sensitivity_2026-11-08.md`,
  `s5_consolidated_2026-11-09.md` — the baseline studies.
- `docs/findings/evidence/gantry_vs_force_2026-11-08/` — raw per-push rows
  and summaries backing the reality-gap numbers.
- `docs/resources/robot-push.mov` — recording of a gantry push.

The policy interface remains `(h, F, τ)`; the internal simulator and SysID
replay remain direct-force by design. The full original description lives in
this repository's PR #2.
