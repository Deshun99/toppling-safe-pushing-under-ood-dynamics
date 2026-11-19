# CLAUDE.md — Project Context

## Implementation rules

- **Implement the requested research behavior first.** Do not add runtime Git
  queries, source hashing, dependency inventories, or generic provenance/metadata
  frameworks unless explicitly requested. Saving experiment settings and results
  does not justify building extra infrastructure.
- **Question an abstraction before renaming it.** If it has no necessary role in
  the requested behavior, remove it rather than give it a more elaborate name.
- **Names must explain the role or action.** Prefer `estimator`, `training_records`,
  and `push_count` over ambiguous `fit`, `data`, and `step`. A method's name must
  describe everything it does. Keep established mathematical notation where it
  is clear; do not make names longer without adding meaning.
- **Reuse one implementation for duplicated behavior.** Check existing owners
  before adding another loader, rollout loop, metric calculation, or conversion.
  Combine actual duplication without forcing unrelated algorithms together.
- **Keep control flow easy to follow.** Keep temporary values local and pass
  helper inputs explicitly. Do not store loop variables or intermediate
  calculations on `self` just to shuttle them between methods.
- **Keep functions typed and at most 30 lines**, including signatures and
  docstrings. Extract coherent operations, not arbitrary line ranges. Avoid
  one-line forwarding methods and classes that merely wrap another class.
- **A simplification must reduce reading effort.** Prefer fewer concepts,
  wrappers, and navigation hops. Group related helpers under an existing owner;
  do not create a new file or class for every helper. Comments should explain
  assumptions or reasons, not repeat the code or describe a generic lifecycle.
- **Keep cleanup focused.** Preserve numerical behavior unless a change is
  requested or a demonstrated bug requires a fix. Update callers and docs, run
  relevant checks, and leave unrelated code alone.

## What this project is

**Simulation-Assisted Toppling-Safe Pushing**

Research question: can a physics simulator serve as an RL agent's **imagination** when adapting to objects with unknown dynamics? An upright cuboid must be pushed toward a target without toppling. Geometry is observable; mass `m`, object-floor friction `μ`, and vertical COM height `h_c` are hidden and must be inferred from motion.

Two uses of the internal simulator are studied:
1. **Imagination as data** — Dyna-style short hypothetical rollouts mixed into SAC's replay buffer.
2. **Imagination as a shield** — a pessimistic filter that simulates candidate actions under the top-K physics hypotheses and rejects any predicted to topple.

All experiments are **sim-to-sim**: a target PyBullet env plays the "unknown system"; a separate headless `p.DIRECT` PyBullet instance is the agent's internal simulator.

## Repo layout

| Path | Contents |
|---|---|
| `docs/project_proposal.md` | Concise proposal (RQ, formulation, method, 6-week timeline) |
| `docs/project_outline.md` | **Canonical detailed spec** — full problem formulation, method, experiments E1–E4, baselines B0–B6, milestones M1–M6, risks |
| `docs/*.pdf/.docx/.pptx` | Reference paper "Active RL Agents That Construct and Refine Their Own Physics Simulators", shortened proposal, 3-min slide deck |
| `docs/resources/` | Flow chart (`flow_chart.drawio` + SVG), method overview image, project thumbnail |
| `src/simulation/` | `modelling/`, `execution/`, target `environment/`, and `tools/` (nested demos and validation) |
| `src/agents/` | `rl/` (SAC and training), `system_identification/` (CMA-ES, SMC², and scoring), and internal `simulator/` |
| `src/evaluation/` | Component-specific evaluation: `rl/` for policies; `system_identification/` groups shared support, comparisons, diagnostics, and demos |
| `src/self_learning/` | Personal PyBullet lessons and playlist |
| `scripts/` | macOS PyBullet build tooling (`build_pybullet_macos.sh`) |
| `.claude/skills/setup-env/` | Skill: interactive dependency onboarding, CPU/CUDA selection, and environment verification (shared with Codex via `.agents/skills/setup-env`) |
| `.claude/skills/run-sim/` | Skill: run/test/debug the simulation (venv, tipping tests, env launch, usage snippets) |
| `requirements.txt` | Pinned deps (see stack below) |
| `README.md` | How-to: setup, tipping tests, GUI demo, env usage, SysID scripts |

## Tech stack

- **Physics**: `pybullet==3.2.7` on all platforms. macOS uses `scripts/build_pybullet_macos.sh` for the Apple SDK zlib patch; the script reads the version from `requirements.txt`.
- **RL**: `gymnasium==0.29.1`, native SAC in `src/agents/rl/sac.py`, `torch==2.3.1` (CPU or CUDA 12.1 wheel; see the setup-env skill).
- **SysID**: CMA-ES via `cma==4.5.0`, included in `requirements.txt`.
- **Setup**: Python 3.11 recommended; use `.claude/skills/setup-env/SKILL.md` for the CPU or NVIDIA GPU setup. Shared pins live in `requirements.txt`.
- **Scientific**: numpy, pandas, matplotlib, scipy, jupyter, tqdm
- Python, no packaging config yet (no `pyproject.toml`); `.gitignore` reserves `artifacts/` for training output

## Current status

**Paper implementation:** [`docs/paper_implementation.md`](../docs/paper_implementation.md)
describes the gantry target executor, measured-force identification replay, and
`--paper` refinement actor–critic training/evaluation preset. The supplied
*Toppling-Safe Pushing Under OOD Dynamics* paper takes precedence over the
older outline and the earlier implementation PRs. Contact-force requests use admittance control;
forecasts apply the request directly and identification replays actual force.
All three paper methods support attention. Learning/OOD performance still
requires fresh validation; keep the implementation PR in draft mode.

**Latest integration (2026-11-02):** [`docs/residual_actor_simulator_handoff.md`](../docs/residual_actor_simulator_handoff.md) tracks the experimental delta actor, trajectory critic and critic-ranked proposals. The runnable `train_delta`/`evaluate_delta` path uses one-step soft TD targets and a likelihood-ratio actor value gradient that accounts for detached action-dependent forecasts; baseline SAC is separate. The [actor audit](../docs/findings/delta_actor_audit_2026-11-02.md) records the gradient correction and COM-coordinate fixes on `residual-actor-simulator-integration-4`. Read the handoff and [historical smoke findings](../docs/findings/residual_integration_smoke_2026-10-28.md) before continuing. The predicted-tilt safety filter was removed on 2026-10-31; action bounds and actual toppling termination remain. Learning/OOD validation and controlled model mismatch remain open; keep the PR in draft mode.

**Implemented: target simulation, CMA-ES/SMC² inference, and SAC training/evaluation.** M1 tipping checks agree with the analytic condition and environment episodes run end-to-end. SAC pretraining uses dynamics randomization (`src/agents/rl/training/train_sac.py`); held-out evaluation and per-object fine-tuning use `src/evaluation/rl/evaluate.py`. The original Dyna imagination/filter loop remains planned; this branch implements the separate experimental delta actor–critic method described above.

`src/simulation/tools/validation/calibrate_m3.py` reports maximum-push outcomes and random-action reward scales. It does not establish that a force limit should be kept or changed. Choose and freeze settings through training-distribution evaluation. Training saves `sac_final.pt` and `config.json` under the printed run directory, normally `artifacts/sac_pretrain/<distribution-name>/<run-id>/`. New checkpoints embed training settings and observation statistics; older checkpoints without embedded settings require their adjacent config.

Historical M2/E4 findings: CMA-ES recovered mass to 0–4% and friction to 0.00–0.05 from sliding excitation. COM height was weakly identifiable without near-tipping pushes (10–30% error with run-to-run variance). The boundary push `(h=0.60H, F=5)` toppled 10/10 in one run and 0/10 in another; full-topple trajectories damaged the fitting loss. See `src/evaluation/system_identification/diagnostics/identifiability.py`. These findings motivate the ensemble and pessimistic filter; rerun experiments to measure current behavior.

Historical M3 findings, before the native PyTorch rewrite: the original signed-distance reward rewarded motion away after passing the target, producing oscillation at 0% success. The environment now uses unsigned remaining distance for reward progress. Frozen-policy success was 50% on eight held-out in-distribution objects; fine-tuning raised four of eight objects to 70–100%, with failures on heavy/high-friction and tall/high-COM objects. These are historical learning results, not validation of current training performance.

**Planned change (professor's recommendation):** a KUKA IIWA arm will execute the pushes in the target environment as a low-level executor beneath the unchanged `(h, F, τ)` action space; the internal simulator stays direct-force. See `.claude/analysis/robot_integration_plan.md`.

**Future iterations:** off-centre force application points (`y` coordinate in the action, inducing yaw) and non-centred (lateral) COM as estimated parameters — extending the task to planar pushing with reorientation.

## Conventions

- Action space: `a_t = (h_t, F_t, τ_t)` — force height `[0.1H, 0.9H]`, magnitude `[0, F_max]`, duration `[0.05, 0.50]` s; fixed push direction, 1D translation with rotational dynamics.
- Episode: ≤ `T_max = 10` decisions; adaptation: ≤ `N_ep = 20` episodes per test object; SysID history persists across episodes.
- Observation `o_t = (d, ψ, ẋ, ψ̇, G)` — estimated latent parameters are **never** appended to the policy observation; they configure the internal simulator only.
- Physics sanity-check formula: `(F/mg)(h − h_c) ≤ w/2 − μ h_c` (used for validation only, never given to the agent).
- Docs use LaTeX math in markdown; flow charts live in `docs/resources/` as drawio sources.

## Deeper analysis

- [`.claude/analysis/project_analysis.md`](analysis/project_analysis.md) — doc inventory, exploration status, gap analysis, dependency notes, next steps.
- [`.claude/analysis/robot_integration_plan.md`](analysis/robot_integration_plan.md) — staged plan for adding the KUKA IIWA robot executor (embodiment, execution layer, reality-gap remapping, risks, timeline).
