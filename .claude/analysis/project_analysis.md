# General Project Analysis

Historical analysis from before implementation. Status and proposed file paths
below describe that earlier stage; use the [project instructions](../CLAUDE.md)
and [current architecture](../../docs/architecture/overall-architecture.md) for
implemented components and current module paths.

*Generated from a full read of `docs/` and `src/self_learning/` (2026-09-28).*

## 1. Executive summary

The project studies **uncertainty-aware physics imagination for safe adaptation**: an RL agent pushes an upright cuboid to a target without toppling it, while the object's mass, floor friction, and vertical COM height are hidden. The agent estimates these parameters online (CMA-ES against dense trajectories), instantiates an ensemble of plausible simulators, and uses them (a) as a Dyna-style data source for SAC and (b) as a pessimistic safety filter on actions.

The contribution is deliberately narrow and empirical — a controlled sim-to-sim comparison of model-free SAC vs. point-estimate vs. ensemble imagination, with graded simulator mismatch (G0–G3) and OOD latent-parameter shifts. The design is mature and unusually well specified (problem formulation, rewards, experiments E1–E4, baselines B0–B6, milestones, risks, minimum viable result are all defined). The gap is that **implementation has not started**: the repo contains only PyBullet tutorial scripts.

## 2. Documentation inventory (`docs/`)

| File | Role | Status |
|---|---|---|
| `project_proposal.md` | Concise proposal: RQ, hypothesis, formulation, method summary, 5-baseline comparison, 6-week timeline | Complete |
| `project_outline.md` | **Canonical spec.** 15 sections: formulation, physics background, reality-gap levels G0–G3, SysID loss, SAC + imagination-as-data/shield loop, protocol (pretrain → per-object adaptation), experiments E1–E4, baselines B0–B6, metrics, RQs/Hs, risks, milestones | Complete and authoritative; supersedes the proposal where they differ (e.g., B0 analytic-controller baseline, normalized params `z = (log m, μ, h_c/H)` only in outline) |
| `Active RL Agents That Construct and Refine Their Own Physics Simulators.pdf` | Key reference paper (title suggests simulators learned/refined online) | Present, unread in this analysis |
| `Proposal Shortened.docx` | Shortened proposal (likely abstract-style) | Present |
| `toppling_safe_pushing_3min_proposal.pptx` | 3-minute pitch deck | Present |
| `resources/flow_chart.drawio` (+ `.svg`) | System method overview flowchart | Editable source retained |
| `resources/image.png`, `project_thumbnail.png` | Figure assets embedded in the proposal | Present |

The outline's protocol summary: pretrain SAC with dynamics randomization over `m ∈ [0.5, 1.5] kg`, `μ ∈ [0.3, 0.7]`, `h_c/H ∈ [0.35, 0.60]`; then per-object adaptation (≤ 20 episodes × ≤ 10 actions) with a persistent SysID history; OOD splits in mass / friction / COM / joint.

## 3. Exploration inventory (`src/self_learning/`)

PyBullet tutorial progression (from a YouTube playlist) — the only code in the repo:

| Script | Covers |
|---|---|
| `lesson1.py` | Connect GUI, gravity, load plane + soccerball, 5000-step sim loop |
| `lesson2.py` | Camera setup, `applyExternalForce` (world frame) on a body |
| `lesson3.py` | Reading body state via `getBasePositionAndOrientation` |
| `lesson4.py` | Ball loading + (velocity/position tracking; projectile-style checks) |
| `lesson4part2.py` | Loading the KUKA IIWA arm, fixed base |
| `lesson5.py` | Joint info enumeration, `resetJointState`, commented-out `setJointMotorControl2` position control |

Assessment: tutorial-grade familiarity with core PyBullet APIs the project actually needs (URDF loading, `applyExternalForce`, base pose/velocity access, headless-capable loops). No gym environment, RL training, SysID, or cuboid model exists yet.

## 4. Method in brief (from the outline)

- **Observation** `o_t = (d, ψ, ẋ, ψ̇, G)`; hidden `θ = (m, μ, h_c)`; estimated parameters configure the simulator only.
- **Action** `a_t = (h, F, τ)` — external force primitive applied by PyBullet directly (no robot arm; embodiment is out of scope).
- **SysID**: replay recorded actions in the internal simulator under candidate parameters; score with a normalized trajectory-discrepancy loss over `(x, ψ, ẋ, ψ̇)`; CMA-ES point estimate + top-K hypotheses `Θ_K` with softmax weights.
- **Imagination as data**: short (1–3 action) hypothetical rollouts into `D_sim`, mixed with target replay at ratio `ρ`.
- **Imagination as shield**: sample `N_c` candidates; simulate under every `θ_k`; reject any predicted to exceed near-topple threshold `ψ_max`; execute the highest-Q safe candidate.
- **Baselines**: B0 analytic controller, B1 model-free SAC, B2 nominal-sim Dyna, B3 point-estimate Dyna, B4 ensemble Dyna, B5 = B4 + filter (proposed), B6 oracle-sim Dyna.
- **Reality-gap levels**: G0 well-specified, G1 observation/action noise (default), G2 unmodelled dynamics, G3 cross-engine (stretch).

## 5. Gap analysis: docs → code

| Need | Status |
|---|---|
| Cuboid generator (geometry G, mass/inertia from `h_c`) | ✗ |
| External-force action primitive with body-relative application point | ✗ (only fixed-body demos so far) |
| Gymnasium env: observation, reward, success/topple detection | ✗ |
| Scripted tipping sanity tests (1/240 s timestep, solver-setting sweep) | ✗ |
| CMA-ES SysID with dense-trajectory replay | ✗ (and `cma` not in requirements) |
| SAC training (native PyTorch) with dynamics randomization | ✗ |
| Dyna replay mixing + pessimistic filter | ✗ |
| Evaluation harness: metrics, seeds, bootstrap CIs | ✗ |

## 6. Dependency notes

- `torch==2.3.1+cu121` is pinned via `--extra-index-url https://download.pytorch.org/whl/cu121` — needs a CUDA-capable machine; consider a CPU fallback pin for portability.
- CMA-ES is named as the SysID optimizer but `requirements.txt` has no `cma` (or `cmaes`) package — must be added.
- Native SAC uses `torch==2.3.1` with `gymnasium==0.29.1`.
- `.gitignore` already reserves `artifacts/` for training outputs; no `pyproject.toml`, no tests dir, no CI.
- `.vscode/settings.json` pins the system Python env manager.

## 7. Suggested next steps (M1 — environment first)

1. Create a package layout (e.g., `src/` or top-level `pushing/`) with an env module, per the outline's M1.
2. Build the cuboid: URDF or `createMultiBody`; set COM via inertial-frame offset + consistent inertia.
3. Implement the force primitive: world-frame push direction, body-relative application point updated per physics step, `F = 0` passive intervals allowed.
4. Success/topple detection: COM projection vs. support region; near-topple threshold `ψ_max`.
5. Scripted tipping tests across timesteps and solver settings **before** RL (explicit risk mitigation in the outline); calibrate `F_max` from pilot tests.
6. Add `cma` to requirements and prototype SysID (M2) in parallel with env validation.

## 8. Key open decisions (from the outline's own risk section)

- `F_max`: must move the heaviest/highest-friction object while leaving unsafe actions available.
- Reward coefficients (`α, c_time, c_step, R_succ, R_topple`): tuned on the training distribution only, frozen afterward.
- Filter threshold `ψ_max` and hypothesis count `K ∈ {4, 8}` are ablation axes, not fixed design choices.
- Identifiability (E4) may force changes to the action space (e.g., passive intervals) — run it early.

## 9. Bottom line

High-quality research design; zero implementation. The outline's own "minimum viable result" (G0/G1 + online SysID + B1/B3/B6 + E2/E4) is the right first target. All subsequent milestones depend on a validated environment, so M1 plus an early E4-style identifiability probe is the highest-leverage work.
