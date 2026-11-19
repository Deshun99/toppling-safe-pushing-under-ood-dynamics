# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands run from the repo root. Use `.venv/bin/python` (created by the setup-env skill).

```bash
# Tests
.venv/bin/python -m pytest                           # full suite with coverage
.venv/bin/python -m pytest tests/test_env.py         # single file
.venv/bin/python -m pytest -k "test_tipping"         # by name pattern
.venv/bin/python -m pytest -x                        # stop on first failure

# Lint
ruff check .                                         # advisory only (CI uses --exit-zero)
ruff check --fix .                                   # auto-fix safe issues

# Physics validation (M1)
.venv/bin/python -m src.simulation.tools.validation.tipping_tests

# GUI environment demo
.venv/bin/python -m src.simulation.tools.demos.env_demo
.venv/bin/python -m src.simulation.tools.demos.env_demo --speed 0 --linger 0  # headless-speed

# SAC pretraining
.venv/bin/python -m src.agents.rl.training.train_sac \
    --distribution-config configs/rl/training_distributions/uniform_excluding_evaluation_regions.json
.venv/bin/python -m src.agents.rl.training.train_sac --smoke  # quick end-to-end check

# SAC held-out evaluation
.venv/bin/python -m src.evaluation.rl.evaluate --model artifacts/sac_pretrain/sac_final.pt
.venv/bin/python -m src.evaluation.rl.evaluate --model path/to/sac_final.pt \
    --split mass_ood friction_ood com_ood joint_ood

# Experimental delta training
.venv/bin/python -m src.agents.rl.training.train_delta \
    --distribution-config configs/rl/training_distributions/uniform_excluding_evaluation_regions.json \
    --smoke --out-dir artifacts/delta/smoke

# TensorBoard
tensorboard --logdir artifacts/tb
```

## Architecture

The system separates the **target environment** (unknown physics) from the **agent's internal simulator** (imagination). Both are PyBullet clients, each on its own client ID; they never share state.

### Layer 1 — Simulation primitives (`src/simulation/`)

- `modelling/` — data types: `Cuboid`/`CuboidSpec` (geometry + hidden physics), `PushAction`, `ActionRecord`, `DenseTrajectory`, action normalization, sampling distributions.
- `execution/` — `ForcePushExecutor` applies timed forces at a specified height in a PyBullet scene; `settling.py` runs passive post-push physics. `robot.py` is a stub for the planned KUKA executor.
- `environment/env.py` — `PushEnv(gym.Env)`: wraps execution, samples objects from an `ObjectSampler`, computes rewards via `RewardModel`, enforces the 10-decision episode limit. Hidden physics (`m`, `μ`, `h_c`) are never in the observation; `last_trajectory` (a `DenseTrajectory`) is stored after each push for the SysID stage.

### Layer 2 — Agent components (`src/agents/`)

- `simulator/internal_sim.py` — `InternalSim`: a headless `p.DIRECT` PyBullet client. `replay(spec, record)` re-executes a stored `ActionRecord` under a candidate `CuboidSpec` and returns a `DenseTrajectory`. This is the "imagination" engine.
- `system_identification/sysid.py` — `SysID` (CMA-ES-backed): after each push, scores candidate physics vectors `z = (log m, μ, h_c/H)` by comparing replayed trajectories against recorded ones via `trajectory_loss`. Returns a point estimate and a top-K weighted ensemble.
- `system_identification/smc2.py` — `SMC2SysID`: particle-based alternative to CMA-ES; same interface.
- `rl/models/sac.py` — `SAC`: tanh-Gaussian actor, twin Q-critics, automatic entropy tuning, raw-observation replay with running-mean normalization. Observation `o = (d, ψ, ẋ, ψ̇, w, L, H)` — geometry is observable, physics is not.
- `rl/models/delta_actor_critic.py` + `delta_controller.py` — **Experimental**: `DeltaController` proposes adjustments to the previous real action, simulates each candidate across the full particle ensemble, and selects the highest-ranked by the trajectory critic. Actor sees observation + previous trajectory; critics additionally see the candidate delta and its ensemble forecast.

### Layer 3 — Training entry points (`src/agents/rl/training/`)

- `train_sac.py` — pretrains SAC with dynamics randomization; parallel `AsyncVectorEnv` workers; saves `sac_final.pt` + `config.json` + `best_model/` under `artifacts/sac_pretrain/<distribution>/<run-id>/`. Checkpoints embed observation statistics and `training_metadata`.
- `train_delta.py` — trains the delta actor–critic; one-step soft TD targets, replay updated after each push. Same artifact layout under `artifacts/delta/`.

### Layer 4 — Evaluation (`src/evaluation/`)

- `rl/evaluator.py` — `Evaluator` owns the episode loop for training validation, GUI playback, and held-out testing. `evaluate(model)` runs frozen; `fine_tune=True` enables online SAC updates with frozen normalization.
- `rl/evaluate.py` — CLI: compares frozen vs. fine-tuning conditions on fixed or sampled object sets; writes `results.json`.
- `system_identification/` — identifiability studies, CMA-ES/SMC² benchmarks, replay-noise calibration.

### Key data flows

**Target episode:** `PushEnv.step(normalized_action)` → `ForcePushExecutor` → `DenseTrajectory` → stored as `last_trajectory` → `SysID.update(record, trajectory)` scores candidates via `InternalSim.replay`.

**Delta episode:** `DeltaController.select_action(obs, geometry, sysid)` → generate normalized candidate deltas → `InternalSim.predict` for each particle × candidate → `DeltaActorCritic` scores forecasts → return best candidate.

**Checkpoint round-trip:** `SAC.save(path)` writes a native `.pt` dict with `model_state`, `obs_stats`, and `training_metadata`. `SAC.load(path, inference_only=True)` skips replay allocation; critics and optimizer states still deserialize.

### Config files (`configs/rl/`)

JSON files describe object sampling distributions and fixed/sampled evaluation sets (geometry + physics bounds, exclusion regions). They are **not** SAC hyperparameter files. Pass via `--distribution-config` / `--evaluation-config` / `--sampler-config`.
