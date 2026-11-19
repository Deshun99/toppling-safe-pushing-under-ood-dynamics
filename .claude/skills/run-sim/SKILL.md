---
name: run-sim
description: Run, test, or debug the pushing simulation — tipping tests, PyBullet env launch (GUI/headless), venv setup, and quick usage snippets. Use whenever the user wants to run the environment, validate the physics, or launch the simulator.
---

# Run the pushing simulation (M1)

Commands assume the repo root and the existing `.venv` (pybullet 3.2.7 patched
build, gymnasium 0.29.1, numpy 1.26.4). No activation needed — use
`.venv/bin/python`.

## Recreate the venv (only if `.venv` is missing or broken)

Use the [setup-env skill](../setup-env/SKILL.md) for interactive dependency
setup, CPU/CUDA selection, and environment verification.

## Physics validation (the M1 deliverable)

```bash
.venv/bin/python -m src.simulation.tools.validation.tipping_tests
```

Expected output: `[PASS] gentle push`, `[PASS] violent push`, and an
"Analytic agreement" grid (100% when the sim matches the no-tip condition
`(F/mg)(h - h_c) <= w/2 - mu h_c`). Non-zero exit = physics regression.

## Quick env usage

```python
import numpy as np
from src.simulation.environment import PushEnv

env = PushEnv(gui=False)  # gui=True opens the PyBullet visualizer
obs, info = env.reset(seed=0)  # info["true_theta"] = (m, mu, h_c) — eval only
done = False
while not done:
    obs, r, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
env.close()
```

- Action: normalized `[-1,1]^3` → `(h, F, τ)` with `h ∈ [0.1H, 0.9H]`,
  `F ∈ [0, F_max]`, `τ ∈ [0.05, 0.50] s`.
- Observation: 7-dim `(d, ψ, ẋ, ψ̇, w, L, H)` — hidden θ never included.
- `info` keys: `true_theta`, `executed`, `toppled`, `success`, `max_tilt`.
- `env.last_trajectory` holds the dense within-action trajectory (M2 SysID).

## Watch the env in the GUI

```bash
.venv/bin/python -m src.simulation.tools.demos.env_demo                            # 1 episode, real-time pace
.venv/bin/python -m src.simulation.tools.demos.env_demo --seed 5 --episodes 3 --speed 0.5
```

Random-agent demo with the PyBullet visualizer. `--speed 1.0` paces the
physics at real time (0 = as fast as possible), `--linger` keeps the final
state on screen. Pacing is wall-clock only (`realtime_scale` on
`ForcePushExecutor`) — trajectories are identical at any speed.

## Scripted pushes without the env (physics debugging)

```python
import pybullet as p
import pybullet_data
from src.simulation.modelling.constants import DT
from src.simulation.modelling.object import Cuboid, CuboidSpec
from src.simulation.execution.force_push import ForcePushExecutor

client = p.connect(p.DIRECT)
p.setGravity(0, 0, -9.81, physicsClientId=client)
p.setTimeStep(DT, physicsClientId=client)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.loadURDF("plane.urdf", physicsClientId=client)
p.changeDynamics(plane, -1, lateralFriction=1.0, physicsClientId=client)  # convention!

spec = CuboidSpec(w=0.08, L=0.08, H=0.20, m=1.0, mu=0.3, h_c=0.10)
cuboid = Cuboid(spec, client)
cuboid.spawn(position=(0, 0, spec.H / 2 + 0.005))
for _ in range(round(0.5 / DT)):  # settle
    p.stepSimulation(physicsClientId=client)
traj = ForcePushExecutor(client).execute(cuboid, h=0.1, F=4.0, tau=0.4)
print(max(abs(t) for t in traj.psi))  # peak tilt (rad)
```

## Debugging checklist

- Timestep is 1/240 s (`p.setTimeStep`); topple = `|ψ| > crit_tilt` where
  `crit_tilt = atan((w/2)/h_c)`.
- The reward progress term uses **unsigned** remaining distance (`env.py`).
  Signed distance paid positive reward for pushing away once the object
  overshot the target — a real exploit that made the first M3 SAC run learn
  target-oscillation at 0% success. Do not "fix" the `abs()` away.
- Friction convention: floor μ = 1.0 so the cuboid's `lateralFriction` IS the
  effective coefficient (Bullet multiplies the two).
- COM sits below the geometric centre via `baseInertialFramePosition` — check
  `src/simulation/modelling/object.py` before suspecting COM placement.
- Spawn at `z = H/2 + 0.005` and settle ~0.5 s before pushing, or the initial
  contact jitter pollutes the trajectory.
- `PushEnv`, training, and calibration default to 50 N through `ACTION_RANGES`.
  Evaluation restores each checkpoint's saved action bounds, including older
  15 N runs; it does not replace them with the current default.
- The harmless `pybullet build time: ...` line on import is normal.

## SysID (M2)

```bash
.venv/bin/python -m src.evaluation.system_identification.diagnostics.identifiability   # E4 identifiability study
.venv/bin/python -m src.evaluation.system_identification.diagnostics.calibrate_scales  # replay noise floor -> LOSS_SCALES
```

- Historical results: m errors 0–4%, mu 0.00–0.05 from sliding excitation;
  h_c 10–30% (weak without near-tipping pushes). Rerun to measure current
  behavior; these values are not acceptance thresholds.
- If the sim changes (timestep, solver, contact settings), rerun the
  calibration and paste the printed scales into `LOSS_SCALES` in
  `src/agents/system_identification/sysid.py`.
- Keep estimation histories **upright-only**: full-topple trajectories are
  chaotic and poison the loss; the tipping boundary itself is bistable
  (same push can give 0/10 or 10/10 topples across runs).

## RL pretraining and held-out evaluation (M3)

```bash
.venv/bin/python -m src.simulation.tools.validation.calibrate_m3    # Maximum-push outcomes and reward scales
.venv/bin/python -m src.agents.rl.training.train_sac --smoke       # short training pipeline check
.venv/bin/python -m src.agents.rl.training.train_sac               # native PyTorch SAC pretraining
.venv/bin/python -m src.evaluation.rl.evaluate \
    --model artifacts/sac_pretrain/sac_final.pt # frozen-policy and fine-tuning evaluation (B1)
```

- SAC pretrains with dynamics randomization using the selected
  `--distribution-config`; `ObjectSampler` applies its bounds and exclusions.
- New checkpoints embed training settings and observation statistics.
  Evaluation reads embedded settings first; older checkpoints without them
  require adjacent `config.json` (including the original `f_max` format).
- Native `.pt` checkpoints include frozen observation statistics. Legacy zip/pickle
  checkpoints require retraining; no separate statistics file is needed.
- `calibrate_m3.py` reports displacement and toppling separately. Failure of
  a maximum-duration push does not justify changing `F_max`; weaker and shorter
  pushes must be considered when assessing safe target-reaching. The script
  does not make a keep/tune decision for the force limit.
- Choose and freeze force/reward settings after training-distribution evaluation.
  Training saves them in `config.json` and the checkpoint under the printed run
  directory, normally `artifacts/sac_pretrain/<distribution-name>/<run-id>/`.
- `--smoke` flags on both scripts give fast end-to-end pipeline checks.
- Monitor training with `tensorboard --logdir artifacts/tb`.

## What to change when milestones land

- M4: point-estimate imagination (B2/B3/B6) on top of the shared pretrained weights.
