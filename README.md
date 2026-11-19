# Simulation-Assisted Toppling-Safe Pushing

University course project. Research question: can a physics simulator serve as an RL
agent's **imagination** when adapting to objects with unknown dynamics? An
upright cuboid must be pushed toward a target without toppling. Geometry is
observable; mass `m`, object-floor friction `μ`, and vertical COM height `h_c`
are hidden and must be inferred from motion.

Policy-selected pushes can execute through a simulated two-axis Cartesian
gantry with admittance-controlled pad contact. The action remains `(h, F, τ)`;
`F` requests contact force. Internal forecasts apply that request directly,
while identification replays measured actual horizontal forces. Direct-force
target execution remains available for component experiments.

The [paper implementation guide](docs/paper_implementation.md) describes the
three-method benchmark, training/evaluation commands, and corrections to the
earlier implementations in [PR #1](https://github.com/Deshun99/toppling-safe-pushing-under-ood-dynamics/pull/1)
and [PR #2](https://github.com/Deshun99/toppling-safe-pushing-under-ood-dynamics/pull/2).

- Full formulation, method, experiments: [`docs/project_outline.md`](docs/project_outline.md)
- Short proposal: [`docs/project_proposal.md`](docs/project_proposal.md)

## Setup

Ask Claude Code or Codex: **“Use the setup-env skill to set up this project.”**

The [setup-env skill](.claude/skills/setup-env/SKILL.md) detects your platform,
asks about CPU/GPU support and environment reuse when relevant, installs
dependencies, and verifies the setup. All dependency setup instructions live in
that skill.

## Running the project

Run commands from the repository root. Python entry points use `python -m src.<package>.<module>`;
imports follow the same `src.*` package paths. For example, run a learning exercise with
`.venv/bin/python -m src.self_learning.lesson1`.

### Automated tests

After installing `requirements.txt` in the configured environment, run:

```bash
.venv/bin/python -m pytest
```

The suite covers cuboid geometry and dynamics, normalized actions, observations,
rewards, episode termination, reset and client isolation, force execution and
wall-clock pacing, SysID replay/loss/ensemble selection, and a small real CMA-ES
run. Physics tests use headless PyBullet; no display or GPU is needed. Coverage
includes branches and writes `coverage.xml`; no minimum coverage is enforced.
Gantry tests exercise kinematics, physical contact, admittance control,
consecutive pushes, force replay, and reported execution failures.

GitHub Actions runs the suite and the existing M1 tipping script on every PR
(including new commits and reopened PRs) and every push to `main` (including
merges). It uses Ubuntu and Python 3.11 with CPU PyTorch, checks dependency
consistency, and uploads coverage and JUnit reports. GUI rendering, tutorial
scripts, and the longer SysID research studies are outside automated coverage.

Physics runs at a fixed 240 Hz (`DT = 1/240` s in `src/simulation/modelling/constants.py`)
throughout the target environment, internal simulator, and tutorial scripts.
The timestep is not configurable; GUI playback speed only changes wall-clock pacing.

### 1. Physics validation (M1 deliverable)

```bash
.venv/bin/python -m src.simulation.tools.validation.tipping_tests
```

Expected: `[PASS] gentle push`, `[PASS] violent push`, and an "Analytic
agreement" grid at 100% (the sim matches the no-tip condition
`(F/mg)(h − h_c) ≤ w/2 − μ h_c`). Non-zero exit = physics regression.

### 2. Watch the environment (GUI demo)

```bash
.venv/bin/python -m src.simulation.tools.demos.env_demo
```

A PyBullet window opens and a random agent pushes the cuboid until it topples
or reaches the target. Each episode prints the object's hidden parameters, every
decision's `(h, F, τ)`, the reward, and the max tilt (radians — 1 rad ≈ 57°).

| Flag | Default | Meaning |
|---|---|---|
| `--seed N` | `0` | seed of the first episode |
| `--episodes K` | `1` | number of episodes (each resamples the object) |
| `--speed S` | `1.0` | physics replay speed: `1.0` = real time, `0` = as fast as possible |
| `--linger S` | `2.0` | seconds the final state stays on screen |
| `--cam-dist D` | unset | debug-visualizer camera distance in m (zoom); `~0.35` is a close-up of the cuboid. Unset leaves the camera alone |

The pacing is wall-clock only — trajectories and physics are identical at any
speed. Examples:

```bash
.venv/bin/python -m src.simulation.tools.demos.env_demo --episodes 5 --speed 0.5 --linger 5   # slow motion
.venv/bin/python -m src.simulation.tools.demos.env_demo --speed 0 --linger 0                  # headless-style, instant
```

### 3. Use the environment in code

```python
from src.simulation.environment import PushEnv

env = PushEnv(gui=False)  # gui=True opens the PyBullet visualizer
obs, info = env.reset(seed=0)  # info["true_theta"] = (m, mu, h_c) — eval only
done = False
while not done:
    obs, r, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
env.close()
```

- Action: normalized `[-1,1]³` → `(h, F, τ)` with `h ∈ [0.1H, 0.9H]`,
  `F ∈ [0, F_max]`, `τ ∈ [0.05, 0.50]` s.
- Observation: 7-dim `(d, ψ, ẋ, ψ̇, w, L, H)` — the hidden θ is never included.
- `info` keys: `true_theta`, `executed`, `toppled`, `success`, `max_tilt`
  (rad), plus the gymnasium conventions `is_success` and
  `TimeLimit.truncated` (used by the M3 RL scripts).
- `env.last_trajectory` holds the dense within-action trajectory (M2 SysID).

Import `PushAction` from `src.simulation.modelling.actions`.
For scripted physics studies, use the same `PushEnv` with a `fixed_spec` and
call `env.push(PushAction(h=h, F=F, tau=tau), warmup_steps=8)` after `reset()`. This accepts physical
units and returns an `ActionRecord` from `src.simulation.modelling.records`, including the start
state, trajectory, and warm-up count used by internal replay. It advances physics
without rewards or episode bookkeeping; use `step()` throughout RL episodes.
The study scripts explicitly handle settling and use `env.replace_object(x=...)`
to replace a fallen object while retaining its physics parameters.

- GUI pacing in code: `PushEnv(gui=True, realtime_scale=1.0)`.

`PushEnv` has no training/test split rules. Its default sampler uses the full
demo bounds in `src/simulation/modelling/sampling.py`. Pass
`object_sampler=your_sampler` to supply an `ObjectSampler` instance whose
`sample(rng)` method returns a `CuboidSpec`; `reset(seed=...)` controls that RNG.
`fixed_spec` takes precedence over any sampler.

The [RL config index](configs/rl/README.md) distinguishes training distributions,
fixed evaluation cases, and sampled evaluation set definitions.

RL training reads explicit geometry/physics bounds and exclusions from
[`configs/rl/training_distributions/uniform_excluding_evaluation_regions.json`](configs/rl/training_distributions/uniform_excluding_evaluation_regions.json) by default.
[`configs/rl/training_distributions/uniform_without_exclusions.json`](configs/rl/training_distributions/uniform_without_exclusions.json) has the same outer
bounds and an explicit empty exclusion list. The shared `ObjectSampler` in
`src/simulation/modelling/sampling.py` owns rejection sampling and config loading. Exclusions
in the supplied training config are joint closed boxes in `(m, μ, h_c/H)`,
independent of geometry. Sampling bounds and exclusions both use `ObjectRanges`: `None`
(or an omitted exclusion field) leaves that parameter unconstrained. Sampling
bounds must specify all six intervals. Exclusions can also constrain geometry. Training and
checkpoint validation both use the selected config. Evaluation set definitions do
not implicitly change either config.
Exclusions may extend outside the training bounds: only their intersection
with the training box can reject samples. A fully disjoint OOD exclusion has
no effect, while a partial overlap removes just the overlapping joint region.
The original configured intervals remain in checkpoint metadata. Nonphysical
intervals and unions covering all training physics are rejected before training.
Action bounds default to `ACTION_RANGES` (50 N maximum); pass an `ActionRanges`
instance through `PushEnv(action_ranges=...)` to configure an experiment.
Training and calibration use the same `ACTION_RANGES` defaults (50 N maximum).
Evaluation restores the action bounds saved with each checkpoint.

### 4. System-identification studies (M2)

```bash
.venv/bin/python -m src.evaluation.system_identification.diagnostics.identifiability   # E4 identifiability study
.venv/bin/python -m src.evaluation.system_identification.diagnostics.calibrate_scales  # replay noise-floor calibration
```

System-identification accuracy depends on the excitation regime and replay
fidelity. `h_c` is weakly identifiable without near-tipping pushes, motivating
the ensemble + pessimistic filter. Run the study to measure current results.

### 5. RL pretraining and held-out evaluation (M3)

```bash
.venv/bin/python -m src.simulation.tools.validation.calibrate_m3                      # F_max + reward diagnostics
.venv/bin/python -m src.agents.rl.training.train_sac --distribution-config configs/rl/training_distributions/uniform_excluding_evaluation_regions.json
.venv/bin/python -m src.agents.rl.training.train_sac --distribution-config configs/rl/training_distributions/uniform_without_exclusions.json
.venv/bin/python -m src.evaluation.rl.evaluate \
    --model artifacts/sac_pretrain/sac_final.pt
```

`train_sac.py` pretrains SAC with dynamics randomization using the chosen JSON
distribution. Estimated parameters never enter the policy observation.
Default output is `artifacts/sac_pretrain/<distribution-name>/<run-id>/`, printed
at startup. `--out-dir` can specify another empty directory; training refuses to
overwrite a nonempty directory. Each run saves `sac_final.pt`, `config.json`, and
`best_model/`. Logs also use a separate run-ID directory.

Both final and best `.pt` files embed `training_metadata`, identical to their
adjacent `config.json`. It includes the resolved `training_distribution` (all
ranges and exclusions, including an empty list), its SHA-256, a unique `run_id`,
the source config path, training arguments, physical action bounds and reward.
The embedded snapshot remains available if the input JSON changes or the weights
file moves. Inspect it without allocating replay:

```python
from src.agents.rl.models.sac import SAC

model = SAC.load("path/to/sac_final.pt", inference_only=True)
print(model.training_metadata["training_distribution"])
print(model.training_metadata["run_id"])
```

Historical checkpoints load with `training_metadata=None`; this means unknown
embedded metadata, not a known empty exclusion list. Their existing sidecars
remain useful. Existing checkpoints and baseline results predate the holes and
do not validate the new split. The default evaluation retains eight fixed objects; sampler configs can test
throughout the reserved regions (see below).
Each best policy is saved as `best_model/best_model.pt` with its own embedded
normalization statistics and adjacent `config.json`. Replay is deliberately
excluded, so every adaptation object starts with an empty target-only buffer.
Frozen checkpoint evaluation and GUI playback use
`SAC.load(path, inference_only=True)` to skip replay allocation. Such models
can predict actions but reject training updates. Critics and optimizer states
still load; the checkpoint format and default training-capable loading are unchanged.
Legacy zip/pickle checkpoints cannot be loaded; retrain to produce native `.pt`
checkpoints. Historical M3 learning results do not validate this implementation.

RL evaluation uses the `src/evaluation/rl/` package. Every `--eval-freq`
training steps, a dedicated environment runs deterministic policy actions on
the same `--n-eval-episodes` seeded cases. CSV and optional TensorBoard logs
include mean/std episode return, success/topple/timeout rates and counts,
mean final position error, and mean push count/time among successful episodes.
Unavailable success-conditioned metrics are omitted from scalar logs and
saved as JSON `null`. Pushing time is the sum of commanded durations, excluding
reset settling and robot approach/retraction.

`src/evaluation/rl/evaluator.py` owns the episode loop for training validation,
GUI playback, and held-out testing. `Evaluator.evaluate(model)` runs a frozen
policy; `fine_tune=True` enables stochastic SAC actions and one update per
transition, with observation normalization still frozen. The shared
`Evaluator.evaluate_checkpoint()` method loads an independent checkpoint copy and
manages its fixed-object environment for each held-out condition. The held-out
CLI uses `SACEvaluation` to run and report paired conditions. `EvaluationSet`
owns fixed or sampled cases and their source configuration; each `EvaluationCaseResult`
keeps its case and both conditions together. `EvaluationMetrics.summarize_progress()`
returns a `ProgressSummary` for either frozen or learning histories.

`evaluations.json` holds summary history with training step counts;
`evaluation_episodes/steps_NNNNNN.json` holds individual episode results,
object specifications, seeds, action bounds, reward coefficients, and task
tolerances. These records include peak within-push tilt, separately from the
environment's terminal-tilt toppling classification. Best-checkpoint selection
continues to use mean episode return. Repeated checkpoint evaluations are
validation; use separate held-out cases for final reported test results.

Watch an existing checkpoint using the same episode measurements:

```bash
.venv/bin/python -m src.evaluation.rl.demos.visualize_trained_sac \
    --model artifacts/sac_pretrain/sac_final.pt --episodes 5 --speed 0.5
```

B1 output retains its existing counts and first-success fields and additionally
includes episode records, evaluation summaries, cumulative fine-tuning curves,
and the fraction of objects that achieve success within the budget.
First-success interaction/time measurements include earlier failed episodes;
objects that never succeed retain `null` values. SysID, shield, robot, and
compute instrumentation and experiment-level confidence intervals remain
separate follow-up integrations; they are not reported as zero by SAC evaluation.

The implementation in `src/agents/rl/models/sac.py` uses a tanh Gaussian actor, twin Q critics,
Polyak target updates, automatic entropy tuning, and raw-observation replay.
Observation normalization is updated during pretraining and frozen during evaluation
and fine-tuning; rewards are unnormalized. Time-limit truncations bootstrap from
the final observation, while true terminations do not bootstrap. Gymnasium workers
run in parallel for `--n-envs > 1`. Collection rounds up to a full vector step;
default gradient steps equal `n-envs` for a 1:1 update-to-transition ratio.

Evaluation and GUI playback restore physical action bounds from embedded
checkpoint metadata first. New checkpoints work without a sidecar, even after
being moved; conflicting sidecars are ignored. Older checkpoints without
embedded metadata require adjacent `config.json`; original `f_max` is supported.
Missing action bounds are an error rather than an implicit 50 N fallback.
New training runs save physical bounds in `action_ranges` and accept `--f-max`
(default: 50 N). The separate top-level `f_max` field is only read for legacy
checkpoints. Training
progress goes to `artifacts/tb/sac_pretrain/` (view with
`tensorboard --logdir artifacts/tb`). CSV metrics are always saved in the run
directory, including when TensorBoard is unavailable.

`evaluate.py` evaluates SAC on 8 fixed objects within the outer bounds
(outline §10, baseline B1). For each object, it compares a frozen pretrained
policy (`--frozen-episodes`, default 1) with a separate copy that learns while
being evaluated (`--fine-tuning-episodes`, default 20), using only that object's
experience. Fine-tuning starts afresh from the checkpoint for every object.
One frozen episode suffices because actions and initial conditions are deterministic
for each fixed object. Repeating it adds no independent evidence.
The script prints per-object topple counts, success rates, and interactions
to first success, plus means. Fine-tuning results measure performance during
learning; there is no separate evaluation after fine-tuning. Results land in
`artifacts/held_out_evaluation/results.json`.

OOD evaluation objects are configured separately in
[`configs/rl/evaluation_sets/fixed_cases.json`](configs/rl/evaluation_sets/fixed_cases.json).
The suite has four cases each for mass OOD, friction OOD, COM OOD and joint OOD,
using the outline's low/high OOD ranges with fixed in-range geometry. Select a
single split using `--split mass_ood`, `friction_ood`, `com_ood` or `joint_ood`:

```bash
.venv/bin/python -m src.evaluation.rl.evaluate \
    --model artifacts/sac_pretrain/sac_final.pt --split mass_ood friction_ood com_ood joint_ood
# Use --split all to include the original eight in-range cases as well.
```

Group names are configuration data; the loader imposes no predefined categories
or OOD labels. `--split` accepts one or more names, or `all`. Use `--out` to
choose the destination; otherwise filenames reflect the selected groups.
The fixed-object JSON is the single definition of the historical eight cases.
Use `--evaluation-config path/to/fixed_cases.json` for a custom fixed set;
`--benchmark-config` remains an alias for existing commands.

For new evaluations, pass the same sampler format used in training:

```bash
.venv/bin/python -m src.evaluation.rl.evaluate \
    --model path/to/sac_final.pt \
    --sampler-config configs/rl/evaluation_sets/sampled_reserved_regions.json --num-objects 20 --seed 42 \
    --out artifacts/held_out_evaluation/holes.json
.venv/bin/python -m src.evaluation.rl.evaluate \
    --model path/to/sac_final.pt \
    --sampler-config configs/rl/evaluation_sets/sampled_mass_extrapolation.json --num-objects 20 --seed 42
```

A single sampler JSON has `name`, `ranges` (all six geometry/physics intervals),
and `exclusions`. For separate regions, use
`{"groups": {"label": <sampler config>, ...}}`. The examples above contain eight
hole groups and two mass groups, respectively. `--num-objects` is **per group**.
Every sampled object stays fixed for all its frozen and fine-tuning episodes.
Smoke mode limits the resolved cases without changing sampled objects or seed
indices. Reports distinguish `requested_num_objects_per_group` from actual
`num_objects_per_group` and include `evaluated_objects_by_group` for all sources.
Training helpers receive the sampler explicitly; only the training entry point
chooses the default config.

Results schema version 2 records the resolved object config, seed, exact object
specifications, global seed indices, checkpoint hash, metadata and summaries.
The two conditions store shared `EvaluationResult` records directly (`episodes`,
`summary`, `metadata`); `adaptation_progress` records first success and learning
curves. Consumers of older results should account for this schema change.
Per-object `membership` distinguishes `training_support`, `excluded_region`,
`outside_ranges` and `unknown`, using the actual checkpoint training metadata,
including geometry bounds and joint exclusions. `outside_checkpoint_ranges`
lists the violated bounds. Legacy evaluation reference ranges are descriptive.

Object sampling uses a separate random stream from policy actions. Given the
same config, object count and seed, selecting a subset of groups preserves its
objects and rollout seed indices. Save and reuse the exact object specifications
as a fixed evaluation set config when comparing methods or training seeds.
`--smoke` takes two objects per selected group and reduces episode budgets.
Checkpoint force bounds remain unchanged during OOD testing, so insufficient
force can legitimately cause timeouts.

`calibrate_m3.py` reports maximum-push outcomes and reward scales from random
episodes. It reports movement and toppling separately and does not recommend
keeping or changing `F_max`: a toppled maximum-duration push does not establish
whether weaker or shorter pushes can reach the target safely. The reward-scale
check also does not measure learned-policy performance. Choose and freeze
settings after suitable training-distribution evaluation (outline §§5.3, 11.1).
Training records those settings in `config.json` and the checkpoint; output is
under the gitignored `artifacts/`.

### Experimental DeltaActorCritic and simulator integration

The draft delta actor–critic method starts each episode with a low probe, then proposes
adjustments to the last executed action. The actor receives the observation and
previous real trajectory; it does not receive absolute actions or physics
hypotheses. A separate controller simulates proposals across the full SMC²
particle ensemble (or CMA-ES hypotheses) and ranks bounded candidates with twin
trajectory critics. Predicted tilt does not veto candidates. If the bounded search finds no
candidate, it stops collection and retains the transitions already executed.

Train a fresh delta actor–critic checkpoint:

```bash
.venv/bin/python -m src.agents.rl.training.train_delta \
    --distribution-config configs/rl/training_distributions/uniform_excluding_evaluation_regions.json --smoke \
    --out-dir artifacts/delta/smoke
```

The output directory must be empty. Remove `--smoke` for the configured training
budget; `--backend cmaes` selects the other estimator. Training now uses one-step
soft TD targets and updates replay after each push, rather than waiting for
Monte Carlo episode returns. Defaults match baseline SAC's 800,000 interactions,
10,000-push warmup, batch size 256, replay capacity 1,000,000, hidden width 256,
and one gradient update per collected interaction after warmup. Probes count
toward the interaction budget, but do not create delta transitions. Warmup uses
uniform feasible deltas without critic ranking. Use `--total-steps`,
`--learning-starts`, `--gradient-steps`, `--batch-size`, and `--replay-capacity`
to change these budgets. `--tau` controls target-critic averaging (default 0.005).
Training resamples object batches until the interaction budget is reached;
`--num-objects` and `--episodes-per-object` control the object schedule, rather
than imposing a 160-episode training limit. Adaptation defaults to no warmup. Each object has its own
SysID history, retained across its episodes. The probe counts toward the episode
budget. Toppling trajectories are excluded from inference.

Choose `--trajectory-feature-strategy interpolate` (default) for fixed-point
interpolation followed by an MLP, or `--trajectory-feature-strategy self-attention`
for attention over every recorded physics sample without interpolation.
`--trajectory-points` controls only the interpolation strategy. The selected
strategy is saved in the checkpoint and restored for evaluation.

The actor receives the seven-value observation and previous trajectory. The critic
also receives the delta and candidate forecast. Neither network receives the
remaining action count. The internal simulator
reconstructs a floor-contact pose from planar state `(x, pitch, vx, wy)`;
it does not restore full 3D pose or velocity. Forecasts can therefore differ from
the target even with correct physical parameters.

Delta training supports two progress-reward strategies through `RewardModel`:
`linear` (the existing default) and `goal_proximity`. Both share the same time
cost, decision cost, success bonus, and toppling penalty. The strategy changes
only progress rewards; observations, action bounds, and success conditions are
unchanged.

For `goal_proximity`, define the distance score

$$
P(d)=-\alpha|d|+B\frac{s}{|d|+s},\qquad
r_t=P(d_{t+1})-P(d_t)-c_{\rm time}\tau_t-c_{\rm step}
+R_{\rm succ}\mathbf{1}_{\rm success}-R_{\rm topple}\mathbf{1}_{\rm topple}.
$$

The linear term gives feedback across the whole approach. The bounded rational
term rewards equal progress more strongly near the goal without diverging at
zero. Staying at any fixed distance receives only the usual costs. Retreating
reverses the progress score, and increasing absolute distance after overshooting
is penalized. This is an undiscounted score difference, not the
`gamma * P(next) - P(current)` shaping rule: it deliberately changes the
discounted reward objective and makes no policy-invariance guarantee.

Select it for a fresh experiment:

```bash
.venv/bin/python -m src.agents.rl.training.train_delta \
    --batch-size 4 --particles 64 --mcmc-steps 1 \
    --trajectory-feature-strategy self-attention --learning-starts 200 \
    --reward-strategy goal_proximity
```

`--reward-strategy` accepts `linear` or `goal_proximity`. Omission defaults to
`linear` for fresh training and restores the saved strategy for checkpoint-based
adaptation/evaluation. With `goal_proximity`, `--proximity-bonus B` defaults to
100 and must be finite and nonnegative; `--proximity-scale s` defaults to 0.25 m
and must be finite and positive. The scale is the distance where the added
closeness score is half its maximum. These options require `goal_proximity`.
The shared defaults remain `alpha=10`, `c_time=1`, `c_step=0.1`, and success and
toppling magnitudes of 100. All settings are saved in run/checkpoint reward
metadata. Checkpoint-based runs restore them and reject conflicting overrides;
legacy reward settings without a strategy restore `linear`.

For a 0.1-second push without success or toppling, the default rewards are:

| Distance change | Linear | Goal proximity |
|---|---:|---:|
| 1.00 m to 0.99 m | -0.100 | +0.061 |
| 0.50 m to 0.49 m | -0.100 | +0.350 |
| 0.10 m to 0.09 m | -0.100 | +2.001 |
| No distance change | -0.200 | -0.200 |

These coefficients are experimental, not validated learning improvements. Compare
success, toppling, and final distance across strategies; their reward totals have
different scales and are not directly comparable.

To additionally penalize stalled goal-proximity pushes, add:

```bash
--no-progress-penalty 1 --no-progress-threshold 0.001
```

This subtracts 1 extra reward when the reduction in absolute goal distance is
at most 0.001 m (1 mm) over a push, including zero progress, retreat, and
overshooting to an equally distant position. The threshold measures net goal
progress, not total displacement, velocity, or progress per second. Progress
strictly above the threshold avoids the extra penalty. It is waived on success,
toppling, or when the final position is within the environment's goal tolerance,
allowing the object to settle near the goal. Time/decision costs still apply.
For a stationary 0.1-second push outside the tolerance, the reward becomes -1.2
with this setting instead of -0.2.

`--no-progress-penalty` defaults to 0 (disabled), and
`--no-progress-threshold` defaults to 0.001 m. Both must be finite and
nonnegative and require `--reward-strategy goal_proximity`. Setting the penalty
to zero disables it; a zero threshold penalizes nonpositive progress only.
Settings are stored in run/checkpoint reward metadata and restored on evaluation
and adaptation. Checkpoints lacking these fields retain the disabled penalty;
conflicting checkpoint overrides are rejected. Existing running processes do
not pick up changed reward settings. This is an additional experimental reward
change, not a validated solution to stationary-policy learning.

To wait for the object to settle before the next decision, add these options to
either `train_delta` or `train_sac`:

```bash
--wait-for-rest --rest-timeout 2
```

The default is disabled, preserving immediate post-push decisions. Passive
physics steps apply no pushing force after the commanded duration. Rest requires
the full 3D linear speed to be at most 0.01 m/s and angular speed at most
0.01 rad/s continuously for 0.05 simulated seconds. Waiting stops early on
toppling. `--rest-timeout` defaults to 2 simulated seconds. A positive cap must
be finite and at least one physics step (1/240 s), rounded down to a whole
number of steps. If the cap expires, the next observation can still contain
motion; `settling_timeout` in the step info reports this. No extra decision is
consumed. Without GUI pacing, this advances simulated time as fast as possible,
not by sleeping for the specified duration.

For an unlimited wait, use `--wait-for-rest --rest-timeout 0`. This waits for
sustained rest or toppling without advancing to the next decision on a timeout.
The same zero setting is saved in checkpoints and applied to delta forecasts.
Rest still uses the velocity thresholds above, not mathematically exact zero
velocity. If persistent motion or numerical jitter prevents the rest condition,
the wait can continue indefinitely; interrupt the process to stop it.

Observations, progress reward, success, and termination are evaluated after the
wait. The time cost includes both commanded push duration and passive waiting.
Step info includes `settling_steps`, `settling_time` in seconds, `settled`, and
`settling_timeout`; `max_tilt` also includes the passive phase. Recorded real
`action_record.traj` and `last_trajectory` remain push-only, preserving matched
SysID replay. Delta candidate and training forecasts include the passive phase
with the same rest settings; SysID replay continues to reproduce only the
recorded force phase. This can increase simulation and attention costs.

Checkpoint environment settings save and restore the wait option and timeout.
SAC periodic/frozen evaluation and visualization restore them; delta evaluation
and adaptation also restore them and reject conflicting overrides. Old
checkpoints default to waiting disabled. `--no-wait-for-rest` explicitly selects
the disabled option for a fresh run; it cannot override a checkpoint trained
with waiting enabled. This changes the task's transition/reward semantics and
requires a fresh training experiment to compare its learning effect.

To isolate learning from parameter-estimation errors, run a small diagnostic:

```bash
.venv/bin/python -m src.agents.rl.training.train_delta \
    --backend known \
    --distribution-config configs/rl/training_distributions/uniform_excluding_evaluation_regions.json \
    --num-objects 8 --episodes-per-object 20 --max-decisions 20 \
    --out-dir artifacts/delta/known_physics
```

`--backend known` bypasses SysID and supplies the true mass, friction and COM
height as one simulator hypothesis. It does not add parameters to the actor's
observation. It is a privileged diagnostic, not unknown-physics evaluation;
the simulator still reconstructs the starting contact state, so known parameters
do not guarantee exact target forecasts. `--particles` and estimator budgets
have no effect in this mode. Pass `--backend known` explicitly when evaluating
the resulting checkpoint; the default backend remains `smc2`.

Episode lines report reward, success, toppling, timeouts, and search stops.
Exact duplicate physics hypotheses share one forecast per candidate while
preserving particle order and weights. The safety filter remains off in every
backend.

Frozen-weight evaluation and per-object adaptation reuse the parent benchmarks:

```bash
.venv/bin/python -m src.evaluation.rl.evaluate_delta \
    --checkpoint artifacts/delta/smoke/delta_final.pt \
    --benchmark-config configs/rl/evaluation_sets/fixed_cases.json --split mass_ood \
    --smoke --out-dir artifacts/delta/ood_smoke
.venv/bin/python -m src.agents.rl.training.train_delta --mode adapt \
    --checkpoint artifacts/delta/smoke/delta_final.pt \
    --sampler-config configs/rl/evaluation_sets/sampled_reserved_regions.json --num-objects 2 \
    --out-dir artifacts/delta/hole_adaptation
```

Each evaluation/adaptation object starts from the same source weights and fresh
inference state. Evaluation freezes actor/critic learning while SysID still
updates. Checkpoint action bounds and the saved finite horizon are restored;
`--max-decisions` explicitly overrides the horizon. New checkpoints use
`delta-td-trajectory-v1`. Loading requires the current model
configuration and network architecture; no legacy checkpoint conversion is
provided. All current forecasts use planar
reconstruction, including when loading checkpoints from the removed full-state
implementation; loading those weights does not reproduce that experiment's
forecast inputs. Vanilla SAC weights require separate training.

Watch the frozen delta controller in the PyBullet GUI:

```bash
.venv/bin/python -m src.evaluation.rl.evaluate_delta \
    --checkpoint artifacts/delta/train_01/delta_final.pt \
    --benchmark-config configs/rl/evaluation_sets/fixed_cases.json --split mass_ood \
    --episodes-per-object 1 --gui --speed 0.5 --linger 3
```

This shows one episode for each of the four mass-OOD objects. `--speed 0.5`
plays target physics at half speed; `--linger 3` pauses before resetting or
closing the window. Playback frames the start and goal, hides the debug panels,
and marks the target with a green line. Candidate simulations and SysID stay headless and can pause
visible motion while computing the next action. Network weights remain frozen;
SysID updates normally. Reports go to a fresh automatic evaluation directory
unless `--out-dir` is specified. GUI playback supports evaluation mode only.

The predicted-tilt safety filter was removed on 2026-10-31. Action bounds,
finite-score checks, and actual toppling termination/rewards still apply.
Existing checkpoints load with the current selection rule. Older reports
describe the filtered controller and remain historical results.

This uses SAC-style one-step soft TD targets with slowly updated twin target
critics and automatic entropy-temperature tuning. Each critic's hypothesis
predictions are posterior-averaged before taking the twin minimum. The target is
`reward + gamma * (1 - terminated) * (next_value - alpha * next_log_prob)`.
Time-limit truncations retain bootstrapping, matching baseline SAC. Next deltas
are sampled directly from the actor, with fresh forecasts from the saved next
observation and updated physics posterior; simulator ranking is an off-policy
collection mechanism, not the TD continuation policy. All executed actor
transitions remain in replay even if candidate search later stops the episode.

Sampling maps the Gaussian into the feasible delta interval around the reference
action, so actor and bootstrap actions satisfy both delta and final-action bounds.
Log densities include the state-dependent scaling. Temperature tuning targets the
same normalized-action entropy as baseline SAC. The actor uses a likelihood-ratio
value gradient so action-dependent forecasts contribute without differentiating
PyBullet. A replay-action value baseline reduces variance, and the entropy term
retains its reparameterized gradient. The delta/trajectory architecture
and ranked collection policy remain differences from baseline SAC; aligned TD and
update budgets alone do not isolate the simulator's contribution. Use controlled
ablations for that claim. Older Monte Carlo delta checkpoints require retraining.

The [2026-11-02 audit](docs/findings/delta_actor_audit_2026-11-02.md) records the
previous wrong-sign actor gradient and two corrected COM-coordinate bugs.
The push-height correction changes physics for objects with an off-center COM;
retrain existing policies before comparing learning results.

Runs save settings in `config.json`, aggregate outcomes and losses in
`results.json`, and episode outcomes in `episodes.jsonl`. Training saves
`delta_final.pt`; adaptation saves independent per-object checkpoints.
Delta training, adaptation, and frozen evaluation also append one row per
executed push to `actions.jsonl` when each episode completes. Probe pushes and
pushes with no gradient update are included. Interrupted, unfinished episodes
are not flushed to this file. Each row contains:

- `push_count`: one-based interaction count within this run, matching update logs.
- `object_index` and `episode_index`: zero-based identifiers matching episode logs.
- `episode_push`: one-based push index; `is_probe` marks the first push.
- `seed`: the episode reset seed.
- `normalized_action`: the three policy coordinates in `[-1, 1]`.
- `action`: physical `height_m`, `force_n`, and commanded `duration_s`.
- `object_parameters`: true `w`, `L`, `H`, `m`, `mu`, and `h_c` from `CuboidSpec`.
  Geometry and COM height are metres, mass is kilograms, and friction is dimensionless.
- Reward, success/toppling flags, termination/truncation flags, and absolute
  remaining goal distance before and after the action, in metres.

Height is measured from the object base along its body axis. The logged force
is the command applied by the current direct-force executor, not net force after
friction. The physical duration excludes optional passive settling; the final
distance and reward include it when enabled. True object settings are logged
for analysis and are not added to policy inputs. Existing run histories cannot
be reconstructed from checkpoints; this logging applies on the next launch.
Python console output is mirrored to `training.log`. Each gradient update is also
written immediately to `updates.jsonl`, with push count, object index, replay size,
losses, critic/actor Q means, TD target range, entropy, temperature, gradient norms,
and actor saturation (the fraction of sampled delta coordinates with absolute
value above 0.99 after mapping the feasible interval to `[-1, 1]`).
`normalized_entropy` removes the feasible interval's
scale, matching the entropy coordinates used for temperature tuning. Gradient
norms are measured before clipping; Q values are predicted returns, not safety
probabilities.

Each update also records the actor's pre-`tanh` Gaussian standard deviation on
that update's sampled replay states, evaluated after the actor update. For each
of `height`, `force`, and `duration`, fields named `actor_std_<coordinate>_*`
contain `min`, `p10`, `median`, `p90`, `max`, `mean`, and `floor_fraction`.
The last field is the fraction clamped at `log_std=-5` (`std` approximately
0.00674); the current upper bound is `log_std=1`. These are minibatch summaries,
not quantiles over the entire replay buffer. They appear in `updates.jsonl`,
saved losses, and the periodic learning log lines. Measurement uses no sampling
and does not alter the training RNG. No exploration limits are changed.

Small std can indicate premature collapse during exploration, but can also be
appropriate after learning. Gaussian std is not the standard deviation of the
final delta: `tanh` saturation and the feasible action interval also compress
its spread. Inspect these summaries together with `actor_saturation`, entropy,
and task outcomes.

`--train-freq N` runs a block of `--gradient-steps M` updates every N real pushes
after warmup. Both counts must be positive and default to 1, preserving the
original schedule. Scheduling uses `(push - learning_start_step) % N == 0`,
including probe pushes, and continues across episodes. For example, warmup 200
and frequency 8 schedule updates at pushes 208, 216, and so on, provided replay
is nonempty. Adaptation starts a new schedule for each independently loaded
object; frozen evaluation never learns. All executed nonprobe transitions still
enter replay, including pushes where no update occurs. Skipped update blocks
are not accumulated or performed later. The nominal update-to-interaction ratio
is `M/N` after warmup; changing it is an experimental choice that can affect
learning, not a behavior-preserving speed optimization. Run/checkpoint arguments
save the selected frequency. Each update records `update_seconds` (wall time of
`train_step`, excluding replay sampling and log writes) in `updates.jsonl`.

To reduce learning work per interaction while retaining batch size 256:

```bash
.venv/bin/python -m src.agents.rl.training.train_delta \
    --distribution-config configs/rl/training_distributions/uniform_excluding_evaluation_regions.json \
    --f-max 15 --batch-size 256 --backend known \
    --trajectory-feature-strategy self-attention --learning-starts 200 \
    --reward-strategy linear --train-freq 8 --gradient-steps 1
```

This runs one update per eight real pushes. It reduces the number of regenerated
forecast batches; the cost of each update and action collection remains.

`--log-every N` prints diagnostics every N gradient updates (default 100), plus the
first update. `--checkpoint-every N` saves at the first episode boundary after
each N training pushes (default 1000, including warmup probes). Both counts must
be positive. Training and adaptation replace `delta_latest.pt` using a temporary
file; frozen evaluation creates no periodic checkpoints. The latest checkpoint
contains model/optimizer settings and weights, but not replay or the live SysID
state, so it does not reproduce an exact interrupted training continuation.

Check a saved critic's rankings with actual candidate pushes from cloned target
states:

```bash
.venv/bin/python -m src.evaluation.rl.diagnose_delta_ranking \
    --checkpoint artifacts/delta/train_01/delta_latest.pt \
    --num-objects 2 --candidates 16 --seed 42
```

The diagnostic loads the checkpoint's controller, action ranges and task settings,
samples objects from its saved distribution configuration path, and follows the
frozen controller for one episode per object. At each nonterminal, nontruncated
state after a push, it samples uniform feasible candidate deltas and compares
critic scores with final-tilt toppling outcomes from the same full Bullet state.
It restores the target state after each candidate and never trains the networks.
`--checkpoint` is required; `--num-objects` defaults to 2, `--candidates` to 16,
and `--seed` to 42. Counts must be positive and the seed nonnegative.
`--out-file` optionally selects the JSONL report; by default it replaces the
checkpoint suffix with `.ranking.jsonl`, overwriting an existing report.

`safe_over_topple_pair_accuracy` measures how often a nontoppling candidate scores
above a toppling candidate from that same state (ties count as one half). It is
null if either group is absent. `best_candidate_toppled` reports whether the
highest-scoring sampled candidate toppled. Reports include candidate actions,
scores, observations, and outcomes. This is a one-push ranking diagnostic;
nontoppling does not imply target success or validate long-horizon Q estimates.

Gradient-check tooling, detailed debug traces, and timing reports have been
removed. The initial smoke
found a force-gradient sign disagreement and OOD toppling: passing unit tests or
the earlier model gate did not establish safe or useful OOD control. See the
[implementation handoff](docs/residual_actor_simulator_handoff.md) and
[smoke findings](docs/findings/residual_integration_smoke_2026-10-28.md).

The delta actor–critic code separates bound checks and forecasting from score selection, and critic
updates from actor updates. The runner separates episode collection, return
labels, replay learning, and reporting. Saved settings include the source
checkpoint path and training distribution; reports do not hash inputs or embed
the source checkpoint's entire metadata history. Older delta actor–critic checkpoints
remain loadable.

### Bayesian system identification with SMC²

`src.agents.system_identification.smc2.SMC2SysID` is an alternative to the existing CMA-ES estimator.
It maintains weighted physics hypotheses `(m, mu, h_c)` using inner particles
for uncertain measured start states, ESS-triggered resampling, and full-history
particle-MCMC moves. This is a **conditional action-replay model**, not a
continuous state tracker between pushes. Noise defaults need calibration.

```python
from src.agents.system_identification.smc2 import SMC2SysID
from src.simulation.modelling.actions import PushAction

history = []  # retain across episodes for the same object
with SMC2SysID(geometry=(w, L, H), n_theta=64, n_state=8, seed=7) as estimator:
    record = target.push(
        PushAction(h=h, F=F, tau=tau), warmup_steps=8
    )  # physical action units
    history.append(record)  # use an upright-data collection protocol
    result = estimator.update(history)  # assimilates only new records
    particles, weights = estimator.particles, estimator.weights
```

`particles` contains `(log(m), mu, h_c/H)`; use the full weighted population
for uncertainty. `result` retains the `SysIDResult` top-K interface, but its
`loss` means negative estimated log likelihood, and `point` is the particle
with greatest aggregated mass. Neither is directly comparable to CMA-ES's
minimum-loss summary. Repeated history is not counted twice; edited or shortened
history is rejected. Create a new estimator for a different object.

```bash
.venv/bin/python -m src.evaluation.system_identification.demos.smc2_demo --n-theta 64 --n-state 8
.venv/bin/python -m pytest tests/test_smc2.py
mkdir -p artifacts/smc2-doc
pdflatex -halt-on-error -interaction=nonstopmode -output-directory=artifacts/smc2-doc docs/smc2.tex
```

The [LaTeX mathematics and implementation note](docs/smc2.tex) explains the
nested filters, observation likelihood, PMMH acceptance rule, project-specific
assumptions, computational cost, and uncertainty limitations. The example
reports full-ensemble means/spreads and ESS/acceptance diagnostics without
passing hidden target parameters to the estimator. Both estimators exclude
warm-up observations from fitting.

Run the headless comparison benchmark before larger inference experiments:

```bash
.venv/bin/python -m src.evaluation.system_identification.comparisons.benchmark_smc2 --smoke --out artifacts/sysid_smoke
.venv/bin/python -m src.evaluation.system_identification.comparisons.benchmark_smc2 --out artifacts/sysid_benchmark
.venv/bin/python -m src.evaluation.system_identification.comparisons.benchmark_smc2 --known-start --development --out artifacts/sysid_known_start
```

The benchmark records each excitation dataset once, shares it across SMC² and
CMA-ES seeds, and evaluates point predictions on separate held-out pushes.
Training retains only upright records; rejected records remain in the dataset
audit. Held-out topples remain in evaluation. True-physics replay errors are
saved separately to measure reconstruction mismatch. Both estimators are
evaluated after each accepted push, with fitting time and replay counts;
their budgets are configurable but are not automatically compute-matched.

Outputs include raw action records, experiment settings,
`metrics.jsonl`, `metrics.csv`, `summary.json`, and per-update SMC² particle
snapshots (`.npz`) and plots (`.png`). Summaries report marginal 95% interval
coverage across inference seeds for the current fixed object and dataset;
this is not population calibration or a joint credible region. The smoke run
uses deliberately tiny budgets and checks the pipeline, not estimator quality.
Output directories must be new to prevent overwriting earlier experiments.

The known-start reference uses `start_std=(0, 0, 0, 0)` and `n_state=1`;
the SMC² library's uncertain-start defaults are unchanged. Both estimators
exclude warm-up observations from scoring while replaying the full warm-up.
The benchmark supplies the same `--sample-stride` (default 12) to both methods.
Indices retain the recording's original phase: with 8 warm-up steps, score
samples 12, 24, … . An empty scoring window raises an error rather than
silently using no evidence. Warm-up exclusion is unconditional.
Use `--development` for development evaluation actions distinct
from the original held-out set. Existing loss/noise scales have not been
recalibrated as part of this scoring-window change.

To diagnose numerical likelihood variance and outer-particle depletion using a
saved benchmark dataset (without selecting settings on held-out pushes):

```bash
.venv/bin/python -m src.evaluation.system_identification.diagnostics.diagnose_smc2 \
    --dataset artifacts/sysid_benchmark_initial/dataset_varied_magnitudes.json \
    --config artifacts/sysid_benchmark_initial/config.json \
    --out artifacts/smc2_diagnosis
```

This repeats likelihood estimates at fixed true physics, profiles COM likelihood
with and without warm-up samples, and compares inner/outer particle counts,
known-start-state likelihoods, and rejuvenation budgets. Known truth is used
for diagnostics and error reporting; fitted variants receive only geometry and
recorded observations. Estimator probes and ablations now exclude warm-up;
rerunning them will not reproduce the historical include-warm-up results.

The known-start grid reference separates sampling failure from likelihood
mismatch without changing either estimator:

```bash
.venv/bin/python -m src.evaluation.system_identification.diagnostics.reference_posterior
```

It loads the saved development dataset, integrates over the full uniform-in-z
prior using volume-weighted cells, and refines two independently initialized
grids. It compares marginal distributions, correlations, 95% intervals, and
development predictive distributions with SMC² after each push. Outputs include
`report.md`, `marginals.png`, `comparison.json`, convergence logs, and grid/particle
snapshots under `artifacts/sysid_reference_posterior/`. Stability is checked
empirically; unconverged steps are explicitly flagged. Reference predictive
integration uses independent Monte Carlo batches with reported errors.

Use `--resume` to reuse likelihood checkpoints and increase `--refinements`;
the record count must match, and likelihoods at 32 evenly spaced cell indices
are recomputed using the current records, object, and scoring stride. Differences
above `1e-8` reject reuse. This is a numerical spot check, not verification of
every cell or of source/dependency versions. Each refinement pass caps new cell
evaluations with
`--max-new-cells` (default 20000); no prior region is discarded. Initial grids
are controlled by `--grids` (default 12 and 24 cells per dimension).

Gradual evidence updates are opt-in for the deterministic known-start case:

```bash
.venv/bin/python -m src.evaluation.system_identification.comparisons.benchmark_smc2 \
    --development --regimes "varied durations" "passive mixed" --seeds 1 2 3 \
    --n-theta 256 --known-start --temper-updates \
    --cma-iterations 10 --cma-restarts 0 --out artifacts/sysid_tempered_run
```

In the API, set `temper_updates=True`, `start_std=(0, 0, 0, 0)` and `n_state=1`.
Each new push is introduced through intermediate likelihood powers, ending at
the full likelihood. Previous pushes remain at full strength. Conditional ESS
controls step sizes using the existing `ess_fraction`; the existing proposal
and `mcmc_steps` are used at each intermediate resampling stage. This follows
the adaptive tempering approach of [Zhou, Johansen & Aston](https://arxiv.org/abs/1303.3123).
The mode rejects uncertain starts and `ess_fraction=1`, and does not change
the default inference path. Stage powers, weights' ESS, and distinct-particle
counts are recorded in `estimator.tempering_history` and benchmark metrics. Extra
stages require extra replays, so compare measured costs as well as accuracy.

The default `proposal="covariance"` kernel shapes Gaussian moves using the
weighted particle covariance before each resampling. It is frozen within the
move stage, regularized, and normalized to keep the same expected squared step
length as the fixed kernel in prior-scaled coordinates. This changes the
proposal shape without acceptance-based step-size tuning. Select
`proposal="fixed"` or `--proposal fixed` to reproduce the previous kernel.
The benchmark and demo also default to covariance. Historical diagnostic
runners explicitly select the fixed kernel to preserve their comparisons.

For a comparison using the saved gradual-update development datasets:

```bash
.venv/bin/python -m src.evaluation.system_identification.comparisons.compare_proposals \
    --out artifacts/sysid_covariance_proposals
```

It checks the fixed control against saved particles and compares covariance
arms with and without cumulative replay ceilings. Budget limits can stop move
attempts but never stop evidence assimilation before the full likelihood.
Actual replay consumption, distinct particles, marginal intervals, prediction
errors, and distances to the separate grid calculation are reported. The
comparison uses random initial populations, without an injected truth particle.

To give CMA-ES the cumulative fitting replay allowances from a saved known-start
SMC² experiment, reusing its exact datasets and inference seeds:

```bash
.venv/bin/python -m src.evaluation.system_identification.comparisons.compare_compute \
    --baseline artifacts/sysid_development_tempered_256 \
    --out artifacts/sysid_compute_matched
```

CMA-ES uses three independent starts per update and refits the full history.
All earlier fits count against the cumulative allowance. Complete-history
evaluations can leave fewer than one history's worth of replays unused.
The API accepts `SysID.update(history, replay_budget=...)` to cap a single
update; this replaces the iteration limit and splits candidate evaluations
across the configured starts. Omitting the cap preserves the iteration budget.
The comparison includes the small-budget baseline, parameter errors, and point
prediction errors. It measures replay calls rather than physics timesteps;
refitting and sequential updating can spend those calls differently.

To distinguish poor truth likelihoods from particle loss in the corrected
known-start setup, run the oracle diagnostic:

```bash
.venv/bin/python -m src.evaluation.system_identification.diagnostics.trace_truth_particle \
    --out artifacts/sysid_truth_particle_trace
```

It replaces one initial particle with exact known physics at equal weight and
traces likelihood ranks, pre/post resampling weights, ancestry and proposal
acceptances. Paired controls keep the original random particle; full and
gradual updates are tested separately. Exact copies, descendants and nearby
hypotheses are tracked separately. This privileged initialization is a
diagnostic, not a performance benchmark or a proposal to protect truth in
production. Outputs include JSON traces, particle snapshots, a report and a
survival plot. Each trial includes `initial_summary`, so the plot also works
when no resampling occurs; those runs are excluded from first-resampling rank
counts. The default input is the saved 256-particle development dataset.
See the [SysID workflow guide](src/evaluation/system_identification/README.md)
for current helper APIs and output-field changes.

To retain the previous best CMA-ES physics as a candidate at each update:

```bash
.venv/bin/python -m src.evaluation.system_identification.comparisons.compare_compute \
    --retain-best --refit-comparison artifacts/sysid_compute_matched \
    --out artifacts/sysid_compute_matched_retained
```

The previous estimate is rescored on the complete current history, with its
replays charged to the same limit before allocating the new search budget.
The optional saved refit comparison must match the baseline's object, seeds,
push-sequence names, and scoring stride, and use the same restart count without
retention. Each included row must also contain the same saved SMC² results.
These checks do not compare the contents of every input file.
In the API, pass `previous_best_parameters=previous_result.point.theta` to `SysID.update`.
This preserves a candidate, not optimizer state or an old loss. It guarantees
no worse current training loss than that candidate, not improved prediction.

## Repo layout

```text
src/
├── simulation/
│   ├── modelling/                 # Cuboids, constants, action records
│   ├── execution/                 # Direct-force and robot executors
│   ├── environment/               # Target Gymnasium environment and rewards
│   └── tools/
│       ├── demos/                 # Interactive environment demo
│       └── validation/            # Tipping checks and M3 calibration
├── agents/
│   ├── rl/
│   │   ├── models/                # SAC, delta actor–critic, controller, neural modules
│   │   └── training/              # SAC pretraining and delta training/adaptation
│   ├── system_identification/
│   │   ├── sysid.py               # CMA-ES estimator
│   │   ├── smc2.py                # Bayesian SMC² estimator
│   │   └── scoring.py             # Shared observation window
│   └── simulator/                 # Agent's internal physics replay
├── evaluation/                    # Evaluation grouped by component
│   ├── rl/                        # Policy rollouts, metrics, evaluation sets
│   │   └── demos/                 # Trained-policy visualization
│   └── system_identification/
│       ├── shared/                # Push protocols, records, metrics, and grids
│       ├── comparisons/           # Estimator accuracy, budgets, proposals
│       ├── diagnostics/           # Likelihood, posterior, truth-particle checks
│       └── demos/                 # Estimator API examples
└── self_learning/                 # Personal PyBullet lessons and playlist
```

`docs/` contains the project specifications and reference materials; `scripts/`
contains only the macOS PyBullet build shell script.

See the [System Identification evaluation guide](src/evaluation/system_identification/README.md)
for the role of each study and which tools are needed for normal evaluation.

For conventions (action space, episode budget, friction convention) and project
status, see [`CLAUDE.md`](.claude/CLAUDE.md).

## AI coding assistants

Claude Code and Codex share the same project instructions and simulation skill:

- `AGENTS.md` → `.claude/CLAUDE.md`
- `.agents/skills/run-sim` → `.claude/skills/run-sim`
- `.agents/skills/setup-env` → `.claude/skills/setup-env`

These are relative symlinks. Edit the files under `.claude/` to keep both
assistants in sync. When adding a Claude skill, add a matching symlink under
`.agents/skills/` for Codex discovery.
