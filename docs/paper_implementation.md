# Toppling-Safe Pushing Under OOD Dynamics: implementation

The supplied paper, *Toppling-Safe Pushing Under OOD Dynamics* (Liu Ernest
Hin Yui and Wong Deshun), governs this implementation. Earlier proposals and
PRs are useful development references, but do not override it. This code
implements the experiment; it does not establish that a fresh training run
will reproduce the paper's success rates.

## Relationship to the earlier PRs

| Reference | Reused foundation | Correction required by the paper |
| --- | --- | --- |
| [PR #1](https://github.com/Deshun99/toppling-safe-pushing-under-ood-dynamics/pull/1) | Two-prismatic-joint URDF, recessed bracket, upright pad, approach and homing sequence | Extend horizontal travel to 1.28 m; replace fixed-speed, motor-force-limited pushing with contact-force admittance control. |
| [PR #2](https://github.com/Deshun99/toppling-safe-pushing-under-ood-dynamics/pull/2) | Shared-client environment integration, execution reports and failure handling | Requested force is a contact-force target, not a motor ceiling. Identification replays measured actual force. Use the three refinement actor–critic methods, rather than the separate B0–B6 Dyna/filter study. |

The new branch starts from `main`; neither earlier PR is merged. Their old
findings and measured force ratios are not results of this implementation.

## Implemented method

The target gantry has pad centre `(xb + 0.12 + qx, yb, zb + 0.65 + qz)`,
with `qx` in `[0, 1.28]` and `qz` in `[-0.63, 0]` metres. The pad measures
`0.03 × 0.10 × 0.04 m` and has lateral friction 0.1. The executor retreats
before changing height, approaches slowly, and starts the timed push after
contact. It limits requested contact heights to the pad window
`[0.02, H - 0.02]`. Consecutive pushes retain robot pose; reset raises the
pad before retracting.

During the timed push, the executor integrates
`v = clip(max(0, block_velocity) + (requested_force - filtered_normal_force), 0, 3)`
into its horizontal position target. The gain is 1 m/(N s); a position servo
with velocity feedforward drives the joint. The servo's 100 N effort ceiling
is separate from the policy's 15 N contact-force bound. Inner position and
velocity gains are 0.001 and 0.01; lower gains avoid repeated impacts and
contact loss observed with PyBullet's defaults. A first-order force filter
uses a 0.02 s time constant. These are implementation choices: the supplied
paper specifies filtered feedback and inner control without these settings.

`ActionRecord` keeps the requested action and measured start of the timed
push. Gantry records additionally contain one actual horizontal force and
application height per physics step. Identification uses this trace, including
zero-force contact gaps. It does not substitute a mean force or the requested
force. The application point approximates the patch by its normal-weighted
centre; planar replay omits vertical/lateral forces and distributed contact
couples. It retains the paper's planar-state reconstruction discrepancy.
Future physics forecasts still apply the requested force directly, without
the gantry or its controller.

All three methods use bounded action refinements and self-attention over the
latest recorded trajectory. The physics method scores all 32 weighted SMC²
hypotheses, averaging each critic before taking the lower critic value.
Estimated parameters configure simulation and do not enter the policy
observation. The prior is uniform in `(log m, mu, h_c/H)` over the paper's
bounds; the measured starting state is treated as known. Covariance-adapted
Metropolis–Hastings proposals maintain diversity after resampling.

The learned model independently interpolates trajectories to 16 evenly
spaced times. Its five scaled channels are elapsed time / 0.5 s,
displacement / 0.25 m, tilt / 0.2 rad, velocity / 1 m/s and angular velocity
/ 2 rad/s. Policy attention keeps variable-length trajectories. Prediction
loss uses examples with saved next-decision context: success/toppling
transitions are excluded, while decision-limit truncations remain eligible.
The actor's likelihood-ratio gradient supports detached simulator forecasts.
Candidate ranking supplies no hard toppling guarantee or safety shield.

Execution failures terminate the episode with an explicit status. A proposed
action's observed terminal reward trains the critic even if approach failed
before a timed trajectory could be recorded. No completed trajectory or
identification observation is fabricated; terminal examples also remain
excluded from learned-model fitting. These failures remain visible in logs.

## Train and evaluate

Run from the repository root with the existing environment:

```bash
.venv/bin/python -m src.agents.rl.training.train_delta \
  --paper --world-model physics --forecast-workers 4 \
  --out-dir artifacts/paper/physics
.venv/bin/python -m src.agents.rl.training.train_delta \
  --paper --world-model learned --out-dir artifacts/paper/learned
.venv/bin/python -m src.agents.rl.training.train_delta \
  --paper --world-model none --out-dir artifacts/paper/model_free
```

`--paper` sets gantry execution, force bound 15 N, 40 decisions, 0.08 m
goal tolerance, attention, 32 hypotheses and two MCMC moves. Training uses
the existing training distribution with reserved in-distribution parameter
combinations excluded. Explicit flags override preset defaults; `--smoke`
reduces the budgets for a pipeline check.

The supplied paper gives 13 curriculum stages between 0.10 and 1.0 m but
does not list intermediate distances or all optimizer/reward settings. For
these unspecified details the preset uses the compatible physics-run
hyperparameters in the repository's
[report source](report/Imagine%20Before%20You%20Push.tex): distances
`0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0`,
200 warm-up pushes, batch size 64, updates every
eight pushes, four candidates per batch, and ten episodes per training
object. It uses goal-proximity reward, toppling penalty 10, inactivity
penalty 1.5, and uncertainty-reduction weight 50 for physics only. Attention
for **all** methods follows the supplied paper wherever the older report
differs. Use `--info-gain-beta 0` for an equal-reward ablation.

Curriculum stages end at the `--curriculum-steps` push cap (10,000 by default
with `--paper`), regardless of success. They can end earlier after **more than
2,000 pushes** at the current distance and **over 90% success in its latest
30 completed episodes** (at least 28 successes). Early advancement occurs at
an episode boundary; the hard cap can interrupt an episode. Both the push
count and success window reset for each distance, including the final stage.
This adds early advancement to the historical fixed-budget curriculum and
does not change historical results.

Evaluate each method's checkpoint separately:

```bash
.venv/bin/python -m src.evaluation.rl.evaluate_delta --paper \
  --checkpoint artifacts/paper/physics/delta_final.pt \
  --out-dir artifacts/paper/evaluation_physics
```

The evaluation preset selects the existing fixed 24-object benchmark and
five episodes per object. Checkpoints restore network architecture and task
settings. Neural weights stay frozen; physics hypotheses persist across an
object's episodes and reset for the next object. Use the final checkpoint
from each corresponding training command for learned/model-free evaluation.
`--gui` enables visualization; `--no-robot` explicitly selects a different
target embodiment for a cross-executor experiment.

## Component validation

```bash
.venv/bin/python -m pytest
.venv/bin/python -m src.simulation.tools.validation.tipping_tests
.venv/bin/python -m src.simulation.tools.validation.gantry_contact \
  --output artifacts/paper/contact.json
.venv/bin/python -m src.evaluation.system_identification.diagnostics.calibrate_scales
```

The contact harness defaults to two seeds, six objects per seed and eight
independent pushes per object. It saves commands, sampled objects and
settings alongside contact-force ratios, duration ratios, signed pad-height
errors, displacement, toppling and failure counts. Seeds, command sampling
and resetting before each push are explicit implementation choices where
the paper does not specify its protocol. Summed normal-force impulse is
reported separately from the horizontal force used for replay. Partial
failures retain available measurements and are counted.

The existing calibration command measures paired replay errors on the
paper's single calibration cuboid. Its error scales measure simulation
discrepancy, not sensor noise, and do not establish accuracy for every
object. Component checks and short training runs validate code paths;
learning quality, multi-seed rankings and hardware transfer require their
own experiments.

## Development measurements for this implementation

The [saved 96-push contact sweep](findings/evidence/paper_gantry/contact.json)
uses the harness defaults (seeds 0 and 1). It recorded no contact failures,
execution failures or stray contacts; 24 pushes toppled. The mean measured
normal/requested force ratio was 1.104 (standard deviation 0.592), mean
contact/commanded duration ratio 0.903, mean signed pad-height error
−0.0118 mm, and mean timed displacement 0.145 m.

These independently sampled development pushes differ from the paper's
reported measurements. They validate that the measurement pipeline runs,
not reproduction of the paper's contact or policy results. Nominal low-height
0.3 s tests at 5 and 10 N also check approximate force tracking and sustained
contact. At requests below sliding friction, the non-retracting law can
retain preload from approach instead of unloading to a small force target.
The [short curriculum check](findings/paper_curriculum_check.md) records
three reduced-budget stages and matched held-out in-distribution evaluations.
Full 13-stage training, OOD policy performance and additional training seeds
remain unmeasured in this change.
