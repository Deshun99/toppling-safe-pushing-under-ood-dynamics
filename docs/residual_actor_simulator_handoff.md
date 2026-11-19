# DeltaActorCritic and internal simulator: implementation handoff

Last maintained: **2026-11-01**. This document describes the current code.
The method is experimental: passing implementation tests does not establish
useful actor gradients, reliable control, or out-of-distribution generalization.
The filename retains the earlier “residual” name; the learner is `DeltaActorCritic`.

## Scope and agreed design

The actor proposes corrections to the last executed action using observed motion.
System identification configures candidate simulations; its raw physical
hypotheses are not neural inputs. This is a separate experiment from baseline
SAC and from the original Dyna/shield proposal in the project outline.

| Decision | Current behavior |
| --- | --- |
| Actor inputs | Seven-value observation and latest real trajectory. No previous absolute action or raw physics parameters. |
| Actor output | Squashed Gaussian distribution over normalized action deltas. |
| Moving reference | Add delta to the last executed action. Rejected proposals never update the reference. |
| Probe | Start every episode with a fixed physical push; reset the reference to that executed probe. |
| Simulator | Forecast each bounded candidate under the inferred hypothesis ensemble. |
| Critic inputs | Observation, previous real trajectory, delta, and candidate forecast. |
| Candidate selection | Uniform feasible delta during warmup; otherwise highest twin-min score after posterior averaging. |
| Safety filter | Removed. Predicted tilt does not reject candidates; action bounds and actual toppling termination remain. |
| Inference history | Persist across episodes of one object; start fresh for each new object. |
| Toppling evidence | Retain complete real records for reporting and learning, but exclude toppled transitions from SysID history. |
| Model mismatch | Controlled mismatch experiments remain future work. |

Search distillation, a learned acceptance gate, differentiable dynamics, and
Dyna replay are possible alternatives, not implemented commitments.
Robot execution remains a separate planned project stage.

## Runtime and data contracts

1. Initialize the object's estimator and simulator. Reset the target episode.
2. Execute the probe, record its trajectory, and assimilate eligible evidence.
   The probe counts toward the action budget but is not an actor training sample.
3. After each execution, append the one-step actor transition to replay and
   perform the configured updates once warmup is over. Probes count as
   interactions but supply no actor transition.
4. During warmup, sample one uniform feasible delta. Otherwise run the actor
   once, sample a batch from its cached feasible distribution, forecast the
   candidates, and score them.
5. Select the highest finite score, or resample until the batch limit is reached.
   Exhaustion stops collection; executed transitions remain available for TD learning.
6. Before execution, retain the selected forecast and historical context. Execute
   the action, update the reference and real trajectory, and assimilate eligible
   evidence. Repeat until success, toppling, timeout, or search exhaustion.
7. Continue updates between pushes, including the terminal transition. Save
   episode outcomes at the end; no episode-return labeling pass is needed.

`ActionRecord(start, action, traj, warmup_steps)` stores the planar starting state,
physical command, dense trajectory, and original warm-up count. `DenseTrajectory`
contains time, x, pitch, x velocity, and pitch angular velocity, sampled during
warm-up and the commanded push; it does not include post-push settling.

`ForecastContext` snapshots the observation, planar state, geometry, action
reference, particles, weights, previous real trajectory, and action bounds.
Actor updates regenerate forecasts from this historical context rather than
substituting the subsequently observed trajectory.

### Simulator reconstruction limitation

`InternalSim.predict` reconstructs a floor-contact pose from `(x, pitch, vx, wy)`.
It does not restore full 3D pose, vertical/lateral velocity, yaw, or contact state.
The full geometric-state implementation was removed. No `BodyState`, full-state
record fields, or geometric-centre trajectory conversion is currently used.
Even known physical parameters therefore do not guarantee exact target forecasts.

### Inference backends

- **SMC²:** use all outer physics particles and posterior weights, not just the
  top-K result summary. Coordinates are `(log mass, friction, relative COM)` and
  are converted to physical parameters using observed geometry. The controller
  uses one known-start inner particle. Exact duplicate hypotheses reuse one
  forecast per candidate while retaining their order and probability weights.
- **CMA-ES:** refit upright history with a bounded replay budget. Its discrepancy
  loss is not directly comparable to SMC² negative log likelihood.
- **Known physics:** bypass estimation and supply true object physics as one
  simulator hypothesis. This is a privileged learning check, not evaluation of
  adaptation to unknown physics. Hidden parameters still do not enter the actor.

SMC² history must remain append-only and unchanged after assimilation. Particle
collapse and poor coverage remain possible; weak probe evidence does not itself
ensure a diverse posterior.

## Learning objective and limitations

The trajectory strategies are mutually exclusive. Set
`DeltaActorCriticConfig.trajectory_feature_strategy` (or the training CLI option
`--trajectory-feature-strategy`) to:

- `"interpolate"` (default): resample to `trajectory_points` (default 16), flatten,
  and encode with an MLP.
- `"self-attention"`: retain every recorded physics sample without interpolation
  or truncation, and encode with a Transformer and learned CLS token.
  `trajectory_points` is unused by this path. Forecasts retain their individual
  lengths in storage; tensor preparation pads them and supplies Boolean masks.

Both paths use five fixed-scale features: elapsed time, relative displacement,
pitch, linear velocity, and angular velocity. Attention receives the four scaled
physical channels and timestamps converted back to seconds since the first
recorded point. Its dropout is zero so collection and candidate ranking remain
repeatable without consuming the global Torch RNG. The actor and both critics
share one trajectory encoder across real and forecast inputs. The critic
optimizer trains this encoder from TD losses;
the actor optimizer trains only the policy head. Actor updates freeze the shared
encoder and both critic heads. The actor value update uses a likelihood-ratio
estimator rather than a direct critic action derivative.

The twin critics now fit one-step soft TD targets. Their predictions are
posterior-averaged before taking the twin minimum. A separate target-critic copy
is updated by Polyak averaging (`tau=0.005`). For each stored transition:

```text
target = reward + gamma * (1 - terminated)
                  * (target_value(next_context, actor_delta) - alpha * log_prob)
```

The next context records the actual next observation, the just-executed real
trajectory, reference action, and updated posterior. Fresh next-action forecasts
use that saved context. True terminations do not bootstrap; time-limit truncations
do, matching baseline SAC. Actor samples define the TD continuation policy;
simulator-ranked selection collects off-policy experience. Search stops no longer
discard already executed transitions.

The actor minimizes negative twin-min value plus entropy. Gaussian samples are
mapped into the intersection of configured delta limits and the feasible final
action range, replacing the previous bound penalty. Log densities include that
mapping. Automatic temperature tuning uses a normalized-action target entropy of
minus the action dimension, adjusted for the feasible interval's scale. The
actor's newly sampled deltas receive fresh simulated forecasts. The simulator
branch remains detached; the value gradient uses the probability of the fixed
sample instead of differentiating the critic while holding forecasts fixed:

```text
Q_sample = min_twin posterior_mean Q(context, delta, forecast(delta))
baseline = min_twin posterior_mean Q(context, replay_delta, replay_forecast)
actor_surrogate = alpha * pathwise_log_prob
                  - stop_gradient(Q_sample - baseline) * fixed_sample_log_prob
```

The fixed-sample density evaluates the current Gaussian at the detached raw
sample. Its likelihood-ratio gradient accounts for the complete sampled value,
including action-dependent forecasts, without requiring a PyBullet derivative.
The replay-action baseline is independent of the new actor noise. Entropy retains
its reparameterized gradient. This replaces the previous partial derivative,
which could have the wrong sign even with an exact critic; see the
[2026-11-02 audit](findings/delta_actor_audit_2026-11-02.md).

This estimator has higher variance than a fully differentiable pathwise update.
Actor proposal probabilities also differ from actions executed after candidate
ranking. This remains an experimental method rather than standard SAC.

Historical [smoke findings](findings/residual_integration_smoke_2026-10-28.md)
record disagreement and OOD toppling under earlier settings; they do not validate
the current implementation.

## Training, evaluation, and compatibility

Use the [README commands](../README.md#experimental-deltaactorcritic-and-simulator-integration)
for fresh training, adaptation, frozen evaluation, and GUI playback.

Current non-smoke CLI defaults:

| Setting | Default |
| --- | --- |
| Object batch / episodes per object | 8 / 20; resample batches until the training interaction budget is reached |
| Fresh episode action limit | 10, including the probe |
| Network hidden width / trajectory points | 256 / 16 |
| Delta scales | `(0.3, 0.4, 0.4)` |
| Learning rate / discount | `3e-4` / `0.99` |
| Training interactions / warmup | 800,000 / 10,000, including probes |
| Updates per push / batch size / replay capacity | 1 / 256 / 1,000,000 |
| Target averaging / temperature | `tau=0.005` / automatic, initially 1 |
| Candidates per batch / maximum batches | 8 / 4 |
| Probe height / force / duration | `0.1H` / `0.5 N` / `0.05 s` |
| Backend / outer particles / MCMC steps | `smc2` / 16 / 2 |
| CMA-ES replay budget | 64 in the CLI; 128 in the direct controller config |

Training shares network weights and replay across objects. Adaptation reloads the
source weights and clears replay for each object. Frozen evaluation disables
network updates but continues SysID updates. Candidate sampling remains
stochastic. Evaluation/adaptation restore checkpoint action bounds and horizon;
`--max-decisions` explicitly overrides the latter. Select the inference backend
explicitly when needed; it defaults to `smc2`, including for known-physics weights.

Checkpoints use `delta-td-trajectory-v1` and must match the current model
configuration and network architecture. Loading does not convert legacy
checkpoints or baseline SAC weights. Current controller settings discard obsolete
`near_topple_fraction` and `use_full_state`
fields. All forecasts now use planar reconstruction: loading weights trained with
the removed full-state implementation does **not** reproduce its forecast inputs.
The 2026-11-02 COM-coordinate fixes also change the physical application height
for off-center-COM objects. Existing weights remain loadable, but new learning
comparisons require retraining under the corrected physics.

Runs save settings in `config.json`, aggregate outcomes and losses in
`results.json`, and episode outcomes in `episodes.jsonl`. Training saves
`delta_final.pt`; adaptation saves independent per-object checkpoints.
Gradient-check tooling, per-push debug traces, timing reports, and controller
diagnostic counters have been removed.

One-time candidate, force, episode-return, and state-reconstruction analysis
scripts were removed from the repository. Their commands are not supported
entry points. Local historical reports under ignored `artifacts/` are not part
of the maintained runtime or evidence for the current code.

## Code ownership and validation

| Path | Responsibility |
| --- | --- |
| `src/agents/rl/models/sac.py`, `nn_modules.py` | Baseline SAC and its neural/replay components. |
| `src/agents/rl/models/delta_actor_critic.py` | Delta actor, trajectory critics, `TrajectoryEncoder` feature extraction, fixed input scales, losses, replay examples, checkpoints. |
| `src/agents/rl/models/delta_controller.py` | Probe, moving reference, inference, bounded search, forecast context. |
| `src/agents/rl/training/train_delta.py` | Per-push collection; `DeltaRun` owns transition replay, update scheduling, and train/adapt/evaluate results. |
| `src/agents/rl/training/delta_utils.py` | Delta action bounds, CLI options, validation, and smoke settings. |
| `src/agents/rl/training/io_utils.py` | Output-directory creation and JSON conversion shared with SAC training. |
| `src/evaluation/rl/evaluate_delta.py` | Frozen-evaluation entry point using the shared delta runner. |
| `src/agents/simulator/internal_sim.py` | Headless planar reconstruction, prediction, replay, and body cleanup. |
| `src/agents/system_identification/` | SMC², CMA-ES, and shared trajectory scoring. |
| `src/simulation/modelling/records.py` | Shared executed-action record. |
| `src/evaluation/rl/evaluator.py` | Baseline SAC evaluation; separate from the delta controller lifecycle. |

Tests cover terminal/truncated TD targets, per-push updates, saved next-context forecasts, actor gradients, checkpoint
round-trips, bounds, reference updates, inference history, and simulator cleanup.
The TD implementation passed **338 tests on 2026-11-01**.
This is implementation validation, not evidence of learning effectiveness.

Open research work:

- Establish critic return calibration on independent states and action budgets.
- Establish that actor updates improve independently measured target returns.
- Measure the effect of planar state reconstruction on forecast fidelity.
- Compare with baseline SAC under matched objects, actuator bounds, interaction
  and compute budgets, and evaluation stochasticity.
- Define controlled model mismatch and train/test separation before claiming
  correction of simulator errors or OOD robustness.
