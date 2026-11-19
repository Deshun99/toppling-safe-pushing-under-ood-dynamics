# Simulation-Assisted Toppling-Safe Pushing

## 0. Summary

An agent must push an upright object to a target without toppling it. The policy commands a push height, force magnitude, and duration. Early stages apply the force directly in simulation; the final stages must execute pushes through a simulated robot arm. Contact remains on the vertical centreline of the cuboid's rear face; arbitrary lateral contact positions are excluded from the core scope. Whether a push is safe and effective depends on latent physical properties that are not directly observable, especially mass, object-floor friction, and vertical centre-of-mass (COM) height.

The agent estimates these properties online from observed object motion and maintains a set of plausible physics hypotheses. It then uses an internal physics simulator in two ways:

1. **Imagination as data**: Dyna-style hypothetical rollouts are added to the replay data used to update a SAC policy.
2. **Imagination as a shield**: candidate actions are simulated across the current hypothesis set, and actions predicted to topple the object under any plausible hypothesis are rejected.

**Headline hypothesis**

> Pessimistic, ensemble-based physics imagination can reduce toppling during adaptation to unfamiliar objects relative to point-estimate imagination and model-free RL, while maintaining task success. The project additionally studies how this benefit changes as the internal simulator becomes increasingly misspecified and which latent-parameter errors are most damaging.

All experiments are sim-to-sim. The target environment and the agent's internal simulator are separate physics environments. The internal simulator is deliberately idealized, while the target environment introduces graded structural mismatch as a controlled proxy for the sim-to-real gap.

---

## 1. Motivation

Model-free RL can require substantial trial-and-error interaction. That is problematic when failed exploratory actions are costly, and toppling is one such failure.

Two geometrically identical objects can respond differently to the same applied force because their latent physical properties differ. For example, a lighter object accelerates more under the same force, a high-friction object resists sliding more strongly, and a high-COM object is more susceptible to rotational instability.

This project treats a physics simulator as the agent's **imagination**. From a short interaction history, the agent estimates plausible physical properties, instantiates them in an internal simulator, and uses the resulting hypothetical experience both to improve the RL policy and to screen potentially unsafe actions before they are executed in the target environment.

The central research question is:

> **Can simulation-generated hypothetical trajectories improve an RL agent's ability to push objects safely and efficiently when mass, friction, and vertical COM are unknown or out of distribution?**

---

## 2. Related Work

| Area | Representative work | Difference in this project |
|---|---|---|
| Imagined experience for RL | Dyna (Sutton, 1990); MBPO (Janner et al., 2019) | The world model is an explicit physics engine with interpretable latent parameters rather than a learned dynamics network |
| Simulator parameter identification | SimOpt (Chebotar et al., 2019); BayesSim (Ramos et al., 2019); TuneNet (Allevato et al., 2020) | Identification is online and object-specific, and feeds an ensemble of plausible simulators |
| Online adaptation via latent dynamics | UP-OSI (Yu et al., 2017); RMA (Kumar et al., 2021) | Adaptation here occurs primarily through simulator-generated experience rather than by directly supplying estimated latent parameters to the policy |
| Dynamics randomization | Peng et al., 2018 | Used during pretraining as a model-free robustness baseline |
| Safe RL via shielding | Alshiekh et al., 2018; Berkenkamp et al., 2017 | The shield uses a physics simulator evaluated across uncertain physical hypotheses |
| Toppling-aware pushing | Lynch & Mason, 1996; Lynch, 1999; later constrained-RL pushing work | This project couples toppling-aware control with online physical-parameter estimation and simulated adaptation |
| COM estimation by pushing | Prior work on sub-critical tipping and dynamic pushing | Estimation is evaluated by its downstream value for safe RL adaptation rather than only by parameter error |

Neither simulated experience for RL nor system identification for manipulation is new. The contribution is a controlled study of **uncertainty-aware physics imagination for safe adaptation** in a task where the relevant physical properties are hidden and must be inferred from motion.

---

## 3. Problem Formulation

### Notation guide

The following symbols are used throughout this outline. COM means centre of mass; a dot denotes a time derivative, a hat denotes an estimate or simulator prediction, and a superscript $*$ denotes hidden ground truth. Subscripts $t$ and $i$ index decisions or recorded actions, $k$ indexes physics hypotheses, and $\max$ denotes a configured upper limit.

| Symbols | Meaning |
|---|---|
| $G=(w,\ell,H)$ | observable cuboid geometry: width along the push direction, depth, and height |
| $\theta=(m,\mu,h_c)$; $z$ | latent mass, object-floor friction coefficient, and COM height; optimization coordinates $z=(\log m,\mu,h_c/H)$ |
| $x_t,d_t,\psi_t$; $\dot{x}_t,\dot{\psi}_t$ | position along the push axis, remaining target distance, and tilt; translational and angular velocity |
| $o_t,s_i$ | policy observation; recorded simulator start state for action $i$ |
| $a_t=(h_t,F_t,\tau_t)$; $F_{\max}$ | action: force-application height, horizontal force magnitude, and duration; maximum allowed force |
| $\pi(a\mid o)$; $Q_\phi(o,a)$ | action policy conditioned on an observation; critic value with learned parameters $\phi$ |
| $T_{\max},N_{\mathrm{ep}}$; $\epsilon,\psi_{\max}$ | maximum decisions per episode and adaptation episodes per object; target-distance tolerance and conservative tilt threshold |
| $g,a$ | gravitational acceleration and translational acceleration in Section 4; elsewhere $a$ denotes an action |
| $\mathcal{H}_t,\xi_i$ | interaction history before decision $t$ and dense trajectory recorded during action $i$; $\tau\in\xi_i$ indexes trajectory sample times in the loss |
| $L(z)$; $\sigma_x,\sigma_\psi,\sigma_v,\sigma_\omega$ | trajectory-discrepancy loss; normalization scales for position, tilt, translational velocity, and angular velocity errors |
| $\hat{z}_t,\hat{\theta}$; $\Theta_K,K,w_k,T$ | fitted parameter estimates; retained hypothesis set, its size, hypothesis weight, and weighting temperature (distinct from geometry width $w$ and decision limit $T_{\max}$) |
| $M,N_c$; $\mathcal{D}_{\mathrm{sim}},\mathcal{D}_{\mathrm{target}},\rho$ | number of imagined rollouts and candidate actions; simulated and target replay buffers and their sampling ratio |
| $\varepsilon$ | deliberately injected relative parameter error in Section 9 (distinct from target tolerance $\epsilon$) |
| $r_t,\alpha,c_{\mathrm{time}},c_{\mathrm{step}}$ | reward, progress weight, elapsed-time penalty rate, and per-decision cost |
| $R_{\mathrm{succ}},R_{\mathrm{topple}}$; $\mathbb{I}_{\mathrm{succ},t},\mathbb{I}_{\mathrm{topple},t}$ | success bonus and toppling penalty magnitudes; binary indicators for those outcomes |

### 3.1 Object

An upright rigid cuboid rests on a horizontal surface. Its observable geometry is

$$
G = (w,\, \ell,\, H),
$$

where $w$ is the width along the push direction, $\ell$ is the depth, and $H$ is the height.

The object's latent physical parameters are

$$
\theta = (m,\, \mu,\, h_c),
$$

where $m$ is mass, $\mu$ is the object-floor friction coefficient, and $h_c$ is the vertical COM height.

The target environment stores the true parameters

$$
\theta^* = (m^*,\, \mu^*,\, h_c^*)
$$

for evaluation only. They are never given directly to the policy or to the system-identification procedure.

For numerical optimization, the latent parameters are represented in normalized form as

$$
z = (\log m,\, \mu,\, h_c/H).
$$

### 3.2 Observation

The task is partially observable because the policy does not observe $m$, $\mu$, or $h_c$.

At decision step $t$, the policy observes

$$
o_t = \left(d_t,\, \psi_t,\, \dot{x}_t,\, \dot{\psi}_t,\, G\right),
$$

where:

- $d_t$ is the remaining distance from the object to the target along the push axis;
- $\psi_t$ is the object's tilt angle about the tipping axis;
- $\dot{x}_t$ is the object's translational velocity along the push axis;
- $\dot{\psi}_t$ is its angular velocity about the tipping axis; and
- $G$ is the observable cuboid geometry.

The object is **not required to settle between decisions**. Therefore $\dot{x}_t$ and $\dot{\psi}_t$ may be non-zero and are necessary to distinguish dynamically different states that have the same position and tilt.

The policy is

$$
\pi(a_t \mid o_t).
$$

Estimated latent parameters are **not supplied directly to the policy**. They are used to configure the internal simulator. This keeps the main experimental question focused on the value of simulated experience rather than the value of privileged parameter estimates as policy inputs.

System identification additionally uses dense within-action trajectories of object position, tilt, translational velocity, and angular velocity.

### 3.3 Action

Each decision step consists of one externally applied force primitive:

$$
a_t = \left(h_t,\, F_t,\, \tau_t\right),
$$

where:

| Symbol | Meaning | Provisional range |
|---|---|---|
| $h_t$ | force-application height on the rear face | $[0.1H,\,0.9H]$ |
| $F_t$ | horizontal force magnitude toward the target | $[0,\,F_{\max}]$ |
| $\tau_t$ | duration for which the force is applied | $[0.05,\,0.50]$ s |

The force direction is fixed toward the target, so the task is restricted to one-dimensional translation while retaining the rotational dynamics required for toppling.

During the interval $\tau_t$, PyBullet applies the external force at the selected body-relative height $h_t$. The force is applied at the lateral centre of the rear face to avoid intentionally inducing yaw. The application point is updated from the object's current pose during the action, while the force direction remains along the target axis.

In direct-force mode, when $\tau_t$ expires, the force is removed and the next decision point occurs immediately. In the required final-stage robot mode, a low-level controller approaches the rear-face centreline at height $h_t$, establishes contact, pushes for $\tau_t$ measured from contact onset, and retracts before the next decision. No mandatory object-settling period is imposed. The policy action remains $(h_t,F_t,\tau_t)$; joint-level control and inverse kinematics belong to the executor. Commanded force is a controller target, not a guarantee of realized contact force.

Robot integration must share the environment's physics client, reset the arm between episodes, report failed or timed-out pushes, and record commanded and realized execution data. Record object motion and monitor failure throughout approach, pushing, and retraction; report total simulated execution time separately from force-application duration. Validate the robot workspace and target distance before evaluation. The internal imagination simulator remains a direct-force model, making robot execution/contact effects an explicit source of model mismatch.

See [Robot Integration Plan](../.claude/analysis/robot_integration_plan.md) for implementation details. Robot execution is required; arbitrary rear-face contact positions are future work.

The three action variables have distinct physical roles:

- $h_t$ controls the moment arm and therefore strongly affects rotational response and toppling tendency;
- $F_t$ controls translational acceleration and the magnitude of the applied tipping moment;
- $\tau_t$ controls the duration of force application and therefore the total impulse delivered to the object.

Allowing $F_t = 0$ also permits the policy to choose a passive interval when it is useful to observe free motion or allow friction to slow the object.

### 3.4 Episode, success and toppling

- **Step**: one force primitive.
- **Episode**: at most $T_{\max}=10$ decisions.
- **Target interaction**: one action executed in the target environment.
- **Success**: the object is within a target tolerance $\epsilon$ and remains below a small tilt tolerance. A small terminal speed tolerance may also be imposed so that merely passing through the goal region at high speed does not count as successful placement.
- **Topple**: terminal failure when the object's COM projection leaves its support region or, equivalently in the planar cuboid approximation, when tilt exceeds the geometry- and COM-dependent critical angle.
- **Near-topple threshold**: a conservative tilt threshold $\psi_{\max}$ used by the optional safety filter before the true toppling condition is reached.

---

## 4. Physics Background

Consider a planar cuboid of width $w$ with horizontally centred COM height $h_c$. A horizontal external force $F$ is applied at height $h$ while the object is sliding.

Under a simple Coulomb-friction approximation,

$$
a \approx \frac{F}{m} - \mu g.
$$

A useful approximate no-tipping condition can be written in terms of the location of the resultant normal force. For forward tipping to be avoided,

$$
\mu h_c + \frac{F}{m g}(h-h_c) \le \frac{w}{2}.
$$

Equivalently,

$$
\frac{F}{m g}(h-h_c) \le \frac{w}{2} - \mu h_c.
$$

This expression is used only as physical intuition and as a sanity check for the simulator; the RL agent does not receive it.

The force-controlled formulation makes all three hidden parameters relevant:

1. **Mass $m$**: for a fixed $F$, lower mass produces greater acceleration and can increase dynamic tipping tendency.
2. **Friction $\mu$**: affects the force required to slide, the translational acceleration during forcing, and deceleration during passive motion.
3. **COM height $h_c$**: changes the restoring moment and strongly affects how close the object is to rotational instability.

A single transition may not uniquely identify all three parameters. For example, low acceleration can result from larger mass, larger friction, or both. Varying force magnitude, force height, and duration across interactions supplies the excitation needed to disambiguate them.

This creates a safety-information trade-off: dynamic pushes are informative about latent physics, but aggressive or high-applied forces can also increase toppling risk.

---

## 5. Environments and Reality-Gap Levels

### 5.1 Internal simulator: the agent's imagination

The internal simulator is an idealized PyBullet model of the cuboid. It is parameterized by a candidate hypothesis

$$
\theta_k = (m_k,\, \mu_k,\, h_{c,k}).
$$

It runs headless using `p.DIRECT`, with multiple simulator clients used in parallel where useful for system identification, hypothetical rollouts, and action filtering.

The internal simulator can be reset to an observed target state and asked to replay the same external-force primitive under different latent-parameter hypotheses.

### 5.2 Target environment

At the start of each test object, the target environment samples hidden ground-truth parameters $\theta^*$.

Mismatch between the internal simulator and target environment is introduced at graded levels:

| Level | Mismatch | Example implementation |
|---|---|---|
| **G0: Well specified** | none | target uses the same model class and solver settings as the internal simulator |
| **G1: Observation and force execution noise** | noisy state estimates and imperfect action execution | noise/latency in pose and velocity; bounded error in $h_t$, $F_t$, and $\tau_t$ |
| **G2: Unmodelled dynamics** | internal simulator cannot exactly represent target dynamics | rolling/spinning friction, different contact stiffness/damping/restitution, bevelled edges, lateral COM offset, non-uniform inertia, spatially varying floor friction, different solver settings |
| **G3: Cross-engine stretch goal** | different contact/friction model | MuJoCo target with PyBullet internal simulator using the same external-force action abstraction |

G0 provides a control condition. G1 is the default mild mismatch. G2 measures robustness to structural model error. G3 is a stretch goal only.

Under G2 and G3, even an oracle that knows the true values of $m$, $\mu$, and $h_c$ still has an imperfect simulator because the model class itself is wrong.

### 5.3 PyBullet implementation notes

- **COM height**: use the inertial-frame offset of the object and adjust inertia consistently.
- **Friction**: set the floor friction convention so that the effective object-floor coefficient is easy to interpret and reproduce.
- **External force**: apply the commanded force every physics step for $\tau_t$ seconds using a body-relative application point and a fixed world-frame push direction.
- **Physics step**: fixed at $1/240$ s (240 Hz), shared by the target and internal simulator; not configurable.
- **Force range**: choose $F_{\max}$ from pilot tests so that the heaviest, highest-friction object can still be moved, while leaving enough headroom for unsafe high-force actions to exist.

---

## 6. Online System Identification

After each target action, the agent has a history

$$
\mathcal{H}_t = \{(s_i,\, a_i,\, \xi_i)\}_{i < t},
$$

where $\xi_i$ is the dense object trajectory recorded during action $i$.

For a candidate normalized parameter vector $z$, each recorded action is replayed in the internal simulator from its recorded start state using the same force height, magnitude, and duration.

A candidate is scored by trajectory discrepancy. One possible loss is

$$
L(z)
=
\sum_{i<t}\sum_{\tau \in \xi_i}
\left[
\frac{(x_\tau-\hat{x}_\tau)^2}{\sigma_x^2}
+
\frac{(\psi_\tau-\hat{\psi}_\tau)^2}{\sigma_\psi^2}
+
\frac{(\dot{x}_\tau-\hat{\dot{x}}_\tau)^2}{\sigma_v^2}
+
\frac{(\dot{\psi}_\tau-\hat{\dot{\psi}}_\tau)^2}{\sigma_\omega^2}
\right].
$$

The point estimate is

$$
\hat{z}_t = \arg\min_z L(z).
$$

The initial implementation uses derivative-free optimization such as CMA-ES because only three latent parameters are estimated.

To represent uncertainty, the system retains the top $K$ distinct low-loss hypotheses

$$
\Theta_K = \{\theta_1,\,\theta_2,\,\ldots,\,\theta_K\},
$$

with optional weights

$$
w_k \propto \exp\left(-L(z_k)/T\right).
$$

The search bounds are wider than the training distribution so that OOD parameter values remain reachable.

Before the first target interaction, the hypothesis set is initialized from the training prior.

---

## 7. Method

**Implementation update (2026-10-28):** the following sections retain the original
SAC/Dyna experiment plan. Draft PR #19 implements a separate moving-reference
residual proposal policy with real-trajectory context, a delta-conditioned
trajectory critic and a rule gate. Its initial training objective is actual
finite-horizon Monte Carlo returns with a partial actor gradient, not ordinary
SAC. See [the living handoff](residual_actor_simulator_handoff.md) for current
decisions and [smoke findings](findings/residual_integration_smoke_2026-10-28.md)
for observed limitations. Dyna replay and model-mismatch experiments remain
deferred in this implementation.

### 7.1 Base SAC policy

A Soft Actor-Critic policy is pretrained over the training distribution of object dynamics.

The actor and critic are

$$
\pi(a \mid o)
$$

and

$$
Q_\phi(o,a).
$$

The policy receives only observable state variables. Estimated mass, friction, and COM are not appended to the policy observation.

This design makes the comparison cleaner: improvements from system identification must act through the internal simulator and the experience it generates, rather than through direct privileged conditioning of the policy.

### 7.2 Imagination as data

After each target interaction and SysID update:

1. Reset the internal simulator to the current target state.
2. Sample one or more hypotheses from $\Theta_K$.
3. Generate $M$ short hypothetical rollouts, typically 1-3 actions long, using the current SAC policy.
4. Store imagined transitions in a simulated replay buffer $\mathcal{D}_{\mathrm{sim}}$.
5. Store real target transitions in $\mathcal{D}_{\mathrm{target}}$.
6. Update SAC using mini-batches drawn from both buffers with a controlled simulated-to-target ratio $\rho$.

The key experimental question is whether these object-specific simulated transitions reduce the number of costly or unsafe target interactions required for adaptation.

### 7.3 Imagination as a shield

Before each target action, the optional pessimistic filter uses the same hypothesis set:

1. Sample $N_c$ candidate actions from the current SAC policy, including the policy mean action.
2. Simulate each candidate for one action under every hypothesis in $\Theta_K$.
3. Record the predicted peak tilt and whether the object topples.
4. Reject a candidate if any plausible hypothesis predicts toppling or a peak tilt above $\psi_{\max}$.
5. Among the remaining actions, execute the one with highest critic value.
6. If no candidate is predicted safe, execute the action with the smallest worst-case predicted tilt.

This separates two possible uses of the simulator:

- **imagination as data** changes the learned actor and critic;
- **imagination as a shield** changes which action is allowed to reach the target environment.

### 7.4 Online loop

```text
Observe target state
        |
        v
Select and execute one external-force action
        |
        v
Record dense object trajectory
        |
        v
Update mass / friction / COM hypotheses with SysID
        |
        v
Generate hypothetical rollouts in the internal simulator
        |
        v
Update SAC from target + simulated replay
        |
        v
Optionally filter the next candidate action across the hypothesis set
        |
        v
Next target action
```

---

## 8. Training and Evaluation Protocol

### 8.1 Phase 1: pretraining

Pretraining is performed entirely in the idealized simulator and does not count as target interaction.

- Sample cuboid geometry and hidden dynamics from the training distribution.
- Train SAC using domain randomization over $m$, $\mu$, and $h_c$.
- Use the same overall pretraining budget for methods that are compared directly.
- No system-identification output is supplied directly to the policy.

#### Potential improvement: low-force exploration and actor initialization

**Status: proposed only; not implemented.** Investigate these changes before
retuning rewards or entropy settings. The seed-42 diagnostic reached 30% success
at 160k steps with a 15 N maximum, while the 50 N run had 0% success and 100%
toppling on the same 20 validation objects. This motivates testing a gentler
initial exploration distribution while retaining access to the full force range;
it does not establish the cause of collapse across training seeds.

- **Low-force random warmup:** during the initial 10,000 transitions, sample
  physical forces from $[0,15]$ N instead of the full $[0,50]$ N. Convert them
  using the unchanged normalized action mapping. Actor initialization alone
  cannot affect this phase because warmup currently bypasses the actor.
- **Low-force actor initialization:** initialize the force mean output around
  5 N. With $F=25(1+\tanh z_F)$, this requires a pre-squash mean of
  $z_F=\operatorname{atanh}(2\cdot5/50-1)\approx-1.10$. Zero the corresponding
  output-row weights and set its bias to this value so the initial mean is
  independent of the observation.
- **Narrower initial force variation:** test an initial pre-squash force standard
  deviation of $0.3$ (log-standard-deviation bias $\log 0.3$), with zero weights
  for that output row. This initially concentrates sampled forces around the
  low-force mean. Automatic entropy tuning may broaden the distribution again.

These values are experimental candidates, not established safe settings; very
small forces may fail to move high-friction objects. Keep the physical bounds
and action mapping fixed throughout training so stored replay actions retain
their meaning. Compare warmup-only, initialization-only, and combined variants
against the unchanged 50 N baseline using matched budgets, object distributions,
and multiple training seeds. Report success, toppling, timeouts, and early force
distributions, then validate the selected model on the reserved holes and OOD
objects. Record any adopted settings in training metadata.

### 8.2 Phase 2: test-time adaptation

For each test object:

- reset the actor and critic to the same pretrained weights;
- keep the object's hidden physical parameters fixed across its adaptation episodes;
- allow up to $N_{\mathrm{ep}}=20$ episodes of at most $T_{\max}=10$ target actions each;
- let the SysID history persist across episodes for that object;
- allow the adaptation method to update its policy from the data sources permitted by that condition.

This makes adaptation runs for different objects independent while allowing learning to accumulate for repeated interaction with the same unfamiliar object.

---

## 9. Experiments

### 9.1 Provisional train and OOD distributions

The exact ranges will be finalized after pilot tests.

The training sampler uses explicit JSON experiment configurations, independent
of `PushEnv`. The shared `ObjectSampler` owns ranges, exclusions, name and sampling;
training selects and records that configuration. `configs/rl/training_distributions/uniform_excluding_evaluation_regions.json` reserves eight joint boxes in
$(m,\mu,h_c/H)$ around the fixed evaluation objects. Each half-width is 5% of
the corresponding outer training span: 0.05 kg, 0.02 friction, and 0.0125 of
object height. Boxes include their boundaries and are clipped to the outer
bounds. Sampling rejects a candidate only if all three hidden parameters lie
inside one box, regardless of geometry. Thus these are interpolation test holes
inside the outer bounds, distinct from the OOD ranges below. Default evaluation uses
the existing eight fixed objects. `--sampler-config configs/rl/evaluation_sets/sampled_reserved_regions.json`
samples each full reserved box, keeping each object fixed during adaptation.
Training and checkpoint validation both follow the selected configuration.
`configs/rl/training_distributions/uniform_without_exclusions.json` disables exclusions with an explicit empty list.
Each final/best checkpoint embeds the resolved distribution, its SHA-256 and
a unique run ID, also recorded in its adjacent `config.json`. Existing pretrained
checkpoints and baseline results used the earlier full-box distribution and do
not establish performance on this new split; a new training run is required.

| Quantity | Training | OOD |
|---|---|---|
| $w,\ell$ | $[0.05,\,0.10]$ m | geometry OOD not required for the core study |
| $H$ | $[0.12,\,0.25]$ m | geometry OOD not required for the core study |
| $m$ | $[0.5,\,1.5]$ kg | $[0.2,\,0.4]\cup[2.0,\,3.0]$ kg |
| $\mu$ | $[0.3,\,0.7]$ | $[0.10,\,0.20]\cup[0.85,\,1.10]$ |
| $h_c/H$ | $[0.35,\,0.60]$ | $[0.20,\,0.30]\cup[0.65,\,0.80]$ |

OOD evaluation is separated into:

1. **Mass OOD**: only $m$ leaves the training range.
2. **Friction OOD**: only $\mu$ leaves the training range.
3. **COM OOD**: only $h_c/H$ leaves the training range.
4. **Joint OOD**: two or three latent properties leave the training range simultaneously.

The implemented SAC evaluation set config `configs/rl/evaluation_sets/fixed_cases.json`
provides four cases per OOD split, alongside the eight existing within-bounds
objects. `evaluate --split mass_ood friction_ood com_ood joint_ood`
runs all sixteen OOD cases;
`--split all` includes all twenty-four cases. Both frozen and fine-tuning
conditions use fixed objects and the shared evaluator. Report per-split results
and classify objects against the checkpoint's actual training ranges and exclusions.
Sampler configurations support arbitrary named groups without fixed category rules. OOD exclusions in a training
config have no effect unless they overlap its training bounds.

### 9.2 E1: Main comparison

Evaluate all main methods on in-distribution objects and on each OOD split under G0, G1, and G2.

Primary outcomes are toppling during adaptation, task success, target-interaction efficiency, and elapsed simulated task time.

### 9.3 E2: Parameter-error sensitivity

Use G0 and deliberately inject known error into one internal-simulator parameter at a time while keeping the other two correct.

For example:

$$
\varepsilon \in \{0,\,\pm 10\%,\,\pm 25\%,\,\pm 50\%\}.
$$

Measure how error in each of $m$, $\mu$, and $h_c$ affects toppling, success, and action selection.

This directly tests which parts of the latent physics must be estimated accurately for imagination to remain useful.

### 9.4 E3: Structural misspecification sweep

Compare point-estimate imagination, ensemble imagination, and oracle simulation across

$$
G0 \rightarrow G1 \rightarrow G2 \rightarrow G3
$$

where G3 is attempted only if time permits.

### 9.5 E4: Identifiability study

Before running the full RL study, perform scripted pushes to establish whether the three hidden parameters can be recovered from the available observations.

Compare excitation regimes such as:

- low-force pushes;
- varied force magnitudes at a fixed height;
- varied force-application heights;
- varied durations;
- mixed sequences that include passive $F=0$ intervals.

For each regime, report the estimation error and spread of the retained hypothesis set for mass, friction, and COM height.

This experiment validates the action space and identifies parameter confounding before expensive RL experiments are run.

---

## 10. Baselines and Ablations

| ID | Method | Target-data adaptation | Uses internal simulator |
|---|---|---|---|
| B0 | Conservative analytic controller using the approximate tipping model and current point estimate | SysID only | only for SysID |
| B1 | Model-free SAC | target replay only | no |
| B2 | Nominal-sim Dyna | target replay + simulations using fixed training-mean parameters | yes |
| B3 | Point-estimate Dyna | target replay + simulations using $\hat{\theta}$ | yes |
| B4 | Ensemble Dyna | target replay + simulations sampled from $\Theta_K$ | yes |
| **B5** | **Ensemble Dyna + pessimistic filter** | same as B4 plus action filtering | yes |
| B6 | Oracle-sim Dyna | target replay + simulations using true $\theta^*$ | yes |

**Ablations**

- point estimate versus $K\in\{4,8\}$ hypotheses;
- Dyna only versus filter only versus Dyna + filter;
- filter threshold $\psi_{\max}$;
- simulated-to-target replay ratio $\rho$;
- rollout horizon for hypothetical trajectories;
- with versus without passive $F=0$ actions, if passive intervals prove important for identifiability.

---

## 11. Reward and Metrics

### 11.1 Reward

The reward encourages the agent to move the object toward the target quickly while avoiding toppling.

Because action duration is itself a decision variable, elapsed time should be penalized directly rather than using only a constant per-step cost:

$$
r_t
=
\alpha\left(d_{t-1}-d_t\right)
-
c_{\mathrm{time}}\tau_t
-
c_{\mathrm{step}}
+
R_{\mathrm{succ}}\,\mathbb{I}_{\mathrm{succ},t}
-
R_{\mathrm{topple}}\,\mathbb{I}_{\mathrm{topple},t}.
$$

Here:

- $d_t$ is the remaining distance to the target after action $t$;
- $c_{\mathrm{time}}>0$ penalizes elapsed force-application time;
- $c_{\mathrm{step}}>0$ is an optional small decision cost that discourages unnecessary chattering;
- $\mathbb{I}_{\mathrm{succ},t}$ is $1$ on successful completion and $0$ otherwise;
- $\mathbb{I}_{\mathrm{topple},t}$ is $1$ when the object topples and $0$ otherwise.

The progress term

$$
d_{t-1}-d_t
$$

is positive when the object moves closer to the target and negative when it moves farther away.

No explicit penalty is placed on object velocity or force magnitude. A fast, forceful action is desirable if it moves the object efficiently without causing toppling or overshoot. The large toppling penalty supplies the safety trade-off, while the elapsed-time term rewards faster completion.

Reward coefficients are tuned only on the training distribution and then frozen for all evaluation conditions.

### 11.2 Metrics

| Metric | Definition |
|---|---|
| **Cumulative topples during adaptation** | total topples across adaptation episodes for each test object |
| Success rate | fraction of evaluation episodes that reach the target without toppling |
| Target interactions to first success | number of target actions before the first successful episode |
| Elapsed simulated time to success | cumulative $\tau_t$ until successful completion |
| Final position error | absolute distance from object to target at episode end |
| SysID error: mass | $|m-\hat{m}|/m$ |
| SysID error: friction | $|\mu-\hat{\mu}|$ |
| SysID error: COM | $|h_c-\hat{h}_c|/H$ |
| Filter statistics | rejection rate and false-safe rate |
| Compute | internal-simulator steps per target action |

For routine SAC checkpoint validation, report mean and standard deviation of
undiscounted episode return, success/topple/timeout rates and counts, mean final
position error, and mean push count and pushing time among successful episodes.
Save per-episode peak absolute tilt as a diagnostic distinct from terminal
toppling classification. Successful-episode means are undefined when there are
no successes and must be stored as `null`, not zero.

For adaptation, also report the fraction of objects that achieve at least one
success within the budget and curves of cumulative success and topples against
target interactions. Interactions and pushing time to first success include
all preceding failed episodes; retain never-successful objects explicitly.

Additional experiment diagnostics should include measured decision latency
(median and a high percentile), held-out displacement/tilt prediction errors,
and missed-topple predictions. Define the filter's observed false-safe rate as
topples among executed actions accepted by the filter; assessing rejected
actions requires separate counterfactual replay experiments. Distinguish
commanded pushing time from total simulated robot execution time.

The shared implementation lives in `src/evaluation/`. Routine SAC validation
and B1 adaptation currently share episode records and aggregation; SysID,
shield, compute, and robot measurements require their respective integrations.

### 11.3 Reporting

Use the same held-out object set for every method to enable paired comparisons. Report mean performance with 95% bootstrap confidence intervals across multiple training seeds and test objects. The exact number of seeds and objects will be chosen according to available compute after a pilot timing study.

Preserve grouping by training seed and test object when constructing confidence
intervals. Use the same seeded cases for repeated checkpoint validation, and
reserve a separate held-out set for final testing. Record denominators and
report outcome categories alongside position error and success-conditioned
efficiency so that unsafe or unsuccessful runs cannot disappear from summaries.

---

## 12. Research Questions and Hypotheses

**RQ1.** Does simulation-augmented adaptation reduce toppling and improve target-interaction efficiency relative to model-free SAC on unfamiliar object dynamics?

**RQ2.** Does maintaining multiple plausible physics hypotheses improve safety over using a single point-estimate simulator?

**RQ3.** Which estimation errors in mass, friction, and vertical COM most strongly degrade the value of simulated experience?

**RQ4.** How does the usefulness of imagination change as the target object moves out of the training distribution and as structural simulator mismatch increases?

**Hypotheses**

- **H1.** Ensemble Dyna + pessimistic filtering will produce fewer cumulative topples than point-estimate Dyna and model-free SAC, while retaining comparable final task success.
- **H2.** COM-height error will strongly affect toppling predictions, while mass and friction errors will strongly affect translational response; under force control, mass is expected to be meaningfully identifiable and task-relevant rather than negligible.
- **H3.** Point-estimate imagination will degrade more sharply under structural mismatch than ensemble-based imagination because an ensemble can partially represent uncertainty in the inferred physics.
- **H4.** The benefit of imagination will be most visible when the target dynamics differ enough from pretraining that the model-free policy requires adaptation, but not so much that the internal simulator becomes systematically misleading.

Imagination is not assumed to help in every regime. Identifying when it hurts is part of the study.

---

## 13. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Mass and friction are confounded by too little excitation | vary force magnitude across pushes; include passive $F=0$ intervals if needed |
| $h_c$ is weakly identifiable from safe pushes | vary force height and magnitude; use dense angular trajectories; quantify the safety-information trade-off in E4 |
| Robot contact execution differs from commanded pushes | validate a simulated robot executor on scripted centreline pushes; record execution error and failures; retain direct-force physics as the internal model |
| PyBullet tipping/contact behaviour is sensitive to solver settings | validate scripted tipping behaviour across timesteps and solver settings before RL training |
| SysID is too slow | parallel `p.DIRECT` clients, warm-started CMA-ES, bounded evaluations, and a short recent-history window |
| SAC adaptation is unstable | keep observation/action spaces low-dimensional, use short rollout horizons, fixed seeds, and conservative learning rates |
| Simulated data hurts because the fitted simulator is wrong | this is explicitly measured using B1-B6, E2, and E3 rather than assumed away |
| Six-week scope becomes too large | G3, active probing, geometric OOD, and point-cloud perception remain stretch goals only |

**Minimum viable result:** implement G0/G1, online SysID, B1/B3/B6, E2/E4, and a validated simulated robot executor with a final-stage policy demonstration. This already answers whether estimated simulated experience helps and whether the three latent parameters are identifiable enough to support the idea.

---

## 14. Milestones

1. **M1 — Environment and physics validation**: direct external-force action primitive, cuboid generator, success/topple logic, scripted tipping tests.
2. **M2 — System identification**: estimate mass, friction, and COM height; run identifiability study E4.
3. **M3 — Base RL**: SAC pretraining with dynamics randomization and model-free adaptation baseline B1.
4. **M4 — Point-estimate imagination**: implement B2, B3, and oracle B6; run main OOD comparisons.
5. **M5 — Uncertainty-aware imagination**: implement B4/B5 and the pessimistic action filter; run ablations. Integrate and validate the simulated robot executor using scripted centreline pushes, including scene reachability, reset, contact timing, and failure reporting.
6. **M6 — Analysis**: parameter-error sensitivity E2, G1/G2 misspecification sweep, confidence intervals, final robot-executed policy demonstration and evaluation, and final report. Report direct-force and robot results separately, including execution failures and measured command-to-contact mismatch.
7. **Stretch**: G3 cross-engine target, active information-seeking actions, geometric OOD, or point-cloud observations.

---

## 15. Scope and Expected Contribution

This project is a controlled sim-to-sim study. It does not claim novelty for simulated experience, system identification, safe action filtering, or toppling-aware pushing individually.

The expected contribution is an empirical characterization of:

1. whether **object-specific physics imagination** reduces unsafe target interactions during adaptation;
2. whether maintaining an **ensemble of plausible mass/friction/COM hypotheses** is safer than relying on one fitted simulator;
3. which **latent-parameter estimation errors** most strongly affect downstream toppling-safe control;
4. how the value of simulated experience changes under **OOD object dynamics** and increasing structural simulator mismatch; and
5. whether mass, friction, and COM height are sufficiently identifiable from the chosen external-force action space to make simulation-assisted adaptation practical.

Required in the final stages: simulated robot-arm pushing through a low-level executor, including inverse kinematics/contact control and a policy demonstration. Pushes remain on the rear face's vertical centreline with action $(h,F,\tau)$; lateral contact position is not a policy variable.

Out of scope for the core six-week project: physical robot experiments, learned joint-level control, arbitrary rear-face contact positions, point-cloud perception, non-cuboid geometry, and full sim-to-real validation.
