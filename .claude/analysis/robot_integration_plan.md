# Robot Integration Plan: Robot-Executed Pushing

*Confirmed scope: simulated robot execution is required in the final project stages (M5 integration/validation, M6 policy demonstration/evaluation). The canonical outline and proposal now include this requirement. Retain `(h, F, τ)` and contact on the rear face's vertical centreline; arbitrary lateral contact positions remain future work. The earlier S1–S6 schedule below is an implementation reference, not an additional requirement to complete robot integration in M1–M3.*

Required integration checks: share the environment physics client; reset/home the arm per episode; validate reachability; report contact failures/timeouts; record commanded and realized execution data; monitor object motion/failure during approach, push, and retraction; and report total simulated execution time. The internal simulator remains direct-force. Physics observations can retain the current format initially, but robot lifecycle, execution results, and phase timing require environment integration changes.

## 0. The one decision that shapes everything

**Keep the force-primitive action space `a = (h, F, τ)` and make the robot an execution layer beneath it.** The policy still commands a push primitive; a low-level controller converts it into arm motion. Two reasons:

1. It preserves everything — the SysID replay scheme, baselines B0–B6, experiments E1–E4, the 6-week timeline — because the internal simulator ("imagination") stays a direct-force PyBullet model. The robot's contact dynamics become *target-env reality* the imagination must cope with, which is exactly the G1/G2 study.
2. The alternative — making the RL action space robotic (EE pose/velocity commands) — is a whole second research project: sample-inefficient, contact-sparse, and it would invalidate the clean baseline comparison. Not recommended for a 6-week scope.

## 1. Embodiment

- **Recommended: fixed-base KUKA IIWA** (7-DOF, ships with `pybullet_data`, already scripted in `src/self_learning/lesson5.py`).
- Alternatives: Franka Panda (same idea, different URDF) or a simple planar pusher (easier to tune, less convincing as "a robot").
- **Hard constraint: reach** (~0.8 m for the IIWA). Place the arm base beside the pushing lane and the target within reach; cap episode push distance at ~0.5–0.7 m, or re-pose the arm between episodes (never mid-episode).

## 2. The execution layer (PyBullet specifics)

PyBullet has no native force control, and rigid-contact force control is stiff/unstable. Standard practice for simulated pushing:

| Approach | How | Verdict |
|---|---|---|
| **Impedance-style position tracking** | IK to pre-push pose at commanded height `h`; approach slowly until `getContactPoints(arm, box)` fires; then track a *virtual setpoint* moving along the push axis whose speed is derived from `F` (e.g., spring-damper law or friction balance); `setJointMotorControl2` in `POSITION_CONTROL` at 240 Hz | **Recommended** — stable, tunable, standard in pushing papers |
| Velocity control with compliance | `VELOCITY_CONTROL` toward a moving target point, soft contact settings | Fallback if position control overshoots |
| Joint-torque force control | `calculateJacobian` + inverse dynamics + feedback from wrist joint reaction forces | Most "correct", most tuning-heavy; only if accurate `F` is required |

Key mechanics:

- **Start timing at contact detection, not command issue** — `τ` can be as short as 0.05 s = 12 physics steps at 1/240 s.
- **Measure the realized push**: `p.getJointState(arm, wrist)[2]` gives joint reaction forces (wrist force approximation); contact normals in PyBullet are unreliable. Log realized `(ĥ, F̂, τ̂)` alongside commanded values.
- **Contact tuning**: `p.changeDynamics` (contactStiffness/Damping), spherical high-friction EE tip, slow approach speed, 1/240 s step.
- End at `τ`, retract arm, next decision. The object may still be moving — already allowed by the formulation.

## 3. Staged integration (maps onto existing milestones)

- **S1 — Scene + reach (with M1)**: arm beside lane, IK to rear-face heights, scripted slow pushes, no agent.
- **S2 — Push executor (with M2)**: wrap the controller behind the exact same `(h, F, τ)` API as the direct-force env. Build an E4-style validation harness: same commands executed both ways, compare object displacement/tilt, quantify executor error.
- **S3 — Target env swap (with M3)**: replace direct force application with the robot executor. Retain the policy observation/action format, while integrating arm reset, execution status, full-phase trajectory recording, and timing/failure accounting into the environment. SysID + internal simulator stay direct-force — deliberately.
- **S4 — Remap reality-gap levels**: the robot gives G1 *for free* (real, measured execution noise instead of injected noise) and adds contact/arm-compliance dynamics to G2. Add a "G-robot" row to the mismatch table and decide the new default level.
- **S5 — Rerun the study (M4–M5)**: B0–B6 + E1–E4 with the robot target.
- **S6 — Docs (M6)**: new outline sections, updated proposal/PPT, new risks, reallocated timeline.

## 4. New risks & mitigations

| Risk | Mitigation |
|---|---|
| Contact jitter/penetration, arm knocks box over | Soft contact settings, spherical tip, slow approach, gain tuning — budget ~1 week contingency |
| Reach limits cut push distance | Shorten lane; re-pose arm between episodes; rail-mounted base as stretch goal only |
| Short `τ` window (12 steps) | Executor timer starts at contact detection; validate at 1/240 s |
| Executed force ≠ commanded | That's the G1 noise — record realized values, optionally replay *realized* actions in SysID |
| Scope creep | Robot is infrastructure. The RL problem, hypotheses, and baselines stay exactly as designed |

## 5. What this does to the science

Adding the robot *strengthens* the core question:

- G1 stops being synthetic noise — it's the actual robot's execution error.
- G2 gains real unmodelled dynamics: contact + arm compliance. Now even the **oracle baseline (B6)** has a wrong simulator model — the imagination must absorb structural mismatch, not just parameter error. A better test of "when does imagination help or hurt" (RQ4/H3/H4).
- Optional extra ablation if time permits: compare the two executor designs (position-tracked vs. torque-controlled pushes) as an execution-noise study.

Additional references:

- Kloss et al., "Accurate Vision-Based Manipulation Through Contact Reasoning", ICRA 2020 — PyBullet pushing with force feedback.
- Bauza & Rodriguez, "A Probabilistic Data-Driven Model for Planar Pushing", ICRA 2017.
- Yu et al., "More Than a Million Ways to Be Pushed", IROS 2016.
- Lynch & Mason (already cited in the outline) anchors the theory.

## 6. Suggested timeline reallocation

| Week | Original milestone | + Robot work |
|---|---|---|
| 1 | M1 env + physics validation | S1 scene/reach in parallel |
| 2 | M2 SysID + E4 | S2 push executor + validation harness |
| 3 | M3 SAC pretrain + B1 | S3 robot target env |
| 4 | M4 point-estimate imagination | Full validation pass with robot target |
| 5 | M5 ensemble + filter | E1/E3 with robot target |
| 6 | E2 + analysis | S6 docs, updated outline/proposal |

## 7. Future iterations (beyond the current project)

Two extensions noted for future work. Both build on the S1–S6 robot executor — the IIWA can already push at lateral offsets within reach, so no new embodiment is required.

### 7.1 Off-centre force application

- **Today**: force is applied at the lateral centre of the rear face, deliberately to avoid inducing yaw (outline §3.3).
- **Future**: add a lateral application-point coordinate to the action, `a = (h, F, τ, y)` with `y ∈ [-L/2, L/2]`, so pushes deliberately induce yaw.
- **Consequences**: state must include yaw and yaw rate about the vertical axis; toppling becomes a 2D support-polygon condition instead of a 1D interval; the task becomes planar pushing with reorientation — Lynch & Mason's stable-pushing mechanics, Bauza & Rodriguez's data-driven planar pushing, and the omni-pusher line of work (Bauza, Hogan, Rodriguez 2018).

### 7.2 Non-centred COM

- **Today**: vertical COM height `h_c` is estimated; lateral COM offset appears only as an unmodelled G2 mismatch source (outline §5.2).
- **Future**: promote lateral (and optionally longitudinal) COM offset to estimated parameters: `θ = (m, μ, h_c, c_x, c_y, I_zz)`.
- **Consequences**: SysID grows to ~6 parameters (CMA-ES still feasible, but identifiability needs its own E4-style study); off-centre pushes become doubly informative since they excite both COM offset and yaw inertia — a richer safety-information trade-off.

### Why the two belong together

Pushes through the COM translate; off-COM pushes rotate, because the friction force at the support contact exerts a moment. With both features the agent must reason about planar dynamics, and the project's core machinery — ensemble hypotheses and the pessimistic filter — becomes a 2D problem where representing uncertainty is plausibly even more valuable.

---

## Bottom line

The robot enters as an executor below the unchanged `(h, F, τ)` interface, the imagination stays a direct-force simulator, and the robot's own imperfections become measured G1/G2 mismatch. The entire study design is preserved, and the project gains a much more compelling robotics story.
