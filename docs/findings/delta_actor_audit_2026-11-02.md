# DeltaActorCritic audit (2026-11-02)

Base: `residual-actor-simulator-integration-3` at
`e5e30b91007b13d1b2da5ea60e845198dcdd7a13`.
Changes: `residual-actor-simulator-integration-4`.

The sweep found a wrong-sign actor learning mechanism and two physical-coordinate
bugs. These can affect known-dynamics runs: knowing mass, friction and COM does
not correct either the actor estimator or the shared actuator implementation.
No trained checkpoint or reward history was available for attributing a specific
run's low reward. The checks below establish defects, not reward improvement.

## Actor value gradient could reward weaker actions

`DeltaActorCritic._prepare_actor_example` regenerated forecasts for each actor
sample, then detached them. `_update_actor` differentiated the critic only through
its separate delta input. That computes a partial derivative of
`Q(context, delta, forecast)` while the actual score is
`Q(context, delta, forecast(delta))`. The omitted term can reverse the gradient.
This limitation was already described in the handoff; the old gradient tests
checked that parameters changed, not that the update improved the complete score.

A controlled oracle makes the failure explicit. Let force delta be `d`, forecast
feature be `f(d) = d`, and both critics be `Q(d, f) = 2*f - d`.
The complete value is `d`, increasing with force; the old update uses derivative
`-1`, reducing force. With entropy disabled, zero initial actor means and learning
rate 0.01, one old actor update moved deterministic force delta from 0 to
approximately -0.004. A forecast-only critic could also leave no value gradient.

The corrected value update uses a likelihood-ratio estimator with a fixed raw
Gaussian sample. The full sampled critic value, including its forecast, supplies
the detached learning signal. The critic value at the replay action is an
action-independent baseline conditional on the replay item. Entropy still uses
its reparameterized derivative. This needs no extra simulator calls or network
inputs and leaves TD targets, temperature tuning and collection unchanged.

The estimator is correct in expectation for the current critic objective, but
has sampling variance. It does not fix critic error, partial observability or the
difference between actor sampling and critic-ranked execution. Fresh multi-seed
training is needed to measure its effect on rewards.

## Commanded height used the wrong physical origin

`Cuboid.rear_face_point` received Bullet's COM pose but subtracted `H/2` from the
requested height. The correct offset is `h - h_c`. For an upright object, the old
application height was `h + h_c - H/2`, not `h`.

For example, `H=0.30`, `h_c=0.05` and commanded `h=0.03` applied force at
`z=-0.07` with the base on the floor. That reverses or distorts the intended
lever arm. Both the target and internal simulator shared the bug, so known
parameters could not reveal it through forecast agreement alone.

The corrected point is measured from the COM. Tests check requested world height
for both low and high COM and face tracking under pitch. This affects baseline
SAC as well as DeltaActorCritic. Existing policies learned different physical
action meanings for off-center COM and require retraining for comparisons.

## Forecast reconstruction shifted the observed COM

`InternalSim._spawn_from_state` passed observed COM x directly to `Cuboid.spawn`,
whose position describes the geometric/link origin. With pitch `psi`, this
shifted reconstructed COM x by `(h_c - H/2) * sin(psi)`.
For `H=0.30`, `h_c=0.05`, `psi=0.2`, requested x=0.30 became about 0.28013.

Reconstruction now compensates for the rotated inertial offset. The state still
uses COM position and velocity, and the reconstructed geometry still contacts
the floor. Regression cases cover positive/negative pitch and low/high COM.
Planar reconstruction still cannot reproduce a full Bullet contact state.

## Training-loop checks and remaining design risks

The review found no accidental error in current/next context timing, historical
physics replay, known-parameter initialization, per-push update scheduling,
terminal transition retention, Polyak target updates, or temperature-gradient
sign. Time-limit bootstrapping follows the existing documented SAC convention.

Three unchanged choices deserve controlled training ablations:

- The actor does not observe its absolute reference action. Identical no-motion
  observations and trajectories can conceal different force references and
  future reachable actions. Hidden physical parameters remain simulator-only.
- Warmup samples local deltas from a small probe on every episode. With 50 N
  bounds and default force delta scale 0.4, the first actor force cannot exceed
  10.5 N. This misses the approximate 15.7 N sliding threshold of a 2 kg,
  friction-0.8 object. Uniform local proposals are not uniform absolute actions.
- Feasible-interval scaling contributes to the physical-action log density in
  TD targets, while automatic temperature tuning uses normalized entropy. At
  initial alpha=1, maximum force and duration reduce the continuation entropy
  bonus by `log(4)`, about 1.386, relative to an interior reference with the same
  normalized policy. This favors interior references. It also penalizes the low
  probe boundary, so it is not solely a bias toward weaker actions.

These choices were not silently retuned as implementation fixes.

## Verification

Run the gradient, physical-coordinate and end-to-end training regressions with:

```bash
.venv/bin/python -m pytest tests/test_delta_actor_critic.py \
    tests/test_object.py tests/test_delta_simulator.py tests/test_delta_training.py \
    --no-cov -q
```

The maintained tests include the wrong-sign oracle and forecast-only value
signal, plus checkpoint/entropy behavior and the physical-coordinate cases.

Final validation: `.venv/bin/python -m pytest -q` passed **420 tests** (13 existing
matplotlib/pyparsing deprecation warnings), with 90% configured aggregate coverage.
The run-sim tipping diagnostic also passed both hard checks and reported 100%
agreement on its approximate analytic grid. `git diff --check` passed.
