# Held-out OOD evaluation of curriculum-trained delta models (2026-11-06)

Frozen evaluation of the three curriculum conditions on the fixed benchmark
`configs/rl/evaluation_sets/fixed_cases.json`. Extends the training-time
diagnostics in `delta_curriculum_2026-11-03.md` with held-out test objects that
were excluded from all training distributions.

## Setup

**Models evaluated** (all from `artifacts/delta/curriculum/`):

| condition | checkpoint | world model | f_max | target dist | training status |
|-----------|-----------|-------------|-------|-------------|-----------------|
| physics | `physics/delta_final.pt` | physics (SMC2, 32 particles) | 15 N | 1.0 m | complete (13 stages) |
| learned | `learned/delta_final.pt` | learned (neural latent) | 50 N | 1.0 m | complete (13 stages) |
| baseline | `baseline/delta_latest.pt` | none (model-free) | 50 N | 0.35 m | **incomplete** (stopped at stage 6/14) |

**Benchmark** (`fixed_cases.json`): 24 fixed objects across five splits. Each
object holds `w=L=0.08 m, H=0.20 m` for the OOD splits so that each split
isolates one physics dimension while holding geometry constant.

| split | n | description |
|-------|---|-------------|
| in_distribution | 8 | varied geometry and physics within training ranges |
| mass_ood | 4 | m ∈ {0.2, 0.4, 2.0, 3.0} kg; training range 0.5–1.5 kg |
| friction_ood | 4 | μ ∈ {0.1, 0.2, 0.85, 1.1}; training range 0.3–0.7 |
| com_ood | 4 | h_c ∈ {0.04, 0.06, 0.13, 0.16} m; training range 35–60% of H |
| joint_ood | 4 | all three parameters simultaneously OOD |

**Protocol**: 5 episodes per object, frozen weights (no online updates or
adaptation), `epsilon=0.08 m` success tolerance, `T_max=40` decisions. Baseline
runs at `target_distance=0.35 m` (embedded in the checkpoint); physics and
learned run at `1.0 m`. Reproduction commands at the end of this file.

## Results

### Success rate by split (episodes / %)

| split | physics | learned | baseline |
|-------|---------|---------|----------|
| in_distribution | 35/40 (88%) | 21/40 (52%) | 0/40 (0%) |
| com_ood | 19/20 (95%) | 8/20 (40%) | 0/20 (0%) |
| friction_ood | 9/20 (45%) | 2/20 (10%) | 0/20 (0%) |
| mass_ood | 5/20 (25%) | 0/20 (0%) | 0/20 (0%) |
| joint_ood | 5/20 (25%) | 0/20 (0%) | 0/20 (0%) |
| **overall** | **73/120 (61%)** | **31/120 (26%)** | **0/120 (0%)** |

### Failure breakdown (topple / timeout / episodes)

| split | physics topple | learned topple | learned timeout |
|-------|---------------|---------------|-----------------|
| in_distribution | 5/40 (12%) | 13/40 (32%) | 6/40 (15%) |
| com_ood | 1/20 (5%) | 11/20 (55%) | 1/20 (5%) |
| friction_ood | 6/20 (30%) | 8/20 (40%) | 10/20 (50%) |
| mass_ood | 10/20 (50%) | 14/20 (70%) | 6/20 (30%) |
| joint_ood | 4/20 (20%) | 10/20 (50%) | 10/20 (50%) |

Baseline toppled on 120/120 episodes (100%); average episode lasted ~4–5 pushes
before toppling, with no timeouts. The incomplete curriculum never learned a safe
pushing style (see below).

### Physics: per-object results

| object | m (kg) | μ | h_c (m) | success |
|--------|--------|---|---------|---------|
| in_dist 0 | 0.7 | 0.40 | 0.064 | 5/5 |
| in_dist 1 | 1.3 | 0.60 | 0.116 | 5/5 |
| **in_dist 2** | **0.5** | **0.30** | **0.132** | **0/5** |
| in_dist 3 | 1.5 | 0.65 | 0.052 | 5/5 |
| in_dist 4–7 | — | — | — | 5/5 each |
| mass_ood (m=0.2) | 0.2 | 0.50 | 0.100 | 0/5 |
| mass_ood (m=0.4) | 0.4 | 0.50 | 0.100 | 0/5 |
| mass_ood (m=2.0) | 2.0 | 0.50 | 0.100 | 5/5 |
| mass_ood (m=3.0) | 3.0 | 0.50 | 0.100 | 0/5 |
| friction_ood (μ=0.10) | 1.0 | 0.10 | 0.100 | 0/5 |
| friction_ood (μ=0.20) | 1.0 | 0.20 | 0.100 | 5/5 |
| friction_ood (μ=0.85) | 1.0 | 0.85 | 0.100 | 4/5 |
| friction_ood (μ=1.10) | 1.0 | 1.10 | 0.100 | 0/5 |
| com_ood (h_c=0.04) | 1.0 | 0.50 | 0.040 | 5/5 |
| com_ood (h_c=0.06) | 1.0 | 0.50 | 0.060 | 5/5 |
| com_ood (h_c=0.13) | 1.0 | 0.50 | 0.130 | 5/5 |
| com_ood (h_c=0.16) | 1.0 | 0.50 | 0.160 | 4/5 |
| joint_ood (m=0.2, μ=0.10) | 0.2 | 0.10 | 0.040 | 0/5 |
| joint_ood (m=0.4, μ=1.10) | 0.4 | 1.10 | 0.160 | 0/5 |
| joint_ood (m=2.0, μ=0.20) | 2.0 | 0.20 | 0.130 | 5/5 |
| joint_ood (m=3.0, μ=0.85) | 3.0 | 0.85 | 0.060 | 0/5 |

The single in_distribution failure (`obj2`, m=0.5, μ=0.30, h_c=0.132) is the
training set's lightest, most slippery object with a high COM — the same
triple-hard configuration that fails in mass_ood and friction_ood.

Physics succeeds on both heavy (m=2.0) and high-com (h_c=0.16) OOD objects,
but fails at the extremes: ultra-light (m=0.2, 0.4), very-slippery (μ=0.10),
and very-sticky (μ=1.10). These are also the hardest SysID cases (see next
section), so failure is likely caused by wrong physics estimates driving bad
candidate ranking.

## SysID accuracy (physics model)

Per-split mean relative errors `|estimate − truth| / |truth|` for the last
episode of each object, plus the mass-free tip-stability margin error
`w/2 − μ·h_c`.

| split | m rel err | μ rel err | h_c rel err | tip-margin abs err (m) |
|-------|-----------|-----------|-------------|------------------------|
| in_distribution | 21% | 33% | 45% | 0.025 |
| com_ood | 33% | 30% | 61% | 0.031 |
| friction_ood | 104% | 184% | 42% | 0.058 |
| mass_ood | 178% | 40% | 33% | 0.013 |
| joint_ood | 62% | 106% | 43% | 0.062 |

The SysID failure pattern matches the task failure pattern:

- **friction_ood** has the worst μ error (184%) and the second-worst tip-margin
  error (0.058 m). The very-slippery and very-sticky objects are outside the
  prior support; the SMC2 posterior gets anchored in the wrong region.
- **mass_ood** has the worst m error (178%) but surprisingly good tip-margin
  (0.013 m). Mass enters the topple condition only through the force scaling
  `F/(mg)`, not the margin `w/2 − μ·h_c`. Because μ is identified reasonably
  well (~40% error) and h_c too (~33%), the tip margin survives despite wrong
  mass. The physics model still fails on ultra-light and very-heavy objects, but
  the mechanism is mass-dependent force calibration (can't apply the right force)
  not topple-boundary knowledge.
- **com_ood** has h_c error of 61% yet 95% success. For the four com_ood objects
  (h_c ∈ {0.04, 0.06, 0.13, 0.16} m) the OOD shift is in one direction at a time
  and μ=0.5 is fully in-distribution, so the tip margin stays recoverable.

## Learned world model: tilt prediction error

Logged `wm_tilt_peak_abs_err` (normalised space) for the learned model, measured
as |predicted peak tilt − actual peak tilt| per executed push.

| split | n | mean | median |
|-------|---|------|--------|
| in_distribution | 35 | 0.053 | 0.040 |
| com_ood | 20 | 0.106 | 0.041 |
| friction_ood | 20 | 0.125 | 0.045 |
| joint_ood | 15 | 0.105 | 0.023 |
| mass_ood | 15 | **2.236** | 0.081 |

`mass_ood` has a mean of 2.24 driven by extreme outlier episodes where the model
wildly mispredicts tilt for out-of-distribution inertia. The median (0.081) is
still higher than in_distribution (0.040), so the distribution is shifted, not
just a few outliers pulling the mean. This confirms the prior hypothesis
(`delta_curriculum_2026-11-03.md` §"Ablation"): the learned world model's
low aggregate MSE coexists with poor tilt-channel accuracy, and the tilt
degradation is worst on OOD physics that changes inertia.

## Interpretation

The held-out results sharpen the training-time picture from the prior findings:

1. **Physics generalises to com_ood (95%) despite never seeing those h_c values
   during training.** The physics simulator is parametric — give it a different
   h_c estimate and it simulates correctly — so the only bottleneck is SysID
   accuracy for that dimension, which is moderate (~61% error). The learned model
   has no equivalent mechanism and drops to 40% on the same split.

2. **Friction OOD is harder than COM OOD for physics, easier for learned.** The
   physics model's SysID posterior was trained on friction 0.3–0.7; at μ=0.10 and
   μ=1.10 it anchors wrong, tips the margin estimate, and the physics advantage
   disappears (45% vs 10% for learned, both bad). The learned model fails more
   uniformly across OOD dimensions rather than having the sharp cliff physics hits.

3. **Mass OOD is the worst split for both.** Physics fails because force
   calibration depends on m; learned fails because inertia shifts dominate the
   trajectory distribution and the world model was never trained there.

4. **The baseline confirms the imagination hypothesis by negation.** Zero success
   and universal toppling at d=0.35 m — a distance the physics model handles at
   71% — shows that world-model-free delta learning does not converge to a safe
   policy even at short distances. The topology of the learned curve and the
   model-free curve in the prior findings is consistent with this: model-free
   collapsed by 0.25 m and never recovered.

5. **Training success rates (from `delta_curriculum_2026-11-03.md`) overstate
   held-out performance.** Physics was ~81% at 1.0 m during training on sampled
   objects; it is 61% overall on fixed held-out objects (88% InD but 25% on hard
   OOD). Learned was ~47% at 1.0 m during training; it is 26% on held-out objects.
   The gap is largest on the mass and joint OOD splits, which sit outside the
   training sampler's support.

## Caveats

- **Baseline comparison is not distance-controlled.** The baseline checkpoint was
  only trained to d=0.35 m (6 of 14 curriculum stages completed), so it evaluates
  at 0.35 m while physics and learned evaluate at 1.0 m. The 0/120 result
  indicates the baseline did not learn a safe policy even at its trained distance;
  it is not a statement about what a fully-trained baseline would do at 1.0 m.
- **5 episodes per object gives wide confidence intervals.** With n=5, a 95%
  success object can look 80% and an 80% success object can look 60–100%. The
  per-split and per-object numbers are indicative, not precise.
- **Single seed.** The curriculum runs were not repeated; the evaluation protocol
  (fixed object set, fixed seed) is reproducible, but the reported numbers reflect
  a single training trajectory.
- **f_max differs between conditions.** Physics was trained with f_max=15 N;
  learned and baseline used f_max=50 N. This is a confound: the learned model has
  a wider action space that may help or hurt on some OOD objects. A matched-f_max
  comparison is not available.

## Reproduction commands

```bash
# Physics
.venv/bin/python -m src.evaluation.rl.evaluate_delta \
  --checkpoint artifacts/delta/curriculum/physics/delta_final.pt \
  --benchmark-config configs/rl/evaluation_sets/fixed_cases.json \
  --world-model physics \
  --particles 32 --candidates 4 --max-batches 2 \
  --episodes-per-object 5 --forecast-workers 4 \
  --out-dir artifacts/evaluation/curriculum/physics

# Learned
.venv/bin/python -m src.evaluation.rl.evaluate_delta \
  --checkpoint artifacts/delta/curriculum/learned/delta_final.pt \
  --benchmark-config configs/rl/evaluation_sets/fixed_cases.json \
  --world-model learned \
  --particles 64 --candidates 4 --max-batches 2 \
  --episodes-per-object 5 \
  --out-dir artifacts/evaluation/curriculum/learned

# Baseline (evaluates at d=0.35 m — target distance embedded in checkpoint)
.venv/bin/python -m src.evaluation.rl.evaluate_delta \
  --checkpoint artifacts/delta/curriculum/baseline/delta_latest.pt \
  --benchmark-config configs/rl/evaluation_sets/fixed_cases.json \
  --world-model none \
  --particles 64 --candidates 4 --max-batches 2 \
  --episodes-per-object 5 \
  --out-dir artifacts/evaluation/curriculum/baseline
```

Results land in `artifacts/evaluation/curriculum/{physics,learned,baseline}/`.
