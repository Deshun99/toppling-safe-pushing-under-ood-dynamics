# SMC² development findings and handoff

Date: 24 October 2026. These are development findings for the final report and presentation, not final experimental conclusions.

For current helper APIs, renamed output fields, and cache checks, see the
[SysID workflow guide](../../src/evaluation/system_identification/README.md).
The measurements and saved outputs below are historical. Statements about
source hashes and dependency records describe those saved runs; current runners
save experiment settings without collecting that information.

## Milestone closure and primary results

Closure date: 24 October 2026. Runtime code, studies and tests are committed at `741767376d7bc72951885f10b1af320361015bd0` (`7417673`). The evidence bundle and closure index are additional documentation; their Git inclusion must be checked separately. The current estimator defaults to covariance-shaped proposals; gradual updates and exact starts remain explicit experiment settings, not library defaults.

Use these as the primary results for the final report and presentation:

1. **Sampler improvement:** [covariance proposal comparison](../../artifacts/sysid_covariance_proposals/report.md). Under the original cumulative replay ceilings, upright position RMSE improves from 0.176 to 0.082 mm (varied durations) and 0.304 to 0.126 mm (passive mixed). Distinct particles increase from 15–42 to 212–229. Mean varied-duration COM distribution distance to the grid improves from 11.05 to 2.50 mm. Actual fitting cost decreases by about 14.4%. These numbers describe the capped experimental arm.
2. **Current estimator comparison:** [production covariance SMC² versus retained CMA-ES](../../artifacts/sysid_covariance_vs_cma_retained/report.md). Normal-schedule SMC² position RMSE is 0.111/0.141 mm versus CMA-ES's 0.070/0.018 mm. Costs are 45,088/45,085 fitting replays. SMC² has lower varied-duration COM error (5.02 versus 8.74 mm); CMA-ES has lower passive-mixed COM error (16.17 versus 33.49 mm). Do not substitute capped-arm errors into this comparison.
3. **Mechanism evidence:** [truth-particle trace](../../artifacts/sysid_truth_particle_trace/report.md) and [separate grid calculation](../../artifacts/sysid_reference_posterior/report.md). Truth scores well when supplied, but exact-copy survival is not equivalent to a correct posterior distribution. Tracing distinguishes accepted moves from resampling losses.

The original small-budget CMA-ES comparison, population-size sweeps, warmup ablations and first tempering experiment are supporting development history. They should not be presented as the current equal-budget comparison. This milestone does not demonstrate population calibration, fresh-object generalization, or improved shield safety.

**Preserved evidence:** see the [bundle guide](evidence/smc2_2026-10-24/README.md), [archive](evidence/smc2_2026-10-24/smc2_evidence.zip), and [manifest](evidence/smc2_2026-10-24/manifest.json). The archive restores selected files to their original `artifacts/...` paths. It includes the final independent grids, all selected-run particle snapshots and numerical records, summary figures, and a closure source snapshot. Intermediate grid checkpoints and repeated per-update plot images are intentionally omitted. No original outputs were moved or deleted.

## Main finding

Maintaining multiple physics hypotheses does not automatically produce reliable uncertainty or better predictions. Our original SMC² implementation could discard plausible hypotheses early, copy the survivors, and report unjustifiably narrow COM intervals. Replay warmup mismatch and noisy inner likelihood estimates further complicated inference.

We corrected the likelihood scoring window, introduced a deterministic known-start experimental mode, and added optional gradual evidence updates. Results improved, but particle exploration and uncertainty reliability remain unresolved. Increasing particle count alone did not consistently resolve these problems. A subsequent covariance-shaped proposal substantially improved particle diversity and agreement with the separate grid calculation under the saved replay ceilings; fresh-object validation remains pending.

The current SMC² setup outperformed a small-budget CMA-ES baseline on mean upright position prediction error in the latest experiments, at substantially greater simulation cost. The first matched-replay comparison produced mixed results under a fresh-refit policy. A follow-up that retained and rescored the previous CMA-ES best estimate achieved lower mean point-prediction errors than SMC² on both sequences, at the same cumulative replay limits. This does not compare posterior uncertainty or establish general algorithm superiority.

## Scope and terminology

- Hidden physics: mass `m`, object–floor friction coefficient `μ`, and vertical centre-of-mass height `h_c`.
- Outer particles are physics hypotheses represented internally as `(log m, μ, h_c/H)`, where `H` is object height.
- Inner particles represent uncertainty in each recorded push's starting state and provide likelihood estimates for an outer hypothesis.
- A particle's weight represents posterior probability mass in the particle approximation; it is not its likelihood alone.
- “COM interval includes truth” means the marginal 95% interval contains the known physical COM. Counts across these runs are not estimates of population calibration.
- The “separate grid calculation,” called a **reference** in older artifacts, numerically integrates the same prior and likelihood. It is a check on the particle approximation, not physical ground truth.

Most comparisons below use one fixed cuboid: width and length 0.08 m, height 0.20 m, mass 1 kg, friction 0.3, and COM height 0.10 m. Seeds 1, 2 and 3 vary inference randomness; target measurements are deterministic simulator states without injected sensor noise.

## Problems, changes, and evidence

### 1. Warmup observations could favour incorrect physics

Reconstructing the object in the internal simulator introduces contact-settling transients that differ from the recorded target motion. The likelihood treated these differences as physics evidence. In the diagnostic case, warmup angular-velocity residuals dominated the apparent COM evidence.

**Evidence:** with mass and friction fixed at truth, an initial COM profile preferred 55 mm over the true 100 mm by 70.28 log-likelihood units. Excluding retained warmup samples moved the preferred value to 95 mm. This was a conditional profile, not a marginalized posterior or a general recovery guarantee.

**Implemented correction:** retain and simulate the complete warmup, but always exclude its observations from fitting. A shared scoring-index helper preserves the original downsampling phase: with eight warmup steps and stride 12, scored indices are 12, 24, and so on. Empty scoring windows raise an error. Both SMC² and CMA-ES use the same scoring window in the benchmark. The former `include_warmup` API option and `--include-warmup` CLI flag have been removed. Historical include-warmup numbers below come from saved experiments; the diagnostic COM profile still compares both windows explicitly within study code. Estimator probes and ablations now always exclude warmup, so rerunning them does not reproduce the original include-warmup diagnosis.

With known starts and 64 outer particles, excluding warmup changed mean upright position RMSE as follows:

| Training sequence | Include warmup | Exclude warmup |
|---|---:|---:|
| Varied durations | 1.239 mm | 0.469 mm |
| Passive mixed | 3.697 mm | 1.520 mm |

This corrects a demonstrated source of misleading evidence. It does not establish perfect replay fidelity or validate the observation model on other objects.

Evidence: [initial diagnosis](../../artifacts/smc2_diagnosis/report.md), [observation-model comparison](../../artifacts/sysid_observation_model_development/report.md).

### 2. Assumed starting-state uncertainty made likelihood estimates noisy

The original estimator sampled uncertain starting states even though the target exposes exact simulator state. Its default 1 cm starting-position standard deviation was a modelling assumption, not measured sensor noise.

**Evidence:** at fixed true physics on one saved push, repeated log-likelihood estimates had standard deviation 5.04 with eight inner particles and 1.15 with 64. Exact starts eliminated this Monte Carlo variation. More inner particles helped mass/friction recovery in the diagnostic ablation, but did not resolve COM uncertainty.

**Implemented experimental mode:** `--known-start` uses `start_std=(0, 0, 0, 0)` and `n_state=1`. This gives a deterministic likelihood for isolating outer-particle sampling failures. The library's starting-state uncertainty defaults remain unchanged.

This is a controlled baseline for the current noise-free simulator, not a solution for realistic noisy state estimates. Switching to exact starts did not uniformly improve prediction accuracy because other failures remained.

Evidence: [initial diagnosis](../../artifacts/smc2_diagnosis/report.md), [observation-model comparison](../../artifacts/sysid_observation_model_development/report.md).

### 3. Weight collapse destroyed diversity, and ESS could hide it

On the original diagnostic dataset, first-update effective sample size (ESS) was only 1.00–1.07 out of 64, with the largest weight holding 96.8–almost 100% of the mass. Resampling copied the few survivors. Two fixed-scale proposal moves per particle did not reliably recover discarded regions.

An exact-start run produced 64 identical particles and subsequently reported ESS = 64. ESS measures weight concentration; equal weights on identical particles do not imply 64 distinct hypotheses. Triggering moves only from low ESS can therefore leave a collapsed population without further exploration.

**Implemented diagnostics:** report distinct particle counts alongside ESS, proposal acceptances, replay counts, particle snapshots, and posterior plots with ground truth. These expose collapse but do not themselves repair it.

**Independent check:** after corrected warmup scoring and exact starts, a separate grid calculation still supported a broad COM distribution, while some SMC² runs were narrow and excluded truth. After three varied-duration pushes, the grid's COM interval was approximately **[43.4, 121.0] mm**; one 64-particle run reported approximately **[82.0, 84.8] mm**. The true COM was 100 mm.

Independent grid resolutions and refinement checks supported numerical stability. This demonstrates loss of plausible probability mass in the particle approximation on this dataset; it does not prove that the likelihood model is correct generally.

Evidence: [initial diagnosis](../../artifacts/smc2_diagnosis/report.md), [grid comparison](../../artifacts/sysid_reference_posterior/report.md).

### 4. Gradual evidence updates partly reduced collapse

**Implemented option:** `temper_updates=True`, exposed in the benchmark as `--temper-updates`. Instead of applying a new push's likelihood at full strength immediately, introduce it through intermediate powers `β`, from zero to one:

```text
intermediate target ∝ prior × likelihood(previous pushes) × likelihood(new push)^β
```

Conditional ESS determines the intermediate increments. Resampling and proposal moves occur between stages. Previous evidence always retains full strength; the final intended posterior is unchanged. Accepted inner-filter bookkeeping retains full cumulative likelihoods.

The proposal distribution itself remains the same fixed-scale random walk, with two moves per resampling stage. More stages therefore mean more total moves and computation.

**Restriction:** this implementation requires zero starting-state uncertainty and one inner particle. It does not apply fractional powers to noisy inner likelihood estimates. Gradual updates are optional and disabled by default.

At 256 outer particles, with identical datasets and seeds:

| Sequence | Update | Position RMSE | COM truth inclusion | Distinct final particles, seeds 1/2/3 | Fitting replays |
|---|---|---:|---:|---|---:|
| Varied durations | Full update | 0.229 mm | 2/3 | 22 / 30 / 14 | 9,516 |
| Varied durations | Gradual | 0.176 mm | 3/3 | 30 / 30 / 42 | 19,929 |
| Passive mixed | Full update | 0.250 mm | 0/3 | 7 / 13 / 4 | 19,688 |
| Passive mixed | Gradual | 0.304 mm | 2/3 | 15 / 16 / 15 | 30,310 |

COM inclusion improved from 2/6 to 5/6 runs, but prediction improvement was mixed. Total fitting cost increased 1.72×. These results do not distinguish the benefit of intermediate targets from the benefit of spending more compute on moves; an equal-budget control is still needed.

Evidence: [gradual-update comparison](../../artifacts/sysid_development_tempered_256/comparison_report.md). Method source: [Zhou, Johansen and Aston, adaptive SMC](https://arxiv.org/abs/1303.3123).

### 5. More outer particles helped inconsistently

Before gradual updates, increasing from 64 to 256 particles improved mean upright position RMSE from 0.469 to 0.229 mm on varied durations and from 1.520 to 0.250 mm on passive mixed. COM truth inclusion nevertheless remained 0/3 on passive mixed.

With gradual updates enabled, the later population-size comparison gave:

| Particles | Varied-duration position RMSE | Passive-mixed position RMSE | COM truth inclusion, both sequences | Total fitting replays |
|---:|---:|---:|---:|---:|
| 256 | 0.176 mm | 0.304 mm | 5/6 | 50,239 |
| 512 | 0.326 mm | 0.110 mm | 4/6 | 81,965 |
| 1,024 | 0.165 mm | 0.143 mm | 5/6 | 192,041 |

At 1,024 particles, only 118–123 distinct hypotheses remained after varied durations and 34–83 after passive mixed. Absolute diversity increased, but substantial duplication remained. On varied durations, 512 particles agreed more closely with the grid's COM distribution than either 256 or 1,024 did on average.

Larger populations were not nested extensions of smaller ones: changing particle count changes initialization and subsequent random-number consumption. These three seeds do not establish a monotonic relationship between population size and accuracy. The evidence does not justify changing the default to 1,024 particles.

Evidence: [64 versus 256](../../artifacts/sysid_development_known_256/comparison_with_64.md), [256/512/1,024 comparison](../../artifacts/sysid_particle_count_comparison/report.md).

## What the training sequences mean

Actions are `(h, F, τ)`: force application height in metres, force in newtons, and duration in seconds. Heights below are fractions of object height `H`.

**Varied durations:** three pushes at `h=0.30H`, `F=5 N`, with durations 0.10, 0.25 and 0.40 s.

**Passive mixed:** six actions combining forced and free motion:

| Action | Height | Force | Duration |
|---|---:|---:|---:|
| Push | 0.30H | 5 N | 0.25 s |
| Free motion | — | 0 N | 0.30 s |
| Push | 0.45H | 5 N | 0.25 s |
| Free motion | — | 0 N | 0.30 s |
| Higher push | 0.60H | 4.5 N | 0.40 s |
| Higher push | 0.70H | 2.3 N | 0.40 s |

The benchmark retains upright records for fitting, preserves rejected records in its audit, and settles or resets before high pushes. Development evaluation uses separate pushes, each starting in a fresh episode. These are prescribed data-collection sequences, not an adaptive exploration policy.

## Comparison with CMA-ES

CMA-ES also searches a population and can retain top-K candidates; it is inaccurate to describe it as considering only one hypothesis during optimization. Its retained candidates and heuristic weights do not constitute a Bayesian posterior.

In the latest shared-data comparisons, CMA-ES achieved upright position RMSE of **0.334 mm** on varied durations and **0.777 mm** on passive mixed. All three gradual-update SMC² population sizes had lower mean errors, although 512 was nearly tied on varied durations.

However, CMA-ES used only **10 iterations without restarts per update**, totalling **5,670 fitting replays**, versus **50,239–192,041** for SMC². CMA-ES was restarted from scratch at each history prefix; SMC² updated sequentially. This original comparison used a small-budget baseline. The later matched-replay experiment below addresses that limitation, with an important distinction between sequential updating and refitting.

Prediction numbers in the tables use each estimator's **selected point hypothesis**, averaged across three seeds. They are not posterior predictive averages. The selected SMC² point is based on particle mass, including exact duplicates; ties can make that summary unstable. Ensemble uncertainty, predictive mixtures, and safety decisions need separate evaluation.

## Follow-up: matched cumulative fitting replay budgets

A subsequent experiment reused the exact 256-particle gradual-update SMC² datasets and seeds. CMA-ES received the same cumulative replay allowance at every history prefix, counting all earlier fits. Each update refit the complete history using three independent starts from the existing initialization. The replay cap replaced the iteration limit. Remaining calls insufficient to evaluate another full-history candidate were left unused.

| Sequence | Method | Position RMSE | Mass error | COM error | Total fitting replays |
|---|---|---:|---:|---:|---:|
| Varied durations | SMC² | 0.176 mm | 0.0370 kg | 21.06 mm | 19,929 |
| Varied durations | Matched CMA-ES | 0.059 mm | 0.0073 kg | 16.34 mm | 19,926 |
| Passive mixed | SMC² | 0.304 mm | 0.0089 kg | 29.14 mm | 30,310 |
| Passive mixed | Matched CMA-ES | 1.648 mm | 0.0550 kg | 41.02 mm | 30,310 |

**Interpretation:** increased compute materially improves CMA-ES on varied durations, where it beats the selected SMC² point prediction on all three seeds. Passive-mixed final results worsen under this particular allocation and restart policy; this is not evidence that additional compute inherently harms CMA-ES.

The strict per-prefix allowance exposes a refit limitation: the last passive-motion update adds only 256 SMC² replay calls per seed. Including unused calls carried forward leaves CMA-ES 258 calls, enough for 43 candidates across three fresh starts, each candidate requiring six history replays. The implementation discards its previous optimizer state and does not carry forward its previous point as a candidate. It can therefore replace a good earlier fit with a poorer short-budget refit. A comparison that retains the previous best estimate was subsequently tested below; retaining optimizer state remains untested. An equal final budget without per-prefix limits would answer a different, offline question.

These limits match replay calls, not physics timesteps or total CPU work. The result compares the current implementations, their different initializations, and one prescribed restart policy. It does not establish universal superiority or posterior calibration. Seeds 1, 2 and 3 use restart seeds `seed`, `seed+1`, and `seed+2`, so restart sets overlap; they are not three fully independent collections of starts.

Reproduce with:

```bash
.venv/bin/python -m src.evaluation.system_identification.comparisons.compare_compute \
  --baseline artifacts/sysid_development_tempered_256 \
  --out artifacts/sysid_compute_matched_rerun
```

Evidence: [matched-compute report](../../artifacts/sysid_compute_matched/report.md), with raw per-update costs and results in `metrics.jsonl` and input/source hashes in `config.json`. API support is `SysID.update(history, replay_budget=...)`; the cap is for that call only. Without it, the original iteration-based path is preserved. The follow-up passed 115 tests plus Ruff checks.

## Follow-up: retain and rescore the previous CMA-ES best estimate

Added an explicit optional `previous_best_parameters=(m, mu, h_c)` argument to `SysID.update`. The caller supplies the previous selected estimate, which is evaluated on the complete current history and competes with the fresh search. Its evaluation is included in the same replay cap. The old loss is never reused, and optimizer state is not retained. The default behavior without an previous best estimate is unchanged. If only one complete-history evaluation fits the budget, the previous best estimate can be rescored without running a new search.

The runner's `--retain-best` mode applies this policy sequentially using the same saved data, inference seeds, three starts, and cumulative replay limits as before:

| Sequence | Method | Position RMSE | Mass error | Friction error | COM error | Total fitting replays |
|---|---|---:|---:|---:|---:|---:|
| Varied durations | SMC² | 0.176 mm | 0.0370 kg | 0.0162 | 21.06 mm | 19,929 |
| Varied durations | CMA-ES, fresh refits | 0.059 mm | 0.0073 kg | 0.0029 | 16.34 mm | 19,926 |
| Varied durations | CMA-ES, retain best | 0.068 mm | 0.0048 kg | 0.0019 | 8.69 mm | 19,926 |
| Passive mixed | SMC² | 0.304 mm | 0.0089 kg | 0.0037 | 29.14 mm | 30,310 |
| Passive mixed | CMA-ES, fresh refits | 1.648 mm | 0.0550 kg | 0.0121 | 41.02 mm | 30,310 |
| Passive mixed | CMA-ES, retain best | 0.038 mm | 0.0012 kg | 0.0005 | 12.91 mm | 30,310 |

**Findings:** retention removes the large passive-motion prediction regressions caused by discarding earlier estimates. It improves passive-mixed mean position RMSE from 1.648 to 0.038 mm. On varied durations it slightly worsens mean position RMSE relative to fresh refits (0.059 to 0.068 mm), while improving mean parameter errors. Retained CMA-ES beats the selected SMC² point's position prediction on all six final seed/sequence pairs. It does not recover every parameter better in every run: on passive mixed seed 2, COM error is approximately 31.33 mm for retained CMA-ES versus 2.83 mm for SMC².

This supports an inexpensive safeguard against short-budget refits losing earlier good candidates. It does not guarantee monotonic held-out prediction improvement: the candidate is selected by current full-history training loss, not by evaluation errors. New evidence can favour a different estimate, and reserving its re-evaluation cost slightly changes the fresh-search candidate allocation.

All 27 per-prefix cumulative limits were verified. The previous point was retained on 13 of the 21 updates with an previous best estimate. A separate post-fit audit re-evaluated previous best estimate training losses and confirmed the selected candidate never had a higher current loss; these verification replays were not used for optimization. The full suite passed 119 tests, including tight budgets with no fresh candidates, previous best estimate re-evaluation costs, and replacement by a better candidate on changed history. Ruff checks passed.

Reproduce with a new output directory:

```bash
.venv/bin/python -m src.evaluation.system_identification.comparisons.compare_compute \
  --retain-best --refit-comparison artifacts/sysid_compute_matched \
  --out artifacts/sysid_compute_matched_retained_rerun
```

Evidence: [retained-estimate comparison](../../artifacts/sysid_compute_matched_retained/report.md), `metrics.jsonl`, `summary.json`, and `audit.json`. Input hashes confirm the same saved baseline. Scope remains one noise-free object, two development sequences and three overlapping sets of restart seeds. The SMC² comparison is its selected point, not a predictive mixture or an uncertainty-aware shield.

## Follow-up: explicitly insert and trace the true physics

Earlier SMC² experiments randomly initialized the outer population. Ground truth was plotted and evaluated separately, not inserted into that population; the separate numerical grid was not an SMC² initialization either. We therefore ran an explicit oracle diagnostic rather than attributing those failures to a supposedly lost exact-truth particle.

The new study replaces slot 0 of 256 particles with `(m=1 kg, μ=0.3, h_c=0.10 m)`, keeping its initial weight equal to all others (1/256). Both full and gradual updates use exact starts, one inner particle, warmup exclusion, the same fixed proposal and two moves per resampling event. Paired controls retain the original slot and RNG seed. There are 24 trials: two sequences × two update modes × three seeds × injected/control.

**Results:**

- At first resampling, truth has the best target likelihood, allowing ties, in all 12 injected trials. At the end, the separately evaluated true full likelihood also outranks the final population or ties its best in all 24 trials. These are ranks against sampled particles, not a proof of the global maximum.
- Full updates retain exact truth copies in all six injected runs. Final exact-copy counts are 243/171/241 for varied durations and 211/250/251 for passive mixed. Corresponding final weight on exact truth is approximately 97.8/89.8/99.7% and 99.99/99.92/99.94%. Thus the corrected likelihood strongly rewards truth when supplied; noisy inner estimates cannot explain its absence here.
- Gradual updates retain an exact truth copy in only one of six final populations. In the other five, the last exact copies disappear through accepted proposal moves during the first push's weak-evidence stages, not by direct deletion of those exact copies at resampling. Subsequent resampling extinguishes their entire ancestry in three runs. Descendants survive in three of six gradual runs, but need not remain near truth.
- In varied-duration gradual seed 1, the first stage reaches beta≈0.0333 and resampling copies truth three times. Accepted moves remove all three exact copies. Their descendants subsequently receive only about 0.005% total weight before resampling removes them during push 2. The virtual true point still has the highest target likelihood against that population. This is a concrete distinction between scoring truth well and having particles at that physics.
- Exact-truth survival is not posterior correctness. The grid still supports broad COM uncertainty. Mean final COM distance to the grid changes from about 14.92 to 18.41 mm with injection for full updates, and from 11.05 to 13.20 mm for gradual updates. Oracle insertion often creates a concentrated atom rather than a better distribution.

**Interpretation:** on this corrected deterministic dataset, the issue is not that truth is assigned a bad likelihood. The population can miss highly competitive physics and fail to recover it through local proposals. Gradual targets legitimately allow moves away from a good point while evidence is weak; those moves are not automatically erroneous. The traces do not justify pinning truth, rejecting all moves away from it, or interpreting ancestry extinction alone as loss of a posterior region. Adequate coverage of the full correlated plausible region remains the objective. Controlled proposal changes should be compared with the grid and under replay budgets, rather than judged only by whether an exact atom survives.

Validation: 123 tests passed at the inference-tracing milestone. Four new tests show that tracing preserves production particles, weights, filter likelihoods, proposal counts and RNG state for both update modes, with and without injection. All 54 control snapshots exactly match the corresponding saved production-run snapshots. Re-evaluating 32 stored grid midpoints across three pushes found zero likelihood discrepancy. These checks validate the instrumentation and support reuse of this grid, not general observation-model correctness.

Reproduce with a new output directory:

```bash
.venv/bin/python -m src.evaluation.system_identification.diagnostics.trace_truth_particle \
  --out artifacts/sysid_truth_particle_trace_rerun
```

Evidence: [trace report](../../artifacts/sysid_truth_particle_trace/report.md), [survival plot](../../artifacts/sysid_truth_particle_trace/truth_survival.png), per-trial JSON ancestry/move events, `summary.json`, `control_parity.json`, and per-push NPZ snapshots. Grid comparison is available for varied durations only. Oracle runs are excluded from performance comparisons; full and gradual diagnostic arms are not compute-matched.

## Follow-up: covariance-shaped proposal moves

The next experiment changes the proposal shape, without adding more particles, more scheduled moves, an oracle particle, or acceptance-based step-size adaptation. `SMC2SysID(proposal="covariance")` estimates the weighted covariance of particles in prior-scaled `(log m, μ, h_c/H)` coordinates immediately before resampling. The matrix is regularized to retain mobility and its trace normalized so the expected squared jump length matches the original `proposal_scale=0.08` random walk. Thus it learns directions and relative spreads, while keeping the overall proposal radius fixed. A collapsed cloud falls back to the original shape.

The matrix is common and frozen throughout each move stage. Each conditional Gaussian proposal remains symmetric, with out-of-bounds proposals rejected rather than clipped. Population covariance adaptation is a standard SMC technique ([Zhou, Johansen & Aston, section 3.3.3](https://arxiv.org/html/1303.3123v2#S3.SS3.SSS3)); the fixed-trace normalization is our experimental design choice, not the paper's complete adaptation scheme. This does not claim exact finite-population posterior draws.

The controlled study uses the same 256 random initial hypotheses, seeds, saved target data, exact starts, likelihood, ESS threshold, and at most two proposal attempts per particle per resampling stage. The capped arms share the original fixed-kernel cumulative fitting replay ceilings at every prefix. A study-only guard stops a move stage before any candidate whose complete likelihood cannot fit the remaining allowance. All new observations still reach full strength (beta=1). The uncapped covariance arm preserves the original move schedule and exposes effects of budget truncation. Actual replay consumption can be below a ceiling; these are matched allowances, not exactly equal spending. Replay-based stopping is itself an adaptive compute policy with finite-population limitations.

| Sequence | Proposal arm | Position RMSE | COM truth inclusion | Distinct final particles, seeds 1/2/3 | Move acceptance | Total fitting replays | Mean COM distance to grid |
|---|---|---:|---:|---|---:|---:|---:|
| Varied durations | Fixed, capped | 0.176 mm | 3/3 | 30 / 30 / 42 | 24.5% | 19,929 | 11.05 mm |
| Varied durations | Covariance, capped | 0.082 mm | 3/3 | 212 / 225 / 214 | 56.8% | 18,071 | 2.50 mm |
| Varied durations | Covariance, uncapped | 0.111 mm | 3/3 | 212 / 216 / 232 | 57.3% | 18,557 | 1.71 mm |
| Passive mixed | Fixed, capped | 0.304 mm | 2/3 | 15 / 16 / 15 | 17.1% | 30,310 | — |
| Passive mixed | Covariance, capped | 0.126 mm | 3/3 | 224 / 229 / 224 | 53.9% | 24,924 | — |
| Passive mixed | Covariance, uncapped | 0.141 mm | 3/3 | 248 / 229 / 224 | 54.3% | 26,531 | — |

**Findings:** both covariance arms preserve substantially more distinct hypotheses, accept more moves, and represent the varied-duration COM distribution much more closely than the fixed kernel. The capped covariance arm improves mean point-prediction errors on both sequences while using 42,995 fitting replays versus 50,239 for the fixed control (approximately 14.4% fewer). Its final COM interval contains truth in all six runs, versus five of six previously. This is conditional inclusion on one object, not evidence of population calibration. The uncapped covariance arm improves the COM grid agreement further on average, while point prediction is slightly worse than the capped arm; distribution quality and the selected point remain different metrics.

The new proposal does not yet beat retained CMA-ES's mean point-prediction errors (0.068 mm and 0.038 mm), although actual replay spending differs. The main new evidence is a better particle approximation at no greater allowed replay cost, rather than a general claim of superiority over CMA-ES.

Validation: 130 tests passed, including covariance shape/positive definiteness, preserved expected jump length, collapsed-cloud fallback, stage freezing, RNG draw counts, complete-evidence likelihood bookkeeping under tight replay caps, and reproducible analytic Gaussian posterior checks for both kernels. All 27 fixed-control update snapshots and replay counts exactly reproduce the saved production run. All 54 capped update records respect the cumulative ceilings. The 32-point × three-record grid likelihood parity check reports zero change. A covariance benchmark smoke run also exercises the CLI.

Reproduce with a new output directory:

```bash
.venv/bin/python -m src.evaluation.system_identification.comparisons.compare_proposals \
  --out artifacts/sysid_covariance_proposals_rerun
```

Evidence: [proposal comparison report](../../artifacts/sysid_covariance_proposals/report.md), [COM distribution plot](../../artifacts/sysid_covariance_proposals/com_distribution_comparison.png), `metrics.jsonl`, `summary.json`, per-stage proposal matrices/cap events, and saved NPZ particles. Following this development experiment, covariance was made the production, benchmark, and demo default at the user's request. `proposal="fixed"` / `--proposal fixed` remains available for controls, and historical diagnostic runners explicitly use it. No acceptance-based step-size adaptation or diversity-triggered rejuvenation policy was added. One global covariance may still handle curved or multimodal regions poorly. Fresh-object/action validation remains pending despite this default change; perform it before claiming generalization or adding further adaptation.

## Follow-up: CMA-ES versus the new covariance-default SMC²

A new production benchmark reran covariance SMC² with its normal two-proposals-per-stage schedule, without the study-only move cap. Its data are identical to the preceding studies, and all 27 particle/weight snapshots exactly match the uncapped covariance arm. Retained-estimate CMA-ES was then refit under the new SMC² cumulative replay allowances at every prefix; earlier fits and previous best estimate re-evaluations are included in its cost. Both use the same three seeds, scoring stride, loss scales and exact recorded starting states.

| Sequence | Method | Position RMSE (mm) | Mass error (kg) | Friction error | COM error (mm) | Total fitting replays |
|---|---|---:|---:|---:|---:|---:|
| passive mixed | Covariance SMC² | 0.141 | 0.0061 | 0.00345 | 33.49 | 26,531 |
| passive mixed | CMA-ES, retain best | 0.018 | 0.0008 | 0.00004 | 16.17 | 26,531 |
| varied durations | Covariance SMC² | 0.111 | 0.0099 | 0.00375 | 5.02 | 18,557 |
| varied durations | CMA-ES, retain best | 0.070 | 0.0044 | 0.00172 | 8.74 | 18,554 |

**Finding:** CMA-ES retains the advantage in mean selected-point position prediction on both sequences, winning five of six final seed/sequence pairs. It also has lower mean mass and friction errors. Covariance SMC² recovers COM more accurately on varied durations (5.02 versus 8.74 mm, better in all three seeds), while CMA-ES has lower mean COM error on passive mixed (16.17 versus 33.49 mm). This distinction reinforces that precise motion prediction is not equivalent to precise recovery of every hidden parameter.

The current SMC² position errors of 0.111/0.141 mm reproduce the uncapped covariance arm. Earlier values of 0.082/0.126 mm came from the capped diagnostic with a different move schedule; do not mix the two results. The new comparison spends 45,088 SMC² fitting replays and 45,085 CMA-ES fitting replays. All 27 cumulative limits and 21 checks of the previous estimate against the current loss were verified; only three CMA-ES calls are unused because they cannot fund another full-history evaluation.

This remains one noise-free object, two development datasets and three inference seeds with overlapping restart sets. It compares selected point hypotheses, not posterior predictive mixtures or safety decisions. SMC² marginal COM intervals contain truth in all six runs, but that is not a population calibration guarantee, and CMA-ES retained candidates do not provide a comparable Bayesian posterior.

Evidence: [new covariance SMC² versus retained CMA-ES report](../../artifacts/sysid_covariance_vs_cma_retained/report.md), [production covariance benchmark summary](../../artifacts/sysid_covariance_benchmark/summary.json), and the comparison’s `validation.json`. The report includes commands for reproducing both stages with new output directories.

## Implementation and validation handoff

Relevant implementation:

- [SMC² estimator](../../src/agents/system_identification/smc2.py): likelihood evaluation, gradual updates, diagnostics, and posterior plots.
- [Shared scoring](../../src/agents/system_identification/scoring.py): scoring indices and warmup exclusion.
- [CMA-ES estimator](../../src/agents/system_identification/sysid.py): common scoring-window integration.
- [Benchmark runner](../../src/evaluation/system_identification/comparisons/benchmark_smc2.py): shared datasets, explicit experiment modes, metrics, and snapshots.
- [Diagnostic runner](../../src/evaluation/system_identification/diagnostics/diagnose_smc2.py): likelihood variance, channel residuals, and controlled ablations.
- [Grid runner](../../src/evaluation/system_identification/diagnostics/reference_posterior.py): separate numerical posterior integration and convergence checks.

At completion of the gradual-update implementation, **107 tests passed** and Ruff lint/format checks passed. Tests cover scoring windows, preserved replay records, deterministic likelihoods, grid integration, conditional ESS, final-target correctness, reproducibility, analytic posterior moments, and preserving previous evidence during intermediate updates. A check of 32 saved grid points across three pushes found zero likelihood difference after adding gradual updates. Later particle-count experiments changed experimental settings only.

This validation status describes the recorded development milestone; rerun checks against the current checkout before final submission.

Example reproduction, using a new output directory:

```bash
.venv/bin/python -m src.evaluation.system_identification.comparisons.benchmark_smc2 \
  --development --regimes "varied durations" "passive mixed" --seeds 1 2 3 \
  --n-theta 256 --known-start --temper-updates --proposal fixed \
  --cma-iterations 10 --cma-restarts 0 \
  --out artifacts/sysid_handoff_rerun
```

Change `--n-theta` to reproduce population-size settings; omit `--temper-updates` for the full-update control. Configurations, raw datasets, per-update metrics, particle snapshots and plots are stored in each run directory. Compare replay counts rather than wall times because some runs executed concurrently.

**Artifact preservation:** `artifacts/` is ignored by Git. The closure [evidence bundle](evidence/smc2_2026-10-24/README.md) preserves selected numerical outputs and figures outside that ignored directory. After the bundle is committed, extract its `artifacts/` entries into a fresh checkout to restore the evidence links. The manifest records what is included and omitted; historical command compatibility and limits on identifying the original source version are documented in the guide.

## Remaining work and suggested next experiments

1. Validate the covariance default on fresh objects and action datasets. Then test any step-size adaptation or extra moves as separate changes under shared replay ceilings; monitor duplication even when ESS is high.
2. Treat CMA-ES that retains its previous best estimate as the stronger point-estimation baseline. If investigating optimizer persistence, compare it against this completed safeguard under the same replay limits; keep restart policy and sequential-versus-refit cost explicit.
3. Evaluate full posterior predictive distributions and toppling probabilities, alongside point prediction and distinct-particle counts. Do not use truth inclusion alone to judge distribution quality.
4. Validate across fresh objects, action sequences and inference seeds. Current development cases have informed implementation choices and cannot serve as an untouched final test set.
5. Study uncertain observations explicitly before extending gradual updates beyond the deterministic known-start case. Starting-state uncertainty, likelihood estimation, and tempering must remain statistically consistent.

An optional covariance-shaped proposal is now implemented and tested. Diversity-triggered rejuvenation, acceptance-based step-size adaptation, and sensor-noise recalibration have not been implemented as part of these findings. COM identifiability and residual observation-model mismatch may still matter; the current evidence does not assign every remaining error to one cause.

## Suggested report or presentation wording

> Initial SMC² inference suffered from misleading warmup residuals and severe particle collapse. Excluding unreliable warmup observations and introducing new evidence gradually improved inference on a controlled simulator study: at 256 hypotheses, the true COM lay inside the reported 95% interval in five of six runs, compared with two of six using full evidence updates. Improvements required more computation and were not uniform across prediction metrics. Increasing particle count alone did not consistently improve uncertainty, motivating better proposal moves and evaluation under matched simulation budgets.

Avoid claiming that SMC² is calibrated, that COM is reliably identified, that tempering has solved collapse, or that SMC² consistently outperforms CMA-ES at equal compute. The present evidence supports none of those stronger conclusions.
