# Imagine Before You Push: Report Handoff and Editorial Rules

**Status:** Instructions and decisions established by the report authors during the revision process, updated 10 November 2026. This is a handoff for any person or coding agent continuing work on the report. It is not a new research result. Where a rule has changed over time, the latest instruction takes precedence.

**Working source:** `docs/report/Imagine Before You Push.tex` in this repository, associated with PR #54. Continue from this file, not an older PDF or a detached copy. Keep the revision audit at `docs/report/REVISION_AUDIT.md` updated. Do not merge the draft PR while outcome metrics remain unverified.

**Latest experiment update (10 November 2026):** The current session is on `report_20261110`; remain on this branch. The branch/PR references above describe the earlier report workflow. Both 15 N replacements completed training through 1.00 m: [model-free](../findings/model_free_15n_curriculum_2026-11-10.md) achieved 89/1,010 successes (8.8%) with 87.7% toppling there; [learned](../findings/learned_15n_curriculum_2026-11-10.md) achieved 438/670 successes (65.4%) with 26.3% toppling. Their [completed frozen benchmarks](../findings/frozen_15n_benchmarks_2026-11-10.md) each use 24 objects, five episodes each and seed 42: model-free succeeds on 8/120 overall (6.7%) and 3/80 OOD (3.8%); learned succeeds on 59/120 overall (49.2%) and 27/80 OOD (33.8%). Both record zero optimizer updates. Physics was not reevaluated here; its existing documented OOD reference is 38/80 (47.5%). The author approved report integration: the source and PDF now use the 15 N comparator training results and three-method held-out table, with the associated abstract, discussion, conclusion and appendix diagnostics updated. Old runs were not pooled with the replacements. Preserve the learned model's higher training success through 0.80 m, physics' lead at 0.90/1.00 m, and the nonuniform OOD split results; do not restore universal physics-dominance or model-free early-stopping claims.

**Approved follow-up:** Section 6.4 now includes OOD toppling counts: 21/80 physics (from the existing physics findings), 30/80 learned and 64/80 model-free (from the verified new logs). Appendix G.2 specifies the learned MLP, inputs, trajectory scaling, training loss, eligible replay examples and optimizer schedule. Appendix D retains 10,000 as the **configured push target** documented by the archived command. Available early physics logs contain about 8,000 pushes per stage; this must not be described as 10,000 observed pushes per stage. The author requested no physics rerun; reported rates retain their actual logged denominators. Contributors remain deferred. No new training-replication study or uncertainty analysis was requested.

## 1. Scientific thesis: do not lose this argument

The **primary research question** is whether an **explicit physics simulator, configured through online system identification (SysID), is a suitable forward model for reinforcement learning under unseen or out-of-distribution (OOD) physical dynamics**, compared with (i) a learned neural world model and (ii) a model-free alternative.

**Toppling-safe pushing is a proxy task, not the principal research objective.** Its safety-critical tipping dynamics make it possible to test the fidelity and usefulness of imagined trajectories when mass, friction and centre-of-mass (COM) height are hidden. Do not recast the paper as primarily about inventing a robot pushing policy, solving the pushing task, or improving a curriculum. The delta actor-critic, system identification, and curriculum are methods that make the forward-model comparison possible.

Maintain the causal and narrative order throughout the abstract, introduction, contributions, experiments, discussion and conclusion:

1. Why a forward model matters under unseen dynamics.
2. Why explicit physics plus online SysID might generalise differently from a learned model.
3. Why toppling-safe pushing is an informative testbed.
4. How the three approaches are implemented and compared.
5. What the experimental evidence shows about safety, OOD behaviour, and imperfect identification.
6. What is and is not established, including limits of sim-to-sim testing.

The report may draw clear, confident conclusions about **observed experimental differences**. Do not replace results with generic hedging, or repeatedly call them preliminary merely because they are research experiments. At the same time, do not claim real-world robustness, guaranteed safety, causal benefit from particle ensembles, or robustness to structural simulator mismatch without supporting experiments.

## 2. Readability and copyediting rules

- **Readability first:** Make the prose flow naturally for a technically literate audience familiar with AI but not necessarily with this specific project.
- Eliminate AI-generated writing patterns: long stacked noun phrases, unnecessary jargon, robotic transitions, sentences that compress motivation, implementation and consequence into a single clause, and abrupt jumps between paragraphs.
- Prefer active voice, direct sentences, clear referents and logically ordered paragraphs. Avoid repetition between Results, Discussion and Conclusion.
- **Keep standard ML terminology**, including forward model, world model, OOD, POMDP, SMC², sequential importance sampling, effective sample size (ESS), Metropolis-Hastings MCMC, posterior, twin critics, SAC, TD targets, Transformer and gradient estimators. Explain their role rather than replacing them with imprecise everyday terms.
- Preserve mathematical precision and scientific meaning while improving readability. A copyedit is not permission to weaken, change or remove a technical result.
- **No em dashes anywhere**, including the LaTeX source. Avoid Unicode em dash characters and LaTeX triple-hyphen em-dash notation. Use commas, colons, parentheses, or separate sentences as appropriate.
- Use consistent terminology: the **physics simulator as a forward model**, the **learned neural world model**, **model-free control**, and the **customised soft actor-critic (SAC) policy called the delta actor-critic**.
- The main report must read independently from start to finish. Appendices provide extra derivations and supporting evidence, not essential definitions or missing pieces of the argument.
- After each substantive revision, read the report again as a whole rather than just verifying isolated paragraphs.

## 3. Non-negotiable content and technical details

### System and reward

- Explain the partially observable problem, the hidden parameters (mass, friction, vertical COM), the action, tipping/stability analysis, and the reward.
- **Keep the explanations of every reward component in the main report**, including potential-based progress shaping, no-progress penalty, time and step costs, terminal rewards and settling requirements.
- Keep **optional information-gain shaping within the Reward subsection**, not Candidate Selection.
- Describe the controller as a **customised SAC policy called the delta actor-critic**, not merely a generic actor-critic.
- Preserve the architectural distinction between the target environment and internal physics simulator. The inferred physics parameters configure the internal simulator; they are not directly appended to neural-network inputs.

### SMC²

- Keep the core SMC² explanation and mathematics in the main report: trajectory likelihood, importance-weight updates, ESS definition, systematic resampling, Metropolis-Hastings MCMC rejuvenation, and the role of adaptive proposals.
- ESS may be explained briefly, but **its definition must remain**. Explain why resampling and MCMC serve different purposes.
- The report's configuration has an inner state filter of size `N_state=1` with known start state; do not imply a nontrivial inner filter was active in the reported experiments. More detailed nested-state likelihood derivation may remain in the appendix.
- Keep the rationale for posterior hypotheses versus a point estimate, but distinguish the *hypothesis* that an ensemble helps robustness from experimentally proven causation.

### Delta actor-critic

- Keep the bounded delta-action formulation, trajectory encoder explanation, posterior-weighted twin-critic score, likelihood-ratio actor gradient, and the implications of a non-differentiable PyBullet model in the main report.
- The trajectory encoder is important enough to explain in the main report, including the Transformer option used in the reported physics-agent experiments.
- More standard SAC training details such as full soft TD-target derivations and temperature tuning can remain in appendices if the main body remains understandable.

### Curriculum

- Retain the **13 real curriculum target distances** from 0.10 m to 1.00 m: 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00 m.
- **Remove all references to a 0.05 m curriculum stage**, including counts or performance claims involving it. The success tolerance was 0.08 m, so starting 0.05 m away made the position criterion trivial. This does **not** mean removing the unrelated 0.05-second minimum push duration.
- Explain why direct training led to a do-nothing policy and how reward shaping, considered-but-not-run supervised critic pre-training, and the reverse curriculum were handled. Do not describe an unexecuted experiment as conducted.
- Keep the important horizon-scaling observation. Present the curriculum as an enabling training technique, not the central research contribution.
- The **historical 50 N model-free run** stopped at 0.35 m after consistent failure beyond 0.15 m. The **completed 15 N replacement**, now used in the report, reached 1.00 m and was evaluated on held-out objects. Retain its measured rates at every distance. Do not restore the old early-stopping and untested-distance statements, pool the two runs, or describe the replacement as universally unsuccessful.

## 4. Experiments, tables, and interpretations

- Do not silently alter numeric results, denominators, model definitions, table rows, or experiment conditions.
- **Force-limit correction verified on 10 November 2026:** The documented physics run used 15 N; the archived learned and model-free configurations resolved to 50 N. In particular, an archived `f_max: null` means no override, not a 15 N cap; the saved `action_ranges.force` is `[0, 50]`. Both new comparator runs resolve to `[0, 15]`. State verified settings accurately rather than claiming that all original force limits matched. Matching force limits does not align every method setting, including the encoder, learning-start setting and physics-only posterior bonus.
- Avoid unsupported or repetitive references to a single training seed. The author explicitly requested removal of that emphasis from the prose.
- Explain why a **learned world model may struggle on OOD physics**: it has not seen those hidden parameter ranges during training. Distinguish this plausible explanation from a separately established causal experiment.
- Preserve the central observation that low aggregate trajectory MSE does not imply correct prediction of the **safety-critical tilt/tipping dynamics**.
- **Updated learned-model diagnostics:** Median trajectory MSE over the first/last 250 training updates is 0.064/0.0032. The logged mean normalised peak-tilt error is 0.032 for 35 in-distribution episodes and 0.070 for 15 mass-OOD episodes, with mass-OOD median 0.021. The diagnostic excludes terminal pushes and is absent from ten episodes, so it cannot explain all topples. Do not reuse the old checkpoint's 0.053/2.24/0.081 values or infer that low MSE alone proves reliable or unreliable tipping forecasts.
- Preserve and discuss SysID imperfections. Mass and friction can be difficult to distinguish from sliding; the low-force initial probe may be weakly informative; longer trajectories can improve identification; COM height can remain difficult to infer safely.
- Discuss the controller's ability to succeed despite imperfect parameter estimates. A posterior particle ensemble **might** explain part of the robustness, but an ensemble-versus-point-estimate ablation is required to demonstrate that mechanism. Keep OOD failures visible as limits to this finding.
- Preserve the distinction between force-calibration errors and errors in the mass-free tip-stability margin.

### Current placement of five original tables

The author-approved layout keeps **original Tables 2, 3 and 4 in the main report** (forward-model ablation, online SysID accuracy, held-out OOD success), while **original Tables 1 and 5 stay in Appendix H** (full curriculum outcomes and detailed OOD SysID errors). LaTeX may renumber the main tables and appendix tables; cite by labels, not hard-coded numbers. Main prose must still communicate the key evidence from appendix tables.

### Main forward-model comparison table, called Table 1 in the current main report

The author asked for **(%) in the column headers** and both **timeout** and **overshooting** metrics in addition to success and toppling. Never mistake `success / topple` notation for a fraction.

**Data integrity is essential:**
- **Latest author-approved outcome split (9 November 2026):** Table 1 uses mutually exclusive S (success), T (all topples, including toppled overshoots), TO (other timeouts with final signed distance `d >= -0.08 m`), and O (overshoot timeouts with `d < -0.08 m`). Each rate uses all episodes at that distance as its denominator. The four counts sum to the episode count; displayed percentages can differ slightly from 100% due to independent rounding to one decimal place. This replaces the earlier overlapping overshoot metric.
- Overshoot percentages must come from episode-level final signed distance or equivalent verified outcome data. Do not invent them or derive them from `100 - success - topple`.
- Check how episodes terminate before interpreting residual outcomes as verified timeouts. Overshoot may be a **subset of timeout**, not an additional mutually exclusive terminal category.
- Define denominators and whether the rates are mutually exclusive, overlapping, or conditional. Percentages need not sum to 100 when overlapping metrics are included.
- Existing unverified overshoot cells marked `?` are placeholders, **not completed analysis**. The uploaded `training_logs.zip` was previously unavailable to the execution runtime. Retry a supported read of the actual ZIP before declaring it inaccessible; if still blocked, report the limitation. Prefer delegated/subagent analysis for large logs when a suitable agent capability is genuinely available, but never claim delegation without using one.
- Keep the GitHub PR as a **draft** until this table is validated.

### Results story

Give priority to the comparison of **physics vs learned vs model-free** and held-out OOD evaluation. The reverse curriculum's success rate supports the feasibility of the experiment; it must not overshadow the actual forward-model research question. Preserve the quantitative SysID robustness analysis and the OOD caveats without repeating all table cells in prose.

## 5. Page layout and appendices

- **Title exactly:** `Imagine Before You Push`. No subtitle and no `November 2026` date.
- **Authors, exactly in this order:** Liu Ernest Hin Yui; Wong Deshun; Harsh Rao Dhanyamraju; Arnav Goel; Alez Ng Rui Jie; Maahir Garg.
- Place the author names **horizontally on one line**, in a smaller size than the main 12-point font. The author block may be smaller than the 12-point minimum for body text. Reduce vertical gaps between title, authors and affiliation.
- Use **12-point main body**, single column, 1-inch margins, and readable tables/figures. Do not use tiny body text or margin tricks to meet the limit.
- The **10-page limit applies through the Conclusion**. References and appendices are outside the main-body limit. Verify pagination after every substantial edit and inspect the first page for blank pages, title spacing or misplaced content.
- Keep the **Relation to Course Material** appendix (`app:course`) as bullet points with the distinction between course material and independently learned material. Do not flatten it into prose.
- Supporting SMC² derivations, standard SAC implementation details, and extended experiment tables can remain in appendices, provided the main body stands alone.
- For selective restoration from Appendix H, the author's priority order was **H.4 (identifiability and task success), H.1 (curriculum outcomes prose, keeping its table in the appendix), H.5 (OOD failures by parameter), H.3 (training-stage dynamics)**.
- Do not reintroduce the unwanted future-direction points: **unmeasured sensitivity to parameter errors**, **loss of success tolerance and manually chosen curriculum**, or **policy-gradient variance and control resolution**.
- Retain future work on **different object shapes and geometry representations**, including point clouds for irregular shapes, and the possible ensemble-versus-point-estimate comparison.
- Do not accidentally add extraneous headings, duplicate appendix sections, blank initial pages, or stale section references.

## 6. GitHub and change-control workflow

- **Work from the existing PR branch.** Push edits to the same branch so discussion continues in PR #54. Do not overwrite `main` or create unrelated replacement reports by default.
- Prefer reviewing the `.tex` source and its compiled rendering. The report PDF on `main` may lag behind the branch source. Do not present an old PDF as validation of the latest commit.
- **Never silently delete complete sections, substantive paragraphs, equations, results, or appendix content.** Previous accidental omissions of Appendix D, nested SMC² details, and information-gain shaping are reasons for this rule. Ask before substantive deletion when possible. If the user has already approved a relocation/condensation, implement that exactly.
- Maintain `docs/report/REVISION_AUDIT.md`. Record explicitly: (a) what moved to the appendix or main body; (b) what was removed rather than relocated; (c) independent editorial choices; (d) remaining blockers; and (e) what was actually validated.
- Run LaTeX compilation, check warnings, labels, captions and table overflow, and inspect the rendered first page and key results pages. Report page count **through the conclusion**, not just total PDF pages.
- Check that the final report meets the readability rules, retains core ML terminology, has no em dashes, and is intelligible **without opening appendices**.
- Do not claim successful compilation, exact page counts, verified numerical data, or completed subagent work unless those checks were performed on the **latest committed revision**.
- Because ChatGPT download links have repeatedly failed for the user, **GitHub branch + PR is the primary delivery channel**, with an audit committed beside the report.

## 7. Project-guideline requirements to verify before submission

The CS4246/CS5446 guidelines require an academic technical report readable by classmates familiar with AI; a maximum 10-page single-column, single-spaced main report with at least 12-point fonts and 1-inch margins; appropriate references; clearly specified team contributions; and transparency about LLM-assisted writing. Do not invent individual author contributions or an AI-use disclosure without the required facts from the authors. Verify any final submission requirements against the provided guidelines rather than assumptions.

## 8. Final review checklist

- [ ] Abstract, Introduction, Results, Discussion and Conclusion clearly answer the **physics-forward-model-under-OOD** research question.
- [ ] Toppling-safe pushing is consistently described as the **proxy task**.
- [ ] Comparative results are preserved, with no invented outcomes or incorrect comparator caveats.
- [ ] SMC² likelihood, importance weighting, ESS, resampling and MCMC are understood from the main body.
- [ ] Delta actor-critic, trajectory encoder, reward components and training rationale remain readable.
- [ ] Main report is self-contained; appendices are optional elaboration.
- [ ] All table data, units, percentage denominators and references are checked.
- [ ] Unverified overshoot/timeout metrics are resolved or clearly marked as outstanding.
- [ ] Thirteen target distances start at 0.10 m, with no 0.05 m curriculum stage.
- [ ] Title, authors, font sizes, margins and title spacing meet the agreed format.
- [ ] No em dashes or unrequested substantive deletions.
- [ ] Latest LaTeX compiles; the conclusion fits within 10 pages; first and key pages are visually inspected.
- [ ] Revision audit accurately states changes, decisions and checks; updated source is committed to the PR branch.

**Governing principle:** Protect the scientific argument and factual evidence first. Improve readability and layout without silently trading away research content.
