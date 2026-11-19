# SAC evaluation: reserved holes and OOD objects (2026-10-26)

Both reports evaluate the same 800,000-step final checkpoint trained with
`configs/rl/with_holes.json` (now
[`uniform_excluding_evaluation_regions.json`](../../configs/rl/training_distributions/uniform_excluding_evaluation_regions.json)), training seed 42 and maximum force 15 N.
SHA-256: `a7b3b9c3c082641cf8517c95cd4a5d083550f97bafa10255c562e946658b9b99`.

Sources: [holes](../../artifacts/sac_pretrain/holes_force15/evaluation_holes.json),
[OOD](../../artifacts/sac_pretrain/holes_force15/evaluation_ood.json),
[validation history](../../artifacts/sac_pretrain/holes_force15/evaluations.json).

**Historical baseline:** the findings below describe LR scale 0.5. The original
raw-result paths linked above were overwritten by the LR-scale-0.2 rerun. The
[learning-rate comparison](#learning-rate-comparison) records the newer results
and links separate snapshots; the original full scale-0.5 trajectories were not
archived here, so only previously recorded summaries can be compared directly.

There are 160 sampled hole objects (20 per region) and 16 fixed OOD objects
(four per category). Membership is recorded as `excluded_region` for all 160
hole objects and `outside_ranges` for all 16 OOD objects. Evaluation seed is 123.
The original runs used five deterministic frozen episodes and twenty stochastic
fine-tuning episodes per object, starting from identical pretrained weights with
fresh replay. All five frozen repeats were identical within each object. This
report collapses them into **one object-level outcome per object**; raw JSON files
retain the original episode records. Frozen counts below therefore use 160 hole
objects or 16 OOD objects, while fine-tuning counts use all 3,200 or 320 episodes.
Rates and means are unchanged. Observation normalization remains frozen.
Fine-tuning uses learning-rate scale 0.5; its scores describe performance during
learning, not a separate final-policy test.

| Evaluation | Success | Topple | Timeout |
|---|---:|---:|---:|
| Holes, frozen (objects) | 55.62% (89/160) | 4.38% (7/160) | 40.00% (64/160) |
| Holes, fine-tuning (episodes) | 54.59% (1747/3200) | 4.62% (148/3200) | 40.78% (1305/3200) |
| OOD, frozen (objects) | 18.75% (3/16) | 31.25% (5/16) | 50.00% (8/16) |
| OOD, fine-tuning (episodes) | 13.12% (42/320) | 27.81% (89/320) | 59.06% (189/320) |

## Training distribution

The evaluated checkpoint records these outer sampling bounds:

| Parameter | Outer training range | Unit |
|---|---|---|
| Width $w$ | [0.05, 0.1] | m |
| Length $L$ | [0.05, 0.1] | m |
| Height $H$ | [0.12, 0.25] | m |
| Mass $m$ | [0.5, 1.5] | kg |
| Object-floor friction $\mu$ | [0.3, 0.7] | dimensionless |
| Relative COM height $h_c/H$ | [0.35, 0.6] | dimensionless |

Training draws uniformly from this joint box, then rejects any object inside
one of the eight reserved holes listed below. The actual training support is
therefore **the outer box minus the union of the holes**, rather than the full
box. Geometry remains sampled over its full ranges; each exclusion jointly
constrains mass, friction, and relative COM height. These bounds come from the
checkpoint metadata recorded in the evaluation report, not the current defaults.

## Reserved holes

These are the joint hidden-parameter regions excluded from training and sampled
for this evaluation. An object belongs to a hole only when **all three** hidden
parameters fall within that row's intervals; boundaries are included. The ranges
below come from the config snapshot stored in the evaluation results.

| Hole | Mass $m$ (kg) | Friction $\mu$ | Relative COM height $h_c/H$ |
|---|---|---|---|
| hole_1 | [0.65, 0.75] | [0.38, 0.42] | [0.3875, 0.4125] |
| hole_2 | [1.25, 1.35] | [0.58, 0.62] | [0.539880952381, 0.564880952381] |
| hole_3 | [0.5, 0.55] | [0.3, 0.32] | [0.5375, 0.5625] |
| hole_4 | [1.45, 1.5] | [0.63, 0.67] | [0.3875, 0.4125] |
| hole_5 | [0.95, 1.05] | [0.48, 0.52] | [0.4875, 0.5125] |
| hole_6 | [1.05, 1.15] | [0.43, 0.47] | [0.440441176471, 0.465441176471] |
| hole_7 | [0.85, 0.95] | [0.53, 0.57] | [0.4375, 0.4625] |
| hole_8 | [1.35, 1.45] | [0.33, 0.37] | [0.4375, 0.4625] |

Geometry varies independently within every hole: width $w$ and length $L$ are
both sampled from $[0.05, 0.10]$ m, and height $H$ from $[0.12, 0.25]$ m.
The COM ratio $h_c/H$ is dimensionless; absolute COM height is that ratio times
the sampled object height. Each hole contributes 20 sampled objects.

Performance for these same regions:

| Group | Frozen success (objects) | Fine-tuning success (episodes) | Frozen topples (objects) | Fine-tuning topples (episodes) |
|---|---:|---:|---:|---:|
| hole_1 | 95.00% | 70.25% | 5.00% | 4.50% |
| hole_2 | 0.00% | 0.50% | 0.00% | 9.00% |
| hole_3 | 60.00% | 26.00% | 30.00% | 16.75% |
| hole_4 | 0.00% | 0.00% | 0.00% | 0.25% |
| hole_5 | 70.00% | 79.50% | 0.00% | 2.00% |
| hole_6 | 85.00% | 90.50% | 0.00% | 0.25% |
| hole_7 | 35.00% | 70.75% | 0.00% | 4.25% |
| hole_8 | 100.00% | 99.25% | 0.00% | 0.00% |

The frozen policy succeeds on 89/160 distinct objects (55.625%) with low overall
toppling. Generalization into excluded regions is useful but highly uneven.
Holes 2 and 4 never succeed when frozen. These reserve heavier, higher-friction
physics. Hole 4 has mean final error 0.896 m against a 1 m target; its median
error is approximately 1 m, indicating little net progress for many cases.
Its flat-sliding friction estimate stays below 15 N, so the force cap alone does
not establish that failure is unavoidable. Action choice, safe force application,
and the ten-decision budget require investigation.

Hole 3 contains light, relatively high-COM objects. It accounts for 6/7 objects that topple
under the frozen policy. Fine-tuning reduces its topple rate from 30% to 16.75%, but
success falls from 60% to 26% and timeout rises from 10% to 57.25%. Safety improves
at the expense of task completion; the logs do not reveal the exact action mechanism.
Hole 1 also loses substantial success (95% to 70.25%). Hole 7 improves most
(35% to 70.75%), followed by holes 5 and 6.

Averaged across all twenty fine-tuning episodes, success changes by -1.03 percentage
points and toppling by +0.25 points. Within the adaptation sequence, success rises
from 51.125% in episodes 1-5 to 57.5% in episodes 16-20; toppling falls from 9.25%
to 2.125%. The final window is still stochastic and contains continued learning.

Fine-tuning achieves at least one success on 116/160 objects, compared with 89/160
for the deterministic frozen policy. It succeeds at least once on 29 of the 71
objects that the frozen policy fails, but two previously successful objects never
succeed during fine-tuning. More stochastic attempts increase opportunity for an
occasional success, so object coverage does not demonstrate improved reliability.

## Outside-range objects

| Group | Frozen success (objects) | Fine-tuning success (episodes) | Frozen topples (objects) | Fine-tuning topples (episodes) |
|---|---:|---:|---:|---:|
| mass_ood | 0.00% | 6.25% | 50.00% | 22.50% |
| friction_ood | 0.00% | 0.00% | 25.00% | 13.75% |
| com_ood | 75.00% | 46.25% | 0.00% | 33.75% |
| joint_ood | 0.00% | 0.00% | 50.00% | 41.25% |

All frozen OOD successes come from the COM category: three of its four objects
succeed under the frozen policy. Mass, friction and joint OOD have zero frozen successes.
Fine-tuning produces five successes on the 0.4 kg mass case, but none on any
friction or joint case. Overall success falls from 18.75% to 13.125%.

The highest-COM object (h_c/H=0.8) is the clearest safety failure: frozen evaluation
succeeds on this object (1/1) without toppling, whereas fine-tuning topples in all 20 episodes.
At h_c/H=0.65, the frozen policy succeeds on this object (1/1), versus 4/20 episodes during fine-tuning, with
7/20 topples. These outcomes motivate checking exploration safety on already
competent policies; they do not isolate the effects of stochasticity versus updates.
The low-COM 0.2 case narrowly misses the position tolerance when frozen (error
0.05175 m against a 0.05 m tolerance) and succeeds in 13/20 adaptation episodes.

The heavy joint case (m=3 kg, mu=0.85) has a nominal flat-sliding friction threshold
mu*m*g=25.02 N, above the actuator cap. Both conditions leave approximately 1 m
position error and time out throughout. This is consistent with inadequate force
for ordinary sliding. The 3 kg mass-only case has a threshold of 14.715 N, leaving
very little headroom below the 15 N cap, and also makes essentially no progress.
These physical estimates do not prove impossibility for every rocking/contact mode.

During OOD adaptation, toppling falls from 37.5% in episodes 1-5 to 15% in episodes
16-20, but success falls from 16.25% to 11.25%. Lower toppling is insufficient
without task completion. Mean final error nevertheless improves from 0.620 m
frozen to 0.532 m during adaptation, showing that some progress is not captured
by the binary success score.

## Interpretation limits and next checks

- Frozen results are reported per distinct object after collapsing the five
  identical repeats: 160 hole configurations and 16 fixed OOD configurations.
  The OOD category percentages each rest on only four objects.
- This is one trained model and one adaptation RNG stream per object. No claim
  of superiority across training or adaptation seeds is supported.
- The contrast changes both learning and action stochasticity. Add a frozen
  stochastic-policy control before attributing performance loss to weight updates.
- Adaptation scores include training episodes. A separate deterministic evaluation
  after adaptation is needed to measure the final adapted policy; those policy
  weights are not saved by these reports and cannot be reconstructed from them.
- The evaluated checkpoint is the final 800k model. Validation selected the 640k
  checkpoint (70% success, 0% topples, mean reward 76.15 versus final 65%, 10%,
  60.78). Evaluate that validation-selected checkpoint separately, preserving these
  results; do not choose between checkpoints based on whichever test score is higher.
- Add action traces (force, height fraction, duration, signed remaining distance,
  speed and tilt) on representative timeout and toppling cases to separate poor
  action selection, overshoot, instability and actuator limitations.
- Broaden the fixed OOD suite and repeat training/adaptation seeds. If a 15 N
  benchmark retains physically demanding objects, report actuator-limited cases
  explicitly. Changing action scaling for the same weights would invalidate the
  learned action semantics. Low-force exploration at a fixed 50 N bound remains
  a proposed experiment in the project outline, not an implemented feature.

## Learning-rate comparison

The new evaluation uses `--lr-scale 0.2`, changing the optimizer learning rate
from 0.00015 (scale 0.5) to 0.00006. `SAC.load()` applies this rate to actor,
critic and entropy-temperature optimizers, not just the actor. The checkpoint
hash, training steps (800k), evaluation seed (123), object count and episode
budgets match the documented baseline. Frozen rates are unchanged in every group.

New raw snapshots: [holes, scale 0.2](../../artifacts/sac_pretrain/holes_force15/evaluation_holes_lr_scale_0p2.json)
and [OOD, scale 0.2](../../artifacts/sac_pretrain/holes_force15/evaluation_ood_lr_scale_0p2.json).
The original scale-0.5 values below come from the earlier tables in this report.

| Fine-tuning metric | Scale 0.5 | Scale 0.2 | Change |
|---|---:|---:|---:|
| Hole successes | 1747/3200 (54.59375%) | 1741/3200 (54.40625%) | -0.1875 pp |
| Hole topples | 148/3200 (4.625%) | 260/3200 (8.125%) | +3.5 pp |
| Hole timeouts | 1305/3200 (40.78125%) | 1199/3200 (37.46875%) | -3.3125 pp |
| OOD successes | 42/320 (13.125%) | 36/320 (11.25%) | -1.875 pp |
| OOD topples | 89/320 (27.8125%) | 108/320 (33.75%) | +5.9375 pp |
| OOD timeouts | 189/320 (59.0625%) | 176/320 (55%) | -4.0625 pp |

| Group | Success, scale 0.5 | Success, scale 0.2 | Topple, scale 0.5 | Topple, scale 0.2 |
|---|---:|---:|---:|---:|
| hole_1 | 70.25% | 70.75% | 4.50% | 8.75% |
| hole_2 | 0.50% | 0.75% | 9.00% | 13.00% |
| hole_3 | 26.00% | 39.50% | 16.75% | 37.25% |
| hole_4 | 0.00% | 0.00% | 0.25% | 0.25% |
| hole_5 | 79.50% | 72.75% | 2.00% | 1.50% |
| hole_6 | 90.50% | 86.00% | 0.25% | 0.50% |
| hole_7 | 70.75% | 66.25% | 4.25% | 3.75% |
| hole_8 | 99.25% | 99.25% | 0.00% | 0.00% |
| mass_ood | 6.25% | 7.50% | 22.50% | 42.50% |
| friction_ood | 0.00% | 0.00% | 13.75% | 15.00% |
| com_ood | 46.25% | 36.25% | 33.75% | 30.00% |
| joint_ood | 0.00% | 1.25% | 41.25% | 47.50% |

Lowering the LR does not improve aggregate success or safety on this run.
Hole success is essentially unchanged, but toppling rises by 112 episodes;
82 of those additional topples occur in hole 3. Its success rises from 26% to
39.5%, while its topple rate rises from 16.75% to 37.25%. The smaller LR trades
fewer timeouts for more successes and more topples in this region. Holes 5, 6
and 7 lose success. Hole 4 remains unsolved.

OOD success falls despite higher object coverage (at least one success on 6/16
objects versus 4/16). The highest-COM object improves from 0/20 successes and
20/20 topples to 3/20 successes and 17/20 topples, but the COM category overall
falls from 46.25% to 36.25% success. The low-COM 0.2 case drops from 13/20 to 6/20
successes. Mass OOD has slightly more successes but much more toppling; friction
OOD remains unsolved; joint OOD gains one successful episode but more toppling.

In adaptation episodes 16-20, hole success is 56.25% at scale 0.2 versus 57.5%
at scale 0.5; toppling is 4.875% versus 2.125%. OOD last-window success is
unchanged at 11.25%, while toppling rises from 15% to 28.75%. These remain
stochastic, learning-enabled windows rather than final-policy evaluation.

This does not support an excessively large LR as the sole explanation for the
previous degradation; it also does not rule out harmful updates. A smaller rate
can slow beneficial adaptation as well as harmful drift, and temperature updates
also changed. Frozen stochastic controls, deterministic post-adaptation tests,
and repeated adaptation seeds are needed to separate these effects. Catastrophic
forgetting would additionally require checking previously learned behavior after
adaptation, rather than inferring it solely from lower during-adaptation scores.
