# B1: gantry-trained SAC on the held-out set, both executors (2026-11-08)

First B1 result for the gantry target (Stage 1 of S5): SAC pretrained in
robot mode, then frozen and per-object fine-tuning on the 8-object
in-distribution set under both executors. The result: the gantry-trained
policy solves its own embodiment (75% frozen at the best checkpoint, zero
topples) but collapses on direct force (0% success, heavy toppling), and
fine-tuning is destructive everywhere — repeating the KUKA-study negative
result on the new embodiment.

## Setup

- Training: `train_sac --robot --target-distance 0.5 --f-max 15`,
  800k steps, 8 workers, uniform-excluding-evaluation-regions
  distribution, push speed 0.12 m/s. Run `artifacts/sac_gantry_lane05`.
- In-training validation oscillated between 0% and 100% success (0%
  topples throughout); the final checkpoint (800k) sat at 20%, so both
  `sac_final.pt` and `best_model/best_model.pt` were evaluated.
- Held-out protocol: `fixed_cases.json` in_distribution split (8 objects),
  seed 42, frozen = 1 episode, fine-tuning = 20 episodes per object with
  frozen normalization. Gantry via checkpoint metadata, force via
  `--no-robot`.
- Evidence: `artifacts/sac_gantry_lane05/results_{final,best}_{robot,force}.json`.

## Results

| Checkpoint | Executor | Frozen success | Tuned success | Frozen topples | Tuned topples |
|---|---:|---:|---:|---:|---:|
| best | gantry | **75.0%** | 41.9% | 0.00 | 0.00 |
| final | gantry | 37.5% | 28.7% | 0.00 | 0.00 |
| best | force | 0.0% | 1.2% | 0.75 | 12.75 |
| final | force | 0.0% | 1.9% | 0.75 | 15.12 |

## What the numbers show

1. **The gantry-trained policy solves its own embodiment.** 75% frozen
   success with zero topples and zero execution failures at the best
   checkpoint. Safety and reliability on the robot target are intact;
   the failure modes are timeouts (stopping short), not toppling.
2. **Checkpoint timing decides the headline number.** The same training
   run gives 75% (best) vs 37.5% (final) frozen success; the SAC policy
   oscillates between "land in the band" and "stop short" strategies.
   Held-out comparisons must therefore evaluate `best_model`, and the
   B1-vs-B2–B6 comparisons should use the same selection rule.
3. **Fine-tuning is destructive, again.** On the gantry, 20 episodes of
   per-object fine-tuning cut success from 75% to 41.9% (best) and from
   37.5% to 28.7% (final), without gaining safety. On direct force it is
   worse: topples climb to 12.75–15.12 per object for 1–2% success. This
   repeats the KUKA-study finding exactly, so it is not embodiment-
   specific: progress-seeking SAC updates from a weak start walk the
   policy into bad pushes.
4. **Cross-executor collapse.** The same gantry-trained policy scores 0%
   on direct force with immediate toppling (0.75 frozen): actions
   calibrated against the velocity-capped executor are unsafe under
   open-loop force. Combined with B0's mirror image (0% force vs 88%
   gantry), the two results together show the embodiment gap dominates
   policy transfer in both directions.
5. **B0 beats B1 on the robot target.** The conservative analytic
   controller reaches 88% (see
   [`b0_conservative_controller_2026-11-08.md`](b0_conservative_controller_2026-11-08.md))
   against B1's 75% best-checkpoint frozen success, with the same zero
   topples. The hand-designed baseline is the strongest method on the
   gantry so far — the bar the imagination baselines (B2–B6) must clear
   is set by B0, not by B1.

## Caveats

- One training seed; the success oscillation means a second seed could
  shift all four cells somewhat, though the KUKA study's qualitative
  pattern (fine-tuning destructive, cross-executor gap) held across its
  own runs.
- Frozen evaluation uses one episode per object, so per-object frozen
  rates are 0/1 decisions; the aggregate 75% is over 8 objects.
