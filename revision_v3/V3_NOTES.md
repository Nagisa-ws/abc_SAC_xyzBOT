# v3 notes (English summary)

## Why evaluation looked worse than training

Measured on your v2 models, 200 held-out scenarios each:

| Cause | Evidence | v3 fix |
|---|---|---|
| Mission 1: eval demanded 5 mm, your curriculum stopped at 2 cm | 100% at 2 cm vs 62% at 5 mm | Eval threshold = last curriculum threshold (`shared/evaluation/task_spec.py`); v3 curriculum ends at 5 mm |
| Mission 3: eval used obstacle margin 1 cm (old level 4), your curriculum ended at 3 cm | 82% vs 77.5% | Eval uses the last curriculum level automatically |
| 10-scenario evals | ±25-point confidence interval | 100 scenarios by default, 95% CI printed |
| Optimistic `best_model` | Mission 3: 400 evals on the same 30 scenarios, mean 0.79, max (saved) 0.90 | Held-out validation picks best vs final, then a 200-scenario test is reported in `final_test.json` |
| Eval robot model lacked the base collision cylinder | 13 vs 14 collision geoms | Added; geometry now identical |

## Why Mission 3 plateaued at 75–80%

At level 3, 26 of 36 Mission 3 failures happened while reaching T1 (self-collision, floor, obstacles). The scenario validator already
computes a collision-free route to T1 (and removes obstacles along it), but the policy couldn't see it. A simple
controller that follows that route scored 99.5% (100% at 5 mm, 99.5% with domain randomization and sensor noise) where your policy
scored 77.5%. A safety filter alone only turned crashes into timeouts (77%).

## What v3 changes

- **Residual policy**: action = reference action + 0.5 × RL correction. The reference follows the validated route
  (`shared/control/reference.py`). The reference action is part of the observation (63 values instead of 57). The
  correction starts at zero, the critic trains first, and an anchor pulls the correction toward zero (Mission 1),
  toward the previous mission's policy (Mission 2), or toward zero again (Mission 3).
- **Mission 3 specifics**: the correction is reset to zero at transfer, it fades out within 2–8 cm of obstacles, and it
  controls position only. In all 4 remaining collisions the correction had rotated the gripper to the 30° limit.
- **One task definition** for training, evaluation, benchmark and the final test.
- **Optional safety filter** (`safety_filter.enabled`, off by default).
- `gradient_steps: 1` everywhere: stable and about 3× faster.

## Results (same 200 unseen scenarios for every column)

| Task | Your v2 | Reference alone | v3 |
|---|---|---|---|
| Mission 1, 5 mm | 66.0% | 90.0% | **93.0%** |
| Mission 1, 2 cm | 100% | — | 99.5% |
| Mission 2, 2 cm | 85.5% | 100% | **99.5%** |
| Mission 3, final level | 70.0% | 99.5% | **100%** |
| Mission 3 + domain randomization (payload, gravity) + sensor noise | 71.5% | 99.0% | **99.5%** |

- Mission 3 contact failures fell from 49/200 to 0/200.
- Cross-track error fell from 24.5 to 14.6 mm (Mission 2) and from 26.0 to 7.7 mm (Mission 3).
- Mission 3 training at the final level (99.6%), in-training eval (100%) and held-out test (100%) now agree.
- **Trade-off:** v3 needs 9–17% more steps per successful episode.
- **Honest note:** most of the gain comes from following the validated route. In Mission 3 the reference alone already
  scores 99.5–100%; the RL correction adds Mission 1 precision (+3 points at 5 mm) and a little speed.

## How to use

- **Included models** (trained here, Missions 1 → 2 → 3, 120k / 150k / 300k steps):
  `python Mission3/eval_mission3.py` (similarly for 1 and 2).
- **Retrain**: `python Mission1/train_mission1.py`, then 2, then 3. The last lines print `[uji held-out]`, the realistic number.
- **v2 checkpoints won't load** (57 vs 63 observation values). Keep your old project copy if you need them.
- **In TensorBoard**: `train/ent_coef`, `train/q_pi_mean` (must not keep rising while success falls),
  `train/anchor_weight`, `curriculum/level`.
