# Mission 3 fix: success dropping to 0% after the critic warm-up

## How to apply

Copy these files over your project (same paths), or run `patch -p1 < perbaikan_misi3.patch` from the project root:

| File | What changed |
|---|---|
| `shared/training/trainer.py` | `ArmSAC.train()`: actor **and** entropy coefficient frozen during warm-up, `ent_coef_max` cap, small pre-tanh mean regularizer, fading actor anchor with a floor, new TensorBoard logs |
| `shared/training/callbacks.py` | `CriticWarmupCallback` now freezes actor + entropy coefficient (the old "reset alpha at the end" is removed) |
| `Mission3/configs/sac_config.yaml` | fixed `ent_coef: 0.005`, `gradient_steps: 1`, anchor `0.4 -> 0.2` over 150k steps, `actor_mean_reg: 1e-3` (your timesteps / eval / checkpoint values are kept) |
| `Mission2/configs/sac_config.yaml` | only `ent_coef_max: 0.02` as a safety cap (your values kept) |
| `README.md`, `LAPORAN_PERUBAHAN.md` | new section "Perbaikan keruntuhan Misi 3" with the analysis and test results |

Then retrain **Mission 3 only**; your existing Mission 1 and 2 checkpoints stay as they are:

```
python Mission3/train_mission3.py
```

Back up `checkpoints/mission3/` first (it is overwritten). Mission 3 now trains roughly 3x faster per step.

## What went wrong (from your TensorBoard logs)

1. **Entropy coefficient runaway during warm-up (the bug).** While the actor was frozen (0-50k steps), SB3 kept
   tuning alpha. The transferred policy's entropy was below the target (-6), and a frozen actor cannot change that,
   so alpha grew from 0.033 to 6.8 (healthy in your Missions 1-2: about 0.007). The critic was trained against
   those inflated targets (critic loss 3 -> 33,600). The old code reset alpha when warm-up ended, but the broken
   critic was still handed to the actor: success fell from 63% to 0% within 10k steps and never recovered.
   Note: your Mission 3 `best_model` was saved at 45k, while the actor was still frozen, so it is really your
   unchanged Mission 2 policy.
2. **Actor chasing critic errors (found while testing the fix).** With alpha fixed but nothing holding the actor,
   the critic's Q estimates inflated (26 -> 71) while real success fell (90% -> 10%). Fewer gradient updates per
   step and a permanent small anchor to the Mission 2 policy fixed this.

## Test results

Run from your own Mission 2 checkpoint with the real Mission 3 config (4 envs, 250k buffer, 50k warm-up), 350k steps:

| | Before (your run) | After |
|---|---|---|
| Eval success after warm-up (60k / 100k / 200k) | 0% / 0% / 0% | 80% / 80% / 90% |
| Curriculum level reached | 3 (stuck) | 4, the final level (at 157k) |
| 100 new level-4 scenarios, best model | 50% (= unchanged Mission 2 policy) | **64%** |
| 100 new level-4 scenarios, final model | **0%** | **64%** |
| Entropy coefficient / critic loss | up to 10.5 / 33,600 | 0.005 / below ~13 |

The in-training evals (20 scenarios) are optimistic; the 100-scenario numbers are the realistic ones.
The main remaining failure is obstacle collisions (about 20%).

Not verified: a full 1.75M-step run, other seeds.

## What to watch in TensorBoard

- `train/ent_coef` stays at 0.005
- `train/critic_loss` stays in the single/low double digits (it was in the thousands before)
- `train/q_pi_mean` must not keep rising while `rollout/success_rate` falls (that is the critic-error symptom)
- `train/anchor_mse` ideally stays below about 0.05
- `curriculum/level` should reach 4

`best_model` (the one exported to ONNX) is chosen by curriculum level first, so a late dip does not overwrite a good result.
