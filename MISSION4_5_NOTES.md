# Missions 4–5: pineapple harvesting in simulation (English notes)

Missions 4 and 5 continue the curriculum from your trained Mission 3 model. **Simulation only:** the robot knows where
the plant is, so there is no camera perception yet (there is a simple model of planning errors, see *Perception error*).

| Mission | Task | Who controls what |
|---|---|---|
| 4 | Reach the grasp pose (crown neck or fruit body) **through the leaves** | Reference controller follows your stargaze line; SAC adds a bounded correction |
| 5 | Full harvest: reach → grip → twist → pull back → lift out of the leaves | Reach and pull-out: reference + SAC. Grip and twist: scripted |

Results: see [Results](#results). Short version: the planner + reference controller does the work; the RL correction
tied it in Mission 4 and was worse in Mission 5.

## Files

```
robot_model/build_harvest_model.py   generates the two XML files below (run again only if you change gripper/pool sizes)
robot_model/ur10e_harvest.xml        UR10e (arm identical to ur10e_train.xml) + PGIA-140-80 gripper + wrist camera
robot_model/scene_harvest.xml        floor + one pineapple plant "pool" (14 rosette, 8 blocking, 4 crown leaves)
shared/harvest/plant.py              random plant layout, writes it into MuJoCo each episode, leaf points & bending
shared/harvest/planner.py            your stargaze max-clearance line applied to leaves; gripper fit; grasp orientation
shared/harvest/checker.py            route validation (IK, self/floor collision, wrist singularity, hard plant parts)
shared/harvest/reference.py          reference controller with orientation (rotates the gripper to the grasp pose)
shared/harvest/report.py             reference-only vs reference+RL on the same test scenes, with breakdowns
Mission4/  envs, eval_envs, configs, debug/harvest_report.py, train_mission4.py, eval_mission4.py
Mission5/  envs, eval_envs, configs, debug/harvest_report.py, debug/render_harvest.py (video), train/eval scripts
```

## The simulated plant

Every episode draws a new plant (all ranges in `Mission4/configs/domain_randomization.yaml`, block `plant`):

- stalk, fruit (diameter 8–14 cm, height 14–24 cm, centre 30–60 cm above the robot base plane), crown neck (3–5.5 cm);
- 10–14 **rosette leaves** from the base (8–14 in the earlier curriculum levels): 45–90 cm long, steep then curving outward;
- 4 short **crown leaves**;
- **blocking leaves** (your request): on each side, **top / front / left / right**, a leaf crossing that side appears
  with probability `p_block` (0.6 at the final level), up to 2 per side, and with probability `p_all_sides` (0.25) all
  four sides are blocked at once. In the 200 final-task test scenes: 34.5% have all 4 sides blocked, 29.5% three, 25% two.
  "Front" = the side facing the robot/vehicle.
- leaves are chains of capsules with **spring hinges**: they bend when pushed and spring back. Stiffness is random
  (1–10 N·m/rad per hinge) because the real value is unknown.
- leaves that would start inside the robot body are removed (the vehicle has just arrived beside the plant).

**All sizes are guesses.** Measure 15–20 real plants and replace the ranges; heights are measured from the robot
base plane, so if the base sits higher than the ground on the vehicle, lower `center_height_range` accordingly.

## Approach line = your stargaze algorithm

`shared/harvest/planner.py` applies your `stargaze.py` concept to the leaves:

- nucleus = grasp point (crown neck centre, or fruit centre for a body grip);
- cube faces **top, front, left, right** (back dropped as the opposite of the initial "front" face, bottom forbidden),
  cube half-size 30 cm (your 15/30/45 cm layers → outer face 30 cm here), 21 × 21 candidate points per face;
- score = distance from the leaf points to the **segment** S → nucleus minus the leaf radius; the best line per face
  is kept and the faces are tried in order of clearance.

Grip mode follows the line direction: crown from above (elevation ≥ 55°) or almost horizontally (−5° … 20°),
fruit body from the side (−10° … 30°). A mode is only used if the part fits between the fingers (next section).
In the 200 final-task test scenes: 45.5% approached from the top, 32% from the front, 22.5% left/right; 4.5% are body grips.

## Gripper (PGIA-140-80 approximation)

From the public datasheet: 80 mm stroke (40 mm per finger), 40–140 N per finger, 95 × 67 × 86 mm body, about 1 kg,
about 0.7 s to close. **Finger gap = [2·offset, 2·offset + 80 mm]**, where `gripper.finger_offset` is how far each
finger is mounted from the centre:

- default `finger_offset: 0.015` → 30–110 mm: the crown neck always fits; the fruit body only when its diameter is
  ≤ ~86 mm (about 10% of the simulated fruits).
- `0.035` → 70–150 mm: fruit bodies fit, but the crown neck no longer does.

With 80 mm stroke, one finger set cannot grip both a 3–5 cm neck and a 10–15 cm fruit. This is a hardware decision.

The wrist camera (`wrist_cam`) is mounted on the gripper bracket (it rotates with link 6) and looks along the approach
direction. It is a collision body too. `Mission5/debug/render_harvest.py` records its view next to an outside view.

## Control structure

Same idea as v3 of Missions 1–3: **action = reference action + bounded SAC correction**. The reference follows the
validated route to S, rotates the gripper to the grasp orientation, then follows the line to the grasp point. Two
limits were added after the first training run made things worse (see *What changed*):

- **progress floor** (`progress_min: 0.4`): the correction can steer around a leaf and change speed, but the motion
  along the reference direction never drops below 40%, so the arm cannot stop in front of the leaves;
- **target gate** (`target_gate: [0.04, 0.10]`): the correction fades out over the last 10 → 4 cm before the grasp
  point; the reference places the fingers.

Observation: the same 63 values as Missions 1–3. The "obstacle" fields now hold the nearest **leaf** points (moving
with the leaves), `ref_action` includes rotation, and the 4th obstacle slot carries the **wrist force** (contact force
on the gripper, what the UR10e's built-in force/torque sensor measures, without inertia). Those 4 inputs and the 3
rotation inputs that were always zero in Mission 3 start with zero weights; the rest of your Mission 3 network is reused.

## Success and failure

| Outcome | Meaning |
|---|---|
| success (M4) | TCP < 1.5 cm from the grasp point and gripper < 12° from the grasp orientation |
| success (M5) | fruit detached and carried to the exit point (pulled back 22 cm along the line, then up 8 cm) |
| leaf_overbend | a leaf hinge bent more than 30° (risk of tearing; **a guess**). Crown leaves don't count: the fingers must press them when gripping the neck, as a hand picker does (5 of 11 reference failures were these) |
| fruit_hit / plant_hit | fingers pressed the crown/fruit > 25 N, or the arm/palm touched the stalk/fruit/neck > 3 N |
| self_collision / floor_contact | as in Missions 1–3 |
| grasp_miss (M5) | after closing, not both fingers touch the crown/fruit |
| twist_fail (M5) | the twist motion is not reachable / collides (mainly side grips: the wrist must swing round the crown) |
| grip_slip (M5) | grip torque `2·μ·F·r` or pull force `2·μ·F` below the stalk's break torque/force (random) |
| fruit_hit_robot / fruit_floor (M5) | the carried fruit hits the arm or the ground |
| timeout | not finished in time |

Leaf damage is measured by **bending**, not by contact force: at 0.3–0.5 m/s the first contact force jumped from
0 to 150–280 N in a single step (an impact spike), which nothing can anticipate and which says little about damage.
The contact force is still reported. Reward: −0.15 × (leaf spring energy / 0.1 J, capped at 3) per step, −20 at timeout.

## Mission 5 is partly modelled by rules (honest note)

- **Grip:** the fingers really close in the physics (force-limited), and the arm shifts sideways while closing so both
  fingers touch (a real UR10e can do this with force mode). Then the fruit is welded to the gripper.
- **Detaching:** not simulated as a breaking stalk. The twist is a real motion (rotation about the fruit axis by a
  random 45–120°, with IK, collisions and leaves). After it, the stalk counts as broken if grip torque ≥ break torque
  (random 0.5–2.5 N·m) and grip pull force ≥ break force (random 20–80 N), with friction μ random 0.4–0.8.
  **These numbers are unknown for real pineapples; measure them.**
- **Not modelled:** twisting the crown may pull the crown off the fruit instead of breaking the stalk. Test this on real
  plants before relying on crown grips.

## Perception error

The planner uses leaf positions "seen" from a look pose: 30% of the leaves are missed and the rest have 2 cm noise
(`perception` block). During the motion, the policy observes the actual leaves nearest to the robot (the wrist
camera sees them up close). Measured with the reference controller alone on the final 200 test scenes: **92.5%** (Mission 4) with a perfect plan,
**83%** with this planning error (an earlier 60-scene check with the old crown-leaf rule gave 93% / 77%). This gap is
where an RL correction would have real work to do.
Set both values to 0 for a perfect plan; replace them with your camera's real detection rate later.

## What changed during development (evidence)

1. **Wrist rotation outrunning the arm.** The target orientation ran up to 70° ahead of the real gripper (wrist joints
   move ~0.05 rad per step); IK solutions jumped and the TCP dropped 50 cm onto the floor although the command was
   upward. Fix: target orientation at most 8° ahead of the actual one; translation slows while the wrist catches up.
2. **Wrist singularity.** Routes passing wrist_2 ≈ ±180° flipped wrist 1/3 by ±0.25 rad per step and caused
   self-collisions. Fix: routes within 15° of the singularity or with joint jumps > 0.6 rad are rejected, and both
   180°-symmetric gripper orientations are tried.
3. **Missed contacts.** MuJoCo's per-body collision tree is built once; the plant geometry changes every episode, so
   finger–crown contacts were sometimes missed (fingers closed straight through the neck). Fix: midphase disabled.
4. **Leaf force spikes** → bending criterion (above).
5. **Four Mission 4 training runs** (each 300k steps, reference vs reference + RL on the same 100 scenes):

   | Run | Change | Reference alone | Reference + RL |
   |---|---|---|---|
   | 1 | perfect plan, free correction | 93% | 86% (stops in front of leaves: timeouts 2 → 6; fingers nudged into the crown: 1 → 3) |
   | 2 | + progress floor, target gate 15 → 6 cm, smaller leaf penalty, timeout penalty | stopped at 41%: eval 86–90% vs 92% | |
   | 3 | + planning error (30% leaves missed, 2 cm noise) | 77% | 78% |
   | 4 (shipped) | + wrist force input, crown-leaf rule, gate 10 → 4 cm | 83% | 84% |

   Run 2 was stopped early because it showed the same pattern; run 3 used the old crown-leaf rule, so its numbers are
   not directly comparable with run 4.

## Results

Same 200 unseen scenes for both columns (seeds 600000+), final curriculum level. Raw data:
`checkpoints/final_eval/mission4_final_eval.json` (script: `checkpoints/final_eval/run_final_eval.py`).

**Mission 4 (reach the grasp pose through the leaves)**

| Condition | Reference alone | Reference + RL |
|---|---|---|
| Final task (with planning error) | 83.0% | 85.0% |
| Perfect plan (all leaves known) | 92.5% | 92.0% |
| Domain randomisation + sensor noise | 85.0% | 84.5% |

- Paired comparison, final task: RL succeeded where the reference failed in 6 scenes, the reverse in 2 (p = 0.29).
  The other conditions: 3 vs 4. **No measurable difference.**
- Remaining failures (final task, RL): leaf overbend 11 (reference 16), timeout 14, plant/fruit touch 5.
- Training's own held-out test (200 other scenes): 82.5%.

**Plainly:** in this simulation the stargaze plan plus the reference controller does almost all of the work, as in
Mission 3. The RL correction did not add a measurable improvement in four attempts. The most useful lever was the
plan itself: knowing all leaves raises the reference from 83% to 92.5%, so better leaf detection matters more than RL here.

**Mission 5 (full harvest: reach, grip, twist, pull back, lift out)**

| Condition | Reference alone | Reference + RL |
|---|---|---|
| Final task (with planning error) | 61.0% | 58.5% |
| Perfect plan (all leaves known) | 70.0% | 60.0% |
| Domain randomisation + sensor noise | 58.5% | 53.0% |

- Paired comparison: final task 11 vs 16 scenes (p = 0.44); perfect plan 9 vs 29 (**p = 0.002, RL worse**); DR + noise
  19 vs 30 (p = 0.15). The Mission 5 RL policy pushes the fingers into the crown more often (fruit_hit 1 → 12 and
  grasp_miss 12 → 22 with a perfect plan). **Use the reference controller alone for Mission 5** (zero RL action;
  `python Mission5/debug/harvest_report.py` shows both).
- Where the reference fails (final task, 78 of 200): almost all **before the fruit is detached**: leaf overbend 26
  (21 while reaching/twisting at the crown, 4 on the way to the line, 1 while pulling back), grasp_miss 15, twist_fail 14 (mostly side grips: the wrist must swing
  round the crown), timeout 13, grip_slip 5, plant/fruit touch 5. **Pulling the fruit out failed only once** (a leaf overbend).
- By approach side (reference): top 71% (91 scenes), left 58%, right 52%, front 50%.
- Training's own held-out test: 59.5% (200 other scenes).

**What this means for the real robot**

1. The plan matters most: knowing all leaves gives +9.5 points (M4) and +9 points (M5). Invest in leaf detection.
2. Top approaches to the crown work best; side grips lose most in the twist. Consider a wrist with a larger twist range,
   or detaching side-gripped fruit by bending instead of twisting (not modelled here).
3. The scripted grasp/twist rules use guessed physics; measure stalk break torque/force and finger friction.
4. RL as built here does not improve on the planner + reference. Treat it as an experiment, not the main controller.

## How to run

```
python Mission4/train_mission4.py     # needs checkpoints/mission3/best_model.zip (your Mission 3 model is included)
python Mission5/train_mission5.py     # needs checkpoints/mission4/best_model.zip
python Mission4/eval_mission4.py      # MuJoCo viewer (settings in Mission4/configs/eval_config.yaml)
python Mission5/eval_mission5.py
python Mission4/debug/harvest_report.py        # reference alone vs reference + RL, 100 scenes
python Mission5/debug/render_harvest.py        # videos -> checkpoints/mission5/videos/
```

Training time on the 2-CPU machine used here: Mission 4 73 min (~70 steps/s), Mission 5 91 min (~56 steps/s), plus
~15 minutes of final validation, test and report each. Colab's free CPU is similar.

## Before a real robot (not done here)

- Perception: 3D positions of fruit, crown and leaves from your camera (multi-view from the wrist camera, or a
  depth camera); measure how many leaves it misses and set `perception`.
- Measure plant sizes, leaf stiffness, stalk break torque/force, finger friction; replace the guessed ranges.
- Decide the fingers (crown vs body) and test whether a crown twist detaches the fruit or only the crown.
- Same 63-value observation builder on the robot, with the same normalisation and the reference controller.
- Speed: the simulation moves up to 0.45 m/s near the plant; start much slower on the real arm, with the UR safety
  limits, an emergency stop, and nobody in the workspace.
