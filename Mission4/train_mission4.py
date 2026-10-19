"""
Mission4/train_mission4.py
--------------------------
Misi 4: mencapai pose cengkeram buah nanas MENEMBUS DAUN (simulasi; posisi tanaman diketahui persis).

Bobot awal dimuat dari Misi 3 (blok 'transfer' di configs/sac_config.yaml), jadi jalankan training Misi 3
lebih dulu (checkpoints/mission3/best_model.zip).

Jalankan (tanpa argumen):
    python train_mission4.py            # dari folder Mission4/
    python Mission4/train_mission4.py   # dari root proyek
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mission4.debug import harvest_report
from Mission4.envs.stage4_harvest_env import Stage4HarvestEnv, DEFAULT_CONFIG, DEFAULT_MODEL
from shared.debug_common import generic_post_training, run_all_sanity
from shared.training import DifficultyCurriculumCallback, MissionHooks, train_mission

MISSION = 4


def pre_debug(ctx):
    dbg = ctx.cfg["debug"]
    run_all_sanity(lambda: ctx.make_raw_env(training=False, randomize=False, seed=1),
                   dbg.get("sanity_n_ik_targets", 10), dbg.get("sanity_n_resets", 5),
                   dbg.get("sanity_n_scripted_episodes", 5))


def build_callbacks(ctx):
    """Kurikulum tanaman (blok `curriculum` di sac_config.yaml)."""
    c = ctx.cfg.get("curriculum", {}) or {}
    if not c.get("enabled", False):
        return []
    cb = DifficultyCurriculumCallback(
        levels=c["levels"], advance_rate=c.get("advance_success_rate", 0.80),
        check_every=c.get("check_every_n_episodes", 50), warmup=c.get("warmup_episodes", 100),
        window=c.get("window_episodes", 100), eval_env=ctx.eval_env,
        state_path=str(ctx.ckpt_dir / "curriculum_state.json"))
    ctx.level_fn = lambda: cb.level
    return [cb]


def post_debug(ctx, result):
    out = ctx.project_root / ctx.cfg["debug"]["output_dir"]
    generic_post_training(ctx, result, out)
    harvest_report.run(ctx, result["onnx_path"], str(out))


def main():
    return train_mission(
        mission_id=MISSION, project_root=ROOT,
        sac_cfg_path=str(ROOT / "Mission4" / "configs" / "sac_config.yaml"),
        env_cls=Stage4HarvestEnv, env_cfg_path=DEFAULT_CONFIG, model_path=DEFAULT_MODEL,
        hooks=MissionHooks(pre_debug=pre_debug, build_callbacks=build_callbacks, post_debug=post_debug))


if __name__ == "__main__":
    main()
