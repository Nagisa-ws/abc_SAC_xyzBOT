"""
shared/evaluation/task_spec.py
------------------------------
SATU definisi "tugas akhir" per misi, dipakai oleh evaluasi, benchmark, uji held-out akhir training, dan
debug pasca-training. Sumbernya Mission{N}/configs/sac_config.yaml (konfigurasi TRAINING), sehingga
evaluasi selalu mengukur tugas yang sama dengan level terakhir kurikulum training.

Masalah sebelumnya (terukur pada model pengguna, 200 skenario held-out):
  * Misi 1: eval_config memakai ambang 5 mm, kurikulum training berhenti di 2 cm -> 62 % vs 100 %.
  * Misi 3: env evaluasi memakai margin rintangan 1 cm (level 4 lama), padahal kurikulum pengguna berakhir
    di 3 cm (level 4 dihapus) -> 77,5 % vs 82 %.

Aturan:
  Misi 1 : ambang sukses = adaptive.thresholds[-1] (bila adaptive aktif), selain itu episode.success_pos_threshold
  Misi 2 : ambang dari env config (episode.success_pos_threshold)
  Misi 3-5: ambang dari env config + kesulitan = curriculum.levels[-1] (bila kurikulum aktif)
  Override eksplisit (eval_config.rollout.success_pos_threshold / difficulty_level) tetap dihormati,
  tetapi dicetak jelas sebagai "BUKAN tugas training".
"""
from __future__ import annotations

from pathlib import Path

from shared.config_utils import load_yaml


def training_config_path(project_root: Path, mission_id: int) -> Path:
    return Path(project_root) / f"Mission{mission_id}" / "configs" / "sac_config.yaml"


def final_task(project_root: Path, mission_id: int, sac_cfg: dict | None = None) -> dict:
    """Return {'success_threshold': float|None, 'difficulty': (level, dict)|None}. None = ikut env config."""
    cfg = sac_cfg if sac_cfg is not None else load_yaml(training_config_path(project_root, mission_id))
    spec = {"success_threshold": None, "difficulty": None}
    if mission_id == 1:
        a = cfg.get("adaptive", {}) or {}
        if a.get("enabled", True) and a.get("thresholds"):
            spec["success_threshold"] = float(a["thresholds"][-1])
    if mission_id >= 3:
        c = cfg.get("curriculum", {}) or {}
        if c.get("enabled", False) and c.get("levels"):
            spec["difficulty"] = (len(c["levels"]) - 1, {k: float(v) for k, v in c["levels"][-1].items()})
    return spec


def apply_final_task(env, project_root: Path, mission_id: int, override_threshold: float | None = None,
                     override_level: int | None = None, sac_cfg: dict | None = None, log=None) -> str:
    """Terapkan tugas akhir ke env (Misi 1/2/3). Return deskripsi satu baris."""
    cfg = sac_cfg if sac_cfg is not None else load_yaml(training_config_path(project_root, mission_id))
    spec = final_task(project_root, mission_id, cfg)
    parts, mismatch = [], []
    thr = spec["success_threshold"]
    if override_threshold is not None:
        if thr is None or abs(float(override_threshold) - thr) > 1e-12:
            mismatch.append(f"ambang {float(override_threshold) * 1000:.1f} mm")
        thr = float(override_threshold)
    if thr is not None:
        if hasattr(env, "set_curriculum"):
            env.set_curriculum(thr, 1.0)
        else:
            env.set_success_threshold(thr)
    parts.append(f"ambang sukses {env.get_success_threshold() * 1000:.1f} mm")

    if mission_id >= 3 and hasattr(env, "set_difficulty"):
        diff = spec["difficulty"]
        if override_level is not None:
            levels = (cfg.get("curriculum", {}) or {}).get("levels", [])
            lvl = int(override_level)
            if diff is None or lvl != diff[0]:
                mismatch.append(f"level kurikulum {lvl}")
            diff = (lvl, {k: float(v) for k, v in levels[lvl].items()})
        if diff is not None:
            env.set_difficulty(diff[0], **diff[1])
            if "obstacle_margin" in diff[1]:
                parts.append(f"level kurikulum {diff[0]} (margin lintasan {diff[1]['obstacle_margin'] * 100:.0f} cm, "
                             f"pendekatan {diff[1]['approach_margin'] * 100:.0f} cm)")
            else:
                parts.append(f"level kurikulum {diff[0]} (" + ", ".join(f"{k}={v:g}" for k, v in diff[1].items()) + ")")
    desc = ", ".join(parts)
    if getattr(env, "residual", False):
        desc += " | struktur policy: RESIDUAL (controller referensi + koreksi RL)"
    if mismatch:
        desc += " | PERHATIAN: override " + ", ".join(mismatch) + " = BUKAN tugas akhir training"
    if log:
        log(f"[tugas] Misi {mission_id}: {desc}")
    return desc
