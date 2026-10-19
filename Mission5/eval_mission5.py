"""
Mission5/eval_mission5.py
-------------------------
Evaluasi Misi 5 (panen lengkap) dengan MuJoCo viewer, memakai policy ONNX terbaru.

Jalankan (tanpa argumen):
    python eval_mission5.py            # dari folder Mission5/
    python Mission5/eval_mission5.py   # dari root proyek

Semua pengaturan ada di configs/eval_config.yaml. Di macOS, jalankan dengan 'mjpython' (syarat viewer).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mission5.eval_envs import Stage5HarvestEvalEnv
from shared.evaluation import run_evaluation


def main():
    return run_evaluation(mission_id=5, project_root=ROOT,
                          eval_cfg_path=str(ROOT / "Mission5" / "configs" / "eval_config.yaml"),
                          env_cls=Stage5HarvestEvalEnv)


if __name__ == "__main__":
    main()
