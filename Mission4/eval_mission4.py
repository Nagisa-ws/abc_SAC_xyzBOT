"""
Mission4/eval_mission4.py
-------------------------
Evaluasi Misi 4 (mencapai pose cengkeram menembus daun) dengan MuJoCo viewer, memakai policy ONNX terbaru.

Jalankan (tanpa argumen):
    python eval_mission4.py            # dari folder Mission4/
    python Mission4/eval_mission4.py   # dari root proyek

Semua pengaturan ada di configs/eval_config.yaml. Di macOS, jalankan dengan 'mjpython' (syarat viewer).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mission4.eval_envs import Stage4HarvestEvalEnv
from shared.evaluation import run_evaluation


def main():
    return run_evaluation(mission_id=4, project_root=ROOT,
                          eval_cfg_path=str(ROOT / "Mission4" / "configs" / "eval_config.yaml"),
                          env_cls=Stage4HarvestEvalEnv)


if __name__ == "__main__":
    main()
