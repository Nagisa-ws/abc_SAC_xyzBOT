"""
Mission4/debug/harvest_report.py
--------------------------------
Debug Misi 4 (dipanggil otomatis oleh train_mission4.py setelah training):
  policy ONNX vs controller referensi saja pada 100 skenario uji yang sama (tugas akhir training),
  rincian per jumlah sisi terhalang daun, per mode/face, gaya daun -> harvest_report.json / .png

Standalone:  python Mission4/debug/harvest_report.py   (memakai ONNX Misi 4 terbaru)
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.evaluation.task_spec import apply_final_task
from shared.harvest.report import compare

MISSION = 4


def run(ctx, onnx_path: str, out_dir: str, n: int = 100, log=print):
    return compare(lambda: ctx.make_raw_env(training=False, randomize=False, seed=11), onnx_path, out_dir, n=n,
                   prepare=lambda e: apply_final_task(e, ctx.project_root, MISSION, sac_cfg=ctx.cfg), log=log)


def main():
    from Mission4.envs.stage4_harvest_env import Stage4HarvestEnv
    onnx = sorted((_ROOT / "models").glob(f"onnx_mission{MISSION}_v*.onnx"),
                  key=lambda p: int(p.stem.rsplit("_v", 1)[1]))[-1]
    out = _ROOT / "checkpoints" / f"mission{MISSION}" / "debug"
    compare(lambda: Stage4HarvestEnv(training=False, randomize=False, seed=11), str(onnx), str(out),
            prepare=lambda e: apply_final_task(e, _ROOT, MISSION))


if __name__ == "__main__":
    main()
