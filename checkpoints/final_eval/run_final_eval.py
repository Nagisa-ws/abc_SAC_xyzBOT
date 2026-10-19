"""Evaluasi akhir Misi 4/5: referensi saja vs referensi+RL pada 200 skenario uji yang sama (seed 600000+),
dgn & tanpa kesalahan persepsi perencana, dan dgn domain randomization + derau sensor.
python checkpoints/final_eval/run_final_eval.py 4|5"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from shared.evaluation.task_spec import apply_final_task
from shared.harvest.report import onnx_predict, run_episodes, summarize

M = int(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
if M == 4:
    from Mission4.envs.stage4_harvest_env import Stage4HarvestEnv as Env
else:
    from Mission5.envs.stage5_harvest_env import Stage5HarvestEnv as Env
onnx = sorted((ROOT / "models").glob(f"onnx_mission{M}_v*.onnx"), key=lambda p: int(p.stem.rsplit("_v", 1)[1]))[-1]
rl = onnx_predict(str(onnx))
out = {"onnx": onnx.name, "n": N, "seed0": 600000}
for cond in ("tugas_akhir", "rencana_sempurna", "dr_derau"):
    for name, fn in (("referensi_saja", lambda o: np.zeros(6)), ("referensi_plus_RL", rl)):
        kw = dict(training=False, randomize=(cond == "dr_derau"), seed=11)
        if cond == "dr_derau":
            kw["obs_noise_std"] = 0.005
        env = Env(**kw)
        apply_final_task(env, ROOT, M)
        if cond == "rencana_sempurna":
            env.cfg["perception"] = {"leaf_miss_prob": 0.0, "leaf_pos_noise": 0.0}
        rows = run_episodes(env, fn, N, 600000)
        s = summarize(rows)
        out[f"{cond}/{name}"] = {"summary": s, "episodes": rows}
        print(f"[M{M}] {cond:16s} {name:18s} sukses {s['success_rate']:.1%} CI {s['ci95'][0]:.0%}-{s['ci95'][1]:.0%} "
              f"langkah(sukses) {s['mean_steps_success']} {s['reasons']}", flush=True)
(ROOT / "checkpoints" / "final_eval_user" / f"mission{M}_final_eval.json").write_text(json.dumps(out, indent=1, default=float))
