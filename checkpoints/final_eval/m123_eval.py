"""Misi 1-3: referensi saja vs referensi+RL (ONNX pengguna) pada 200 skenario uji (seed 600000+), tugas akhir training."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from shared.evaluation.task_spec import apply_final_task
from shared.harvest.report import onnx_predict, run_episodes, summarize
from Mission1.envs.stage1_reach_env import Stage1ReachEnv
from Mission2.envs.stage2_path_env import Stage2PathEnv
from Mission3.envs.stage3_obstacle_env import Stage3ObstacleEnv
out = {}
for M, Env in ((1, Stage1ReachEnv), (2, Stage2PathEnv), (3, Stage3ObstacleEnv)):
    rl = onnx_predict(str(ROOT / "models" / f"onnx_mission{M}_v1.onnx"))
    for cond in ("tugas_akhir", "dr_derau"):
        for name, fn in (("referensi_saja", lambda o: np.zeros(6)), ("referensi_plus_RL", rl)):
            kw = dict(training=False, randomize=(cond == "dr_derau"), seed=11)
            if cond == "dr_derau":
                kw["obs_noise_std"] = 0.005
            env = Env(**kw)
            apply_final_task(env, ROOT, M)
            rows = run_episodes(env, fn, 200, 600000)
            s = summarize(rows)
            out[f"M{M}/{cond}/{name}"] = s
            print(f"[M{M}] {cond:12s} {name:18s} {s['success_rate']:.1%} CI {s['ci95'][0]:.0%}-{s['ci95'][1]:.0%} steps {s['mean_steps_success']} {s['reasons']}", flush=True)
(ROOT / "checkpoints" / "final_eval_user" / "mission123_eval.json").write_text(json.dumps(out, indent=1, default=float))
