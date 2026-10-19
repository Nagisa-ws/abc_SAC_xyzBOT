"""
Mission5/debug/render_harvest.py
--------------------------------
Rekam video panen (tanpa layar, render offscreen MuJoCo) -> checkpoints/mission5/videos/*.mp4 (+ .gif kecil)

  * kiri  : kamera luar (mengikuti tanaman)
  * kanan : kamera pergelangan (wrist_cam di link 6) -- gambaran apa yang dilihat kamera biasa pengguna

Jalankan:
    python Mission5/debug/render_harvest.py                 # policy ONNX Misi 5 terbaru, 3 skenario
    python Mission5/debug/render_harvest.py --reference     # controller referensi saja (tanpa RL)
Di Linux tanpa layar, butuh EGL (default di sini) atau OSMesa: MUJOCO_GL=egl / osmesa.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mujoco
import numpy as np


def render_episode(env, predict, seed: int, out_path: Path, fps: int = 25, size=(480, 640), max_frames: int = 900):
    import imageio.v2 as imageio
    obs, info0 = env.reset(seed=seed)
    h, w = size
    r_out = mujoco.Renderer(env.model, h, w)
    r_cam = mujoco.Renderer(env.model, h, w)
    cam = mujoco.MjvCamera()
    lay = env.pool.layout
    cam.lookat[:] = lay.fruit_center
    cam.distance, cam.azimuth, cam.elevation = 1.7, 205.0, -22.0
    opt = mujoco.MjvOption()
    frames, done, info = [], False, info0
    every = max(1, int(round(1.0 / (fps * env.model.opt.timestep * env.n_substeps))))
    counter = [0]

    def grab(force=False):
        if force or counter[0] % every == 0:
            r_out.update_scene(env.data, cam, opt)
            r_cam.update_scene(env.data, "wrist_cam")
            frames.append(np.concatenate([r_out.render(), r_cam.render()], axis=1))
        counter[0] += 1

    env.frame_callback = grab                 # langkah internal (jari menutup, twist) ikut terekam
    while not done and len(frames) < max_frames:
        obs, r, te, tr, info = env.step(predict(obs))
        done = te or tr
        grab(force=done)
    env.frame_callback = None
    for _ in range(fps):                       # tahan frame terakhir 1 s
        frames.append(frames[-1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=fps, macro_block_size=8)
    gif = out_path.with_suffix(".gif")
    imageio.mimsave(str(gif), [f[::2, ::2] for f in frames[::2]], duration=2.0 / fps, loop=0)
    r_out.close()
    r_cam.close()
    return info.get("termination_reason"), info0


def main():
    from Mission5.envs.stage5_harvest_env import Stage5HarvestEnv
    from shared.evaluation.task_spec import apply_final_task
    from shared.harvest.report import onnx_predict
    use_ref = "--reference" in sys.argv
    env = Stage5HarvestEnv(training=False, randomize=False, seed=3)
    apply_final_task(env, _ROOT, 5)
    if use_ref:
        predict, tag = (lambda o: np.zeros(6)), "referensi"
    else:
        onnx = sorted((_ROOT / "models").glob("onnx_mission5_v*.onnx"), key=lambda p: int(p.stem.rsplit("_v", 1)[1]))[-1]
        predict, tag = onnx_predict(str(onnx)), "rl"
    out = _ROOT / "checkpoints" / "mission5" / "videos"
    seeds = [int(a) for a in sys.argv[1:] if a.isdigit()] or [700001, 700002, 700003]
    for s in seeds:
        reason, info0 = render_episode(env, predict, s, out / f"harvest_{tag}_seed{s}.mp4")
        print(f"[video] seed {s}: {reason} | mode {info0.get('grip_mode')} dari {info0.get('approach_face')} | "
              f"sisi terhalang: {info0.get('blocked_sides') or '-'} -> {out / f'harvest_{tag}_seed{s}.mp4'}")


if __name__ == "__main__":
    main()
