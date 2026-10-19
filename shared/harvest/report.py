"""
shared/harvest/report.py
------------------------
Laporan Misi 4-5: policy (ONNX) vs controller referensi saja, pada skenario uji yang SAMA (seed tetap),
pada tugas akhir training (level terakhir kurikulum). Rincian per jumlah sisi terhalang daun, per mode/face
cengkeram, gaya daun, dan alasan gagal. Hasil -> JSON + PNG.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def run_episodes(env, predict, n: int, seed0: int, max_steps: int | None = None) -> list[dict]:
    rows = []
    for i in range(n):
        obs, info0 = env.reset(seed=seed0 + i)
        steps, leaf_sum, done = 0, 0.0, False
        info = info0
        while not done:
            obs, r, te, tr, info = env.step(predict(obs))
            steps += 1
            leaf_sum += float(info.get("leaf_force", 0.0))
            done = te or tr or (max_steps is not None and steps >= max_steps)
        rows.append({
            "seed": seed0 + i, "reason": info.get("termination_reason"), "success": info.get("termination_reason") == "success",
            "steps": int(info.get("steps_total", steps)), "n_blocked": int(info0.get("n_blocked", 0)),
            "blocked_sides": info0.get("blocked_sides", ""), "grip_mode": info0.get("grip_mode"),
            "face": info0.get("approach_face"), "line_clearance": float(info0.get("line_clearance", np.nan)),
            "leaf_force_max": float(info.get("leaf_force_max", 0.0)), "leaf_force_mean": leaf_sum / max(steps, 1),
            "scenario_ok": bool(info0.get("scenario_ok", True)), "phase": int(info.get("phase", 0)),
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    k = sum(r["success"] for r in rows)
    out = {"n": n, "success_rate": k / max(n, 1), "ci95": _wilson(k, n),
           "reasons": dict(Counter(r["reason"] for r in rows)),
           "mean_steps_success": float(np.mean([r["steps"] for r in rows if r["success"]])) if k else None,
           "leaf_force_max_median": float(np.median([r["leaf_force_max"] for r in rows])) if n else None,
           "leaf_force_mean_mean": float(np.mean([r["leaf_force_mean"] for r in rows])) if n else None}
    by_b = defaultdict(list)
    for r in rows:
        by_b[r["n_blocked"]].append(r["success"])
    out["by_blocked_sides"] = {str(b): {"n": len(v), "success_rate": float(np.mean(v))} for b, v in sorted(by_b.items())}
    by_m = defaultdict(list)
    for r in rows:
        by_m[f"{r['grip_mode']}/{r['face']}"].append(r["success"])
    out["by_mode_face"] = {kk: {"n": len(v), "success_rate": float(np.mean(v))} for kk, v in sorted(by_m.items())}
    return out


def onnx_predict(onnx_path: str):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return lambda o: sess.run(["action"], {"observation": o[None, :].astype(np.float32)})[0][0]


def compare(env_factory, onnx_path: str, out_dir: str, n: int = 100, seed0: int = 600_000,
            prepare=None, log=print, tag: str = "") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, fn in (("referensi_saja", lambda o: np.zeros(6)), ("referensi_plus_RL", onnx_predict(onnx_path))):
        env = env_factory()
        if prepare is not None:
            prepare(env)
        rows = run_episodes(env, fn, n, seed0)
        res[name] = {"summary": summarize(rows), "episodes": rows}
        s = res[name]["summary"]
        log(f"[laporan{tag}] {name:18s}: sukses {s['success_rate']:.1%} (CI95 {s['ci95'][0]:.0%}-{s['ci95'][1]:.0%}) "
            f"| gaya daun maks median {s['leaf_force_max_median']:.1f} N | {s['reasons']}")
    (out / "harvest_report.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for j, (name, r) in enumerate(res.items()):
            b = r["summary"]["by_blocked_sides"]
            xs = [int(k) for k in b]
            ax[0].bar(np.array(xs) + (j - 0.5) * 0.38, [b[str(x)]["success_rate"] for x in xs], 0.38, label=name)
            lf = [e["leaf_force_max"] for e in r["episodes"]]
            ax[1].hist(np.clip(lf, 0, 150), bins=30, alpha=0.6, label=name)
        ax[0].set_xlabel("jumlah sisi terhalang daun (atas/depan/kiri/kanan)")
        ax[0].set_ylabel("success rate")
        ax[0].set_ylim(0, 1.05)
        ax[0].legend()
        ax[1].set_xlabel("gaya daun maksimum per episode (N, dipotong 150)")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(out / "harvest_report.png", dpi=110)
        plt.close(fig)
    except Exception as e:                       # plot tidak boleh menggagalkan laporan
        log(f"[laporan] plot gagal: {e}")
    return res
