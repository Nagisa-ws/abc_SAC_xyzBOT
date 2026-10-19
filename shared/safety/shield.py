"""
shared/safety/shield.py
-----------------------
Safety filter (shield) satu-langkah untuk aksi Cartesian -> IK.

Masalah yang diatasi (terukur pada model pengguna, 200 skenario held-out):
  * Misi 3: 29 dari 36 kegagalan adalah KONTAK (self-collision 11, kontak lantai 9, rintangan 9);
    26 dari 36 terjadi di fase A (home -> T1), median di langkah 68, jauh dari T1 (level 3).
  * Misi 2: 11 dari 19 kegagalan adalah self-collision / kontak lantai, semuanya di fase A.
Policy hanya mengatur pose TCP; IK memilih konfigurasi sendi tanpa sadar tabrakan, sehingga policy tidak
bisa "melihat" bahwa sebuah gerakan TCP akan melipat lengan ke badan/lantai/rintangan.

Cara kerja (tiap langkah, sebelum perintah dikirim ke aktuator):
  1. Aksi policy a -> target sendi q_cmd lewat IK (fungsi env._command, TANPA mengubah state env).
  2. Prediksi: konfigurasi q_cmd (dan titik tengah q_now -> q_cmd) diperiksa pada SALINAN model MuJoCo
     dengan margin kontak diperbesar (`contact_margin`) + clearance kapsul ke rintangan (`obstacle_margin`).
  3. Bila aman -> a dipakai apa adanya (biaya: 2x kinematika + deteksi tabrakan).
     Bila tidak -> beberapa kandidat aksi dievaluasi (diskalakan, tanpa rotasi, geser per sumbu, diam);
     dipilih kandidat AMAN yang paling dekat dengan a. Bila tak ada yang aman: kandidat dengan margin
     terbesar (menjauh dari bahaya).
Pasangan body yang SUDAH berada dalam margin pada pose home (mis. base-shoulder yang selalu bersentuhan
<1 mm) diabaikan karena bukan bahaya nyata.

Filter adalah bagian dari ENV (bukan policy), jadi dipakai identik saat training dan evaluasi. Untuk robot
nyata, filter yang sama dijalankan dengan model kinematik robot (MuJoCo) di depan perintah aktuator.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from shared.envs.robot_geometry import obstacle_clearance


@dataclass
class ShieldConfig:
    enabled: bool = False
    contact_margin: float = 0.01      # m; kontak self/lantai diprediksi dalam jarak ini = tidak aman
    obstacle_margin: float = 0.015    # m; clearance kapsul-rintangan di bawah ini = tidak aman
    check_midpoint: bool = True       # periksa juga titik tengah q_now -> q_cmd
    penalty: float = 0.0              # reward -= penalty * ||a_exec - a||^2 (saat training dgn filter)

    @classmethod
    def from_dict(cls, d: dict | None) -> "ShieldConfig":
        d = dict(d or {})
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class SafetyShield:
    def __init__(self, env, cfg: ShieldConfig):
        self.env, self.cfg = env, cfg
        m = mujoco.MjModel.from_xml_path(env.model_path)
        col = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
        m.geom_margin[col] = float(cfg.contact_margin)
        self.m, self.d = m, mujoco.MjData(m)
        # pasangan body yang sudah "bersentuhan" dalam margin pada pose home -> diabaikan
        mujoco.mj_resetDataKeyframe(m, self.d, env.home_key_id)
        mujoco.mj_fwdPosition(m, self.d)
        self.ignore = {self._pair(c) for c in self.d.contact[: self.d.ncon]}
        self.n_calls = self.n_interventions = 0

    def _pair(self, c):
        b1, b2 = int(self.m.geom_bodyid[c.geom1]), int(self.m.geom_bodyid[c.geom2])
        return (min(b1, b2), max(b1, b2))

    # ---- skor keamanan konfigurasi (>= 0 aman; < 0 = seberapa dalam melanggar margin, meter) ----
    def score(self, q_arm: np.ndarray) -> float:
        env, m, d = self.env, self.m, self.d
        d.qpos[:] = env.data.qpos
        d.qpos[env.arm_qpos_adr] = q_arm
        mujoco.mj_fwdPosition(m, d)
        s = np.inf
        for c in d.contact[: d.ncon]:
            if self._pair(c) not in self.ignore:
                s = min(s, float(c.dist) - self.cfg.contact_margin)
        if env.obstacle_points.shape[0]:
            clear, _ = obstacle_clearance(env.caps, d, env.obstacle_points, env.obstacle_radii)
            s = min(s, float(np.min(clear)) - self.cfg.obstacle_margin)
        return s

    def _eval(self, action, q_now):
        cmd = self.env._command(action)
        s = self.score(cmd["q_cmd"])
        if self.cfg.check_midpoint and s >= 0:
            s = min(s, self.score(0.5 * (q_now + cmd["q_cmd"])))
        return s, cmd

    def _candidates(self, a: np.ndarray):
        pos, rot = a[:3], a[3:]
        out = [np.r_[pos * 0.5, rot * 0.5], np.r_[pos * 0.25, rot * 0.25], np.r_[pos, 0 * rot],
               np.r_[pos * 0.5, 0 * rot]]
        mag = max(float(np.linalg.norm(pos)), 0.3)
        for ax in range(3):
            for sgn in (1.0, -1.0):
                e = np.zeros(3); e[ax] = sgn
                out.append(np.r_[np.clip(0.4 * pos + 0.6 * mag * e, -1, 1), 0 * rot])
        out.append(np.zeros(6))
        return out

    def filter(self, action: np.ndarray):
        """Return (aksi_dieksekusi, cmd, info)."""
        self.n_calls += 1
        q_now = self.env.data.qpos[self.env.arm_qpos_adr].copy()
        s0, cmd0 = self._eval(action, q_now)
        if s0 >= 0:
            return action, cmd0, {"shield_intervened": False, "shield_score": s0}
        self.n_interventions += 1
        best = (s0, 0.0, action, cmd0)                   # (skor, -jarak ke aksi asli, aksi, cmd)
        for c in self._candidates(action):
            s, cmd = self._eval(c, q_now)
            dist = float(np.sum((c - action) ** 2))
            key = (min(s, 0.0), -dist) if s < 0 else (0.0, -dist)
            if key > (min(best[0], 0.0), best[1]):
                best = (s, -dist, c, cmd)
        s, negd, a_exec, cmd = best
        return a_exec, cmd, {"shield_intervened": True, "shield_score": s, "shield_delta": float(-negd)}
