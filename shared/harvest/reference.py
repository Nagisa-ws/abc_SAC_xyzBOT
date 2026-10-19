"""
shared/harvest/reference.py
---------------------------
Controller referensi Misi 4-5 = controller referensi Misi 1-3 (shared/control/reference.py) + ORIENTASI.

  posisi  : fase 0 ikuti waypoint rute home -> S (lookahead), fase lain lurus ke target aktif env
  rotasi  : putar target orientasi menuju orientasi yang diminta env (`env.reference_quat()`):
            fase 0 = orientasi waypoint (slerp home -> orientasi cengkeram), fase berikutnya = orientasi cengkeram.
  aksi    = clip(gain * error / delta_maks), norma posisi & rotasi masing-masing <= 1.

Controller ini TIDAK tahu tentang daun (hanya mengikuti garis max-clearance dari perencana stargaze);
mendorong / menghindari daun yang masih menghalangi adalah tugas koreksi RL.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import mujoco

from ik.ik_solver import _quat_to_axis_angle_error
from ik.pose_utils import quat_angle


@dataclass
class HarvestReferenceConfig:
    gain: float = 0.8
    lookahead: int = 2
    rot_gain: float = 0.8
    # gerak posisi diperlambat bila orientasi aktual tertinggal dari orientasi yang diminta (pergelangan lebih
    # lambat dari translasi): skala = clip(1 - (lag - lag_free) / lag_span, lag_min_scale, 1)
    lag_free_deg: float = 8.0
    lag_span_deg: float = 25.0
    lag_min_scale: float = 0.15

    @classmethod
    def from_dict(cls, d: dict | None):
        d = dict(d or {})
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class HarvestReference:
    def __init__(self, cfg: HarvestReferenceConfig | None = None):
        self.cfg = cfg or HarvestReferenceConfig()

    def target(self, env) -> np.ndarray:
        tcp = env.tcp_pos
        tgt = env.active_target()
        wps = getattr(env, "ref_approach_wps", None)
        if env.phase == 0 and wps is not None and len(wps) > 0:
            i = int(np.argmin(np.linalg.norm(wps - tcp[None, :], axis=1)))
            env._ref_wp_index = i
            tgt = wps[min(i + self.cfg.lookahead, len(wps) - 1)]
        return tgt

    def act(self, env) -> np.ndarray:
        a = np.zeros(6)
        v = (self.target(env) - env.tcp_pos) / env.max_pos_delta * self.cfg.gain
        n = float(np.linalg.norm(v))
        a[:3] = v if n <= 1.0 else v / n
        q_des = env.reference_quat()
        if q_des is not None:
            q_act = np.zeros(4)
            mujoco.mju_mat2Quat(q_act, env.data.site_xmat[env.tcp_site_id])
            lag = np.degrees(quat_angle(q_act, q_des))
            c = self.cfg
            a[:3] *= float(np.clip(1.0 - (lag - c.lag_free_deg) / c.lag_span_deg, c.lag_min_scale, 1.0))
            err = _quat_to_axis_angle_error(env.target_quat, q_des)
            w = err / env.max_rot_delta * self.cfg.rot_gain
            n = float(np.linalg.norm(w))
            a[3:] = w if n <= 1.0 else w / n
        return a
