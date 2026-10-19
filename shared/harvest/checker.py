"""
shared/harvest/checker.py
-------------------------
Validasi rute Misi 4-5 (dijalankan saat reset, memakai IK yang sama dgn env).

Beda dgn shared/envs/validation.py (Misi 1-3):
  * kontak dgn DAUN diabaikan (daun lentur boleh tersentuh/terdorong; itu tugas RL),
  * orientasi berubah sepanjang rute (slerp dari orientasi home ke orientasi cengkeram),
  * clearance terhadap bagian KERAS tanaman (batang, buah, leher crown) diperiksa:
      - seluruh robot >= `approach_margin` di rute home -> S,
      - badan robot selain jari >= `line_margin` di garis S -> nucleus,
      - jari terhadap batang & bagian keras selain objek yang dicengkeram >= `finger_margin`.
"""
from __future__ import annotations

import mujoco
import numpy as np

from shared.envs.robot_geometry import RobotCapsules, points_to_segments, world_segments
from shared.envs.validation import ReachabilityChecker


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(np.clip(d, -1, 1))
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


def robot_geom_mask(model: mujoco.MjModel, base_body: int) -> np.ndarray:
    """True untuk geom milik robot (subtree body 'base')."""
    return model.body_rootid[model.geom_bodyid] == model.body_rootid[base_body]


def robot_capsules_only(caps: RobotCapsules, model: mujoco.MjModel, base_body: int) -> RobotCapsules:
    keep = model.body_rootid[caps.body_ids] == model.body_rootid[base_body]
    idx = np.where(keep)[0]
    return RobotCapsules(body_ids=caps.body_ids[idx], p0_local=caps.p0_local[idx], p1_local=caps.p1_local[idx],
                         radii=caps.radii[idx], names=[caps.names[i] for i in idx])


class HarvestChecker(ReachabilityChecker):
    def __init__(self, model, ik_solver, caps: RobotCapsules, robot_geoms: np.ndarray, floor_geom: int,
                 finger_cap_mask: np.ndarray, pos_tol=2e-3, rot_tol=0.03, waypoint_step=0.04):
        super().__init__(model, ik_solver, caps, pos_tol=pos_tol, waypoint_step=waypoint_step)
        self.robot_geoms = np.asarray(robot_geoms, bool)
        self.floor_geom = int(floor_geom)
        self.finger_cap_mask = np.asarray(finger_cap_mask, bool)
        self.rot_tol = float(rot_tol)
        self.last_quats: np.ndarray | None = None
        # singularitas pergelangan UR (wrist_2 ~ 0 / +-pi: sumbu wrist_1 & wrist_3 segaris) dan lompatan
        # konfigurasi antar-waypoint ditolak: di dekatnya IK membalik wrist_1/wrist_3 +-0.25 rad/langkah dan
        # TCP menyimpang jauh dari garis (terukur: TCP turun 28 cm, lalu self-collision).
        self.wrist2_index = 4
        self.wrist_singular_margin = np.deg2rad(15.0)
        self.max_joint_jump = 0.6

    def has_contact(self, q_full: np.ndarray) -> bool:
        """Hanya kontak robot-robot dan robot-lantai (daun & tanaman ditangani terpisah)."""
        d = self._data
        d.qpos[:] = q_full
        mujoco.mj_forward(self.model, d)
        for i in range(d.ncon):
            c = d.contact[i]
            r1, r2 = self.robot_geoms[c.geom1], self.robot_geoms[c.geom2]
            if r1 and r2:
                return True
            if (r1 and c.geom2 == self.floor_geom) or (r2 and c.geom1 == self.floor_geom):
                return True
        return False

    def rollout_q(self, q_full_start, waypoints, quats):
        """Ikuti waypoint dgn orientasi per-waypoint. Return (ok, q_akhir, segs)."""
        q_full = np.array(q_full_start, dtype=float, copy=True)
        segs = []
        for wp, q in zip(waypoints, quats):
            q_arm, info = self.ik.solve(q_full, wp, q)
            if info["position_error"] > self.pos_tol or info["rotation_error"] > self.rot_tol:
                return False, q_full, segs
            if np.max(np.abs(q_arm - q_full[self.ik.qpos_adr])) > self.max_joint_jump:
                return False, q_full, segs
            if abs(np.sin(q_arm[self.wrist2_index])) < np.sin(self.wrist_singular_margin):
                return False, q_full, segs
            q_full[self.ik.qpos_adr] = q_arm
            if self.has_contact(q_full):
                return False, q_full, segs
            segs.append(world_segments(self.caps, self._data))
        return True, q_full, segs

    def approach_q(self, q_start, p_from, p_to, quat_from, quat_to, base_xy):
        """Rute lurus lalu polar home -> S, orientasi slerp. Return (ok, q, segs, wps, quats)."""
        for wps in (self.waypoints(p_from, p_to), self.polar_waypoints(p_from, p_to, base_xy)):
            n = len(wps)
            # orientasi selesai berputar pada 70% rute (sisa rute = orientasi cengkeram)
            quats = np.array([slerp(quat_from, quat_to, min(1.0, (i + 1) / max(1, 0.7 * n))) for i in range(n)])
            ok, q, segs = self.rollout_q(q_start, wps, quats)
            if ok:
                return True, q, segs, np.array(wps), quats
        return False, q_start, [], None, None

    def hard_clearance(self, segs, pts: np.ndarray, rad: np.ndarray, cap_mask: np.ndarray | None = None) -> float:
        """Clearance minimum permukaan robot (kapsul terpilih) - titik keras sepanjang segs."""
        if pts.shape[0] == 0 or not segs:
            return np.inf
        cm = np.ones(self.caps.n, bool) if cap_mask is None else cap_mask
        best = np.inf
        for a, b in segs:
            dist, _ = points_to_segments(pts, a[cm], b[cm])
            surf = dist - self.caps.radii[cm][None, :] - rad[:, None]
            best = min(best, float(surf.min()))
        return best
