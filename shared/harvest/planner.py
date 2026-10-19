"""
shared/harvest/planner.py
-------------------------
Perencana garis pendekatan Misi 4-5 = konsep "stargaze" pengguna (stargaze.py / obstacles/stargaze.py)
diterapkan ke DAUN nanas:

  * nucleus  = titik cengkeram (mode "crown": tengah leher crown; mode "body": tengah buah)
  * kubus    = kubus berpusat di nucleus, sisi 2*L, DIPUTAR menghadap robot:
               face DEPAN = menghadap base robot (kendaraan), ATAS = +z, KIRI/KANAN.
               Face BELAKANG dibuang (berseberangan dgn face awal "depan"), face BAWAH terlarang (tanah)
               -> persis aturan stargaze: initial face + face tetangga, opposite diabaikan, forbidden dihapus.
  * kandidat = grid titik S pada face yang diizinkan; garis S -> nucleus
  * skor     = clearance garis = jarak minimum titik-daun ke SEGMEN S->nucleus dikurangi radius daun
               (segmen, bukan garis tak hingga: daun di belakang nucleus tidak menghalangi)
  * pilih    = kandidat dgn clearance MAKSIMUM (max-clearance, seperti tombol "Find Max Clearance Path")

Tambahan untuk gripper:
  * mode cengkeram per kandidat ditentukan sudut elevasi garis:
      crown : dari atas (elevasi >= crown_top_min_deg) ATAU dari samping hampir datar
              (crown_side_min_deg <= elevasi <= crown_side_max_deg)
      body  : dari samping (elevasi di [body_min_deg, body_max_deg]) dan hanya bila diameter buah muat di jari
  * hanya mode yang muat di bukaan jari yang dipakai (lihat `GripperSpec`)
  * kandidat diurutkan; env memvalidasi IK/tabrakan dan mencoba kandidat berikutnya bila gagal.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from obstacles.stargaze import ConstellationGenerator


@dataclass
class GripperSpec:
    finger_offset: float = 0.015      # m, celah dalam jari saat menutup penuh = 2*offset
    stroke_per_finger: float = 0.04   # m (PGIA-140-80: 80 mm total)
    grip_clearance: float = 0.012     # m, ruang bebas per sisi saat jari terbuka melewati objek
    max_force: float = 140.0          # N per jari
    grip_force_frac: float = 0.8      # gaya cengkeram yang dipakai (fraksi max)
    min_squeeze: float = 0.006        # m, objek harus lebih lebar dari celah minimum + ini (agar benar-benar terjepit)

    @property
    def gap_min(self) -> float:
        return 2.0 * self.finger_offset

    @property
    def gap_max(self) -> float:
        return 2.0 * (self.finger_offset + self.stroke_per_finger)

    def fits(self, width: float) -> bool:
        """Objek selebar `width` bisa dicengkeram (jari bisa melewatinya terbuka & menjepitnya)."""
        return (width >= self.gap_min + self.min_squeeze) and (width + 2 * self.grip_clearance <= self.gap_max)

    def open_cmd(self, width: float) -> float:
        """Perintah bukaan per jari (m) untuk melewati objek selebar `width` dgn ruang bebas."""
        need = 0.5 * (width + 2 * self.grip_clearance) - self.finger_offset
        return float(np.clip(need + 0.004, 0.0, self.stroke_per_finger))

    @classmethod
    def from_dict(cls, d: dict | None) -> "GripperSpec":
        d = dict(d or {})
        return cls(**{k: float(d[k]) for k in cls.__dataclass_fields__ if k in d})


@dataclass
class ApproachPlan:
    start: np.ndarray        # S (= T1)
    target: np.ndarray       # nucleus (= T2, posisi TCP saat cengkeram)
    mode: str                # crown | body
    face: str
    clearance: float         # clearance garis terhadap daun (m), bisa negatif (garis menembus daun)
    elevation_deg: float
    quat: np.ndarray         # orientasi TCP saat cengkeram (wxyz)


def grasp_quat(approach_dir: np.ndarray, x_hint: np.ndarray) -> np.ndarray:
    """Orientasi TCP: sumbu z alat = arah pendekatan; sumbu x alat (arah menutup jari) HORIZONTAL
    (tegak lurus sumbu buah yang vertikal) dan sedekat mungkin dgn `x_hint` (minim putaran pergelangan)."""
    z = approach_dir / (np.linalg.norm(approach_dir) + 1e-12)
    zw = np.array([0.0, 0.0, 1.0])
    if abs(z[2]) > 0.97:                              # dari atas: x bebas di bidang horizontal
        x = x_hint - np.dot(x_hint, z) * z
        x[2] = 0.0
        if np.linalg.norm(x) < 1e-6:
            x = np.array([1.0, 0.0, 0.0])
    else:
        x = np.cross(zw, z)                           # horizontal & tegak lurus z
        if np.dot(x, x_hint) < 0:
            x = -x
    x = x - np.dot(x, z) * z
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(z, x)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([x, y, z]).ravel())
    return q


class ApproachPlanner:
    def __init__(self, cfg: dict, gripper: GripperSpec):
        self.cfg = dict(cfg)
        self.gripper = gripper
        self.L = float(self.cfg.get("cube_half_size", 0.30))
        self.res = int(self.cfg.get("resolution", 21))
        self.faces = list(self.cfg.get("faces", ["top", "front", "left", "right"]))
        self.modes = list(self.cfg.get("grip_modes", ["crown", "body"]))

    def _frame(self, nucleus, base_pos):
        f = np.asarray(base_pos, float) - np.asarray(nucleus, float)
        f[2] = 0.0
        f /= np.linalg.norm(f) + 1e-12
        z = np.array([0.0, 0.0, 1.0])
        left = np.cross(z, f)
        return f, left, z

    def _face_points(self, nucleus, base_pos):
        f, left, z = self._frame(nucleus, base_pos)
        normals = {"top": z, "front": f, "left": left, "right": -left}
        g = np.linspace(-self.L, self.L, self.res)
        G1, G2 = [a.ravel() for a in np.meshgrid(g, g)]
        pts, names = [], []
        for name in self.faces:
            n = normals[name]
            # dua sumbu bidang face
            u = f if name in ("top",) else z
            v = np.cross(n, u)
            P = nucleus[None, :] + self.L * n[None, :] + G1[:, None] * u[None, :] + G2[:, None] * v[None, :]
            pts.append(P)
            names += [name] * len(P)
        return np.vstack(pts), np.array(names)

    def _mode_ok(self, mode, elev):
        c = self.cfg
        if mode == "crown":
            return elev >= float(c.get("crown_top_min_deg", 55)) or (float(c.get("crown_side_min_deg", -5)) <= elev
                                                                   <= float(c.get("crown_side_max_deg", 20)))
        return float(c.get("body_min_deg", -10)) <= elev <= float(c.get("body_max_deg", 30))

    def plan(self, layout, leaf_pts: np.ndarray, leaf_rad: np.ndarray, base_pos: np.ndarray,
             x_hint: np.ndarray, reach_filter=None, top_k: int = 6) -> list[ApproachPlan]:
        nuclei = {"crown": layout.neck_center, "body": layout.fruit_center}
        widths = {"crown": 2 * layout.neck_r, "body": 2 * layout.fruit_r}
        plans = []
        for mode in self.modes:
            if not self.gripper.fits(widths[mode]):
                continue
            N = nuclei[mode]
            S, names = self._face_points(N, base_pos)
            if reach_filter is not None:
                keep = np.asarray(reach_filter(S), bool)
                S, names = S[keep], names[keep]
            if len(S) == 0:
                continue
            d = S - N[None, :]
            elev = np.degrees(np.arctan2(d[:, 2], np.linalg.norm(d[:, :2], axis=1)))
            ok = np.array([self._mode_ok(mode, e) for e in elev])
            S, names, elev = S[ok], names[ok], elev[ok]
            if len(S) == 0:
                continue
            if leaf_pts.shape[0]:
                dist = ConstellationGenerator.segment_clearance(leaf_pts - N[None, :], S - N[None, :])
                clear = (dist - leaf_rad[:, None]).min(axis=0)
            else:
                clear = np.full(len(S), np.inf)
            # kandidat terbaik PER FACE (beragam arah; env mencoba berurutan bila validasi IK/tabrakan gagal)
            order = []
            for fname in dict.fromkeys(names.tolist()):
                idx = np.where(names == fname)[0]
                order.append(int(idx[np.argmax(clear[idx])]))
            for i in order:
                app = N - S[i]
                plans.append(ApproachPlan(start=S[i].copy(), target=N.copy(), mode=mode, face=str(names[i]),
                                          clearance=float(clear[i]), elevation_deg=float(elev[i]),
                                          quat=grasp_quat(app, x_hint)))
        plans.sort(key=lambda p: -p.clearance)
        return plans[:top_k]
