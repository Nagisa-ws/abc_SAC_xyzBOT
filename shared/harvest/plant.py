"""
shared/harvest/plant.py
-----------------------
Tanaman nanas acak untuk Misi 4-5 (simulasi dulu, tanpa persepsi kamera).

Model MuJoCo (robot_model/scene_harvest.xml) berisi SATU tanaman "pool" dengan jumlah daun maksimum.
Tiap episode `PlantSampler.sample()` membuat tata letak acak (`PlantLayout`) dan `PlantPool.apply()`
menuliskannya ke MjModel: posisi tanaman, ukuran batang/buah/leher crown, panjang & arah & kekakuan tiap
daun. Daun yang tidak dipakai disembunyikan (tanpa kontak, transparan).

Bagian tanaman (koordinat tanaman: origin = pangkal batang di tanah, z ke atas):
  * batang (stalk)       : kapsul keras, tinggi diacak
  * buah                 : kapsul keras (radius & tinggi diacak)
  * leher crown (neck)   : kapsul keras di atas buah = titik cengkeram mode "crown"
  * daun crown (tuft)    : 4 daun pendek fleksibel di atas leher
  * daun roset           : daun panjang fleksibel dari pangkal batang, condong ke atas lalu melengkung keluar
  * daun PENGHALANG      : daun fleksibel yang melintang di sisi ATAS / DEPAN / KIRI / KANAN buah, masing-masing
                           muncul dgn peluang `p_block` (permintaan pengguna: daun kadang menutupi buah dari
                           sisi mana saja, kadang keempat sisi sekaligus).
  "Depan" = sisi buah yang menghadap base robot (kendaraan); kiri/kanan relatif terhadap arah itu.

SEMUA rentang ukuran adalah TEBAKAN awal (ukuran nyata di lapangan acak dan belum diukur). Ganti nilai di
Mission4/configs/domain_randomization.yaml (blok `plant`) setelah mengukur 15-20 tanaman nyata.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

SIDES = ("top", "front", "left", "right")


# ---------------------------------------------------------------------- #
# Tata letak
# ---------------------------------------------------------------------- #
@dataclass
class LeafSpec:
    kind: str                 # rosette | block | tuft
    root: np.ndarray          # (3,) koordinat tanaman (tuft: relatif tuft_root)
    direction: np.ndarray     # (3,) arah segmen pertama
    seg_len: float
    bend: float               # rad, tekukan (pitch) antar segmen (+ = melengkung ke bawah)
    radius: float
    stiffness: float          # N.m/rad per engsel
    damping: float
    side: str = ""            # untuk daun penghalang


@dataclass
class PlantLayout:
    ground: np.ndarray        # (3,) posisi dunia pangkal batang
    stalk_h: float
    stalk_r: float
    fruit_r: float
    fruit_half: float         # setengah panjang bagian silinder kapsul buah
    neck_r: float
    neck_half: float
    fruit_mass: float
    leaves: list = field(default_factory=list)
    blocked_sides: tuple = ()

    # titik-titik penting (koordinat DUNIA)
    @property
    def fruit_center(self) -> np.ndarray:
        return self.ground + np.array([0.0, 0.0, self.stalk_h + self.fruit_r + self.fruit_half])

    @property
    def fruit_top_z(self) -> float:
        return float(self.fruit_center[2] + self.fruit_half + self.fruit_r)

    @property
    def neck_center(self) -> np.ndarray:
        c = self.fruit_center.copy()
        c[2] = self.fruit_top_z + self.neck_half + 0.5 * self.neck_r
        return c

    @property
    def fruit_height(self) -> float:
        return 2.0 * (self.fruit_half + self.fruit_r)


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / (np.linalg.norm(v) + 1e-12)


def _u(rng, r):
    return float(rng.uniform(float(r[0]), float(r[1])))


class PlantSampler:
    """Membuat PlantLayout acak dari blok `plant` config + parameter kurikulum."""

    def __init__(self, cfg: dict):
        self.cfg = dict(cfg)
        self.level = dict(self.cfg.get("default_level", {}))

    def set_level(self, **kw):
        self.level.update(kw)

    def _lv(self, key, default):
        return self.level.get(key, default)

    def sample(self, rng: np.random.Generator, ground: np.ndarray, front_dir: np.ndarray) -> PlantLayout:
        c = self.cfg
        fr = c["fruit"]
        fruit_r = 0.5 * _u(rng, fr["diameter_range"])
        H = _u(rng, fr["height_range"])
        fruit_half = max(0.01, 0.5 * H - fruit_r)
        zc = _u(rng, fr["center_height_range"])
        stalk_h = max(0.10, zc - fruit_half - fruit_r)
        lay = PlantLayout(
            ground=np.asarray(ground, dtype=float).copy(),
            stalk_h=stalk_h, stalk_r=0.5 * _u(rng, c["stalk"]["diameter_range"]),
            fruit_r=fruit_r, fruit_half=fruit_half,
            neck_r=0.5 * _u(rng, c["crown"]["neck_diameter_range"]),
            neck_half=0.5 * _u(rng, c["crown"]["neck_length_range"]),
            fruit_mass=_u(rng, fr["mass_range"]),
        )
        stiff = self._lv("stiffness_range", c["leaf"]["stiffness_range"])
        damp = c["leaf"]["damping_range"]

        def leaf_props():
            return dict(radius=0.5 * _u(rng, c["leaf"]["width_range"]),
                        stiffness=_u(rng, stiff), damping=_u(rng, damp))

        # ---- daun roset ----
        ro = c["rosette"]
        n_lo, n_hi = self._lv("rosette_count", ro["count_range"])
        n = int(rng.integers(int(n_lo), int(n_hi) + 1))
        az0 = rng.uniform(0, 2 * np.pi)
        for i in range(n):
            az = az0 + 2 * np.pi * i / max(n, 1) + rng.normal(0, 0.25)
            rho = _u(rng, ro["root_radius_range"])
            root = np.array([rho * np.cos(az), rho * np.sin(az), _u(rng, ro["root_height_range"])])
            el = np.deg2rad(_u(rng, ro["elevation_deg_range"]))
            d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            L = _u(rng, ro["length_range"])
            lay.leaves.append(LeafSpec("rosette", root, d, L / ro.get("segments", 4),
                                       np.deg2rad(_u(rng, ro["bend_deg_range"])), **leaf_props()))

        # ---- daun crown (tuft) ----
        tf = c["crown"]
        for i in range(int(tf.get("tuft_leaves", 4))):
            az = 2 * np.pi * i / 4 + rng.normal(0, 0.3)
            el = np.deg2rad(_u(rng, tf["tuft_elevation_deg_range"]))
            d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            L = _u(rng, tf["tuft_length_range"])
            p = leaf_props()
            p["radius"] = 0.5 * _u(rng, tf["tuft_width_range"])
            lay.leaves.append(LeafSpec("tuft", np.zeros(3), d, L / 2, np.deg2rad(10.0), **p))

        # ---- daun penghalang (atas / depan / kiri / kanan) ----
        bl = c["block"]
        p_side = float(self._lv("p_block", bl.get("p_block", 0.4)))
        max_per = int(self._lv("max_per_side", bl.get("max_per_side", 2)))
        p_all = float(self._lv("p_all_sides", bl.get("p_all_sides", 0.0)))
        f = _unit(np.r_[front_dir[:2], 0.0])
        left = np.cross([0.0, 0.0, 1.0], f)
        normals = {"top": np.array([0.0, 0.0, 1.0]), "front": f, "left": left, "right": -left}
        all_sides = rng.random() < p_all
        fc_local = lay.fruit_center - lay.ground
        blocked = []
        n_block_max = int(c.get("pool", {}).get("n_block", 8))
        for side in SIDES:
            if not (all_sides or rng.random() < p_side):
                continue
            k = int(rng.integers(1, max_per + 1))
            for _ in range(k):
                if sum(1 for l in lay.leaves if l.kind == "block") >= n_block_max:
                    break
                nrm = normals[side]
                dist = _u(rng, bl["distance_range"])
                L = _u(rng, bl["length_range"])
                if side == "top":
                    C = np.array([fc_local[0], fc_local[1], lay.fruit_top_z - lay.ground[2]
                                  + 2 * lay.neck_half + dist])
                    C[:2] += rng.uniform(-0.05, 0.05, 2)
                    ang = rng.uniform(0, 2 * np.pi)
                    u = np.array([np.cos(ang), np.sin(ang), rng.uniform(-0.25, 0.25)])
                else:
                    C = fc_local + nrm * (lay.fruit_r + dist)
                    C[2] += rng.uniform(-0.6, 0.9) * (lay.fruit_half + lay.fruit_r)
                    tang = np.cross([0.0, 0.0, 1.0], nrm)
                    C += tang * rng.uniform(-0.05, 0.05)
                    psi = np.deg2rad(rng.uniform(-60, 60))
                    u = np.cos(psi) * tang + np.sin(psi) * np.array([0.0, 0.0, 1.0])
                u = _unit(u)
                if rng.random() < 0.5:
                    u = -u
                frac = rng.uniform(0.35, 0.65)
                root = C - frac * L * u
                if root[2] < 0.03:                 # akar di bawah tanah -> balik arah
                    u = -u
                    root = C - frac * L * u
                if root[2] < 0.03:
                    continue
                lay.leaves.append(LeafSpec("block", root, u, L / bl.get("segments", 3),
                                           np.deg2rad(rng.uniform(-6, 6)), side=side, **leaf_props()))
            blocked.append(side)
        lay.blocked_sides = tuple(blocked)
        return lay


# ---------------------------------------------------------------------- #
# Pool di MjModel
# ---------------------------------------------------------------------- #
def _quat_from_x_axis(d: np.ndarray) -> np.ndarray:
    x = _unit(d)
    up = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
    y = _unit(np.cross(up, x))
    z = np.cross(x, y)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([x, y, z]).ravel())
    return q


def _quat_pitch(angle: float) -> np.ndarray:
    return np.array([np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0])


class PlantPool:
    def __init__(self, model: mujoco.MjModel):
        self.m = m = model
        nid = lambda t, n: mujoco.mj_name2id(m, t, n)
        B, G, J, S = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_SITE
        self.plant_body = nid(B, "plant")
        self.tuft_root = nid(B, "tuft_root")
        self.g_stalk, self.g_fruit, self.g_neck = nid(G, "stalk"), nid(G, "fruit_att"), nid(G, "neck")
        self.s_fruit, self.s_grasp = nid(S, "fruit_center"), nid(S, "grasp_target")
        self.free_body = nid(B, "fruit_free")
        self.g_free = [nid(G, n) for n in ("fruit_free_body", "fruit_free_neck", "fruit_free_tuft")]
        self.free_qadr = int(m.jnt_qposadr[nid(J, "fruit_free_joint")])
        self.free_dadr = int(m.jnt_dofadr[nid(J, "fruit_free_joint")])
        self.chains = {"rosette": [], "block": [], "tuft": []}
        for kind, pre in (("rosette", "ros"), ("block", "blk"), ("tuft", "tuft")):
            i = 0
            while nid(B, f"{pre}{i}_s0") >= 0:
                segs = []
                s = 0
                while nid(B, f"{pre}{i}_s{s}") >= 0:
                    segs.append(dict(body=nid(B, f"{pre}{i}_s{s}"), geom=nid(G, f"{pre}{i}_g{s}"),
                                     joints=[nid(J, f"{pre}{i}_j{s}a"), nid(J, f"{pre}{i}_j{s}b")]))
                    s += 1
                self.chains[kind].append(segs)
                i += 1
        self.leaf_geoms = np.array([sg["geom"] for k in self.chains for ch in self.chains[k] for sg in ch])
        self.leaf_geom_set = set(int(g) for g in self.leaf_geoms)
        self.hard_geom_set = {int(self.g_stalk), int(self.g_fruit), int(self.g_neck)}
        self.tuft_geom_set = set(int(sg["geom"]) for ch in self.chains["tuft"] for sg in ch)
        self.leaf_qadr = np.array([m.jnt_qposadr[j] for k in self.chains for ch in self.chains[k]
                                   for sg in ch for j in sg["joints"]])
        self.leaf_dadr = np.array([m.jnt_dofadr[j] for k in self.chains for ch in self.chains[k]
                                   for sg in ch for j in sg["joints"]])
        self._leaf_rgba = {int(g): m.geom_rgba[g].copy() for g in self.leaf_geoms}
        self._hard_rgba = {g: m.geom_rgba[g].copy() for g in self.hard_geom_set}
        self.active = []            # list[(segs, LeafSpec)]
        self.layout: PlantLayout | None = None

    # ---------------- tulis layout ke model ----------------
    @staticmethod
    def _set_capsule(m, g, radius, half):
        m.geom_size[g, 0] = radius
        m.geom_size[g, 1] = half
        m.geom_rbound[g] = radius + half
        if hasattr(m, "geom_aabb"):
            m.geom_aabb[g, :3] = 0.0
            m.geom_aabb[g, 3:] = [radius, radius, radius + half]

    def _enable_geom(self, g, on: bool, rgba=None):
        m = self.m
        if on:
            m.geom_contype[g], m.geom_conaffinity[g] = (2, 0) if g in self.leaf_geom_set else (4, 0)
            if rgba is not None:
                m.geom_rgba[g] = rgba
        else:
            m.geom_contype[g] = m.geom_conaffinity[g] = 0
            m.geom_rgba[g, 3] = 0.0

    def apply(self, lay: PlantLayout):
        m = self.m
        self.layout = lay
        m.body_pos[self.plant_body] = lay.ground
        # batang
        self._set_capsule(m, self.g_stalk, lay.stalk_r, 0.5 * lay.stalk_h)
        m.geom_pos[self.g_stalk] = [0, 0, 0.5 * lay.stalk_h]
        # buah & leher
        fc = lay.fruit_center - lay.ground
        self._set_capsule(m, self.g_fruit, lay.fruit_r, lay.fruit_half)
        m.geom_pos[self.g_fruit] = fc
        nc = lay.neck_center - lay.ground
        self._set_capsule(m, self.g_neck, lay.neck_r, lay.neck_half)
        m.geom_pos[self.g_neck] = nc
        m.site_pos[self.s_fruit] = fc
        for g in self.hard_geom_set:
            self._enable_geom(g, True, self._hard_rgba[g])
        m.body_pos[self.tuft_root] = nc + np.array([0, 0, lay.neck_half + lay.neck_r])
        # buah lepas (dipakai Misi 5): ukuran & massa sama dgn buah
        gb, gn, gt = self.g_free
        self._set_capsule(m, gb, lay.fruit_r, lay.fruit_half)
        self._set_capsule(m, gn, lay.neck_r, lay.neck_half)
        m.geom_pos[gn] = [0, 0, nc[2] - fc[2]]
        m.geom_pos[gt] = [0, 0, nc[2] - fc[2] + lay.neck_half + 0.09]
        m.body_mass[self.free_body] = lay.fruit_mass
        r, h = lay.fruit_r, lay.fruit_half + lay.fruit_r
        m.body_inertia[self.free_body] = lay.fruit_mass * np.array([(3 * r * r + 4 * h * h) / 12] * 2 + [r * r / 2])

        # daun: isi per jenis, sisanya disembunyikan
        self.active = []
        by_kind = {"rosette": [], "block": [], "tuft": []}
        for spec in lay.leaves:
            by_kind[spec.kind].append(spec)
        for kind, chains in self.chains.items():
            specs = by_kind[kind]
            for ci, segs in enumerate(chains):
                spec = specs[ci] if ci < len(specs) else None
                for si, sg in enumerate(segs):
                    g, b = sg["geom"], sg["body"]
                    if spec is None:
                        self._enable_geom(g, False)
                        continue
                    self._enable_geom(g, True, self._leaf_rgba[int(g)])
                    self._set_capsule(m, g, spec.radius, 0.5 * spec.seg_len)
                    m.geom_pos[g] = [0.5 * spec.seg_len, 0, 0]
                    if si == 0:
                        m.body_pos[b] = spec.root
                        m.body_quat[b] = _quat_from_x_axis(spec.direction)
                    else:
                        m.body_pos[b] = [spec.seg_len, 0, 0]
                        m.body_quat[b] = _quat_pitch(spec.bend)
                    for j in sg["joints"]:
                        m.jnt_stiffness[j] = spec.stiffness
                        m.dof_damping[m.jnt_dofadr[j]] = spec.damping
                if spec is not None:
                    self.active.append((segs, spec))

    # ---------------- state ----------------
    def reset_state(self, data: mujoco.MjData):
        data.qpos[self.leaf_qadr] = 0.0
        data.qvel[self.leaf_dadr] = 0.0
        q = self.free_qadr
        data.qpos[q:q + 7] = [0, 0, -3, 1, 0, 0, 0]
        data.qvel[self.free_dadr:self.free_dadr + 6] = 0.0
        for g in self.g_free:
            self.m.geom_contype[g] = self.m.geom_conaffinity[g] = 0

    def active_leaf_joints(self):
        """(qpos_adr, stiffness, is_tuft) engsel daun aktif (roset/penghalang/crown yang tidak disembunyikan)."""
        if getattr(self, "_aj_layout", None) is not self.layout or getattr(self, "_aj_n", -1) != len(self.active):
            adr, k, kind = [], [], []
            for segs, spec in self.active:
                if self.m.geom_contype[segs[0]["geom"]] == 0:
                    continue
                for sg in segs:
                    for j in sg["joints"]:
                        adr.append(self.m.jnt_qposadr[j])
                        k.append(self.m.jnt_stiffness[j])
                        kind.append(spec.kind == "tuft")
            self._aj = (np.array(adr, dtype=int), np.array(k), np.array(kind, dtype=bool))
            self._aj_layout, self._aj_n = self.layout, len(self.active)
        return self._aj

    def _cache_active(self, per_seg: int):
        geoms, half, reff, tuft = [], [], [], []
        for segs, spec in self.active:
            for sg in segs:
                geoms.append(sg["geom"])
                half.append(0.5 * spec.seg_len)
                reff.append(float(np.sqrt(spec.radius ** 2 + (spec.seg_len / per_seg / 2) ** 2)))
                tuft.append(spec.kind == "tuft")
        self._ag = np.array(geoms, dtype=int)
        self._ah = np.array(half)
        self._ar = np.array(reff)
        self._at = np.array(tuft, dtype=bool)
        self._per_seg = per_seg

    def leaf_points(self, data: mujoco.MjData, per_seg: int = 4, include_tuft: bool = True):
        """Titik sampel sepanjang sumbu daun aktif (dunia) + radius efektif (bola yang menutup kapsul)."""
        if getattr(self, "_per_seg", None) != per_seg or getattr(self, "_cache_layout", None) is not self.layout:
            self._cache_active(per_seg)
            self._cache_layout = self.layout
        sel = np.ones(len(self._ag), bool)
        if not include_tuft:
            sel &= ~self._at
        sel &= self.m.geom_contype[self._ag] != 0
        g = self._ag[sel]
        if g.size == 0:
            return np.zeros((0, 3)), np.zeros(0)
        c = data.geom_xpos[g]                                   # (G,3)
        ax = data.geom_xmat[g].reshape(-1, 3, 3)[:, :, 2]       # (G,3)
        h = self._ah[sel][:, None]
        t = ((np.arange(per_seg) + 0.5) / per_seg)[None, :, None]   # (1,P,1)
        pts = (c - h * ax)[:, None, :] + (2 * h)[:, :, None] * t * ax[:, None, :]
        rad = np.repeat(self._ar[sel], per_seg)
        return pts.reshape(-1, 3), rad

    def hard_points(self, data: mujoco.MjData, n: int = 6, which=("stalk", "fruit", "neck")):
        """Titik sepanjang sumbu kapsul keras (dunia) + radius kapsul."""
        gid = {"stalk": self.g_stalk, "fruit": self.g_fruit, "neck": self.g_neck}
        pts, rad = [], []
        for w in which:
            g = gid[w]
            if self.m.geom_contype[g] == 0:
                continue
            c = data.geom_xpos[g]
            ax = data.geom_xmat[g].reshape(3, 3)[:, 2]
            h = self.m.geom_size[g, 1]
            ts = np.linspace(-h, h, n)
            pts.append(c[None, :] + ts[:, None] * ax[None, :])
            rad.append(np.full(n, self.m.geom_size[g, 0]))
        if not pts:
            return np.zeros((0, 3)), np.zeros(0)
        return np.vstack(pts), np.concatenate(rad)

    def prune_leaves_near(self, data: mujoco.MjData, caps, keepout: float) -> int:
        """Sembunyikan daun yang (pada pose robot saat ini = home) lebih dekat dari `keepout` ke badan robot.
        Robot tiba di samping tanaman; daun yang menembus badan robot di awal episode tidak realistis
        (gaya kontak awal raksasa). Return jumlah daun yang dipangkas."""
        from shared.envs.robot_geometry import obstacle_clearance
        n = 0
        keep = []
        for segs, spec in self.active:
            g = np.array([sg["geom"] for sg in segs])
            c = data.geom_xpos[g]
            ax = data.geom_xmat[g].reshape(-1, 3, 3)[:, :, 2]
            h = 0.5 * spec.seg_len
            t = np.linspace(-1, 1, 5)[None, :, None]
            pts = (c[:, None, :] + h * t * ax[:, None, :]).reshape(-1, 3)
            clear, _ = obstacle_clearance(caps, data, pts, np.full(len(pts), spec.radius))
            if float(np.min(clear)) < keepout:
                for gi in g:
                    self._enable_geom(int(gi), False)
                n += 1
            else:
                keep.append((segs, spec))
        self.active = keep
        self._cache_layout = None
        self._aj_layout = None
        return n

    def hide_attached_fruit(self):
        """Buah terlepas: sembunyikan buah/leher/crown yang menempel di tanaman."""
        for g in (self.g_fruit, self.g_neck):
            self._enable_geom(g, False)
        for g in self.tuft_geom_set:
            self._enable_geom(g, False)
        self._aj_layout = None
        self._cache_layout = None
