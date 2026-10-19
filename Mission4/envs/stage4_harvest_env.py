"""
Mission4/envs/stage4_harvest_env.py
-----------------------------------
Misi 4: MENCAPAI POSE CENGKERAM buah nanas MENEMBUS DAUN (simulasi, posisi tanaman diketahui persis).

Skenario per episode (lihat shared/harvest/):
  1. Tanaman acak: batang, buah, leher crown, daun roset lentur, daun crown, dan daun PENGHALANG yang
     melintang di sisi atas / depan / kiri / kanan buah (masing-masing dgn peluang p_block; kadang keempatnya).
  2. Perencana stargaze (konsep pengguna): garis max-clearance dari face kubus (atas/depan/kiri/kanan) ke
     titik cengkeram. Mode cengkeram (crown / body) mengikuti arah garis & ukuran objek vs bukaan jari.
  3. Validasi: rute home -> S (T1) dan garis S -> titik cengkeram (T2) layak secara IK, tanpa self-collision /
     lantai, dan aman terhadap bagian KERAS tanaman. Daun TIDAK divalidasi (boleh tersentuh/didorong).

Fase:
  0 = home -> S  (rute pendekatan + gripper diputar ke orientasi cengkeram)
  1 = S -> titik cengkeram sepanjang garis (jari terbuka melewati leher crown / buah)

Sukses: fase 1, TCP < success_pos_threshold dari titik cengkeram, error orientasi < success_orient_deg.
Gagal : self_collision / floor_contact, plant_hit (badan robot menabrak batang/buah/leher, atau jari menabrak
        bagian keras selain objek), fruit_hit (jari menekan objek terlalu keras), leaf_overforce (gaya daun total
        melewati batas = merusak tanaman), timeout.

Kontak daun = LUNAK: reward -w * gaya daun (dinormalisasi). Ini yang membuat koreksi RL bermakna: controller
referensi hanya mengikuti garis, tidak tahu cara mengurangi dorongan ke daun yang masih menghalangi.

Observasi = layout 63 nilai yang SAMA dgn Misi 1-3 (shared/envs/obs_layout.py):
  rel_target/path_* = rute & garis, obs_rel/obs_dist/min_clearance = 4 titik DAUN (dan batang) terdekat yang
  BERGERAK mengikuti daun, ref_action = aksi controller referensi (posisi + rotasi).
"""
from __future__ import annotations

import copy
from pathlib import Path

import mujoco
import numpy as np

from ik.pose_utils import integrate_quat, limit_quat_deviation, perpendicular_distance_to_segment, quat_angle
from shared.envs.base_arm_env import BaseArmEnv
from shared.envs.obs_layout import OBS_FAR
from shared.envs.robot_geometry import build_robot_capsules, obstacle_clearance
from shared.harvest import (ApproachPlanner, GripperSpec, HarvestReference, HarvestReferenceConfig,
                            PlantPool, PlantSampler)
from shared.harvest.checker import HarvestChecker, robot_capsules_only, robot_geom_mask

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = str(_PROJECT_ROOT / "robot_model" / "scene_harvest.xml")
DEFAULT_CONFIG = str(_PROJECT_ROOT / "Mission4" / "configs" / "domain_randomization.yaml")

_RZ_PI = np.array([0.0, 0.0, 0.0, 1.0])     # rotasi 180 deg di sumbu z alat (gripper simetris)


def _mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


class Stage4HarvestEnv(BaseArmEnv):
    MISSION_ID = 4

    def __init__(self, model_path: str = DEFAULT_MODEL, config_path: str = DEFAULT_CONFIG,
                 obs_noise_std: float | None = None, training: bool = True,
                 seed: int | None = None, randomize: bool | None = None):
        super().__init__(model_path=model_path, config_path=config_path, obs_noise_std=obs_noise_std,
                         training=training, seed=seed, randomize=randomize)
        m = self.model
        hv = self.cfg["harvest"]
        self.hv = hv
        self.pool = PlantPool(m)
        self.sampler = PlantSampler(self.cfg["plant"])
        self.gripper = GripperSpec.from_dict(self.cfg.get("gripper"))
        self.planner = ApproachPlanner(self.cfg["planner"], self.gripper)
        ps = dict(self.cfg.get("policy_structure", {}) or {})
        self.reference = HarvestReference(HarvestReferenceConfig.from_dict(ps.get("reference")))

        name2id = mujoco.mj_name2id
        G, B, J = mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT
        self.floor_geom = name2id(m, G, "floor")
        self.robot_geoms = robot_geom_mask(m, self.base_body_id)
        self.finger_bodies = [name2id(m, B, "finger_left"), name2id(m, B, "finger_right")]
        self.finger_geoms = {name2id(m, G, "finger_left_geom"), name2id(m, G, "finger_right_geom")}
        self.finger_qadr = [int(m.jnt_qposadr[name2id(m, J, n)]) for n in ("finger_left_joint", "finger_right_joint")]
        self.weld_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "fruit_in_gripper")
        # jari dipasang dgn offset dari sumbu tengah (config gripper.finger_offset)
        r_f = float(m.geom_size[name2id(m, G, "finger_left_geom"), 0])
        m.body_pos[self.finger_bodies[0], 0] = self.gripper.finger_offset + r_f
        m.body_pos[self.finger_bodies[1], 0] = -(self.gripper.finger_offset + r_f)

        # kapsul robot saja (tanpa kapsul gripper sintetis Misi 1-3; jari & palm sudah kapsul sendiri)
        caps = build_robot_capsules(m, "__tanpa_kapsul_gripper__", self.gripper_radius)
        self.caps = robot_capsules_only(caps, m, self.base_body_id)
        self.finger_cap_mask = np.isin(self.caps.body_ids, self.finger_bodies)
        val = self.cfg.get("validation", {})
        self.checker = HarvestChecker(m, self.ik_solver, self.caps, self.robot_geoms, self.floor_geom,
                                      self.finger_cap_mask, pos_tol=float(val.get("ik_pos_tol", 2e-3)),
                                      rot_tol=float(val.get("ik_rot_tol", 0.03)),
                                      waypoint_step=float(val.get("waypoint_step", 0.04)))
        self._approach_margin = float(val.get("approach_hard_margin", 0.03))
        self._line_margin = float(val.get("line_hard_margin", 0.01))
        self._finger_margin = float(val.get("finger_hard_margin", 0.004))
        self._top_k = int(val.get("plans_to_try", 4))
        self._home_keepout = float(val.get("home_leaf_keepout", 0.05))
        self._n_pruned = 0
        self._n_leaf_missed = 0

        ct = self.cfg["contact"]
        self._f_hard = float(ct.get("hard_force_limit", 3.0))
        self._f_touch = float(ct.get("target_touch_limit", 25.0))
        self._f_leaf_max = float(ct.get("leaf_force_limit", 1e9))
        self._f_leaf_ref = float(ct.get("leaf_force_ref", 20.0))
        self._bend_max = float(ct.get("leaf_bend_limit_rad", 1.0))
        self._energy_ref = float(ct.get("leaf_energy_ref", 0.5))
        self._ignore_tuft_bend = bool(ct.get("crown_leaves_count_as_damage", False)) is False
        self._orient_tol = np.deg2rad(float(self.cfg["episode"].get("success_orient_deg", 12.0)))
        self._t1_thr = float(self.cfg["episode"].get("t1_reach_threshold", 0.03))
        gate = ps.get("target_gate", [0.06, 0.15])
        self._target_gate = None if gate is None else (float(gate[0]), float(gate[1]))
        self._progress_min = float(ps.get("progress_min", 0.4))
        self._orient_lead = np.deg2rad(float(self.cfg.get("action", {}).get("max_orient_lead_deg", 8.0)))

        self.plan = None
        self._plans = []
        self.ref_approach_quats = None
        self._ref_wp_index = 0
        self._difficulty_id = -1
        self._grip_open = self.gripper.stroke_per_finger
        self._leaf_force = 0.0
        self._leaf_force_max = 0.0
        self._leaf_force_sum = 0.0
        self._leaf_bend = 0.0
        self._leaf_bend_max = 0.0
        self._leaf_energy = 0.0
        self._hard_force = 0.0
        self._touch_force = 0.0
        self._fruit_robot_force = 0.0
        self._target_geom = self.pool.g_neck
        self._plant_reason = None
        self._hard_pts = np.zeros((0, 3))
        self._hard_rad = np.zeros(0)
        self._x_hint = self.data.site_xmat[self.tcp_site_id].reshape(3, 3)[:, 0].copy()
        sub = set(self.finger_bodies) | {self.payload_body_id}
        self.gripper_geoms = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in sub}
        self._wrist_force = np.zeros(3)
        ob = dict(self.cfg.get("observation", {}) or {})
        # slot rintangan ke-4 diganti gaya pergelangan (3) + besarnya (1), dibagi wrist_force_scale (N)
        self._force_in_obs = bool(ob.get("wrist_force_in_last_obstacle_slot", False))
        self._force_scale = float(ob.get("wrist_force_scale", 20.0))

    # ------------------------------------------------------------------ #
    # Kurikulum
    # ------------------------------------------------------------------ #
    def set_difficulty(self, level: int, **params):
        """Parameter level (sac_config.yaml -> curriculum.levels): p_block, max_per_side, p_all_sides,
        rosette_min, rosette_max, stiffness_max (+ kunci lain diabaikan)."""
        self._difficulty_id = int(level)
        lv = {}
        if "p_block" in params:
            lv["p_block"] = float(params["p_block"])
        if "max_per_side" in params:
            lv["max_per_side"] = int(params["max_per_side"])
        if "p_all_sides" in params:
            lv["p_all_sides"] = float(params["p_all_sides"])
        if "rosette_min" in params or "rosette_max" in params:
            base = self.cfg["plant"]["rosette"]["count_range"]
            lv["rosette_count"] = (int(params.get("rosette_min", base[0])), int(params.get("rosette_max", base[1])))
        if "stiffness_max" in params:
            lo = self.cfg["plant"]["leaf"]["stiffness_range"][0]
            lv["stiffness_range"] = (float(lo), float(params["stiffness_max"]))
        self.sampler.set_level(**lv)

    # ------------------------------------------------------------------ #
    # Skenario
    # ------------------------------------------------------------------ #
    def _reach_filter(self, pts):
        rel = pts - self.base_position[None, :]
        r = np.linalg.norm(rel[:, :2], axis=1)
        R = self.hv["start_reach"]
        return (r >= R["r_min"]) & (r <= R["r_max"]) & (pts[:, 2] >= R["z_min"]) & (pts[:, 2] <= R["z_max"])

    def _sample_scenario(self):
        pl = self.hv["plant_placement"]
        R = self._rng.uniform(*pl["radius_range"])
        az = np.deg2rad(self._rng.uniform(*pl["azimuth_deg_range"]))
        ground = self.base_position + np.array([R * np.cos(az), R * np.sin(az), 0.0])
        ground[2] = 0.0
        layout = self.sampler.sample(self._rng, ground, self.base_position - ground)
        self.pool.apply(layout)
        self.pool.reset_state(self.data)
        self.data.eq_active[self.weld_id] = 0
        mujoco.mj_forward(self.model, self.data)
        self._n_pruned = self.pool.prune_leaves_near(self.data, self.caps, self._home_keepout)
        leaf_pts, leaf_rad = self._perceived_leaf_points()
        self._plans = self.planner.plan(layout, leaf_pts, leaf_rad, self.base_position, self._x_hint,
                                        reach_filter=self._reach_filter, top_k=self._top_k)
        self._hard_pts, self._hard_rad = self.pool.hard_points(self.data)
        self.plan = self._plans[0] if self._plans else None
        if self.plan is None:                      # tidak ada garis layak -> skenario ditolak validasi
            self.T1 = self.T2 = layout.neck_center.copy()
        else:
            self.T1, self.T2 = self.plan.start.copy(), self.plan.target.copy()

    def _perceived_leaf_points(self):
        """Titik daun yang DIPAKAI PERENCANA (garis stargaze), mensimulasikan persepsi kamera dari pose 'look':
        sebagian daun tidak terdeteksi (tertutup daun lain) dan posisinya berderau. Observasi policy selama gerak
        tetap memakai posisi daun aktual di dekat robot (kamera pergelangan melihat daun dari dekat).
        perception.leaf_miss_prob = 0 dan leaf_pos_noise = 0 -> perencana tahu semua daun persis."""
        pts, rad = self.pool.leaf_points(self.data)
        pc = self.cfg.get("perception", {}) or {}
        miss, noise = float(pc.get("leaf_miss_prob", 0.0)), float(pc.get("leaf_pos_noise", 0.0))
        if (miss <= 0 and noise <= 0) or len(pts) == 0:
            return pts, rad
        per = self.pool._per_seg * 1                         # titik per segmen
        seg_ids = np.arange(len(pts)) // per
        # satu daun = beberapa segmen; daun hilang utuh (bukan titik acak)
        leaf_of_seg = np.repeat(np.arange(len(self.pool.active)), [len(sg) for sg, _ in self.pool.active
                                                                   if self.model.geom_contype[sg[0]["geom"]] != 0])
        keep_leaf = self._rng.random(len(self.pool.active)) >= miss
        keep = keep_leaf[leaf_of_seg[seg_ids]] if len(leaf_of_seg) > seg_ids.max() else np.ones(len(pts), bool)
        noisy = pts + self._rng.normal(0.0, noise, pts.shape) if noise > 0 else pts
        self._n_leaf_missed = int((~keep_leaf).sum())
        return noisy[keep], rad[keep]

    def _hard_subset(self, which):
        return self.pool.hard_points(self.data, which=which)

    def _validate_scenario(self) -> bool:
        self.plan = None
        if not self._plans:
            return False
        chk = self.checker
        q0 = self.data.qpos.copy()
        tcp0 = self.data.site_xpos[self.tcp_site_id].copy()
        quat0 = np.zeros(4)
        mujoco.mju_mat2Quat(quat0, self.data.site_xmat[self.tcp_site_id])
        variants = []
        for plan in self._plans:                # gripper simetris: x alat +-  (putar 180 deg di sumbu z alat)
            variants.append(plan)
            alt = copy.copy(plan)
            alt.quat = _mul(plan.quat, _RZ_PI)
            variants.append(alt)
        for plan in variants:
            width = 2 * (self.pool.layout.neck_r if plan.mode == "crown" else self.pool.layout.fruit_r)
            opening = self.gripper.open_cmd(width)
            q0[self.finger_qadr] = opening
            ok, q1, segs_a, wps, quats = chk.approach_q(q0, tcp0, plan.start, quat0, plan.quat,
                                                       self.base_position[:2])
            if not ok:
                continue
            if chk.hard_clearance(segs_a, self._hard_pts, self._hard_rad) < self._approach_margin:
                continue
            line = chk.waypoints(plan.start, plan.target)
            ok, _, segs_b = chk.rollout_q(q1, line, [plan.quat] * len(line))
            if not ok:
                continue
            if chk.hard_clearance(segs_b, self._hard_pts, self._hard_rad, ~self.finger_cap_mask) < self._line_margin:
                continue
            target = "neck" if plan.mode == "crown" else "fruit"
            others = tuple(w for w in ("stalk", "fruit", "neck") if w != target)
            p_o, r_o = self._hard_subset(others)
            p_t, r_t = self._hard_subset((target,))
            if chk.hard_clearance(segs_b, p_o, r_o, self.finger_cap_mask) < self._finger_margin:
                continue
            if chk.hard_clearance(segs_b, p_t, r_t, self.finger_cap_mask) < self._finger_margin:
                continue
            # rencana diterima
            self.plan = plan
            self.T1, self.T2 = plan.start.copy(), plan.target.copy()
            self.ref_approach_quats = quats
            chk.last_wps = wps
            self._grip_open = opening
            self._target_geom = self.pool.g_neck if plan.mode == "crown" else self.pool.g_fruit
            self.data.qpos[self.finger_qadr] = opening
            self.data.ctrl[self.gripper_actuator_id] = opening
            mujoco.mj_forward(self.model, self.data)
            return True
        return False

    def _on_reset(self):
        self._leaf_force = self._leaf_force_max = self._leaf_force_sum = 0.0
        self._leaf_bend = self._leaf_bend_max = self._leaf_energy = 0.0
        self._hard_force = self._touch_force = self._fruit_robot_force = 0.0
        self._plant_reason = None
        self._ref_wp_index = 0
        self._ct_sum, self._ct_n = 0.0, 0

    # ------------------------------------------------------------------ #
    # Referensi orientasi
    # ------------------------------------------------------------------ #
    def reference_quat(self):
        if self.plan is None:
            return None
        if self.phase == 0 and self.ref_approach_quats is not None and len(self.ref_approach_quats):
            i = min(self._ref_wp_index + self.reference.cfg.lookahead, len(self.ref_approach_quats) - 1)
            return self.ref_approach_quats[i]
        return self.plan.quat

    def orientation_error(self) -> float:
        if self.plan is None:
            return 0.0
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self.data.site_xmat[self.tcp_site_id])
        return min(quat_angle(q, self.plan.quat), quat_angle(q, _mul(self.plan.quat, _RZ_PI)))

    # ------------------------------------------------------------------ #
    # Struktur residual Misi 4-5: koreksi RL yang TIDAK bisa menghentikan robot & tidak mengganggu presisi akhir
    # ------------------------------------------------------------------ #
    def _compose_residual_action(self, a_rl: np.ndarray) -> np.ndarray:
        """aksi = a_ref + g * skala * a_RL, dengan dua pembatas (hasil uji v1 Misi 4: RL 86% vs referensi 93%,
        kegagalan baru = robot BERHENTI di depan daun (timeout 2 -> 6) dan koreksi di dekat buah (fruit_hit 1 -> 3)):
          1. progres: komponen posisi searah a_ref dijaga >= `progress_min` x |a_ref| -> RL boleh membelok di
             samping daun & mengubah kecepatan, tetapi tidak bisa diam/mundur (tidak ada minimum lokal 'berhenti');
          2. gerbang target: g = clip((d - d0) / (d1 - d0), 0, 1), d = jarak TCP ke titik cengkeram -> 6-15 cm
             terakhir dikerjakan controller referensi (presisi jari di sekitar leher crown / buah)."""
        ref = self._ref_action
        g = 1.0
        if self._target_gate is not None and self.T2 is not None and self.phase >= 1:
            d0, d1 = self._target_gate
            dist = float(np.linalg.norm(self.T2 - self.tcp_pos))
            g = float(np.clip((dist - d0) / max(d1 - d0, 1e-9), 0.0, 1.0))
        self._gate = g
        a = ref + g * self.residual_scale * a_rl
        rp = ref[:3]
        n = float(np.linalg.norm(rp))
        if n > 1e-6 and self._progress_min > 0:
            u = rp / n
            along = float(np.dot(a[:3], u))
            lo = self._progress_min * n
            if along < lo:
                a[:3] += (lo - along) * u
        return np.clip(a, -1.0, 1.0)

    # ------------------------------------------------------------------ #
    # Perintah aksi -> sendi (orientasi target tidak boleh jauh mendahului orientasi aktual)
    # ------------------------------------------------------------------ #
    def _command(self, action: np.ndarray) -> dict:
        """Seperti BaseArmEnv._command, tetapi orientasi target dibatasi maks `max_orient_lead` dari orientasi
        AKTUAL gripper. Misi 4-5 memutar gripper sampai ~90-180 deg; sendi pergelangan hanya bisa bergerak
        ~0.05 rad/langkah, sehingga tanpa batas ini target orientasi berlari jauh di depan robot, solusi IK
        melompat, perubahan sendi terpotong `max_joint_delta`, dan TCP menyimpang (terukur: TCP turun 50 cm
        ke lantai walau perintah posisi ke atas)."""
        pos_delta = action[:3] * self.max_pos_delta
        rot_delta = action[3:] * self.max_rot_delta
        q_now = self.data.qpos[self.arm_qpos_adr]
        actual = np.zeros(4)
        mujoco.mju_mat2Quat(actual, self.data.site_xmat[self.tcp_site_id])
        base_q = limit_quat_deviation(self.target_quat, actual, self._orient_lead)
        quat = integrate_quat(base_q, rot_delta)
        quat = limit_quat_deviation(quat, actual, self._orient_lead + self.max_rot_delta)
        q_target, ik_info = self.ik_solver.solve(self.data.qpos, self.tcp_pos + pos_delta, quat)
        dq = q_target - q_now
        big = float(np.max(np.abs(dq)))
        if big > self.max_joint_delta:          # skala SERAGAM (arah gerak di ruang sendi tetap), bukan clip per sendi
            dq *= self.max_joint_delta / big
        return {"q_cmd": q_now + dq, "quat": quat, "ik_info": ik_info}

    # ------------------------------------------------------------------ #
    # Kontak & clearance
    # ------------------------------------------------------------------ #
    def _update_derived_state(self):
        m, d = self.model, self.data
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        self._manip = self.ik_solver.manipulability_from_data(d)

        self._contact_reason = None
        leaf_f = hard_f = touch_f = fruit_robot_f = 0.0
        f6 = np.zeros(6)
        wrist_f = np.zeros(3)
        leaf_set, hard_set, free_set = self.pool.leaf_geom_set, self.pool.hard_geom_set, set(self.pool.g_free)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            r1, r2 = bool(self.robot_geoms[g1]), bool(self.robot_geoms[g2])
            if r1 and r2:
                self._contact_reason = "self_collision"
                continue
            if (r1 and g2 == self.floor_geom) or (r2 and g1 == self.floor_geom):
                self._contact_reason = self._contact_reason or "floor_contact"
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            fn = abs(float(f6[0]))
            other = g2 if r1 else g1 if r2 else None
            if other is not None:
                rg = g1 if r1 else g2
                if rg in self.gripper_geoms:
                    # gaya pada gripper (dunia) = yang diukur sensor gaya/torsi pergelangan UR10e (tanpa inersia)
                    fw = c.frame.reshape(3, 3).T @ f6[:3]
                    wrist_f += fw if rg == g2 else -fw
                if other in leaf_set:
                    leaf_f += fn
                elif other in hard_set:
                    if rg in self.finger_geoms and other == self._target_geom:
                        touch_f += fn
                    else:
                        hard_f += fn
                elif other in free_set:
                    fruit_robot_f += fn
            elif (g1 in free_set and g2 in leaf_set) or (g2 in free_set and g1 in leaf_set):
                leaf_f += fn                                # buah yang dibawa menabrak daun (Misi 5)
        self._leaf_force, self._hard_force, self._touch_force = leaf_f, hard_f, touch_f
        self._fruit_robot_force = fruit_robot_f
        self._wrist_force = wrist_f
        self._leaf_force_max = max(self._leaf_force_max, leaf_f)
        self._leaf_force_sum += leaf_f
        # tekukan daun (sudut engsel dari bentuk istirahat) & energi pegas daun = ukuran "kerusakan" yang halus.
        # Gaya kontak sesaat TIDAK dipakai sbg kriteria gagal: pada 0.3-0.5 m/s gaya kontak pertama melonjak
        # 0 -> 150-280 N dalam satu langkah (impuls tumbukan), tidak bisa diantisipasi & bukan ukuran kerusakan.
        # Daun crown (tuft) tidak dihitung sbg KERUSAKAN tanaman: jari yang menjepit leher crown pasti menekan
        # daun crown (bagian buah yang dipanen; pemanen manual juga begitu). Terukur: 5 dari 11 kegagalan tekuk
        # referensi = daun crown di 2-4 cm terakhir. Energinya tetap masuk penalti kelembutan.
        adr, k, tuft = self.pool.active_leaf_joints()
        if adr.size:
            th = d.qpos[adr]
            dmg = ~tuft if self._ignore_tuft_bend else np.ones_like(tuft)
            self._leaf_bend = float(np.max(np.abs(th[dmg]))) if dmg.any() else 0.0
            self._leaf_energy = float(0.5 * np.sum(k * th * th))
        else:
            self._leaf_bend = self._leaf_energy = 0.0
        self._leaf_bend_max = max(self._leaf_bend_max, self._leaf_bend)

        pts, rad = self.pool.leaf_points(d)
        sp, sr = self.pool.hard_points(d, n=5, which=("stalk",))
        self.obstacle_points = np.vstack([pts, sp]) if len(sp) else pts
        self.obstacle_radii = np.concatenate([rad, sr]) if len(sr) else rad
        if self.obstacle_points.shape[0] > 0:
            self._obs_clear, self._obs_witness = obstacle_clearance(self.caps, d, self.obstacle_points,
                                                                    self.obstacle_radii)
            self._min_clear = float(np.min(self._obs_clear))
        else:
            self._obs_clear, self._obs_witness, self._min_clear = np.zeros(0), np.zeros((0, 3)), np.inf

    def _get_obs(self) -> np.ndarray:
        obs = super()._get_obs()
        if self._force_in_obs:
            from shared.envs.obs_layout import OBS_SLICES as S
            f = np.clip(self._wrist_force / self._force_scale, -5.0, 5.0)
            r0 = S["obs_rel"].start + 9
            obs[r0:r0 + 3] = f
            obs[S["obs_dist"].start + 3] = min(float(np.linalg.norm(self._wrist_force)) / self._force_scale, 5.0)
        return obs

    def _plant_termination(self):
        """(penalti, terminated, alasan) untuk kontak tanaman."""
        if self._hard_force > self._f_hard:
            return self._w("plant_hit_penalty", -50.0), True, "plant_hit"
        if self._touch_force > self._f_touch:
            return self._w("plant_hit_penalty", -50.0), True, "fruit_hit"
        if self._leaf_bend > self._bend_max:
            return self._w("leaf_overforce_penalty", -50.0), True, "leaf_overbend"
        if self._leaf_force > self._f_leaf_max:
            return self._w("leaf_overforce_penalty", -50.0), True, "leaf_overforce"
        return 0.0, False, None

    def _leaf_reward(self) -> float:
        # penalti energi tekuk daun (halus, bisa diantisipasi dari posisi daun yang teramati)
        r = -abs(self._w("leaf_energy_penalty", 0.15)) * min(self._leaf_energy / self._energy_ref,
                                                              self._w("leaf_energy_cap", 3.0))
        self._rc["leaf"] = r
        return r

    # ------------------------------------------------------------------ #
    # Reward
    # ------------------------------------------------------------------ #
    def _approach_reward(self):
        """Fase 0-1 (dipakai juga Misi 5). Return (reward, sukses_cengkeram)."""
        tcp = self.tcp_pos
        if self.phase == 0:
            dist = float(np.linalg.norm(self.T1 - tcp))
            reward = self._common_reward_terms(dist)
            if dist < self._t1_thr:
                self.phase = 1
                self._prev_dist = float(np.linalg.norm(self.T2 - tcp))
                b = self._w("phase_transition_bonus", 5.0)
                self._rc["phase_bonus"] = b
                reward += b
            return reward, False
        dist = float(np.linalg.norm(self.T2 - tcp))
        reward = self._common_reward_terms(dist)
        d_perp = perpendicular_distance_to_segment(tcp, self.T1, self.T2)
        cap = float(self.hv.get("cross_track_cap", 0.08))
        r_ct = -abs(self._w("cross_track_penalty", 150.0)) * min(d_perp, cap) ** 2
        ang = self.orientation_error()
        r_or = -abs(self._w("orientation_penalty", 0.5)) * ang
        self._rc["cross_track"], self._rc["orientation"] = r_ct, r_or
        self._ct_sum += d_perp
        self._ct_n += 1
        reward += r_ct + r_or
        return reward, (dist < self.success_pos_threshold and ang < self._orient_tol)

    def _compute_reward_and_termination(self):
        if self.plan is None:
            return 0.0, True, "no_plan"
        reward, at_grasp = self._approach_reward()
        reward += self._leaf_reward()
        pen, term, reason = self._contact_termination()
        if term:
            return reward + pen, True, reason
        pen, term, reason = self._plant_termination()
        if term:
            self._rc["plant"] = pen
            return reward + pen, True, reason
        if at_grasp:
            r = self._w("success_reward", 40.0)
            self._rc["success"] = r
            return reward + r, True, "success"
        return reward + self._timeout_term(), False, None

    def _timeout_term(self) -> float:
        """Penalti pada langkah terakhir episode (timeout) -> 'berhenti di depan daun' tidak lebih murah daripada
        mencoba lewat."""
        if self.step_count >= self.max_steps:
            r = self._w("timeout_penalty", -20.0)
            self._rc["timeout"] = r
            return r
        return 0.0

    # ------------------------------------------------------------------ #
    def _get_info(self) -> dict:
        info = super()._get_info()
        lay = self.pool.layout
        info["min_clearance"] = float(self._min_clear) if np.isfinite(self._min_clear) else float(OBS_FAR)
        info["difficulty_id"] = int(self._difficulty_id)
        info["leaf_force"] = float(self._leaf_force)
        info["leaf_force_max"] = float(self._leaf_force_max)
        info["wrist_force"] = float(np.linalg.norm(self._wrist_force))
        info["leaf_bend_deg"] = float(np.degrees(self._leaf_bend))
        info["leaf_bend_max_deg"] = float(np.degrees(self._leaf_bend_max))
        info["leaf_energy"] = float(self._leaf_energy)
        info["leaf_impulse"] = float(self._leaf_force_sum * self.model.opt.timestep * self.n_substeps)
        info["orient_err_deg"] = float(np.degrees(self.orientation_error()))
        if self.plan is not None:
            info.update(grip_mode=self.plan.mode, approach_face=self.plan.face,
                        line_clearance=float(self.plan.clearance), approach_elev_deg=float(self.plan.elevation_deg))
        if lay is not None:
            info.update(blocked_sides=",".join(lay.blocked_sides), n_blocked=len(lay.blocked_sides),
                        fruit_diameter=2 * lay.fruit_r, neck_diameter=2 * lay.neck_r,
                        n_leaves=sum(1 for l in lay.leaves if l.kind != "tuft"))
        if self.T1 is not None and self.T2 is not None:
            info["cross_track"] = perpendicular_distance_to_segment(self.tcp_pos, self.T1, self.T2)
        if self._ct_n > 0:
            info["cross_track_mean"] = self._ct_sum / self._ct_n
        return info
