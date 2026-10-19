"""
Mission5/envs/stage5_harvest_env.py
-----------------------------------
Misi 5: PANEN LENGKAP (simulasi) = Misi 4 + cengkeram + putar (twist) + tarik + angkat keluar dari daun.

Fase:
  0 = home -> S (rute pendekatan)                                   [RL + referensi]
  1 = S -> titik cengkeram sepanjang garis stargaze                  [RL + referensi]
      -> urutan PANEN otomatis (skrip, tidak dikontrol RL):
         a. jari menutup (aktuator gaya terbatas) sampai menjepit objek; cek KEDUA jari menyentuh objek
         b. putar (twist) gripper mengelilingi sumbu buah sampai sudut patah batang (acak), TCP tetap
         c. batang dianggap patah bila kapasitas torsi cengkeraman >= torsi patah batang (acak), selain itu
            cengkeraman SELIP (gagal). Buah lalu ikut gripper (weld) dan bertabrakan dgn daun.
  2 = tarik mundur sepanjang garis sejauh retreat_distance (titik R) sambil membawa buah   [RL + referensi]
  3 = angkat ke titik keluar E = R + (0, 0, lift_height)                                [RL + referensi]

Sukses: buah terbawa sampai E (|TCP - E| < exit_threshold) tanpa kegagalan.
Gagal tambahan Misi 5: grasp_miss (jari tidak menjepit objek), twist_fail (IK/tabrakan saat memutar),
grip_slip (torsi/gaya cengkeraman < kebutuhan patah batang), fruit_hit_robot (buah menabrak lengan),
fruit_floor (buah menyentuh tanah), ditambah semua kegagalan Misi 4.

CATATAN MODEL (jujur): kontak cengkeram & patahnya batang TIDAK disimulasikan dgn gesekan penuh. Cengkeraman =
penjepitan fisik jari (dicek dari kontak) + weld; patah batang = aturan torsi:
    torsi_cengkeram = 2 * mu * F_jari * r_objek   (F_jari = grip_force_frac * 140 N, mu acak)
    sukses bila torsi_cengkeram >= torsi_patah (acak) dan gaya tarik 2*mu*F_jari >= gaya_patah (acak).
Rentang torsi/gaya patah batang & leher crown nanas belum diketahui -> rentang lebar di config, ukur di lapangan.
Risiko nyata yang TIDAK dimodelkan: twist pada crown bisa mencabut crown dari buah (bukan buah dari batang).
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from ik.pose_utils import perpendicular_distance_to_segment, project_point_on_segment
from Mission4.envs.stage4_harvest_env import Stage4HarvestEnv, _mul

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = str(_PROJECT_ROOT / "robot_model" / "scene_harvest.xml")
DEFAULT_CONFIG = str(_PROJECT_ROOT / "Mission5" / "configs" / "domain_randomization.yaml")


def _quat_z(angle: float) -> np.ndarray:
    return np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])


class Stage5HarvestEnv(Stage4HarvestEnv):
    MISSION_ID = 5

    def __init__(self, model_path: str = DEFAULT_MODEL, config_path: str = DEFAULT_CONFIG,
                 obs_noise_std: float | None = None, training: bool = True,
                 seed: int | None = None, randomize: bool | None = None):
        super().__init__(model_path=model_path, config_path=config_path, obs_noise_std=obs_noise_std,
                         training=training, seed=seed, randomize=randomize)
        h = self.cfg["harvest_action"]
        self.ha = h
        self._exit_thr = float(h.get("exit_threshold", 0.03))
        self._lift_h = float(h.get("lift_height", 0.08))
        self._retreat = float(h.get("retreat_distance", 0.22))
        self.E = None
        self.R = None
        self.q_hold = None
        self.grasp_point = None
        self._detached = False
        self._harvest_log: dict = {}
        self._extra_steps = 0
        self.frame_callback = None
        G = mujoco.mjtObj.mjOBJ_GEOM
        self.finger_geom_l = mujoco.mj_name2id(self.model, G, "finger_left_geom")
        self.finger_geom_r = mujoco.mj_name2id(self.model, G, "finger_right_geom")

    # ------------------------------------------------------------------ #
    def _on_reset(self):
        super()._on_reset()
        self.data.eq_active[self.weld_id] = 0
        self._detached = False
        self.q_hold = None
        self.grasp_point = None
        self.E = None
        self.R = None
        self._harvest_log = {}
        self._extra_steps = 0

    def active_target(self) -> np.ndarray:
        if self.phase == 2:
            return self.R
        if self.phase == 3:
            return self.E
        return super().active_target()

    def _segment(self):
        if self.phase == 2:
            return self.grasp_point, self.R
        if self.phase == 3:
            return self.R, self.E
        return self.T1, self.T2

    def _path_features(self, tcp: np.ndarray):
        a, b = self._segment()
        if a is None or b is None:
            return np.zeros(3), np.zeros(3), 0.0
        seg = b - a
        direction = seg / (np.linalg.norm(seg) + 1e-9)
        t, proj = project_point_on_segment(tcp, a, b)
        return direction, proj - tcp, t

    def reference_quat(self):
        # setelah batang patah: kembali ke orientasi sebelum twist (rute garis layak secara IK dgn orientasi itu;
        # orientasi hasil twist bisa membuat titik mundur di luar jangkauan pergelangan)
        if self.phase >= 2 and self.plan is not None:
            return self.plan.quat
        return super().reference_quat()

    # ------------------------------------------------------------------ #
    # Urutan panen (skrip)
    # ------------------------------------------------------------------ #
    def _sim(self, n_ctrl_steps: int = 1):
        for _ in range(n_ctrl_steps):
            for _ in range(self.n_substeps):
                mujoco.mj_step(self.model, self.data)
            self._extra_steps += 1
            if self.frame_callback is not None:          # perekam video (Mission5/debug/render_harvest.py)
                self.frame_callback()
        self._update_derived_state()

    def _hold_arm(self):
        self.data.ctrl[self.arm_actuator_ids] = self.data.qpos[self.arm_qpos_adr]

    def _attach_fruit(self):
        """Pindahkan buah lepas ke pose buah di tanaman, aktifkan weld ke gripper, sembunyikan buah tanaman."""
        m, d, pool = self.model, self.data, self.pool
        fc = d.geom_xpos[pool.g_fruit].copy()
        q_f = np.zeros(4)
        mujoco.mju_mat2Quat(q_f, d.geom_xmat[pool.g_fruit])
        qa = pool.free_qadr
        d.qpos[qa:qa + 3] = fc
        d.qpos[qa + 3:qa + 7] = q_f
        d.qvel[pool.free_dadr:pool.free_dadr + 6] = 0.0
        for g in pool.g_free:
            m.geom_contype[g], m.geom_conaffinity[g] = 4, 3
        pool.hide_attached_fruit()
        mujoco.mj_forward(m, d)
        # relpose weld = pose buah dalam frame body gripper
        b1, b2 = int(m.eq_obj1id[self.weld_id]), int(m.eq_obj2id[self.weld_id])
        p1, p2 = d.xpos[b1], d.xpos[b2]
        R1 = d.xmat[b1].reshape(3, 3)
        q1, q2 = d.xquat[b1].copy(), d.xquat[b2].copy()
        q1c = q1.copy()
        q1c[1:] *= -1
        relq = np.zeros(4)
        mujoco.mju_mulQuat(relq, q1c, q2)
        m.eq_data[self.weld_id, 0:3] = 0.0
        m.eq_data[self.weld_id, 3:6] = R1.T @ (p2 - p1)
        m.eq_data[self.weld_id, 6:10] = relq
        d.eq_active[self.weld_id] = 1

    def _harvest_sequence(self):
        """Return (bonus_reward, gagal_reason|None)."""
        h, lay, plan = self.ha, self.pool.layout, self.plan
        rng = self._rng
        log = {}
        target = self._target_geom
        self.grasp_point = self.tcp_pos.copy()
        # --- a. menutup jari, dgn penyesuaian (compliance) lengan di sumbu tutup jari ---
        # Gripper paralel pada objek yang diam: bila TCP meleset beberapa mm di sumbu tutup, jari pertama yang
        # menyentuh menghentikan keduanya (jari tersinkron) -> hanya satu jari menjepit. Robot nyata memakai mode
        # gaya/compliance; di sini lengan menggeser TCP ke tengah objek selama jari menutup.
        self._hold_arm()
        self.data.ctrl[self.gripper_actuator_id] = 0.0
        n_close = int(h.get("close_steps", 35))               # 0.7 s
        obj_c = self.data.geom_xpos[target].copy()
        for _ in range(n_close):
            x_t = self.data.site_xmat[self.tcp_site_id].reshape(3, 3)[:, 0]
            off = float(np.dot(obj_c - self.tcp_pos, x_t))
            goal = self.tcp_pos + np.clip(0.5 * off, -0.004, 0.004) * x_t
            q_arm, _ = self.ik_solver.solve(self.data.qpos, goal, self.target_quat)
            self.data.ctrl[self.arm_actuator_ids] = q_arm
            self._sim(1)
            if self._contact_reason is not None:
                return 0.0, self._contact_reason
        self.grasp_point = self.tcp_pos.copy()
        d = self.data
        touch_l = touch_r = False
        for i in range(d.ncon):
            c = d.contact[i]
            pair = {int(c.geom1), int(c.geom2)}
            if target in pair and self.finger_geom_l in pair:
                touch_l = True
            if target in pair and self.finger_geom_r in pair:
                touch_r = True
        width = 2 * (lay.neck_r if plan.mode == "crown" else lay.fruit_r)
        open_now = float(d.qpos[self.finger_qadr[0]])
        log.update(grasp_touch_left=touch_l, grasp_touch_right=touch_r, finger_open_after_close=open_now)
        if not (touch_l and touch_r):
            self._harvest_log = log
            return 0.0, "grasp_miss"
        self.data.ctrl[self.gripper_actuator_id] = open_now      # jepit pada lebar objek
        self._attach_fruit()
        bonus = self._w("grasp_bonus", 10.0)

        # --- b. twist mengelilingi sumbu buah (vertikal) lewat IK, TCP tetap ---
        ang_break = np.deg2rad(rng.uniform(*h["twist_break_deg_range"]))
        step = np.deg2rad(float(h.get("twist_step_deg", 4.0)))
        q0 = np.zeros(4)
        mujoco.mju_mat2Quat(q0, d.site_xmat[self.tcp_site_id])
        sign = self._plan_twist_direction(q0, ang_break, step)
        log["twist_dir"] = sign
        if sign == 0.0:
            self._harvest_log = log
            return bonus, "twist_fail"
        theta = 0.0
        while theta < ang_break:
            theta = min(ang_break, theta + step)
            q_t = _mul(_quat_z(sign * theta), q0)
            q_arm, info = self.ik_solver.solve(d.qpos, self.grasp_point, q_t)
            q_now = d.qpos[self.arm_qpos_adr]
            dq = q_arm - q_now
            big = float(np.max(np.abs(dq)))
            if big > self.max_joint_delta:
                dq *= self.max_joint_delta / big
            d.ctrl[self.arm_actuator_ids] = q_now + dq
            self._sim(1)
            for _ in range(4):                                  # tunggu sendi menyusul (gerak pergelangan besar)
                if np.max(np.abs(d.ctrl[self.arm_actuator_ids] - d.qpos[self.arm_qpos_adr])) < 0.03:
                    break
                self._sim(1)
            if self._contact_reason is not None:
                self._harvest_log = log
                return bonus, self._contact_reason
            if self._leaf_bend > self._bend_max:
                self._harvest_log = log
                return bonus, "leaf_overbend"
            if info["position_error"] > 0.01 or np.linalg.norm(self.tcp_pos - self.grasp_point) > 0.03:
                log["twist_deg_reached"] = float(np.degrees(theta))
                self._harvest_log = log
                return bonus, "twist_fail"
        for _ in range(3):                                      # sendi menyusul target terakhir
            self._sim(1)
        log["twist_deg"] = float(np.degrees(ang_break))
        mujoco.mju_mat2Quat(self.target_quat, d.site_xmat[self.tcp_site_id])
        self.q_hold = self.target_quat.copy()

        # --- c. aturan patah batang vs kapasitas cengkeraman ---
        mu = rng.uniform(*h["friction_range"])
        F = self.gripper.grip_force_frac * self.gripper.max_force
        r_obj = 0.5 * width
        tau_cap, pull_cap = 2 * mu * F * r_obj, 2 * mu * F
        tau_need = rng.uniform(*h["break_torque_range"])
        pull_need = rng.uniform(*h["break_pull_range"])
        log.update(mu=mu, tau_cap=tau_cap, tau_need=tau_need, pull_cap=pull_cap, pull_need=pull_need)
        self._harvest_log = log
        if tau_cap < tau_need or pull_cap < pull_need:
            return bonus, "grip_slip"
        self._detached = True
        return bonus + self._w("detach_bonus", 10.0), None

    def _plan_twist_direction(self, q0, ang, step) -> float:
        """Pilih arah putar (+/-) yang layak secara IK & tanpa self-collision/lantai (simulasi kinematik
        di data scratch checker, seperti perencana gerak robot nyata). 0 = tidak ada yang layak."""
        chk = self.checker
        best, best_score = 0.0, -np.inf
        for sign in (1.0, -1.0):
            q_full = self.data.qpos.copy()
            ok, theta, min_s2 = True, 0.0, 1.0
            while theta < ang:
                theta = min(ang, theta + step)
                q_arm, info = self.ik_solver.solve(q_full, self.grasp_point, _mul(_quat_z(sign * theta), q0))
                if info["position_error"] > 0.003 or np.max(np.abs(q_arm - q_full[self.arm_qpos_adr])) > 0.5:
                    ok = False
                    break
                q_full[self.arm_qpos_adr] = q_arm
                if chk.has_contact(q_full):
                    ok = False
                    break
                min_s2 = min(min_s2, abs(np.sin(q_arm[4])))
            if ok and min_s2 > best_score:
                best, best_score = sign, min_s2
        return best

    # ------------------------------------------------------------------ #
    # Reward
    # ------------------------------------------------------------------ #
    def _carry_termination(self):
        if self._fruit_robot_force > float(self.cfg["contact"].get("fruit_robot_limit", 5.0)):
            return self._w("plant_hit_penalty", -50.0), True, "fruit_hit_robot"
        d = self.data
        free = set(self.pool.g_free)
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 == self.floor_geom and c.geom2 in free) or (c.geom2 == self.floor_geom and c.geom1 in free):
                return self._w("plant_hit_penalty", -50.0), True, "fruit_floor"
        return 0.0, False, None

    def _compute_reward_and_termination(self):
        if self.plan is None:
            return 0.0, True, "no_plan"
        if self.phase <= 1:
            reward, at_grasp = self._approach_reward()
            reward += self._leaf_reward()
            pen, term, reason = self._contact_termination()
            if term:
                return reward + pen, True, reason
            pen, term, reason = self._plant_termination()
            if term:
                self._rc["plant"] = pen
                return reward + pen, True, reason
            if not at_grasp:
                return reward + self._timeout_term(), False, None
            bonus, fail = self._harvest_sequence()
            self._rc["harvest"] = bonus
            reward += bonus
            if fail is not None:
                pen = self._w("harvest_fail_penalty", -30.0)
                self._rc["harvest_fail"] = pen
                return reward + pen, True, fail
            self.phase = 2
            out = self.T1 - self.grasp_point
            L = float(np.linalg.norm(out))
            self.R = self.grasp_point + out * min(1.0, self._retreat / max(L, 1e-9))
            self.E = self.R + np.array([0.0, 0.0, self._lift_h])
            self._prev_dist = float(np.linalg.norm(self.R - self.tcp_pos))
            return reward, False, None

        # fase 2-3: membawa buah keluar
        tcp = self.tcp_pos
        tgt = self.active_target()
        dist = float(np.linalg.norm(tgt - tcp))
        reward = self._common_reward_terms(dist)
        a, b = self._segment()
        d_perp = perpendicular_distance_to_segment(tcp, a, b)
        cap = float(self.hv.get("cross_track_cap", 0.08))
        r_ct = -abs(self._w("cross_track_penalty", 150.0)) * min(d_perp, cap) ** 2
        self._rc["cross_track"] = r_ct
        reward += r_ct + self._leaf_reward()
        for fn in (self._contact_termination, self._plant_termination, self._carry_termination):
            pen, term, reason = fn()
            if term:
                return reward + pen, True, reason
        if self.phase == 2 and dist < float(self.cfg["episode"].get("t1_reach_threshold", 0.03)):
            self.phase = 3
            self._prev_dist = float(np.linalg.norm(self.E - tcp))
            b = self._w("phase_transition_bonus", 5.0)
            self._rc["phase_bonus"] = b
            return reward + b, False, None
        if self.phase == 3 and dist < self._exit_thr:
            r = self._w("success_reward", 40.0)
            self._rc["success"] = r
            return reward + r, True, "success"
        return reward + self._timeout_term(), False, None

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # langkah internal urutan panen ikut menghabiskan waktu episode
        if self._extra_steps:
            self.step_count += self._extra_steps
            self._extra_steps = 0
            if not terminated and self.step_count >= self.max_steps:
                truncated = True
                info["termination_reason"] = "timeout"
                info["is_success"] = False
        info["steps_total"] = int(self.step_count)
        info["detached"] = bool(self._detached)
        info.update({f"harvest_{k}": v for k, v in self._harvest_log.items()})
        return obs, reward, terminated, truncated, info
