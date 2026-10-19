"""
robot_model/build_harvest_model.py
----------------------------------
Membangun model MuJoCo untuk Misi 4-5 (panen nanas):

  robot_model/scene_harvest.xml  =  UR10e (kinematika & dinamika lengan IDENTIK dgn ur10e_train.xml)
                                    + gripper paralel listrik tipe DH-Robotics PGIA-140-80 (aproksimasi)
                                    + kamera pergelangan (link 6 / wrist 3)
                                    + satu tanaman nanas "pool" (batang, buah, leher crown, daun fleksibel)

Jalankan ulang hanya bila ukuran pool / gripper diubah:
    python robot_model/build_harvest_model.py

Gripper (aproksimasi dari datasheet publik PGI/PGIA-140-80, HARUS diverifikasi dgn unit asli):
  * badan 95 x 67.1 x 86 mm, massa ~1.0 kg (+ jari ~0.25 kg, + kamera ~0.1 kg)
  * stroke total 80 mm (40 mm per jari), gaya cengkeram 40-140 N per jari, waktu menutup ~0.7 s
  * jari: batang 110 mm, dipasang dgn OFFSET dari sumbu tengah -> rentang celah jari
    = [2*offset, 2*offset + 80 mm]. Offset diatur di Mission4/configs/domain_randomization.yaml
    (gripper.finger_offset); geometri XML di sini memakai nilai default dan disesuaikan env saat runtime.

Tanaman (semua ukuran DIACAK per episode oleh env; di XML hanya nilai placeholder):
  * "pool" tetap: N_ROSETTE daun roset (4 segmen), N_BLOCK daun penghalang (3 segmen),
    N_TUFT daun crown (2 segmen). Daun yang tidak dipakai disembunyikan (tanpa kontak).
  * Tiap sambungan segmen = 2 engsel (tekuk atas/bawah & samping) dgn pegas -> daun melentur saat didorong
    dan kembali ke bentuk semula. gravcomp=1 (bentuk istirahat = bentuk yang digambar generator).

Kelas kontak (bit contype/conaffinity):
  robot  : contype 1, conaffinity 1|2|4   -> kena lantai, dirinya sendiri, daun, bagian keras tanaman
  lantai : 1 / 1
  daun   : contype 2, conaffinity 0       -> HANYA kena robot (dan buah yang sedang dibawa); tidak antar-daun
  keras  : contype 4, conaffinity 0       -> batang, buah (masih menempel), leher crown
  buah-dibawa (setelah lepas): contype 4, conaffinity 1|2 -> kena daun & lantai; dikecualikan dgn gripper
"""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "ur10e_train.xml"
OUT_ROBOT = HERE / "ur10e_harvest.xml"
OUT_SCENE = HERE / "scene_harvest.xml"

N_ROSETTE, SEG_ROSETTE = 14, 4
N_BLOCK, SEG_BLOCK = 8, 3
N_TUFT, SEG_TUFT = 4, 2

FINGER_OFFSET = 0.015      # m, jarak permukaan dalam jari ke sumbu tengah saat menutup penuh (default)
FINGER_LEN = 0.11
FINGER_R = 0.009
PALM_HALF = (0.0475, 0.0336, 0.043)     # 95 x 67.1 x 86 mm
TCP_DEPTH = 0.065                        # TCP = titik tengah bantalan jari (dari ujung depan badan gripper)

GRIPPER_XML = f"""

                  <!-- GRIPPER PGIA-140-80 (aproksimasi). Nama body 'Servo-v1' dipertahankan (payload env). -->
                  <body name="Servo-v1" gravcomp="1" pos="0 0.1 0" quat="-1 1 0 0">
                    <inertial mass="1.35" pos="0 0 0.045" diaginertia="0.0016 0.0013 0.0011"/>
                    <geom name="palm" class="link_geom" type="box" material="black"
                          size="{PALM_HALF[0]} {PALM_HALF[1]} {PALM_HALF[2]}" pos="0 0 {PALM_HALF[2]}"/>
                    <!-- 2 kapsul (sumbu x alat) yang membungkus box palm -> dipakai perhitungan clearance
                         (box tidak ikut kapsul). Tanpa kontak fisik (kontak fisik = box 'palm'). -->
                    <geom name="palm_caps_a" class="link_geom" type="capsule" rgba="0 0 0 0" contype="0" conaffinity="0"
                          fromto="-0.0139 0 0.026 0.0139 0 0.026" size="0.0336"/>
                    <geom name="palm_caps_b" class="link_geom" type="capsule" rgba="0 0 0 0" contype="0" conaffinity="0"
                          fromto="-0.0139 0 0.060 0.0139 0 0.060" size="0.0336"/>
                    <site name="tcp_site" pos="0 0 {2 * PALM_HALF[2] + TCP_DEPTH:.4f}" size="0.006" rgba="1 0 0 1"/>
                    <site name="palm_front" pos="0 0 {2 * PALM_HALF[2]:.4f}" size="0.004" rgba="0 1 0 1"/>
                    <!-- KAMERA PERGELANGAN (ikut link 6 / wrist 3). Kamera biasa (RGB) pada bracket di sisi +y
                         gripper, menghadap arah pendekatan (sumbu z alat). -->
                    <geom name="wrist_camera" class="link_geom" type="capsule" material="black"
                          fromto="-0.025 0.052 0.03 0.025 0.052 0.03" size="0.016"/>
                    <camera name="wrist_cam" pos="0 0.052 0.05" xyaxes="-1 0 0 0 1 0" fovy="70"/>

                    <body name="finger_left" gravcomp="1" pos="{FINGER_OFFSET + FINGER_R} 0 {2 * PALM_HALF[2]}">
                      <inertial mass="0.12" pos="0 0 0.055" diaginertia="0.00012 0.00012 0.00001"/>
                      <joint name="finger_left_joint" type="slide" axis="1 0 0" range="0 0.04" damping="20"/>
                      <geom name="finger_left_geom" class="link_geom" type="capsule" material="jointgray"
                            size="{FINGER_R} {FINGER_LEN / 2}" pos="0 0 {FINGER_LEN / 2}" friction="0.8 0.02 0.001"/>
                    </body>
                    <body name="finger_right" gravcomp="1" pos="{-(FINGER_OFFSET + FINGER_R)} 0 {2 * PALM_HALF[2]}">
                      <inertial mass="0.12" pos="0 0 0.055" diaginertia="0.00012 0.00012 0.00001"/>
                      <joint name="finger_right_joint" type="slide" axis="-1 0 0" range="0 0.04" damping="20"/>
                      <geom name="finger_right_geom" class="link_geom" type="capsule" material="jointgray"
                            size="{FINGER_R} {FINGER_LEN / 2}" pos="0 0 {FINGER_LEN / 2}" friction="0.8 0.02 0.001"/>
                    </body>
                  </body>
"""


def _leaf_chain(prefix: str, n_seg: int, rgba: str, indent: str) -> str:
    """Rantai kapsul sepanjang sumbu +x lokal. Panjang/arah/kekakuan diatur env saat runtime."""
    seg_len = 0.15
    out, close = [], []
    for s in range(n_seg):
        pos = "0 0 0" if s == 0 else f"{seg_len} 0 0"
        ind = indent + "  " * s
        out.append(f'{ind}<body name="{prefix}_s{s}" pos="{pos}" gravcomp="1">')
        out.append(f'{ind}  <inertial mass="0.02" pos="{seg_len / 2} 0 0" diaginertia="0.00002 0.00004 0.00004"/>')
        out.append(f'{ind}  <joint name="{prefix}_j{s}a" type="hinge" axis="0 1 0" class="leafjoint"/>')
        out.append(f'{ind}  <joint name="{prefix}_j{s}b" type="hinge" axis="0 0 1" class="leafjoint"/>')
        out.append(f'{ind}  <geom name="{prefix}_g{s}" class="leaf" size="0.012 {seg_len / 2}" '
                   f'pos="{seg_len / 2} 0 0" quat="0.70710678 0 0.70710678 0" rgba="{rgba}"/>')
        close.append(f"{ind}</body>")
    return "\n".join(out + close[::-1])


def build_robot_xml() -> str:
    src = SRC.read_text(encoding="utf-8")
    src = src.replace('<mujoco model="ur10e_with_gripper_train">', '<mujoco model="ur10e_harvest">')
    # robot juga bertabrakan dgn daun (bit 2) & bagian keras tanaman (bit 4)
    src = src.replace('<geom type="capsule" group="1" contype="1" conaffinity="1" rgba="0.6 0.6 0.65 1"/>',
                      '<geom type="capsule" group="1" contype="1" conaffinity="7" rgba="0.6 0.6 0.65 1"/>')
    # ganti blok gripper lama (Servo-v1 + 2 jari slide) dgn PGIA-140-80
    start = src.index("                  <!-- GRIPPER ASSEMBLY")
    end = src.index("                </body>\n              </body>\n            </body>", start)
    src = src[:start] + GRIPPER_XML.strip("\n") + "\n\n" + src[end:]
    # equality, actuator, keyframe diganti di scene (nq berubah) -> hapus dari file robot
    src = re.sub(r"\s*<!-- Sinkronisasi.*?</equality>", "", src, flags=re.S)
    src = re.sub(r"\s*<actuator>.*?</actuator>", "", src, flags=re.S)
    src = re.sub(r"\s*<keyframe>.*?</keyframe>", "", src, flags=re.S)
    src = src.replace("VERSI TRAINING (SIMPLIFIED).",
                      "VERSI PANEN (Misi 4-5) - DIBANGKITKAN oleh build_harvest_model.py, jangan diedit manual.\n"
                      "    Lengan identik dgn ur10e_train.xml; gripper = aproksimasi PGIA-140-80.")
    return src


def build_scene_xml() -> str:
    leaves = []
    for i in range(N_ROSETTE):
        leaves.append(_leaf_chain(f"ros{i}", SEG_ROSETTE, "0.25 0.55 0.2 1", "      "))
    for i in range(N_BLOCK):
        leaves.append(_leaf_chain(f"blk{i}", SEG_BLOCK, "0.35 0.6 0.15 1", "      "))
    tuft = [_leaf_chain(f"tuft{i}", SEG_TUFT, "0.3 0.6 0.25 1", "        ") for i in range(N_TUFT)]

    n_leaf_joints = 2 * (N_ROSETTE * SEG_ROSETTE + N_BLOCK * SEG_BLOCK + N_TUFT * SEG_TUFT)
    qpos_home = ["-1.5708 -1.5708 1.5708 -1.5708 -1.5708 0", "0.04 0.04"]
    qpos_home += ["0"] * n_leaf_joints
    qpos_home += ["0 0 -3 1 0 0 0"]          # buah lepas: diparkir di bawah lantai sampai terlepas
    ctrl_home = "-1.5708 -1.5708 1.5708 -1.5708 -1.5708 0 0.04"

    return f"""<mujoco model="ur10e harvest scene (Misi 4-5)">
  <!-- DIBANGKITKAN oleh robot_model/build_harvest_model.py - jangan diedit manual. -->
  <include file="ur10e_harvest.xml"/>

  <!-- midphase (BVH per body) DIMATIKAN: env mengubah posisi/ukuran geom tanaman tiap episode, sedangkan BVH
       dihitung sekali saat kompilasi -> pasangan kontak jari-leher crown / robot-batang bisa terlewat
       (terukur: jari menutup penuh menembus leher crown tanpa kontak). Broadphase memakai rbound/aabb yang
       diperbarui env. -->
  <option timestep="0.002" integrator="implicitfast">
    <flag midphase="disable"/>
  </option>
  <size memory="40M"/>

  <default>
    <default class="leaf">
      <geom type="capsule" contype="2" conaffinity="0" condim="3" friction="0.4 0.01 0.001"
            solref="0.01 1" solimp="0.9 0.95 0.002 0.5 2" group="2"/>
    </default>
    <default class="leafjoint">
      <joint stiffness="3" damping="0.08" armature="0.0005" limited="false"/>
    </default>
    <default class="hardplant">
      <geom contype="4" conaffinity="0" condim="3" friction="0.8 0.02 0.001" group="2"/>
    </default>
  </default>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20" offwidth="640" offheight="480"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.35 0.28 0.2" rgb2="0.3 0.24 0.17"
      markrgb="0.5 0.4 0.3" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.05"/>
    <material name="fruit" rgba="0.85 0.6 0.15 1"/>
    <material name="stalk" rgba="0.4 0.5 0.25 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 2.0" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" contype="1" conaffinity="1"/>

    <!-- TANAMAN (posisi/ukuran diatur env tiap episode). Origin body = pangkal batang di tanah. -->
    <body name="plant" pos="0 0.7 0">
      <geom name="stalk" class="hardplant" type="capsule" size="0.02 0.15" pos="0 0 0.15" material="stalk"/>
      <geom name="fruit_att" class="hardplant" type="capsule" size="0.06 0.05" pos="0 0 0.4" material="fruit"/>
      <geom name="neck" class="hardplant" type="capsule" size="0.022 0.015" pos="0 0 0.52" rgba="0.4 0.55 0.2 1"/>
      <site name="fruit_center" pos="0 0 0.4" size="0.01" rgba="1 1 0 0.5"/>
      <site name="grasp_target" pos="0 0 0.52" size="0.008" rgba="0 1 1 0.8"/>
      <body name="tuft_root" pos="0 0 0.55">
{chr(10).join(tuft)}
      </body>
{chr(10).join(leaves)}
    </body>

    <!-- BUAH SETELAH LEPAS (freejoint). Diparkir jauh; env memindahkannya ke pose buah saat batang patah. -->
    <body name="fruit_free" pos="0 0 -3">
      <freejoint name="fruit_free_joint"/>
      <inertial mass="1.5" pos="0 0 0" diaginertia="0.006 0.006 0.003"/>
      <geom name="fruit_free_body" type="capsule" size="0.06 0.05" contype="0" conaffinity="0"
            material="fruit" friction="0.8 0.02 0.001"/>
      <geom name="fruit_free_neck" type="capsule" size="0.022 0.015" pos="0 0 0.12" contype="0" conaffinity="0"
            rgba="0.4 0.55 0.2 1"/>
      <geom name="fruit_free_tuft" type="capsule" size="0.04 0.07" pos="0 0 0.22" contype="0" conaffinity="0"
            rgba="0.3 0.6 0.25 1"/>
    </body>
  </worldbody>

  <contact>
    <exclude body1="fruit_free" body2="Servo-v1"/>
    <exclude body1="fruit_free" body2="finger_left"/>
    <exclude body1="fruit_free" body2="finger_right"/>
    <exclude body1="fruit_free" body2="wrist_3_link"/>
  </contact>

  <equality>
    <!-- kedua jari bergerak simetris -->
    <joint name="finger_sync" joint1="finger_left_joint" joint2="finger_right_joint" polycoef="0 1 0 0 0"
           solref="0.004 1"/>
    <!-- buah dipegang gripper (diaktifkan env setelah cengkeraman valid) -->
    <weld name="fruit_in_gripper" body1="Servo-v1" body2="fruit_free" active="false" solref="0.004 1"/>
  </equality>

  <actuator>
    <general class="size4" name="shoulder_pan" joint="shoulder_pan_joint"/>
    <general class="size4" name="shoulder_lift" joint="shoulder_lift_joint"/>
    <general class="size3_limited" name="elbow" joint="elbow_joint"/>
    <general class="size2" name="wrist_1" joint="wrist_1_joint"/>
    <general class="size2" name="wrist_2" joint="wrist_2_joint"/>
    <general class="size2" name="wrist_3" joint="wrist_3_joint"/>
    <!-- ctrl = bukaan per jari (m), 0 = menutup penuh, 0.04 = membuka penuh; gaya dibatasi 140 N -->
    <position name="gripper_actuator" joint="finger_left_joint" ctrlrange="0 0.04" kp="3000"
              forcerange="-140 140"/>
  </actuator>

  <keyframe>
    <key name="home" qpos="{' '.join(qpos_home)}" ctrl="{ctrl_home}"/>
  </keyframe>
</mujoco>
"""


def main():
    OUT_ROBOT.write_text(build_robot_xml(), encoding="utf-8")
    OUT_SCENE.write_text(build_scene_xml(), encoding="utf-8")
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(OUT_SCENE))
    print(f"OK: {OUT_SCENE.name} nq={m.nq} nv={m.nv} nbody={m.nbody} ngeom={m.ngeom}")


if __name__ == "__main__":
    main()
