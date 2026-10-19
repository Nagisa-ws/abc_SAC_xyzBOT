# curriculum_arm_rl (versi SAC)

Curriculum reinforcement learning untuk lengan **UR10e 6-DoF + gripper** di **MuJoCo**: **SAC** (Stable-Baselines3) + IK,
tiga misi bertingkat (reaching → menyusuri jalur → menghindari rintangan), ekspor ke **ONNX**, dan **controller
hibrida RL + LQR** untuk presisi tingkat milimeter.

> **English summary.** This is a SAC port of [Nagisa-ws/ABC_XYZbot](https://github.com/Nagisa-ws/ABC_XYZbot)
> (MIT), a three-stage curriculum RL project for a UR10e arm in MuJoCo (damped-least-squares IK, unified
> observation so weights transfer Mission 1 → 2 → 3), trained with **SAC**. **v3** restructures the controller as
> **residual RL**: a reference controller follows the collision-free route that the scenario validator already
> computes, and the SAC policy learns a bounded correction on top. It also makes evaluation measure exactly the
> final training task, with held-out test scenarios and confidence intervals. See
> [v3](#v3-struktur-residual-dan-evaluasi-konsisten) for the evidence and results. Documentation below is in Indonesian.

## Isi

- [v3: struktur residual dan evaluasi konsisten](#v3-struktur-residual-dan-evaluasi-konsisten)
- [Misi](#misi)
- [Status dan hasil](#status-dan-hasil)
- [Instalasi](#instalasi)
- [Mulai cepat](#mulai-cepat)
- [Controller hibrida (RL + LQR)](#controller-hibrida-rl--lqr)
- [Struktur proyek](#struktur-proyek)
- [Keputusan desain](#keputusan-desain)
- [Reproduksibilitas](#reproduksibilitas)
- [Batasan](#batasan)
- [Pemecahan masalah](#pemecahan-masalah)
- [Dokumen lain, lisensi, dan keamanan](#dokumen-lain-lisensi-dan-keamanan)

## v3: struktur residual dan evaluasi konsisten

### Kenapa hasil evaluasi lebih rendah dari training (terukur pada model v2 pengguna, 200 skenario held-out)

| Penyebab | Bukti | Perbaikan v3 |
|---|---|---|
| **Misi 1: ambang evaluasi ≠ training.** `eval_config` menuntut 5 mm, kurikulum berhenti di 2 cm | 2 cm: **100 %**, 5 mm: **62 %** (gagal = timeout ±40 mm dari target) | Ambang evaluasi = ambang terakhir kurikulum (`shared/evaluation/task_spec.py`); kurikulum v3 berakhir di 5 mm |
| **Misi 3: tugas evaluasi lebih sulit dari training.** Env eval memakai margin rintangan 1 cm (level 4 lama), kurikulum berakhir di 3 cm | level 3: **82 %**, margin eval: **77,5 %** | Evaluasi memakai level terakhir kurikulum secara otomatis |
| **Evaluasi 10 skenario.** CI 95 % ±25 poin | `eval_log.csv` Anda: 6/10, konsisten dgn 77,5 % ± derau | Default 100 skenario, ringkasan mencetak CI 95 % |
| **best_model optimistis.** 400 evaluasi pada 30 skenario yang sama | Misi 3: rata-rata 0,79, maks (= best) 0,90 | Evaluasi lebih jarang/lebih banyak; di akhir training best vs final dipilih pada set validasi, lalu diuji pada 200 skenario baru (`final_test.json`) |
| Model eval tanpa collision silinder base | model eval kurang 1 geom collision | Ditambahkan; geometri collision kini identik |

### Kenapa Misi 3 mentok di 75–80 %

Pada 200 skenario di level 3 (82 % sukses), 26 dari 36 kegagalan Misi 3 terjadi di **fase A** (home → T1): self-collision,
kontak lantai, dan rintangan, median di langkah 68 (pada margin evaluasi: 26 dari 45). Validasi skenario sudah menghitung jalur pendekatan yang **dijamin aman** (lurus atau
polar mengitari base; rintangan di sepanjang jalur itu dipangkas), tetapi policy tidak bisa melihatnya dan memilih
rute sendiri. Uji tanpa training ulang:

| Pada margin evaluasi Misi 3 (200 skenario) | Sukses |
|---|---|
| Policy RL langsung (v2) | 77,5 % |
| + safety filter (`shared/safety/shield.py`) | 77 % (kegagalan kontak 38 → 1, tetapi menjadi timeout) |
| + target diganti titik pemandu pada jalur aman | 78,5 % |
| **Controller referensi sederhana yang mengikuti jalur aman** | **99,5 %** (5 mm: 100 %; + DR & derau sensor: 99,5 %) |

### Struktur v3

- **Policy residual** (`policy_structure.type: residual` di `domain_randomization.yaml`):
  `aksi = clip(a_ref + 0.5·a_RL[pos], a_RL[rot])`. `a_ref` = `shared/control/reference.py` (ikuti jalur pendekatan
  tervalidasi, lalu garis T1 → T2). `a_ref` masuk observasi (`ref_action`, observasi 57 → 63 dimensi). Actor dimulai dgn
  koreksi ≈ 0; critic dilatih dulu (actor beku), lalu actor ditarik ke koreksi nol (Misi 1 dan 3) / ke policy misi
  sebelumnya (Misi 2) dgn bobot yang meluruh ke floor. `type: direct` mengembalikan struktur lama.
- **Misi 3 khusus** (ditemukan saat menguji, 3 iterasi):
  1. koreksi warisan Misi 2 menyimpang 15–19 mm dari jalur (cross-track eval) → tabrakan 10–16 % pada margin 1 cm. Maka koreksi **direset
     ke 0** saat transfer (fitur & critic tetap dipakai) dan jangkar menarik ke **koreksi nol** (`actor_anchor_target: zero`);
  2. koreksi memotong jalur demi kecepatan dekat rintangan → **gerbang clearance** (`residual_obstacle_gate: [0.02, 0.08]`):
     koreksi dipudarkan ke 0 bila clearance < 8 cm (nol penuh < 2 cm);
  3. pada 4 dari 4 sisa tabrakan, koreksi rotasi telah memutar gripper sampai batas 30° (21,6–30° saat tabrakan;
     referensi saja: 0°, sukses pada skenario yang sama), padahal jalur divalidasi dgn orientasi nominal → di Misi 3 koreksi **hanya posisi** (`residual_scale_rot: 0`).
- **Satu definisi tugas** untuk training, evaluasi, benchmark, uji akhir (`shared/evaluation/task_spec.py`).
- **Uji held-out otomatis** di akhir setiap training: validasi 100 skenario (pilih best/final) + uji 200 skenario baru,
  dicetak sebagai `[uji held-out]` dan disimpan di `checkpoints/missionN/final_test.json`.
- **Safety filter opsional** (`safety_filter.enabled`, default mati): cek tiap aksi satu langkah ke depan dgn model
  robot, ganti aksi yang akan menabrak dgn kandidat aman terdekat. Untuk robot nyata.
- `gradient_steps: 1` (update-to-data 0,25) di semua misi: lebih stabil dan ±3× lebih cepat.

### Hasil v3

Model v3 yang disertakan (`checkpoints/missionN/`, `models/onnx_missionN_v1.onnx`) dilatih berantai Misi 1 → 2 → 3
(120k / 150k / 300k langkah, satu seed, CPU 2 core: ±16 / 21 / 46 menit). Semua kolom diuji pada **200 skenario baru
yang sama** (seed 600000+, deterministik, tugas akhir training, tanpa randomisasi kecuali disebut). Angka dalam kurung =
CI 95 %.

| Tugas | v2 Anda (direct) | Referensi saja | **v3 residual** |
|---|---|---|---|
| Misi 1, ambang 5 mm | 66,0 % (59–72) | 90,0 % (85–93) | **93,0 % (89–96)** |
| Misi 1, ambang 2 cm | 100 % | — | 99,5 % |
| Misi 2, ambang 2 cm | 85,5 % (80–90) | 100 % | **99,5 % (97–100)** |
| Misi 3, level akhir (margin 1 cm) | 70,0 % (63–76) | 99,5 % | **100 % (98–100)** |
| Misi 3, level akhir + domain randomization + derau sensor | 71,5 % (65–77) | 99,0 % | **99,5 % (97–100)** |

| Aspek lain | v2 Anda | v3 |
|---|---|---|
| Kegagalan kontak Misi 3 (self-collision + lantai + rintangan) | 49 / 200 | 0 / 200 |
| Cross-track Misi 2 / Misi 3 | 24,5 / 26,0 mm | 14,6 / 7,7 mm |
| Training vs uji held-out Misi 3 | training ±81 %, eval selama training rata-rata 79 % (best 90 %), held-out 70–77,5 % | training 99,6 % (level akhir), eval 100 %, held-out 100 % |
| Langkah per episode sukses (Misi 1 5 mm / Misi 2 / Misi 3) | 98 / 118 / 123 | 107 / 129 / 144 (**lebih lambat 9–17 %**) |

Angka v2 di tabel diagnosis atas (seed 50000+) dan tabel ini (seed 600000+) berbeda beberapa poin (mis. Misi 1 5 mm:
62 vs 66 %; Misi 3 margin akhir: 77,5 vs 70 %) karena set skenario dan model scene berbeda; keduanya di dalam CI.

Skrip evaluasi pada model yang disertakan (100 skenario, seed 1000+, model `scene_eval.xml`): Misi 1 91/100 (5 mm),
Misi 2 99/100, Misi 3 100/100. Uji held-out akhir training (`final_test.json`): 95,5 % / 99,5 % / 100 %.

**Peran RL (jujur):** sebagian besar kenaikan berasal dari mengikuti jalur tervalidasi, bukan dari RL. Benchmark Misi 3
(`benchmark_hybrid_mission3.py`): referensi saja 100 % (2 cm) / 100 % (5 mm), referensi + RL 100 % / 99 %. RL
menyumbang presisi Misi 1 (+3 poin pada 5 mm), sedikit kecepatan (Misi 3 pada 2 cm: 130 vs 134 langkah, pada 5 mm sama: 151; Misi 2: 129 vs 144), tetapi
di Misi 2 menambah cross-track (14,6 vs 8,1 mm) karena reward menukar ketelitian dengan kecepatan. Untuk memberi RL peran
lebih besar, beri tugas yang tidak bisa dipenuhi referensi (batas waktu ketat, penyejajaran orientasi, rintangan bergerak).

**Kompatibilitas:** observasi kini 63 dimensi, jadi checkpoint/ONNX v2 (57 dimensi) **tidak bisa** dipakai; latih ulang
Misi 1 → 2 → 3. Dengan struktur residual, training jauh lebih singkat (default 300k / 300k / 500k langkah).
Hibrida LQR (`hybrid.enabled`) hanya untuk policy `direct`; pada policy residual diabaikan otomatis.

## Misi

Setiap misi dilatih terpisah (`python Mission<N>/train_mission<N>.py`, tanpa argumen CLI). Misi N+1 memuat bobot Misi N
lewat blok `transfer` di `sac_config.yaml`. Definisi misi, observasi, aksi, dan reward **sama** dengan versi PPO.

| Misi | Tugas                                               | Yang ditambahkan                                                                                                                                                         |
| ---- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1    | Reaching ke target acak                             | Ambang sukses **adaptif**: naik ke level berikutnya bila success rate ≥ 85 %; hanya `reward_success` yang membesar, suku reward lain tetap                               |
| 2    | Reaching T1, lalu menyusuri **garis lurus T1 → T2** | Penalti cross-track `−w·d²` (d dibatasi 8 cm), penanda fase di observasi, panjang segmen 0,15–0,50 m                                                                     |
| 3    | Seperti Misi 2, dengan **rintangan**                | Konstelasi titik rintangan (terinspirasi konsep *stargaze*), shaping rintangan berbasis potensial, tabrakan = −100 dan episode berakhir, **kurikulum kesulitan** 5 level |

Aksi: 6 dimensi (3 delta posisi TCP maks 5 cm/langkah, 3 delta rotasi kerangka dunia maks 0,08 rad/langkah) → IK (damped
least squares) → target sendi → aktuator posisi MuJoCo. Langkah kontrol 20 ms (10 × 2 ms). Observasi **63 dimensi (v3; 57 di v1-v2), sama
di semua misi** (dimensi yang belum relevan bernilai konstan). Aksi SAC adalah Gaussian ter-squash `tanh`, jadi selalu
di [−1, 1], persis rentang aksi env.

## Status dan hasil

**Hasil terkini: lihat [v3](#v3-struktur-residual-dan-evaluasi-konsisten).** Bagian di bawah adalah catatan verifikasi
pipeline dari versi SAC pertama (v1) dan tetap berlaku.

Yang **sudah diverifikasi** (smoke test 3.000 langkah per misi, 2 env, Misi 1 → 2 → 3 berantai, CPU):

- sanity check pra-training, training SAC, transfer bobot, warm-up critic, evaluasi `best_model`, kurva TensorBoard,
  plot debug pasca-training berjalan tanpa galat untuk ketiga misi;
- ekspor ONNX berhasil dan cocok dengan checkpoint (selisih maks ≈ 1e-7 pada 301 observasi nyata; ≈ 1e-6 terhadap
  `SAC.predict` deterministik);
- transfer: bobot input dimensi baru di actor dan kedua critic Q bernilai 0, dan actor hasil transfer menghasilkan aksi
  **identik** dengan actor misi sebelumnya walau dimensi baru diacak;
- warm-up critic: parameter actor tidak berubah sama sekali selama warm-up, sementara critic belajar;
- ONNX SAC dipakai apa adanya oleh evaluasi (`eval_mission3.py`, headless), controller hibrida, dan
  `benchmark_hybrid_mission3.py`.

Uji lanjutan (juga tanpa galat):

- ukuran config **asli** (4 env, buffer 250 ribu per env, batch 256; ±2,3 GB RAM terpakai) pada 30 ribu / 20 ribu / 20 ribu
  langkah untuk Misi 1 / 2 / 3 berantai;
- kurikulum dipaksa naik: ketujuh level ambang Misi 1 dan kelima level kesulitan Misi 3, dengan buffer kecil yang terisi
  penuh dan berputar (wraparound) serta checkpoint periodik;
- Ctrl+C di tengah training: model terbaik tersimpan, ONNX diekspor dan diverifikasi, semua proses worker tertutup.

Indikasi awal saja, bukan hasil: pada Misi 1 level ambang 10 cm, success rate training mencapai ±50 % dalam ±10 ribu langkah
(config uji dengan buffer 20 ribu dan batch 64).

Yang **belum** diverifikasi: kualitas dan konvergensi policy pada training penuh, kelayakan hyperparameter, waktu
training penuh, Windows native/macOS, dan viewer interaktif.

## Perbaikan keruntuhan Misi 3

**Gejala** (training pengguna, 1,75 juta langkah): success Misi 3 bertahan ±77–87 % selama warm-up critic, lalu jatuh
ke **0 % dalam ±10 ribu langkah setelah warm-up selesai** (50k) dan tidak pernah pulih. `best_model` Misi 3 tersimpan di
langkah 45k, yaitu saat actor masih beku, sehingga "hasil" Misi 3 itu sebenarnya policy Misi 2 yang belum diubah.

**Penyebab 1 — koefisien entropi lepas kendali selama warm-up (bug).** Selama warm-up actor dibekukan, tetapi SB3
tetap memperbarui alpha otomatis. Entropi policy hasil transfer (log π ≈ 10,5 dengan std 0,25) berada di bawah
`target_entropy: auto` (−6), dan actor beku tidak bisa menaikkannya, sehingga alpha tumbuh eksponensial: 0,033 → 6,8
(nilai sehat di Misi 1–2: ±0,007). Critic dilatih dengan target `−alpha·log π` raksasa (critic_loss 3 → 33.600).
Versi lama me-reset alpha di akhir warm-up, tetapi critic yang rusak tetap diserahkan ke actor → keruntuhan. Setelah
itu alpha naik lagi ke ±0,9 (target −6 tetap tak tercapai), sehingga bonus entropi mengalahkan reward.

**Penyebab 2 — actor mengejar galat critic (ditemukan saat menguji perbaikan 1).** Dengan alpha diperbaiki dan jangkar
yang meluruh ke 0, keruntuhan hanya tertunda: Q(s,π) naik 26 → 71 sementara success eval turun 90 % → 10 % tepat saat
jangkar habis (update-to-data 1, γ 0,995).

**Perbaikan** (`shared/training/trainer.py`, `shared/training/callbacks.py`, `Mission3/configs/sac_config.yaml`):

1. Warm-up membekukan actor **dan** alpha (`ArmSAC.actor_frozen`); reset alpha di akhir warm-up dibuang.
2. Misi 3 memakai alpha **tetap** `ent_coef: 0.005`; untuk mode otomatis tersedia batas `ent_coef_max`.
3. `gradient_steps: 1` (update-to-data 0,25 dengan 4 env): overestimation hilang dan training ±3× lebih cepat.
4. Jangkar actor ke policy Misi 2 (`actor_anchor_coef 0.4` → `actor_anchor_floor 0.2` dalam 150k langkah setelah warm-up).
5. Regularisasi kecil mean pra-tanh (`actor_mean_reg: 1e-3`) dan log TensorBoard baru: `train/q_pi_mean`,
   `train/anchor_weight`, `train/anchor_mse`, `train/actor_mu_abs`.

**Hasil uji** (dari checkpoint Misi 2 milik pengguna, config Misi 3 asli: 4 env, buffer 250k, warm-up 50k; eval 20
skenario tetap per putaran, jadi ±10 poin derau):

| Run | eval 60k | eval 100k | eval 200k | eval akhir | level kurikulum | 100 skenario level 4 (deterministik) |
|---|---|---|---|---|---|---|
| Sebelum perbaikan (training pengguna) | 0 % | 0 % | 0 % | 0 % (1,75M) | 3 | best 50 %*, final **0 %** |
| Perbaikan 1 saja (alpha) + jangkar ke 0, UTD 1 | 80 % | 70 % | **10 %** | — | 3 | — |
| + UTD 0,25, floor jangkar 0,1 | 75 % | 75 % | 90 % | 70 % (300k) | 4 (152k) | — |
| **Config yang dipakai** (UTD 0,25, floor 0,2) | 80 % | 80 % | 90 % | 85 % (350k) | 4 (157k) | best **64 %**, final **64 %** |

\* `best_model` pengguna = policy Misi 2 yang belum diubah (tersimpan saat actor masih beku).

Kolom "eval" = evaluasi 20 skenario selama training (dipakai memilih `best_model`, jadi optimistis). Kolom terakhir
memakai 100 skenario lain (seed 70000+) pada level kurikulum terakhir: angka itulah yang realistis. Pada run yang
dipakai, alpha tetap 0,005, critic_loss tetap di bawah ±13 (sebelumnya 33.600), Q(s,π) tidak menggelembung, dan
success training kembali ke 74–80 % di 300–350k. Kegagalan utama yang tersisa: tabrakan rintangan (±20 %). Uji ini
hanya sampai 350k langkah dengan satu seed; perilaku pada 1,75 juta langkah belum diverifikasi.

**Yang perlu dilakukan:** salin file perbaikan, lalu jalankan ulang **Misi 3 saja** (`python Mission3/train_mission3.py`);
checkpoint Misi 1–2 yang ada tetap dipakai. Pantau di TensorBoard: `train/ent_coef` harus tetap 0,005,
`train/q_pi_mean` tidak boleh naik terus sementara `rollout/success_rate` turun, dan `train/anchor_mse` sebaiknya tetap
di bawah ±0,05. `best_model` (yang diekspor ke ONNX) dipilih per level kurikulum, jadi penurunan di akhir training tidak
merusak hasil yang sudah tersimpan.

## Instalasi

```
python -m venv .venv && source .venv/bin/activate        # Python 3.10+ (smoke test: 3.13)
pip install -r requirements.txt
```

Smoke test dijalankan dengan: `mujoco 3.15`, `gymnasium 1.4`, `stable-baselines3 2.9`, `torch 2.14`, `numpy 2.5`,
`onnx 1.23`, `onnxruntime 1.29`, `tensorboard 2.21` (Linux, CPU). `requirements.txt` hanya memberi batas bawah; untuk hasil
yang dapat diulang, simpan `pip freeze` dari lingkungan Anda.

- GPU **tidak** diperlukan (MLP kecil; `device: cpu` di config lebih cepat).
- Viewer membutuhkan layar; tanpa `DISPLAY` evaluasi otomatis berjalan *headless* (CSV dan plot tetap dibuat).
- macOS: viewer harus dijalankan dengan `mjpython`. Windows native dan macOS belum diuji.
- Model **training** tidak memakai mesh; folder `robot_model/assets/` (±40 MB) hanya dibutuhkan untuk tampilan viewer.
- RAM: replay buffer default 4 env × 250 ribu transisi ≈ 0,5 GB; turunkan `sac.buffer_size` bila RAM terbatas.

## Mulai cepat

### A. Melatih dari awal (satu-satunya jalur; belum ada model terlatih)

```
python Mission1/train_mission1.py
python Mission2/train_mission2.py      # butuh checkpoints/mission1/best_model.zip + best_vecnormalize.pkl
python Mission3/train_mission3.py      # butuh checkpoints/mission2/best_model.zip + best_vecnormalize.pkl
tensorboard --logdir checkpoints
```

Setiap `train_missionN.py` menjalankan: sanity check (IK, env, controller referensi) → training SAC dengan TensorBoard →
`best_model` dan `final_model` → ekspor ONNX berversi ke `models/onnx_missionN_vK.onnx` → `verify_onnx` pada observasi
nyata → plot debug di `checkpoints/missionN/debug/`.

**Kecepatan.** SAC memperbarui jaringan pada tiap langkah env (update-to-data = 1), jadi per langkah jauh lebih mahal
daripada PPO. Di mesin uji (2 core CPU, 2–4 env) terukur ±130–320 langkah/detik (melambat setelah update dimulai);
waktu training penuh **belum diukur**, tetapi 4 juta langkah pada ±130 langkah/detik berarti ±8 jam.
Untuk mempercepat: turunkan `sac.gradient_steps` (mis. 2 dengan `n_envs: 4` → update-to-data 0,5; sampel-efisiensi
lebih rendah), perbanyak `n_envs` bila ada core, atau kurangi `run.timesteps`.

- **Ctrl+C aman**: model terbaik tetap disimpan, ONNX diekspor dan diverifikasi. Tekan **sekali** lalu tunggu
1–2 menit; pesan `BrokenPipeError` di akhir hanya noise saat menutup proses worker.
- **Belum ada fitur resume**: menjalankan ulang `train_missionN.py` memulai dari awal dan menimpa `best_model.zip`
(ONNX tidak tertimpa, nomor versinya naik). Replay buffer **tidak** disimpan di checkpoint. Cadangkan `checkpoints/` lebih dulu.
- Jika checkpoint misi sebelumnya belum ada, Misi 2 dan 3 berhenti dengan pesan yang jelas sebelum training dimulai.

**Konfigurasi Misi 1.** `adaptive.thresholds` default `[0.10, 0.08, 0.06, 0.04, 0.02, 0.01, 0.005]` (7 level) dengan
`run.timesteps: 4_000_000`; jumlah langkah tersebut titik awal dan belum tentu cukup untuk mencapai level terakhir.
Pantau `curriculum/level` di TensorBoard dan naikkan `timesteps` bila perlu.

### B. Evaluasi viewer

```
python Mission3/eval_mission3.py                 # evaluasi dengan MuJoCo viewer
python Mission3/benchmark_hybrid_mission3.py     # RL saja vs hibrida, ambang 2 cm dan 5 mm
python Mission2/benchmark_hybrid_mission2.py
```

Pengaturan di `Mission*/configs/eval_config.yaml` (jumlah skenario, seed, ambang, ONNX yang dipakai). ONNX default `latest`
= versi tertinggi di `models/`; isi `paths.onnx_model` dengan path eksplisit untuk mengunci model.

## Controller hibrida (RL + LQR)

Lapisan ini **tidak bergantung pada algoritma**: ia hanya memakai file ONNX (input observasi mentah 63-dim, output aksi
6-dim) dan vektor observasi yang sama, sehingga kode `shared/control/` tidak diubah. Controller LQR mengambil alih
**posisi** TCP saat gripper dalam 8 cm dari T1 atau saat fase T1→T2 dimulai; aksi rotasi tetap dari RL.

```
# Mission*/configs/eval_config.yaml
hybrid:
  enabled: true        # RL + LQR (posisi)
  orientation: true    # opsional: sejajarkan moncong dengan jalur (hanya jalur menurun curam)
```

Saat aktif, terminal mencetak `[eval] mode HIBRIDA: ...`; bila baris itu tidak muncul, evaluasi memakai RL saja.
Parameter ada di `shared/control/controller.yaml`. Untuk ambang presisi, set `rollout.t1_reach_threshold` dan `rollout.success_pos_threshold` ke `0.005`.

Deploy di luar proyek:

```
from shared.control import make_hybrid_from_onnx
policy = make_hybrid_from_onnx("models/onnx_mission3_v1.onnx")   # parameter: shared/control/controller.yaml
policy.reset()                  # WAJIB di awal tiap episode (controller menyimpan riwayat posisi)
action = policy(obs_mentah)     # observasi mentah 63-dim (v3); normalisasi sudah ada di dalam ONNX
```

**Baca hasil hibrida dengan hati-hati.** Pada smoke test, policy SAC yang nyaris belum dilatih (3.000 langkah) tetap
mencapai 2/2 sukses pada evaluasi hibrida Misi 3, sedangkan "RL saja" 0/3. Controller LQR (dan orientasi) menanggung
sebagian besar pekerjaan di fase presisi dan penyusuran jalur, jadi success rate hibrida **bukan** ukuran kualitas
policy RL; bandingkan kolom "RL saja". Angka hibrida versi PPO di `KONTROL_HIBRIDA.md` bukan hasil SAC.

Catatan penting:

- Penyejajaran moncong hanya untuk jalur **menurun curam** (< 60° dari "lurus ke bawah"). Pada jalur mendatar/menanjak,
penyejajaran menurunkan success 11–18 poin pada versi PPO (self-collision, kontak lantai), jadi tidak diaktifkan.
- Saat orientasi aktif, batas kerucut orientasi env (30°) dilonggarkan (`policy.cone_request`; ditangani `bind_to_env`).
Pada robot nyata, antarmuka perintah harus menerima rentang orientasi lebih lebar pada tahap itu.
- `max_pos_delta` dan `max_rot_delta` di `controller.yaml` harus sama dengan config env (benchmark memeriksanya).

Detail dan batasan: [`KONTROL_HIBRIDA.md`](KONTROL_HIBRIDA.md).

## Struktur proyek

Tata letak folder dan file **sama** dengan proyek PPO; satu-satunya perubahan nama adalah `ppo_config.yaml` →
`sac_config.yaml` (lokasi sama).

```
Mission1|2|3/
  envs/                  env training (Stage1ReachEnv, Stage2PathEnv, Stage3ObstacleEnv)
  eval_envs/             subclass env training + scene_eval.xml (observasi identik dengan training)
  configs/               sac_config.yaml · domain_randomization.yaml (config env) · eval_config.yaml
  debug/                 plot khusus misi (threshold sweep / cross-track / clearance rintangan)
  train_missionN.py      entry point training
  eval_missionN.py       entry point evaluasi (viewer)
  benchmark_hybrid_missionN.py   uji RL saja vs hibrida (Misi 2 dan 3)
shared/
  envs/                  BaseArmEnv, obs_layout (observasi 63-dim), robot_geometry (kapsul), validation, path_task
  training/              trainer generik (ArmSAC), callbacks (adaptive, kurikulum kesulitan, best-model, warm-up critic), transfer
  evaluation/            evaluasi viewer + marker T1/T2/garis/rintangan
  exporter/              to_onnx (actor SAC, berversi), verify_onnx
  debug_common/          rollout, sanity check, plot, pembaca TensorBoard
  control/               controller LQR, controller orientasi, policy hibrida, benchmark
ik/                      IK damped-least-squares + util pose
obstacles/               generator konstelasi rintangan (+ visualizer)
robot_model/             ur10e_train.xml / ur10e_eval.xml / scene_*.xml / assets (mesh)
checkpoints/             keluaran training per misi (model terbaik, plot debug, log evaluasi); kosong di awal
models/                  ONNX hasil ekspor (onnx_missionN_vK.onnx + .json metadata); kosong di awal
```

Semua parameter ada di YAML. Daftar perubahan per file: [`LAPORAN_PERUBAHAN.md`](LAPORAN_PERUBAHAN.md).

## Keputusan desain

Bagian yang tidak berubah dari versi PPO (observasi terpadu, kompensasi gravitasi, validasi skenario saat reset,
rintangan berbasis kapsul, shaping potensial, kurikulum kesulitan Misi 3, pemilihan `best_model` berdasarkan
(level, success rate, ketelitian)) dijelaskan di README proyek asal. Yang khusus SAC:

- **Ekspor ONNX = aksi deterministik actor**: `tanh(mu(latent_pi(obs_ternormalisasi)))`, normalisasi observasi dibakar ke
  graph. Critic Q tidak diekspor. Antarmuka ONNX identik dengan versi PPO.
- **VecNormalize dengan off-policy aman**: SB3 menyimpan observasi **asli** di replay buffer dan menormalisasi saat
  sampling dengan statistik terkini, jadi data lama tidak "basi". Pada Misi 2 dan 3 statistik tetap dibekukan setelah transfer
  (alasannya sama dengan PPO: input actor tidak boleh bergeser).
- **Transfer bobot** (`shared/training/transfer.py`): bobot input dimensi baru di-nol-kan pada actor **dan** kedua critic Q
  (target critic disalin ulang), sehingga awal Misi N+1 = perilaku Misi N. Koefisien entropi tidak disalin
  (`copy_ent_coef: false`). SAC tidak punya `log_std` global: `log_std_init` mengganti head `log_std` menjadi konstan
  (std pra-tanh seragam); `log_std_max` milik PPO tidak ada padanannya dan dihapus. Padanan "derau aksi harus menyusut"
  di Misi 2 adalah `ent_coef: auto_0.01` + `target_entropy: -8`.
- **`ArmSAC`**: pada training hasil transfer, aksi dikumpulkan dengan policy yang dimuat sejak langkah pertama. Perilaku
  bawaan SB3 (aksi acak seragam selama `learning_starts`) akan membuang policy hasil transfer dan mengisi buffer dengan data
  tak representatif. `learning_starts` tetap menentukan kapan gradient pertama dihitung.
- **Warm-up critic** (padanan `CriticWarmupCallback` PPO): actor **dan koefisien entropi** dibekukan, hanya kedua
  critic Q yang dilatih `critic_warmup_steps` langkah (lihat [Perbaikan keruntuhan Misi 3](#perbaikan-keruntuhan-misi-3)).
- **Jangkar actor** (`actor_anchor_coef`, `actor_anchor_steps`): setelah warm-up, actor ditarik ke aksi policy hasil
  transfer (atau ke koreksi nol pada residual) dengan bobot yang meluruh linear ke `actor_anchor_floor` (skala mean|Q| ala TD3+BC). `train/anchor_weight` di TensorBoard.
- **Hyperparameter SAC** (`sac_config.yaml`): `gamma 0.995`, `tau 0.005`, `batch_size 256`, jaringan 256×256 untuk actor
  dan critic, aktivasi `relu` (PPO memakai `tanh`; aktivasi harus sama di semua misi), `ent_coef: auto_<awal>`. Nilai awal
  entropi dan `target_entropy` per misi adalah **dugaan awal**, belum ditala.
- **Kurva TensorBoard**: panel entropi/value loss PPO diganti `train/ent_coef` dan `train/critic_loss`. Tag grafik alasan
  terminasi diperbaiki menjadi `reasons/*_rate` (yang benar-benar ditulis `ReasonRateCallback`; kode PPO asal membaca
  `failure_breakdown/*`, tag yang tidak ditulis callback mana pun).

## Reproduksibilitas

- SAC diseed (`run.seed`), tetapi training off-policy multi-proses (`SubprocVecEnv`) belum diuji bit-per-bit; jangan
  berasumsi hasil identik antar eksekusi atau antar mesin. Perubahan `torch_num_threads`, `n_envs`, atau `gradient_steps`
  mengubah hasil.
- Sebaran antar seed belum diukur. Rantai Misi 1 → 2 → 3 mewariskan perbedaan tiap tahap.
- Simpan `pip freeze`, config, dan `best_model_info.json` bersama tiap model yang akan dibandingkan.

## Batasan

- **v3 diverifikasi dgn satu seed per misi** dan langkah 120k / 150k / 300k; sebaran antar seed belum diukur.
- **Kinerja tinggi sebagian berasal dari konstruksi tugas**: validasi skenario memangkas rintangan yang disapu badan robot
  di sepanjang jalur referensi, jadi jalur itu dijamin aman. Controller referensi memanfaatkan jaminan itu. Rintangan yang
  tidak diketahui/bergerak, atau jalur yang tidak divalidasi, tidak tercakup.
- Pada robot nyata, jalur referensi harus dihitung oleh planner yang sama (IK + cek kontak dgn model robot) sebelum gerak,
  dan `ref_action` dihitung dari state terukur. Policy ONNX v3 mengeluarkan **koreksi**, bukan aksi penuh.
- Gerbang rintangan hanya bekerja < 8 cm dari rintangan; ia tidak membatasi seberapa jauh koreksi membawa lengan dari jalur
  di ruang bebas. Keamanan juga bergantung pada koreksi terlatih yang tetap dekat jalur (jangkar koreksi-nol, floor 0,2).
  Terukur: model iterasi c (dilatih dgn rotasi) yang dijalankan tanpa rotasi menabrak 2/120 di fase A; model v3 yang
  disertakan 0 tabrakan pada semua uji (200 + 200 + 100 + 200 skenario).
- v3 lebih lambat 9–17 % per episode sukses dibanding v2 (controller referensi melambat dekat target).
- Misi 1 pada 5 mm: sisa kegagalan = timeout dekat target (±7 %); training lebih lama cenderung membantu (eval naik dari
  60–80 % ke 92–96 % antara 40k dan 120k langkah).
- Checkpoint/ONNX v2 (57 dimensi) tidak kompatibel dgn v3.
- Safety filter mengubah tabrakan menjadi berhenti, bukan sukses (terukur pada policy direct); default mati.
- Kurikulum ambang adaptif Misi 1 menaikkan `reward_success` saat level naik, sedangkan replay buffer masih memuat
  transisi dengan reward level lama (efeknya mereda seiring buffer terisi).
- Tidak ada resume training dan replay buffer tidak disimpan.
- Belum diuji: Windows native, macOS, jendela viewer interaktif, derau sensor nyata, hardware nyata.

## Pemecahan masalah

| Gejala                                                           | Penyebab / solusi                                                                                                                     |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `could not initialize GLFW` / viewer tidak terbuka               | Tidak ada layar. Evaluasi otomatis headless; atau jalankan dengan `DISPLAY` (macOS: `mjpython`)                                       |
| Misi 2/3 berhenti dengan `... tetapi file belum ada: ...`        | Latih misi sebelumnya dulu (belum ada checkpoint yang disertakan)                                                                     |
| Error saat `SAC.load` pada zip lama                              | Checkpoint dari versi PPO tidak kompatibel; latih ulang dengan SAC                                                                    |
| Training lambat                                                  | Normal untuk SAC. Turunkan `sac.gradient_steps`, tambah `n_envs`, atau kurangi `run.timesteps`                                       |
| RAM tinggi / proses di-kill                                      | Turunkan `sac.buffer_size` (per env) atau `n_envs`                                                                                    |
| Success rate Misi 2/3 anjlok tepat setelah warm-up               | Lihat [Perbaikan keruntuhan Misi 3](#perbaikan-keruntuhan-misi-3). Cek `train/ent_coef` di TensorBoard: bila naik terus, pakai `ent_coef` tetap atau `ent_coef_max` |
| `train/ent_coef` naik terus                                      | `target_entropy` lebih tinggi dari entropi policy. Pakai `ent_coef` tetap (mis. 0.005) atau turunkan `target_entropy`              |
| `ValueError: max_pos_delta env != controller.yaml`               | Samakan `max_pos_delta`/`max_rot_delta` di `controller.yaml` dengan config env                                                        |
| `Belum ada onnx_missionN_v*.onnx di ...` pada evaluasi/benchmark | Jalankan `train_missionN.py` atau salin ONNX ke `models/`                                                                             |

## Dokumen lain, lisensi, dan keamanan

- [`KONTROL_HIBRIDA.md`](KONTROL_HIBRIDA.md): controller hibrida, pengukuran (versi PPO), dan batasan.
- [`LAPORAN_PERUBAHAN.md`](LAPORAN_PERUBAHAN.md): daftar perubahan PPO → SAC dan hal yang belum tervalidasi.
- [`robot_model/assets/README.md`](robot_model/assets/README.md): daftar mesh dan sumbernya.

**Lisensi.** Proyek turunan dari [Nagisa-ws/ABC_XYZbot](https://github.com/Nagisa-ws/ABC_XYZbot); kode dan gripper dirilis di
bawah [lisensi MIT](LICENSE) dengan pemberitahuan hak cipta asli dipertahankan. Mesh UR10e dari MuJoCo Menagerie tetap
memakai lisensinya sendiri (lihat Atribusi).

**Atribusi.** Mesh lengan UR10e dan model dasarnya berasal dari [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (`universal_robots_ur10e`); lisensinya mengikuti
berkas `LICENSE` di sana (lihat [`robot_model/assets/README.md`](robot_model/assets/README.md)). Gripper dirancang sendiri oleh
pemilik repo asal dan mengikuti lisensi kode di atas.

**Keamanan.** `best_vecnormalize.pkl` dan `*.zip` dari Stable-Baselines3 memakai `pickle`; **muat hanya dari sumber yang
Anda percaya**. ONNX berformat protobuf dan tidak memakai `pickle`.
