# Laporan Perubahan: PPO → SAC

Proyek ini adalah port dari [Nagisa-ws/ABC_XYZbot](https://github.com/Nagisa-ws/ABC_XYZbot) (commit `9016ddd`) yang
mengganti **algoritma pembelajaran** PPO menjadi **SAC** (Stable-Baselines3). Struktur folder, logika misi, lingkungan,
IK, rintangan, controller hibrida, dan antarmuka ONNX dipertahankan. Dokumen ini mencatat apa yang diubah, mengapa, dan
apa yang sudah serta belum divalidasi.

## 1. Perubahan per file

| File | Perubahan |
|---|---|
| `Mission{1,2,3}/configs/ppo_config.yaml` → `sac_config.yaml` | Diganti nama (lokasi sama). Blok `ppo:` → `sac:` (buffer, learning_starts, tau, ent_coef, target_entropy, dst.). Blok `transfer` diganti `copy_ent_coef`, `log_std_max` dihapus, `critic_warmup_steps` dikecilkan. `run.timesteps` diturunkan (4M / 2M / 3M), ditambah `log_interval_episodes`. Blok `normalization`, `curriculum`, `adaptive`, `export`, `debug` **tidak berubah**. |
| `Mission{1,2,3}/train_mission{N}.py` | Memanggil `train_mission(sac_cfg_path=...)` dengan `sac_config.yaml`; docstring. Hook sanity/callback/debug tidak berubah. |
| `shared/training/trainer.py` | `build_ppo` → `build_sac`; kelas `ArmSAC(SAC)` (aksi dari policy hasil transfer sejak langkah 0); `PPO.load` → `SAC.load`; `learn(log_interval=...)`; metadata ONNX memuat `"algorithm": "SAC"`. Pipeline 8 langkah, VecEnv, VecNormalize, callback, penyimpanan, ekspor tidak berubah. |
| `shared/training/transfer.py` | Memuat `SAC`; menolkan bobot input dimensi baru di actor (`latent_pi[0]`) **dan** kedua critic Q (`q_networks[i][0]`), lalu menyalin ulang target critic; `log_std_init` mengganti head `actor.log_std` menjadi konstan; opsi `copy_ent_coef`; log std aksi dihitung dari observasi nyata. Logika statistik normalisasi (dimensi lama dipertahankan, dimensi baru diisi dari rollout nyata) tidak berubah. |
| `shared/training/callbacks.py` | `CriticWarmupCallback` ditulis ulang untuk SAC (bekukan seluruh parameter actor; kembalikan `log_ent_coef` + state optimizer saat selesai). Callback lain tidak berubah selain komentar. |
| `shared/training/__init__.py` | Mengekspor `ArmSAC`. |
| `shared/exporter/to_onnx.py` | `NormalizedOnnxablePolicy` memakai `features_extractor` + `latent_pi` + `mu` + `tanh` (aksi deterministik actor SAC) menggantikan `mlp_extractor` + `action_net` + clamp PPO. Input `observation` (57-dim mentah) dan output `action` (6-dim) tidak berubah. CLI re-ekspor memakai `SAC.load`. |
| `shared/exporter/verify_onnx.py` | `PPO.load` → `SAC.load` pada CLI; fungsi verifikasi tidak berubah. |
| `shared/debug_common/post_training.py` | `PPO.load` → `SAC.load`. |
| `shared/debug_common/common.py` | Panel kurva: entropi/value loss PPO → `train/ent_coef` dan `train/critic_loss`; tag alasan terminasi `failure_breakdown/*` → `reasons/*` (tag yang benar-benar ditulis `ReasonRateCallback`). |
| `shared/envs/obs_layout.py` | Hanya satu kata di docstring (`PPO.load()` → `model.load()`). |
| `.gitignore` | Menambah `checkpoints/**/*replay_buffer*`. |
| `README.md`, `KONTROL_HIBRIDA.md`, `LAPORAN_PERUBAHAN.md` | Ditulis ulang / diberi catatan versi SAC. Angka hasil versi PPO tidak diklaim untuk SAC. |

**Tidak diubah (identik byte-per-byte dengan proyek asal):** semua env (`envs/`, `eval_envs/`), `shared/envs/`,
`shared/control/` (controller hibrida), `shared/evaluation/`, `ik/`, `obstacles/`, `robot_model/`, `eval_config.yaml`,
`domain_randomization.yaml`, skrip `eval_mission*.py` dan `benchmark_hybrid_mission*.py`, `LICENSE`, `requirements.txt`.

## 2. Tidak disertakan (dan alasannya)

- `checkpoints/mission*/*`, `models/onnx_*`: hasil training PPO. Tidak kompatibel dengan SAC dan bukan hasil SAC.
  Folder `checkpoints/` dan `models/` tetap ada, kosong.
- `Colab_Eval_if you want try in_Colab.zip` dan `evaluasi_colab.ipynb`: paket evaluasi Colab berisi model PPO; notebook
  memanggil `shared.evaluation.video_render` yang tidak ada di snapshot repo asal, jadi tidak dapat diverifikasi.
- `LAPORAN_PERUBAHAN.md` asal (audit kode awal, dokumen historis): digantikan dokumen ini.

## 3. Pemetaan konsep PPO → SAC

| PPO (asal) | SAC (di sini) |
|---|---|
| `n_steps`, `n_epochs`, `clip_range`, `gae_lambda`, `target_kl`, `max_grad_norm`, `vf_coef` | Tidak ada padanan (on-policy). Diganti `buffer_size`, `learning_starts`, `batch_size`, `tau`, `train_freq`, `gradient_steps` |
| `ent_coef` konstan (0.003 / 0.0) | `ent_coef: auto_<awal>` (otomatis) + `target_entropy` |
| `log_std` global (`log_std_init`, `log_std_max`) | Head `actor.log_std` bergantung observasi; `log_std_init` menggantinya konstan; `log_std_max` dihapus |
| `net_arch: {pi, vf}` | `net_arch: {pi, qf}` (actor dan dua critic Q) |
| `value_net` dibekukan/dilatih lebih dulu | Kedua critic Q dilatih lebih dulu, actor dibekukan |
| Aksi = Gaussian lalu clip ke [-1, 1] | Aksi = `tanh(Gaussian)`; deterministik = `tanh(mu)` |
| `activation_fn: tanh` | `activation_fn: relu` (bisa diubah; harus sama di semua misi) |

## 4. Validasi yang dilakukan

Smoke test 3.000 langkah per misi (2 env, Misi 1 → 2 → 3 berantai, CPU, SB3 2.9.0, torch 2.14.1):

- Pipeline lengkap berjalan tanpa galat untuk ketiga misi (sanity check, training, transfer, warm-up, evaluasi, `best_model`,
  ekspor ONNX, verifikasi, plot debug).
- ONNX vs checkpoint: selisih maks ≈ 1,2e-7 (301 observasi nyata); ONNX vs `SAC.predict` deterministik ≈ 1e-6; aksi dalam [-1, 1].
- Transfer: bobot input dimensi baru bernilai 0 di actor, `qf0`, `qf1`, dan target critic; aksi actor hasil transfer identik
  (selisih 0) dengan actor misi sebelumnya walau dimensi baru diacak.
- Warm-up: perubahan parameter actor = 0 selama warm-up, perubahan parameter critic > 0.
- `ArmSAC`: dengan flag aktif aksi berasal dari policy; dengan flag mati perilaku bawaan SB3 (aksi acak) tetap.
- `eval_mission3.py` (headless, hibrida aktif) dan `benchmark_hybrid_mission3.py` berjalan dengan ONNX SAC tanpa perubahan kode.

Uji lanjutan (tanpa galat):

- Ukuran config asli (4 env, buffer 250 ribu per env, batch 256) pada 30k / 20k / 20k langkah, Misi 1 → 2 → 3 berantai; RAM ±2,3 GB.
- Kurikulum dipaksa naik (`advance_rate: 0`): 7 level ambang Misi 1 (`set_curriculum`) dan 5 level kesulitan Misi 3
  (`set_difficulty`); buffer 4 × 1.500 transisi diisi 16.000 langkah (wraparound); checkpoint periodik tertulis.
- SIGINT di tengah training Misi 1: `best_model` tersimpan, ONNX diekspor dan terverifikasi, semua worker tertutup, proses selesai bersih.

## 5. Belum tervalidasi

- **Kualitas policy**: tidak ada training penuh. Success rate, kecepatan konvergensi, dan apakah kurikulum Misi 1 mencapai
  ambang 5 mm dalam 4M langkah belum diketahui.
- **Hyperparameter**: seluruh nilai SAC (terutama `ent_coef` awal, `target_entropy`, `critic_warmup_steps`, `log_std_init`,
  `learning_rate` fine-tuning, `buffer_size`) adalah dugaan awal, belum ditala.
- **Waktu training**: hanya terukur ±130–320 langkah/detik pada uji singkat (2 core, 2–4 env); belum diukur pada training penuh.
- **Kurikulum adaptif Misi 1 di bawah off-policy**: replay buffer memuat reward dengan `reward_success` level lama.
  Dampaknya belum diukur.
- **Reproduksibilitas multi-proses**, sebaran antar seed, Windows/macOS, viewer interaktif, derau sensor, dan hardware nyata.
- **Controller hibrida dengan model SAC terlatih**: angka di `KONTROL_HIBRIDA.md` berasal dari model PPO.

## 6. Perbaikan: success Misi 3 runtuh ke 0 % setelah warm-up

Detail lengkap dan tabel hasil: README.md, bagian "Perbaikan keruntuhan Misi 3".

| File | Perubahan |
|---|---|
| `shared/training/trainer.py` | `ArmSAC.train()` (salinan `SAC.train()` SB3 2.9 + tambahan): actor **dan** alpha tidak diperbarui saat `actor_frozen`; batas `ent_coef_max`; `actor_mean_reg`; jangkar actor ke policy hasil transfer (`set_anchor`, meluruh ke `actor_anchor_floor`); log `train/q_pi_mean`, `train/anchor_weight`, `train/anchor_mse`, `train/actor_mu_abs`. `build_sac` membaca `ent_coef_max`, `actor_mean_reg`; `train_mission` memasang jangkar setelah transfer. |
| `shared/training/callbacks.py` | `CriticWarmupCallback` memakai flag `model.actor_frozen` (actor + alpha beku); reset alpha di akhir warm-up dibuang. |
| `Mission3/configs/sac_config.yaml` | `ent_coef: 0.005` (tetap, dulu `auto_0.03`), `ent_coef_max: 0.02`, `actor_mean_reg: 1e-3`, `gradient_steps: 1` (dulu -1), `actor_anchor_coef: 0.4`, `actor_anchor_steps: 150000`, `actor_anchor_floor: 0.2`. |
| `Mission2/configs/sac_config.yaml` | Hanya `ent_coef_max: 0.02` (pengaman). Perbaikan warm-up di kode juga menghapus lonjakan alpha 22× yang terjadi di akhir warm-up Misi 2. |

**Penyebab (terukur dari log TensorBoard pengguna dan uji ulang):**

1. Alpha otomatis tetap diperbarui selama actor beku. Entropi policy hasil transfer (log π ≈ 10,5) di bawah target −6,
   jadi alpha naik 0,033 → 6,8, critic dilatih dengan target entropi raksasa (critic_loss → 33.600), dan critic itu
   menghancurkan actor saat warm-up selesai (63 % → 0 % dalam 10k langkah). Alpha lalu naik lagi ke ±0,9.
2. Setelah alpha diperbaiki, actor masih mengejar galat critic bila tidak ditahan: dengan jangkar yang habis total dan
   UTD 1, Q(s,π) naik 26 → 71 sementara success eval turun 90 % → 10 %.

**Validasi:** dari checkpoint Misi 2 pengguna, config Misi 3 asli (4 env, buffer 250k, warm-up 50k), 350k langkah:
kurikulum mencapai level terakhir (157k), tidak ada keruntuhan, dan pada 100 skenario level 4 baru best = final = 64 %
(sebelumnya best 50 % = policy Misi 2 yang belum diubah; final 0 %). Uji mekanika: actor dan alpha tidak berubah selama
warm-up, critic berubah; jangkar meluruh; checkpoint tetap bisa dimuat dengan `SAC.load()` biasa.

**Belum divalidasi:** training 1,75 juta langkah penuh, seed lain, dan perubahan ini pada training ulang Misi 1–2.

## 7. v3: struktur residual dan evaluasi konsisten

Detail dan tabel hasil: README.md, bagian "v3: struktur residual dan evaluasi konsisten".

| File | Perubahan |
|---|---|
| `shared/control/reference.py` (baru) | Controller referensi: ikuti jalur pendekatan tervalidasi (fase A), lurus ke T2 (fase B), rotasi 0 |
| `shared/envs/base_arm_env.py` | `policy_structure` (direct/residual): aksi = clip(a_ref + skala·a_RL); `ref_action` di observasi; `_command()` murni (dipakai safety filter); `safety_filter` opsional; `model_path` disimpan |
| `shared/envs/obs_layout.py` | Field `ref_action` (6) di akhir: OBS_DIM 57 → 63 |
| `shared/safety/` (baru) | Safety filter satu langkah (kontak self/lantai dgn margin, clearance rintangan), default mati |
| `shared/evaluation/task_spec.py` (baru) | Satu definisi tugas akhir per misi dari `sac_config.yaml` (ambang terakhir kurikulum Misi 1, level terakhir kurikulum Misi 3) |
| `shared/evaluation/viewer_eval.py` | Memakai `task_spec`, CI 95 % Wilson, peringatan bila < 50 skenario, hibrida diabaikan pada policy residual, override `difficulty_level` / `safety_filter` |
| `shared/control/benchmark.py` | Kesulitan Misi 3 dari `task_spec`; pada policy residual membandingkan "Referensi saja" vs "Referensi + RL" |
| `shared/training/trainer.py` | Uji held-out akhir (validasi best vs final + uji 200 skenario, `final_test.json`); `export.source: auto`; inisialisasi actor residual (koreksi ≈ 0), warm-up critic dan jangkar koreksi-nol untuk residual tanpa transfer; `set_anchor(target="zero")` |
| `shared/debug_common/` | Controller skrip = controller referensi; studi robustness memakai tugas akhir |
| `robot_model/ur10e_eval.xml` | Silinder collision base ditambahkan (geometri collision kini identik dgn model training) |
| `Mission*/configs/domain_randomization.yaml` | Blok `policy_structure` (residual) dan `safety_filter` (mati) |
| `Mission*/configs/sac_config.yaml` | Langkah 300k / 300k / 500k, `gradient_steps: 1`, eval tiap 25k × 50 skenario, `final_validation/test_episodes`, Misi 1 kurikulum sampai 5 mm + `residual_*`, Misi 2 jangkar ke policy Misi 1, Misi 3 koreksi direset + jangkar koreksi-nol + kurikulum 5 level |
| `Mission*/configs/eval_config.yaml` | 100 skenario, ambang/level = tugas training (null), hibrida mati |

**Kompatibilitas:** checkpoint/ONNX v2 (57 dimensi) tidak bisa dimuat; latih ulang Misi 1 → 2 → 3.

### 7.1 Tambahan Misi 3 (ditemukan saat validasi, tiap langkah diukur)

| Iterasi | Konfigurasi | Hasil pada level akhir (eval 50 skenario selama training) |
|---|---|---|
| a | koreksi warisan Misi 2 + jangkar ke Misi 2 | 84–88 %, tabrakan 10–16 % (referensi saja ~100 %) |
| b | koreksi direset ke 0 + jangkar koreksi-nol | 84–98 %, tabrakan berulang, training turun ke 83 % |
| c | + gerbang clearance `[0.02, 0.08]` | 90–100 %; pada 4/4 tabrakan gerbang = 0 tetapi gripper telah diputar koreksi rotasi sampai batas 30° (21,6–30° saat tabrakan) |
| d | + koreksi hanya posisi (`residual_scale_rot: 0`) | **100 % pada 25 evaluasi berturut-turut (60k–300k)**, uji held-out 200/200 |

File terkait: `shared/envs/base_arm_env.py` (`residual_obstacle_gate`), `shared/training/trainer.py`
(`reset_residual_output`, `actor_anchor_target`), `Mission3/configs/domain_randomization.yaml`,
`Mission3/configs/sac_config.yaml`.

### 7.2 Validasi v3

- Uji regresi: dengan safety filter mati, kode baru menghasilkan episode identik per seed dgn kode lama (40 skenario).
- Integrasi: env residual dgn koreksi nol = controller referensi (Misi 1 5 mm 36/40, Misi 3 40/40).
- Smoke test Misi 1 → 2 → 3 (sanity, training, transfer, uji held-out, ekspor & verifikasi ONNX, plot debug) tanpa galat.
- `eval_mission1/2/3.py` dan `benchmark_hybrid_mission3.py` pada model yang disertakan: 91/100, 99/100, 100/100;
  benchmark referensi vs referensi + RL 100/100 % (2 cm) dan 100/99 % (5 mm).
- Perbandingan 200 skenario yang sama (v2 vs referensi vs v3): lihat README bagian "Hasil v3".

## 8. Misi 4-5: panen nanas (simulasi)

Dasar: model Misi 1-3 hasil training pengguna (uji held-out 100 % / 100 % / 100 %, 200 skenario) disalin ke
`checkpoints/mission{1,2,3}` dan `models/`. Rincian desain, alasan, dan angka: [`MISSION4_5_NOTES.md`](MISSION4_5_NOTES.md).

| File | Perubahan |
|---|---|
| `robot_model/build_harvest_model.py` (baru) | Membangkitkan `ur10e_harvest.xml` (lengan identik + gripper PGIA-140-80 + kamera pergelangan) dan `scene_harvest.xml` (tanaman pool: 14 daun roset, 8 penghalang, 4 crown; midphase dimatikan) |
| `shared/harvest/` (baru) | `plant.py` (tanaman acak, daun lentur, titik & tekukan daun), `planner.py` (stargaze pada daun, kecocokan jari, orientasi cengkeram), `checker.py` (validasi rute: IK, tabrakan, singularitas pergelangan, bagian keras), `reference.py` (referensi + orientasi), `report.py` (referensi vs RL) |
| `Mission4/`, `Mission5/` (baru) | env, eval env, config, train/eval, `debug/harvest_report.py`; `Mission5/debug/render_harvest.py` (video) |
| `shared/envs/base_arm_env.py` | massa nominal payload dibaca dari model (Misi 1-3 tetap 3.424 kg); komposisi aksi residual dipindah ke `_compose_residual_action()` (perilaku Misi 1-3 identik) |
| `shared/envs/obs_layout.py` | `ACTIVE_DIMS[4]`, `ACTIVE_DIMS[5]` (layout 63 nilai yang sama) |
| `shared/training/transfer.py` | opsi `reinit_constant_dims`, `refit_all_stats`, `reinit_fields` |
| `shared/training/callbacks.py`, `shared/evaluation/task_spec.py` | kurikulum & tugas akhir generik untuk Misi 3-5 |
| `checkpoints/final_eval/run_final_eval.py` (baru) | evaluasi akhir 200 skenario, referensi vs RL, 3 kondisi |

Validasi: model Misi 3 pengguna dgn kode baru tetap 100 % (50 skenario uji). Pipeline Misi 4/5 (sanity → training →
validasi → uji held-out → ONNX → verifikasi ONNX → laporan) diuji end-to-end lalu dilatih penuh.
