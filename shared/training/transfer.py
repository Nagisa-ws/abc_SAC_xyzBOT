"""
shared/training/transfer.py
---------------------------
Transfer bobot Misi N -> Misi N+1 (curriculum), versi SAC.

Karena observasi terpadu berdimensi tetap (obs_layout.OBS_DIM), arsitektur sama
dan bobot bisa dimuat langsung. Yang perlu ditangani adalah dimensi yang BARU
AKTIF di misi baru (mis. fitur path di Misi 2, fitur rintangan di Misi 3):

  * Di misi lama dimensi itu konstan -> varians statistik normalisasi ~ 0 dan
    bobot input-nya tidak pernah mendapat gradien (masih acak/ortogonal).
  * Bila dibiarkan, observasi baru ternormalisasi -> nilai ekstrem (kena clip)
    dan bobot acak mengacaukan policy hasil Misi N.

Penanganan:
  1. Bobot layer pertama untuk dimensi baru di-NOL-kan pada actor DAN kedua critic Q (qf0, qf1; target
     critic disalin ulang), sehingga di awal Misi N+1 perilaku = policy Misi N, lalu fitur baru dipelajari
     bertahap. (Input critic = [observasi, aksi]; dimensi observasi ada di depan, indeksnya sama.)
  2. obs_rms: statistik dimensi LAMA dipertahankan (input actor tidak bergeser ->
     perilaku policy lama terjaga). Statistik dimensi BARU diisi dari data nyata env
     misi baru (rollout dengan policy hasil transfer, `warm_stats_steps` langkah),
     bukan tebakan mean 0 / var 1. `count` dibuat cukup besar (`obs_rms_count`) agar
     statistik tidak melompat pada iterasi awal.
  3. Eksplorasi. SAC tidak punya parameter `log_std` global seperti PPO: std aksi dikeluarkan oleh head
     `actor.log_std` (bergantung observasi) dan keseimbangan eksplorasi diatur koefisien entropi.
       * `log_std_init` (opsional): head log_std diganti konstan (bobot = 0, bias = nilai), jadi std aksi
         (sebelum tanh) seragam = exp(nilai) di awal. Mengganti std yang sudah dipelajari menghapus
         ketergantungannya pada observasi; pakai hanya bila memang ingin eksplorasi baru (mis. Misi 3).
       * koefisien entropi TIDAK disalin dari misi sebelumnya kecuali `copy_ent_coef: true`; nilai awalnya
         dari `sac.ent_coef` di config misi baru.
  4. Optimizer (Adam) tidak dipindahkan: moment lama tidak relevan untuk reward/misi baru.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from shared.envs.obs_layout import ACTIVE_DIMS, OBS_DIM

_MIN_VAR = 1.0e-4        # lantai varians untuk dimensi baru (hindari pembagian ~0)


def _load_vecnorm_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _collect_raw_obs(model: SAC, vecnorm, n_steps: int) -> np.ndarray:
    """Observasi MENTAH dari env misi baru, digerakkan oleh policy hasil transfer (stokastik)."""
    inner = vecnorm.venv
    raw = inner.reset()
    out = [raw.copy()]
    for _ in range(max(1, n_steps // max(1, inner.num_envs))):
        act, _ = model.predict(vecnorm.normalize_obs(raw), deterministic=False)
        raw, _, _, _ = inner.step(act)
        out.append(raw.copy())
    return np.concatenate(out, axis=0)


def _first_linear_layers(model: SAC):
    """Layer Linear pertama actor dan kedua critic Q (menerima observasi di dimensi depan)."""
    layers = [model.policy.actor.latent_pi[0]]
    layers += [q[0] for q in model.policy.critic.q_networks]
    return layers


def policy_action_std(model: SAC, normed_obs: np.ndarray) -> np.ndarray:
    """Rata-rata std aksi (sebelum tanh) per dimensi aksi pada `normed_obs` - untuk log/diagnosis."""
    obs = torch.as_tensor(normed_obs, dtype=torch.float32, device=model.device)
    with torch.no_grad():
        _, log_std, _ = model.policy.actor.get_action_dist_params(obs)
    return torch.exp(log_std).mean(dim=0).cpu().numpy()


def transfer_from_previous(model: SAC, vecnorm, tcfg: dict, project_root: Path,
                           target_mission: int, log=print) -> dict:
    """Muat bobot + statistik normalisasi Misi sebelumnya ke `model` & `vecnorm` (in-place).

    tcfg: {source_model, source_vecnormalize, source_mission, zero_new_input_weights,
           obs_rms_count, warm_stats_steps, log_std_init, copy_ent_coef}
    """
    src_model = Path(project_root) / tcfg["source_model"]
    src_norm = Path(project_root) / tcfg["source_vecnormalize"]
    if not src_model.exists():
        raise FileNotFoundError(
            f"Checkpoint Misi {tcfg.get('source_mission')} tidak ditemukan: {src_model}\n"
            f"Jalankan training misi sebelumnya dulu.")
    if not src_norm.exists():
        raise FileNotFoundError(f"Statistik VecNormalize tidak ditemukan: {src_norm}")

    prev = SAC.load(str(src_model), device="cpu")
    if prev.observation_space.shape != model.observation_space.shape:
        raise ValueError(f"Dimensi observasi berbeda: {prev.observation_space.shape} vs "
                         f"{model.observation_space.shape}")
    model.policy.load_state_dict(prev.policy.state_dict(), strict=True)

    src_mission = int(tcfg.get("source_mission", target_mission - 1))
    new_dims = ACTIVE_DIMS[target_mission] & ~ACTIVE_DIMS[src_mission]
    old = _load_vecnorm_pickle(str(src_norm))
    src_var = np.array(old.obs_rms.var, dtype=np.float64)
    if tcfg.get("reinit_constant_dims", False):
        # Dimensi yang KONSTAN di misi sumber (var ~ 0) tetapi bervariasi di misi baru diperlakukan sbg dimensi
        # baru. Contoh Misi 3 -> 4: ref_action rotasi selalu 0 di Misi 3; di Misi 4 controller referensi memutar
        # gripper. Dengan statistik lama (var ~ 0) nilai ternormalisasi meledak ke batas clip (+-10).
        const = src_var < 1e-6
        if np.any(const & ~new_dims):
            log(f"[transfer] {int(np.sum(const & ~new_dims))} dimensi konstan di Misi {src_mission} "
                f"diperlakukan sbg dimensi baru: {np.where(const & ~new_dims)[0].tolist()}")
        new_dims = new_dims | const
    if tcfg.get("reinit_fields"):
        # field yang ARTINYA berubah di misi baru (mis. Misi 4: slot rintangan ke-4 = gaya pergelangan)
        from shared.envs.obs_layout import OBS_SLICES
        for name, offs in dict(tcfg["reinit_fields"]).items():
            st = OBS_SLICES[name].start
            for o in offs:
                new_dims[st + int(o)] = True
        log(f"[transfer] field berubah arti -> dimensi baru: {dict(tcfg['reinit_fields'])}")
    idx = torch.as_tensor(np.where(new_dims)[0], dtype=torch.long)

    if tcfg.get("zero_new_input_weights", True) and len(idx) > 0:
        with torch.no_grad():
            for first in _first_linear_layers(model):
                first.weight[:, idx] = 0.0
    # target critic = salinan critic (setelah penolan bobot di atas)
    model.policy.critic_target.load_state_dict(model.policy.critic.state_dict())

    # ---- statistik normalisasi ----
    mean = np.array(old.obs_rms.mean, dtype=np.float64)
    var = src_var.copy()
    # refit_all_stats: statistik SEMUA dimensi diisi ulang dari data misi baru (dipakai bila distribusi
    # observasi berubah total, mis. Misi 4: tanaman & daun, bukan konstelasi titik)
    refit = np.ones_like(new_dims) if tcfg.get("refit_all_stats", False) else new_dims.copy()
    mean[refit], var[refit] = 0.0, 1.0                      # sementara
    mean[new_dims], var[new_dims] = 0.0, 1.0
    vecnorm.obs_rms.mean, vecnorm.obs_rms.var = mean, var
    vecnorm.obs_rms.count = float(tcfg.get("obs_rms_count", 50_000.0))

    warm = int(tcfg.get("warm_stats_steps", 0))
    raw = None
    if warm > 0 and refit.any():
        if tcfg.get("refit_all_stats", False):
            # statistik sementara yang wajar sebelum rollout: dari misi sumber utk dimensi lama
            m0, v0 = np.array(old.obs_rms.mean, dtype=np.float64), src_var.copy()
            m0[new_dims], v0[new_dims] = 0.0, 1.0
            vecnorm.obs_rms.mean, vecnorm.obs_rms.var = m0, np.maximum(v0, _MIN_VAR)
        raw = _collect_raw_obs(model, vecnorm, warm)
        mean[refit] = raw[:, refit].mean(axis=0)
        var[refit] = np.maximum(raw[:, refit].var(axis=0), _MIN_VAR)
        vecnorm.obs_rms.mean, vecnorm.obs_rms.var = mean, var
        log(f"[transfer] statistik {int(refit.sum())} dimensi diisi dari {len(raw)} observasi nyata misi baru")

    # ---- eksplorasi ----
    actor = model.policy.actor
    with torch.no_grad():
        if tcfg.get("log_std_init") is not None:
            actor.log_std.weight.zero_()
            actor.log_std.bias.fill_(float(tcfg["log_std_init"]))
        if tcfg.get("copy_ent_coef", False) and getattr(model, "log_ent_coef", None) is not None \
                and getattr(prev, "log_ent_coef", None) is not None:
            model.log_ent_coef.copy_(prev.log_ent_coef.to(model.log_ent_coef.device))

    # diagnosis: std aksi pada observasi nyata (ternormalisasi dengan statistik transfer)
    if raw is None:
        raw = _collect_raw_obs(model, vecnorm, 512)
    sample = vecnorm.normalize_obs(raw[:512]).astype(np.float32)
    std = policy_action_std(model, sample)

    info = {"source_model": str(src_model), "n_new_dims": int(new_dims.sum()),
            "new_dims": np.where(new_dims)[0].tolist(), "obs_dim": OBS_DIM}
    log(f"[transfer] bobot Misi {src_mission} dimuat dari {src_model.name}; "
        f"{info['n_new_dims']} dimensi baru aktif (bobot di-nol-kan: "
        f"{bool(tcfg.get('zero_new_input_weights', True))}) | std aksi (pra-tanh) = "
        f"{np.round(std, 2).tolist()}")
    return info
