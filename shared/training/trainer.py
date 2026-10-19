"""
shared/training/trainer.py
--------------------------
Pipeline training generik untuk semua misi (algoritma: SAC, Stable-Baselines3):

  1. sanity check (IK / env / controller skrip)            [debug tools]
  2. VecEnv (Subproc/Dummy) + VecNormalize (+ env eval terpisah)
  3. SAC (hyperparameter dari sac_config.yaml); opsional TRANSFER dari misi sebelumnya
  4. callback: reason-rate, (adaptive), progress/ETA, best-model eval, checkpoint
  5. TensorBoard (SAC bawaan + callback di atas)
  6. simpan final + best; ekspor ONNX berversi -> models/onnx_missionN_vK.onnx
  7. verify_onnx pada observasi mentah NYATA                [exporter]
  8. debug pasca-training (kurva, replay, ONNX vs checkpoint, robustness)

train_missionN.py hanya menyiapkan env class + hook khusus misi lalu memanggil
`train_mission()`.

Catatan SAC (off-policy) yang berbeda dari PPO:
  * Ada replay buffer; SB3 menyimpan observasi ASLI (belum dinormalisasi) di buffer dan menormalisasi
    saat sampling dengan statistik VecNormalize terkini, jadi VecNormalize aman dipakai.
  * Policy = Gaussian ter-squash tanh, aksi selalu di [-1, 1]; eksplorasi diatur koefisien entropi `ent_coef`
    (otomatis: "auto_<nilai awal>") dan `target_entropy`, bukan `ent_coef` konstan ala PPO.
  * `learning_starts` bawaan SB3 mengumpulkan aksi ACAK seragam; pada training hasil TRANSFER itu akan
    membuang policy yang sudah dimuat, maka `ArmSAC` memakai policy hasil transfer sejak langkah pertama.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch as th
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from shared.config_utils import load_yaml
from shared.envs.obs_layout import OBS_DIM
from shared.exporter import (export_policy_to_onnx, next_versioned_onnx_path, verify_onnx,
                             vecnormalize_stats_from_pkl, write_onnx_metadata)
from shared.training.callbacks import (BestModelEvalCallback, CriticWarmupCallback,
                                       ProgressCallback, ReasonRateCallback)
from shared.training.transfer import transfer_from_previous

_ACT = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}


class ArmSAC(SAC):
    """SAC Stable-Baselines3 + penyesuaian untuk curriculum dengan transfer bobot.

    1. `act_with_policy_from_start`: SB3 memakai aksi acak seragam selama `learning_starts`. Setelah transfer
       bobot itu membuang policy yang sudah bagus. Bila True, aksi dikumpulkan dengan policy yang dimuat sejak
       langkah pertama; `learning_starts` tetap menentukan kapan gradient pertama dihitung.

    `train()` di bawah adalah salinan `SAC.train()` SB3 2.9 dengan tambahan berikut (semua bisa dimatikan;
    nilai default = perilaku SB3 asli):

    2. `actor_frozen` (dipakai CriticWarmupCallback): actor DAN koefisien entropi tidak diperbarui; hanya
       critic yang dilatih. Sebelumnya koefisien entropi otomatis tetap diperbarui selama actor beku. Bila
       entropi policy hasil transfer berada di bawah `target_entropy` (Misi 3: log_pi ~10.5 vs target -6),
       actor yang beku tidak bisa menaikkan entropinya, sehingga alpha tumbuh eksponensial (terukur: 0.033 ->
       6.8 dalam 50k langkah) dan critic dilatih dengan target -alpha*log_pi raksasa (critic_loss 3 -> 33.600).
       Critic rusak itu diserahkan ke actor saat warm-up selesai -> success 63% -> 0% dalam 10k langkah.
       (Arah sebaliknya di Misi 2: alpha menyusut 20x selama warm-up lalu di-reset naik 22x sekaligus.)
    3. `ent_coef_max`: batas atas koefisien entropi otomatis (pengaman terhadap runaway yang sama setelah
       warm-up; pada Misi 3 alpha naik lagi ke ~0.9 setelah warm-up karena target -6 tetap tak tercapai).
    4. `actor_mean_reg`: regularisasi kecil mean pra-tanh actor, coef * mean(mu^2) (ada di implementasi SAC
       asli Haarnoja dengan 1e-3; SB3 tidak memilikinya). Pengaman ringan agar mean actor tidak melayang ke
       daerah tanh jenuh, tempat gradient Q tidak sampai ke actor. BUKAN penyebab keruntuhan Misi 3
       (|mu| terukur ~1.3 setelah transfer).
    5. Jangkar actor (`set_anchor`): setelah transfer, actor_loss += w_t * mean|Q| * mean((a - a_prior)^2),
       a_prior = aksi deterministik actor hasil transfer (salinan beku). w_t meluruh linear dari
       `actor_anchor_coef` ke `actor_anchor_floor` selama `actor_anchor_steps` langkah setelah warm-up.
       Skala mean|Q| mengikuti TD3+BC (coef 0.4 = kekuatan bawaan TD3+BC). Mencegah actor mengejar galat
       critic (overestimation). Terukur pada Misi 3 dgn jangkar meluruh ke 0: Q(s,pi) naik 26 -> 71 sementara
       success eval turun 90% -> 10% tepat saat jangkar habis; karena itu ada `actor_anchor_floor` > 0.
       `train/q_pi_mean` di TensorBoard memantau gejala ini.
    Checkpoint tetap bisa dimuat dengan `SAC.load()` biasa (salinan actor jangkar tidak disimpan).
    """

    act_with_policy_from_start: bool = False
    actor_frozen: bool = False
    ent_coef_max: float | None = None
    actor_mean_reg: float = 0.0
    actor_anchor_coef: float = 0.0
    actor_anchor_steps: int = 0
    actor_anchor_floor: float = 0.0

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["_anchor_actor"]

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if self.act_with_policy_from_start:
            learning_starts = 0
        return super()._sample_action(learning_starts, action_noise, n_envs)

    # ---- jangkar actor (dipanggil setelah transfer) ----
    def set_anchor(self, coef: float, steps: int, floor: float = 0.0, target: str = "prior"):
        """target 'prior' = aksi actor saat ini (hasil transfer); 'zero' = koreksi nol (struktur residual:
        tetap dekat controller referensi kecuali critic yakin koreksi menguntungkan)."""
        import copy
        self.actor_anchor_coef, self.actor_anchor_steps = float(coef), int(steps)
        self.actor_anchor_floor = min(float(floor), float(coef))
        self._anchor_zero = str(target).lower() == "zero"
        self._anchor_actor = None
        if self.actor_anchor_coef > 0 and not self._anchor_zero:
            self._anchor_actor = copy.deepcopy(self.actor).eval()
            for prm in self._anchor_actor.parameters():
                prm.requires_grad_(False)
        self._anchor_start = None            # diisi saat actor pertama kali dilatih

    def anchor_weight(self) -> float:
        active = getattr(self, "_anchor_actor", None) is not None or getattr(self, "_anchor_zero", False)
        if not active or self.actor_anchor_coef <= 0 or self.actor_anchor_steps <= 0 or self.actor_frozen:
            return 0.0
        if self._anchor_start is None:
            self._anchor_start = self.num_timesteps
        frac = (self.num_timesteps - self._anchor_start) / self.actor_anchor_steps
        return max(self.actor_anchor_floor, self.actor_anchor_coef * max(0.0, 1.0 - frac))

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        import torch.nn.functional as F
        from stable_baselines3.common.utils import polyak_update

        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, mu_abs, anchor_losses, q_pi = [], [], [], [], []
        update_actor = not self.actor_frozen
        w_anchor = self.anchor_weight()
        log_ent_max = None if self.ent_coef_max is None else float(np.log(self.ent_coef_max))

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            if self.use_sde:
                self.actor.reset_noise()

            mean_actions, log_std, kw = self.actor.get_action_dist_params(replay_data.observations)
            actions_pi, log_prob = self.actor.action_dist.log_prob_from_params(mean_actions, log_std, **kw)
            log_prob = log_prob.reshape(-1, 1)
            mu_abs.append(float(mean_actions.detach().abs().mean()))

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            # alpha TIDAK diperbarui selama actor beku (lihat docstring butir 2)
            if update_actor and ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                if log_ent_max is not None:
                    with th.no_grad():
                        self.log_ent_coef.clamp_(max=log_ent_max)

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if update_actor:
                q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
                min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
                q_pi.append(float(min_qf_pi.detach().mean()))
                actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
                if self.actor_mean_reg > 0:
                    actor_loss = actor_loss + self.actor_mean_reg * (mean_actions ** 2).mean()
                if w_anchor > 0:
                    with th.no_grad():
                        if self._anchor_zero:
                            a_prior = th.zeros_like(mean_actions)
                        else:
                            a_prior = self._anchor_actor(replay_data.observations, deterministic=True)
                        q_scale = min_qf_pi.abs().mean()
                    anchor = ((th.tanh(mean_actions) - a_prior) ** 2).mean()
                    anchor_losses.append(anchor.item())
                    actor_loss = actor_loss + w_anchor * q_scale * anchor
                actor_losses.append(actor_loss.item())
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        if actor_losses:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            # Q(s, pi(s)) rata-rata: naik terus sementara success turun = overestimation yang dieksploitasi actor
            self.logger.record("train/q_pi_mean", np.mean(q_pi))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/actor_mu_abs", np.mean(mu_abs))      # > ~3 = aksi jenuh
        self.logger.record("train/anchor_weight", w_anchor)
        if anchor_losses:
            self.logger.record("train/anchor_mse", np.mean(anchor_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


@dataclass
class TrainContext:
    mission_id: int
    project_root: Path
    cfg: dict
    env_cls: type
    env_cfg_path: str
    model_path: str
    ckpt_dir: Path
    logdir: Path
    models_dir: Path
    train_env: object = None
    eval_env: object = None
    model: SAC | None = None
    level_fn: Callable[[], int] = field(default=lambda: 0)
    extras: dict = field(default_factory=dict)

    def make_raw_env(self, training: bool = False, randomize: bool | None = None, seed: int = 0):
        """Satu env mentah (bukan VecEnv) untuk debug/verifikasi."""
        return self.env_cls(model_path=self.model_path, config_path=self.env_cfg_path,
                            training=training, randomize=randomize, seed=seed)


@dataclass
class MissionHooks:
    pre_debug: Callable[[TrainContext], None] | None = None
    build_callbacks: Callable[[TrainContext], list] | None = None
    post_debug: Callable[[TrainContext, dict], None] | None = None


# ---------------------------------------------------------------------- #
def _make_env_thunk(ctx_env_cls, model_path, env_cfg_path, seed, training, randomize):
    def _init():
        env = ctx_env_cls(model_path=model_path, config_path=env_cfg_path, seed=seed,
                          training=training, randomize=randomize)
        return Monitor(env)
    return _init


def _set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _lr(cfg_sac: dict):
    lr = float(cfg_sac.get("learning_rate", 3e-4))
    if str(cfg_sac.get("lr_schedule", "constant")) == "linear":
        return lambda progress: lr * max(progress, 0.05)
    return lr


def _ent_coef(value):
    """'auto' / 'auto_0.1' (otomatis, nilai awal) tetap string; angka -> koefisien entropi tetap."""
    if isinstance(value, str):
        return value
    return float(value)


def _target_entropy(value):
    return "auto" if str(value) == "auto" else float(value)


def build_sac(cfg: dict, train_env, logdir: Path, seed: int, device: str,
              act_with_policy_from_start: bool = False) -> ArmSAC:
    p = cfg["sac"]
    arch = list(p.get("net_arch", [256, 256]))
    qf_arch = list(p.get("qf_net_arch", arch))
    policy_kwargs = dict(net_arch=dict(pi=arch, qf=qf_arch),
                         activation_fn=_ACT[str(p.get("activation_fn", "relu")).lower()])
    model = ArmSAC(
        "MlpPolicy", train_env, learning_rate=_lr(p), buffer_size=int(p["buffer_size"]),
        learning_starts=int(p["learning_starts"]), batch_size=int(p["batch_size"]),
        tau=float(p["tau"]), gamma=float(p["gamma"]), train_freq=int(p.get("train_freq", 1)),
        gradient_steps=int(p.get("gradient_steps", -1)),
        ent_coef=_ent_coef(p.get("ent_coef", "auto")),
        target_entropy=_target_entropy(p.get("target_entropy", "auto")),
        policy_kwargs=policy_kwargs, tensorboard_log=str(logdir), seed=seed, device=device,
        verbose=1)
    model.act_with_policy_from_start = bool(act_with_policy_from_start)
    model.ent_coef_max = None if p.get("ent_coef_max") is None else float(p["ent_coef_max"])
    model.actor_mean_reg = float(p.get("actor_mean_reg", 0.0))
    return model


def init_residual_actor(model: SAC, log_std_init: float = -1.6):
    """Struktur residual: actor mulai dgn koreksi ~0 (perilaku awal = controller referensi) dan derau kecil."""
    actor = model.policy.actor
    with torch.no_grad():
        actor.mu.weight.mul_(0.01)
        actor.mu.bias.zero_()
        actor.log_std.weight.mul_(0.01)
        actor.log_std.bias.fill_(float(log_std_init))


# ---------------------------------------------------------------------- #
def heldout_test(ctx: TrainContext, model_zip: str, vecnorm_pkl: str, n: int, seed0: int) -> dict:
    """Uji deterministik pada n skenario held-out (seed tidak dipakai saat memilih model), tugas AKHIR misi
    (shared/evaluation/task_spec.py), env tanpa derau/randomisasi = sama dengan eval_missionN.py."""
    from collections import Counter
    from shared.evaluation.task_spec import apply_final_task
    from shared.evaluation.viewer_eval import wilson_interval
    model = SAC.load(model_zip, device="cpu")
    mean, var, clip, eps = vecnormalize_stats_from_pkl(vecnorm_pkl)
    std = np.sqrt(var + eps)
    env = ctx.make_raw_env(training=False, randomize=False, seed=seed0)
    apply_final_task(env, ctx.project_root, ctx.mission_id, sac_cfg=ctx.cfg)
    reasons = Counter()
    for i in range(n):
        obs, _ = env.reset(seed=seed0 + i)
        info = {}
        for _ in range(env.max_steps + 5):
            a, _ = model.predict(np.clip((obs - mean) / std, -clip, clip).astype(np.float32), deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            if term or trunc:
                break
        reasons[info.get("termination_reason") or "timeout"] += 1
    k = reasons.get("success", 0)
    lo, hi = wilson_interval(k, n)
    return {"n": n, "seed0": seed0, "success_rate": k / n, "ci95": [lo, hi], "reasons": dict(reasons)}


# ---------------------------------------------------------------------- #
def collect_raw_observations(ctx: TrainContext, model_zip: str, vecnorm_pkl: str,
                             n_steps: int = 300, log=print) -> np.ndarray:
    """Observasi MENTAH nyata (env mode training, dgn noise) yang dihasilkan policy."""
    from shared.debug_common.common import make_predict_fn_sb3
    model = SAC.load(model_zip, device="cpu")
    mean, var, clip, eps = vecnormalize_stats_from_pkl(vecnorm_pkl)
    fn = make_predict_fn_sb3(model, mean, var, clip, eps)
    env = ctx.make_raw_env(training=True, randomize=True, seed=777)
    obs, _ = env.reset(seed=777)
    out = [obs]
    for _ in range(n_steps):
        obs, _, term, trunc, _ = env.step(fn(obs))
        out.append(obs)
        if term or trunc:
            obs, _ = env.reset()
    return np.array(out, dtype=np.float32)


def export_and_verify(ctx: TrainContext, model_zip: str, vecnorm_pkl: str, meta: dict,
                      log=print) -> dict:
    ecfg = ctx.cfg["export"]
    model = SAC.load(model_zip, device="cpu")
    mean, var, clip, eps = vecnormalize_stats_from_pkl(vecnorm_pkl)
    onnx_path = next_versioned_onnx_path(str(ctx.models_dir), ecfg["name_prefix"])
    export_policy_to_onnx(model, mean, var, clip, eps, str(onnx_path), OBS_DIM)
    write_onnx_metadata(str(onnx_path), {"mission": ctx.mission_id, "algorithm": "SAC",
                                         "obs_dim": OBS_DIM, "source_model": model_zip,
                                         "vecnormalize": vecnorm_pkl, **meta})
    log(f"[onnx] diekspor -> {onnx_path}")
    obs_batch = collect_raw_observations(ctx, model_zip, vecnorm_pkl, log=log)
    max_d, mean_d, ok = verify_onnx(model, mean, var, clip, eps, str(onnx_path), OBS_DIM,
                                    tol=float(ecfg.get("verify_tolerance", 1e-4)),
                                    obs_batch=obs_batch)
    log(f"[onnx] verifikasi ({len(obs_batch)} observasi nyata): max|diff|={max_d:.2e} "
        f"mean|diff|={mean_d:.2e} -> {'SUKSES' if ok else 'GAGAL'}")
    return {"onnx_path": str(onnx_path), "max_diff": max_d, "mean_diff": mean_d, "verified": ok}


# ---------------------------------------------------------------------- #
def train_mission(*, mission_id: int, project_root: Path, sac_cfg_path: str, env_cls: type,
                  env_cfg_path: str, model_path: str, hooks: MissionHooks | None = None,
                  log=print) -> dict:
    hooks = hooks or MissionHooks()
    root = Path(project_root)
    cfg = load_yaml(sac_cfg_path)
    run, rt = cfg["run"], cfg.get("runtime", {})
    torch.set_num_threads(int(rt.get("torch_num_threads", 1)))
    device = str(rt.get("device", "cpu"))
    seed = int(run.get("seed", 0))
    _set_seeds(seed)

    # gagal-cepat: transfer butuh checkpoint misi sebelumnya (sebelum sanity check/pembuatan env)
    tc = cfg.get("transfer", {}) or {}
    if tc.get("enabled", False):
        for key in ("source_model", "source_vecnormalize"):
            if not (root / tc[key]).exists():
                raise FileNotFoundError(
                    f"Misi {mission_id} memuat bobot dari Misi {tc.get('source_mission', mission_id - 1)}, "
                    f"tetapi file belum ada: {root / tc[key]}\n"
                    f"Jalankan training Misi {tc.get('source_mission', mission_id - 1)} lebih dulu "
                    f"(python train_mission{tc.get('source_mission', mission_id - 1)}.py).")

    ctx = TrainContext(
        mission_id=mission_id, project_root=root, cfg=cfg, env_cls=env_cls,
        env_cfg_path=str(env_cfg_path), model_path=str(model_path),
        ckpt_dir=root / run["ckpt_dir"], logdir=root / run["logdir"],
        models_dir=root / cfg["export"]["onnx_dir"])
    for d in (ctx.ckpt_dir, ctx.logdir, ctx.models_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ---- 1. debug pra-training ----
    if hooks.pre_debug and cfg.get("debug", {}).get("run_sanity_check_before_training", True):
        log(f"\n=== Misi {mission_id}: sanity check pra-training ===")
        hooks.pre_debug(ctx)

    # ---- 2. environment ----
    n_envs = int(run["n_envs"])
    thunks = [_make_env_thunk(env_cls, ctx.model_path, ctx.env_cfg_path, seed + 1000 * i,
                              True, None) for i in range(n_envs)]
    kind = str(run.get("vec_env", "auto"))
    vec_cls = SubprocVecEnv if (kind == "subproc" or (kind == "auto" and n_envs > 1)) else DummyVecEnv
    norm = cfg["normalization"]
    train_env = VecNormalize(vec_cls(thunks), norm_obs=True, norm_reward=bool(norm.get("norm_reward", False)),
                             clip_obs=float(norm["clip_obs"]), epsilon=float(norm["epsilon"]))
    eval_env = VecNormalize(
        DummyVecEnv([_make_env_thunk(env_cls, ctx.model_path, ctx.env_cfg_path, seed + 99_999,
                                     False, False)]),
        norm_obs=True, norm_reward=False, training=False,
        clip_obs=float(norm["clip_obs"]), epsilon=float(norm["epsilon"]))
    ctx.train_env, ctx.eval_env = train_env, eval_env

    # ---- 3. SAC (+ transfer) ----
    tcfg = cfg.get("transfer", {}) or {}
    env_yaml = load_yaml(ctx.env_cfg_path)
    residual = str((env_yaml.get("policy_structure", {}) or {}).get("type", "direct")).lower() == "residual"
    model = build_sac(cfg, train_env, ctx.logdir, seed, device,
                      act_with_policy_from_start=bool(tcfg.get("enabled", False)) or residual)
    ctx.model = model
    if residual:
        log("[struktur] policy RESIDUAL: aksi = controller referensi (jalur tervalidasi) + koreksi RL "
            f"(skala posisi {env_yaml['policy_structure'].get('residual_scale_pos', 0.5)}, "
            f"rotasi {env_yaml['policy_structure'].get('residual_scale_rot', 1.0)})")
        if not tcfg.get("enabled", False):
            p = cfg["sac"]
            init_residual_actor(model, float(p.get("residual_log_std_init", -1.6)))
            log("[struktur] actor diinisialisasi dgn koreksi ~0 -> perilaku awal = controller referensi")
            # jangkar ke koreksi NOL: tanpa ini actor mengikuti critic yang masih acak dan merusak perilaku
            # awal (terukur: held-out 90% -> 0-25% dalam 3k langkah)
            model.set_anchor(float(p.get("residual_anchor_coef", 0.4)), int(p.get("residual_anchor_steps", 100_000)),
                             float(p.get("residual_anchor_floor", 0.1)), target="zero")
            log(f"[struktur] jangkar koreksi-nol: coef {model.actor_anchor_coef} -> {model.actor_anchor_floor} "
                f"dalam {model.actor_anchor_steps:,} langkah")
    if tcfg.get("enabled", False):
        transfer_from_previous(model, train_env, tcfg, root, mission_id, log=log)
        if residual and tcfg.get("reset_residual_output", False):
            # Fitur actor & critic dari misi sebelumnya dipertahankan, tetapi KOREKSI direset ke ~0: koreksi
            # yang berguna di misi sebelumnya bisa merugikan di misi baru (terukur Misi 3: koreksi warisan
            # Misi 2 menyimpang ~15 mm dari jalur -> tabrakan 10-16% pada margin rintangan 1 cm, padahal
            # referensi saja ~100%).
            init_residual_actor(model, float(tcfg.get("residual_log_std_init", -1.6)))
            log("[transfer] koreksi residual direset ke ~0 (perilaku awal = controller referensi)")
        target = str(tcfg.get("actor_anchor_target", "prior"))
        model.set_anchor(float(tcfg.get("actor_anchor_coef", 0.0)), int(tcfg.get("actor_anchor_steps", 0)),
                         float(tcfg.get("actor_anchor_floor", 0.0)), target=target)
        if model.actor_anchor_coef > 0:
            log(f"[transfer] jangkar actor ({'koreksi nol' if target == 'zero' else 'policy misi sebelumnya'}) aktif: "
                f"coef {model.actor_anchor_coef} meluruh ke {model.actor_anchor_floor} dalam "
                f"{model.actor_anchor_steps:,} langkah setelah warm-up")
        # eval env memakai stats train (disinkron oleh callback), tetapi inisialisasi juga:
        eval_env.obs_rms = train_env.obs_rms
        # Bekukan statistik normalisasi. Bila terus diperbarui, mean/var dimensi lama bergeser ke
        # distribusi misi baru -> input actor berubah -> policy hasil transfer rusak MESKIPUN bobot
        # actor tidak diubah.
        if tcfg.get("freeze_obs_rms", True):
            train_env.training = False
            log("[transfer] statistik normalisasi observasi DIBEKUKAN (tidak diperbarui selama training)")

    # ---- 4. callbacks ----
    reason_cb = ReasonRateCallback(window=100)
    extra = hooks.build_callbacks(ctx) if hooks.build_callbacks else []
    total = int(run["timesteps"])
    warmup = []
    if tcfg.get("enabled", False) and int(tcfg.get("critic_warmup_steps", 0)) > 0:
        warmup = [CriticWarmupCallback(int(tcfg["critic_warmup_steps"]))]
    elif residual and int(cfg["sac"].get("residual_critic_warmup_steps", 0)) > 0:
        # residual tanpa transfer: critic belajar menilai controller referensi dulu, actor (koreksi ~0) beku
        warmup = [CriticWarmupCallback(int(cfg["sac"]["residual_critic_warmup_steps"]))]
    callbacks = [reason_cb, *warmup, *extra,
                 ProgressCallback(total, float(rt.get("progress_print_interval_sec", 15)), reason_cb),
                 BestModelEvalCallback(eval_env, int(run.get("eval_episodes", 20)),
                                       int(run.get("eval_freq", 25_000)), str(ctx.ckpt_dir),
                                       level_fn=lambda: ctx.level_fn(),
                                       eval_seed=int(run.get("eval_seed", 424_242)),
                                       eval_at_start=bool(tcfg.get("enabled", False))),
                 CheckpointCallback(max(1, int(run["ckpt_freq"]) // n_envs), str(ctx.ckpt_dir / "periodic"),
                                    name_prefix=f"mission{mission_id}", save_vecnormalize=True)]

    # ---- 5. training ----
    log(f"\n=== Misi {mission_id}: training SAC {total:,} steps | {n_envs} env ({vec_cls.__name__}) "
        f"| TensorBoard: tensorboard --logdir {ctx.logdir} ===")
    t0 = time.time()
    interrupted = False
    try:
        model.learn(total_timesteps=total, callback=CallbackList(callbacks),
                    log_interval=int(run.get("log_interval_episodes", 10)),
                    tb_log_name=f"mission{mission_id}", progress_bar=False)
    except KeyboardInterrupt:
        interrupted = True
        log("\n[train] dihentikan pengguna (Ctrl+C) - menyimpan & mengekspor model terbaik...")
    wall = time.time() - t0

    # ---- 6. simpan ----
    final_zip = ctx.ckpt_dir / "final_model.zip"
    final_pkl = ctx.ckpt_dir / "final_vecnormalize.pkl"
    model.save(str(final_zip))
    train_env.save(str(final_pkl))
    best_zip, best_pkl = ctx.ckpt_dir / "best_model.zip", ctx.ckpt_dir / "best_vecnormalize.pkl"
    have_best = best_zip.exists() and best_pkl.exists()
    source = str(cfg["export"].get("source", "auto")).lower()
    log(f"[train] selesai dalam {wall / 60:.1f} menit.")

    # ---- 6b. validasi + uji held-out (seed TIDAK dipakai untuk memilih checkpoint selama training) ----
    # best_model dipilih sbg maksimum dari puluhan/ratusan evaluasi pada skenario yang SAMA -> angkanya
    # optimistis (terukur: Misi 3 pengguna, 400 evaluasi rata-rata 0.80, best 0.90). Di sini best vs final
    # dibandingkan pada set VALIDASI terpisah, lalu yang terpilih diuji pada set UJI terpisah lagi.
    n_val, n_test = int(run.get("final_validation_episodes", 100)), int(run.get("final_test_episodes", 200))
    val, test = {}, None
    if source == "auto" and have_best and n_val > 0:
        for tag, (z, pk) in {"best": (best_zip, best_pkl), "final": (final_zip, final_pkl)}.items():
            val[tag] = heldout_test(ctx, str(z), str(pk), n_val, int(run.get("validation_seed", 300_000)))
            log(f"[validasi] {tag:5s}: {val[tag]['success_rate']:.1%} pada {n_val} skenario {val[tag]['reasons']}")
        use_best = val["best"]["success_rate"] >= val["final"]["success_rate"]
    else:
        use_best = source in ("best", "auto") and have_best
    src_zip, src_pkl = (best_zip, best_pkl) if use_best else (final_zip, final_pkl)
    if n_test > 0:
        test = heldout_test(ctx, str(src_zip), str(src_pkl), n_test, int(run.get("test_seed", 400_000)))
        lo, hi = test["ci95"]
        log(f"[uji held-out] {'best' if use_best else 'final'}: {test['success_rate']:.1%} "
            f"(CI 95% {lo:.0%}-{hi:.0%}) pada {n_test} skenario baru {test['reasons']}  <- angka yang realistis")
        (ctx.ckpt_dir / "final_test.json").write_text(json.dumps(
            {"source": "best" if use_best else "final", "validation": val, "test": test,
             "timesteps": int(model.num_timesteps)}, indent=2), encoding="utf-8")
    log(f"[train] sumber ONNX: {'best' if use_best else 'final'}")

    # ---- 7. ONNX + verifikasi ----
    export = export_and_verify(ctx, str(src_zip), str(src_pkl),
                               {"source": "best" if use_best else "final",
                                "timesteps": int(model.num_timesteps), "interrupted": interrupted,
                                "heldout_test": test})
    result = {"final_model": str(final_zip), "best_model": str(best_zip) if best_zip.exists() else None,
              "interrupted": interrupted, "wall_sec": wall, **export, "tb_logdir": str(ctx.logdir),
              "heldout_test": test, "validation": val}

    # ---- 8. debug pasca-training ----
    if hooks.post_debug and cfg.get("debug", {}).get("enabled", True):
        log(f"\n=== Misi {mission_id}: debug pasca-training ===")
        try:
            hooks.post_debug(ctx, result)
        except Exception as e:                                   # debug tidak boleh menggagalkan hasil
            log(f"[debug] peringatan: {type(e).__name__}: {e}")

    train_env.close()
    eval_env.close()
    log(f"\nSELESAI. ONNX: {export['onnx_path']}\nTensorBoard: tensorboard --logdir {ctx.logdir}")
    return result
