"""PPO with a variable number of chargers (algorithm.md Alg. 2; DD11, DD14, DD15, DD21; safety_layer.md Sec. 9).

* Gaussian policy over raw per-charger actions a; executed power p = Pi_t(P_ref a) (or the fallback, or the
  physical clip for penalty baselines) is part of the environment; the likelihood ratio uses the RAW sample.
* log-prob and entropy sum only over influential tokens infl = occ & (u - l > eps_num) (DD21); fallback steps are
  masked out of the actor loss (kept for the critic); the actor loss uses the FIXED minibatch size as normaliser.
* beta regulariser L_reg = sum_w mean_{i in occ} ((xbar_i - Pi(xbar)_i)/P_ref)^2 / B_mb with xbar = P_ref mu and
  Pi(xbar) under stop-gradient (Moreau identity: exact gradient of 0.5 dist^2, S 9).
* Learning reward r / (N r_ref), r_ref = c_ref P_ref dt (DD15); penalty baselines use r_pen / (N r_ref).
* GAE(gamma, lambda) with true terminals at t = T; clipped surrogate, value loss, entropy bonus, Adam with
  linear LR decay, global grad-norm clipping, target-KL early stop of the epoch loop.
* Mixed-N batches: buffers of width N_env padded to N_max at update time (Lemma N1(iv)).
"""
from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import Cfg, config_hash8, exp_id as make_exp_id, save_config
from ..constants import C_REF, P_REF
from ..envs.scenarios import TrainMixture
from ..envs.vec_env import EnvSpec, VecEnv
from ..envs.wrapper import WrapperOptions
from ..models.attention_policy import AttentionActor, AttentionCritic
from ..models.mlp_policy import MLPActor, MLPCritic
from ..safety.projection import project_batched
from ..utils import RunMeta, append_csv, set_global_seed
from .buffers import RolloutBuffer

LOG2PI = math.log(2 * math.pi)

TRAIN_LOG_COLUMNS = [
    "exp_id", "env_steps", "wall_clock_s", "episode_return_eur_mean", "episode_return_eur_std",
    "frac_unserved_mean", "tr_violation_kwh_mean", "policy_loss", "value_loss", "entropy", "approx_kl",
    "clip_frac", "beta_reg", "proj_dist_mean", "lr",
    # extras
    "update", "n_episodes", "penalty_return_eur_mean", "infeasible_frac", "fallback_frac", "epochs_run",
    "n_evs_admission_reduced_mean", "n_admission_not_restored_mean", "sps", "update_time_s", "rollout_time_s",
]


# --------------------------------------------------------------------------------------- builders
def build_networks(cfg) -> tuple[nn.Module, nn.Module]:
    m = cfg["model"]
    if m["arch"] == "attention":
        return (AttentionActor(m["d_model"], m["n_layers"], m["n_heads"], m["head_hidden"]),
                AttentionCritic(m["d_model"], m["n_layers"], m["n_heads"], m["head_hidden"]))
    if m["arch"] == "mlp":
        return MLPActor(m["n_max"], m["mlp_hidden"]), MLPCritic(m["n_max"], m["mlp_hidden"])
    raise ValueError(f"unknown architecture {m['arch']}")


def conformal_provider(cfg, root: Path | None = None):
    c = cfg["conformal"]
    if not c["enabled"]:
        return None
    from ..safety.conformal import ConformalProvider
    cache = Path(c["cache_dir"])
    if root is not None and not cache.is_absolute():
        cache = root / cache
    return ConformalProvider(c["alpha"], cache, c["n_train"], c["n_cal"], c["predictor"], c["uncalibrated"])


def wrapper_options(cfg, root: Path | None = None, admission: bool | None = None) -> WrapperOptions:
    s, r = cfg["safety"], cfg["reward"]
    return WrapperOptions(safety=bool(s["enabled"]), lower_bound=bool(s.get("lower_bound", True)),
                          admission=bool(s["admission"] if admission is None else admission),
                          admission_existing_targets=s["admission_existing_targets"], eps_q=float(s["eps_q"]),
                          forecast_rel_std=float(cfg["features"]["forecast_rel_std"]), w_E=float(r["w_E"]),
                          w_P=float(r["w_P"]), penalty_dense=bool(r["dense"]),
                          conformal_provider=conformal_provider(cfg, root))


def train_mixture(cfg) -> TrainMixture:
    sc = dict(cfg["scenario"])
    profiles = sc.pop("profiles")
    rhos = sc.pop("rho_tr")
    base = {k: v for k, v in sc.items() if v is not None}
    return TrainMixture(profiles=list(profiles), rho_tr=[float(x) for x in rhos], base=base)


def env_specs(groups, run_seed: int) -> list[EnvSpec]:
    specs, k = [], 0
    for g in groups:
        for _ in range(int(g["count"])):
            specs.append(EnvSpec(N=int(g["N"]), index=k, run_seed=run_seed))
            k += 1
    return specs


# ------------------------------------------------------------------------------------- batching
def batch_obs(obs_list, device) -> dict:
    """Pad a list of Observations (different N) into tensors."""
    Nmax = max(o.tokens.shape[0] for o in obs_list)
    E = len(obs_list)
    tok = np.zeros((E, Nmax, obs_list[0].tokens.shape[1]), np.float32)
    occ = np.zeros((E, Nmax), bool)
    infl = np.zeros((E, Nmax), bool)
    for k, o in enumerate(obs_list):
        n = o.tokens.shape[0]
        tok[k, :n] = o.tokens
        occ[k, :n] = o.occ
        infl[k, :n] = o.infl
    glob = np.stack([o.glob for o in obs_list])
    return {"tokens": torch.as_tensor(tok, device=device), "glob": torch.as_tensor(glob, device=device),
            "occ": torch.as_tensor(occ, device=device), "infl": torch.as_tensor(infl, device=device)}


def gaussian_logp(a, mu, log_std):
    return -0.5 * ((a - mu) / torch.exp(log_std)) ** 2 - log_std - 0.5 * LOG2PI


# ---------------------------------------------------------------------------------------- update
def ppo_update(actor, critic, opt, batch: dict, pc: dict, generator: torch.Generator, p_ref: float = P_REF) -> dict:
    """One PPO update (epochs x minibatches) on a flattened, padded batch (Alg. 2 'update')."""
    B = batch["tokens"].shape[0]
    mb = int(pc["minibatch"])
    params = list(actor.parameters()) + list(critic.parameters())
    clip = float(pc["clip"])
    beta = float(pc["beta"])
    stats = {k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "beta_reg")}
    epochs_run = 0
    dev = batch["tokens"].device
    for _ in range(int(pc["epochs"])):
        epochs_run += 1
        perm = torch.randperm(B, generator=generator, device="cpu").to(dev)
        kls = []
        for s in range(0, B, mb):
            idx = perm[s:s + mb]
            Bm = idx.numel()
            tok, glob, occ = batch["tokens"][idx], batch["glob"][idx], batch["occ"][idx]
            infl = batch["infl"][idx].to(torch.float32)
            w = batch["w"][idx].to(torch.float32)
            mu, log_std = actor(tok, glob, occ)
            v = critic(tok, glob, occ)
            lp_tok = gaussian_logp(batch["a"][idx], mu, log_std)
            logp_new = (lp_tok * infl).sum(-1)
            logratio = logp_new - batch["logp"][idx]
            ratio = torch.exp(logratio)
            A = batch["adv"][idx]
            if pc.get("norm_adv", True) and Bm > 1:
                A = (A - A.mean()) / (A.std() + 1e-8)
            surr = torch.min(ratio * A, torch.clamp(ratio, 1 - clip, 1 + clip) * A)
            L_clip = (w * surr).sum() / Bm                                   # fixed normaliser (DD21)
            n_infl = infl.sum(-1)
            ent_tok = 0.5 + 0.5 * LOG2PI + log_std
            H_row = torch.where(n_infl > 0, (ent_tok * infl).sum(-1) / n_infl.clamp_min(1.0), torch.zeros_like(n_infl))
            H = (w * H_row).sum() / Bm
            L_reg = torch.zeros((), device=dev)
            if beta > 0:
                xbar = p_ref * mu
                feas = batch["feasible"][idx]
                with torch.no_grad():
                    pbar = xbar.detach().to(torch.float64).clone()
                    if feas.any():
                        pf, _, _ = project_batched(pbar[feas], batch["l"][idx][feas], batch["u"][idx][feas],
                                                   batch["Ctil"][idx][feas], batch["Crev"][idx][feas], occ[feas])
                        pbar[feas] = pf
                    pbar = pbar.to(xbar.dtype)
                occf = occ.to(torch.float32)
                reg_row = ((((xbar - pbar) / p_ref) ** 2) * occf).sum(-1) / occf.sum(-1).clamp_min(1.0)
                wreg = w * feas.to(torch.float32)
                L_reg = (wreg * reg_row).sum() / Bm
            L_vf = ((v - batch["ret"][idx]) ** 2).mean()
            loss = -L_clip + float(pc["vf_coef"]) * L_vf - float(pc["ent_coef"]) * H + beta * L_reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, float(pc["max_grad_norm"]))
            opt.step()
            with torch.no_grad():
                keep = w > 0
                if keep.any():
                    kl = ((ratio - 1) - logratio)[keep].mean().item()
                    cf = ((ratio - 1).abs() > clip)[keep].float().mean().item()
                else:
                    kl, cf = 0.0, 0.0
            kls.append(kl)
            stats["policy_loss"].append(-L_clip.item())
            stats["value_loss"].append(L_vf.item())
            stats["entropy"].append(H.item())
            stats["approx_kl"].append(kl)
            stats["clip_frac"].append(cf)
            stats["beta_reg"].append(L_reg.item())
        if pc.get("target_kl") is not None and np.mean(kls) > float(pc["target_kl"]):
            break
    out = {k: float(np.mean(v)) if v else float("nan") for k, v in stats.items()}
    out["epochs_run"] = epochs_run
    return out


# -------------------------------------------------------------------------------------- trainer
class PPOTrainer:
    def __init__(self, cfg: Cfg, out_dir: str | Path, seed: int, device: str = "cpu", resume: bool = False,
                 root: Path | None = None, exp_id: str | None = None):
        self.cfg = cfg
        self.seed = int(seed)
        self.device = torch.device(device)
        self.root = Path(root) if root else Path.cwd()
        self.h8 = config_hash8(cfg)
        self.exp_id = exp_id or make_exp_id(cfg["method"], cfg.get("_name", "cfg"), self.seed, self.h8)
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        tt = int(cfg["train"]["torch_threads"])
        if tt > 0:
            torch.set_num_threads(tt)
        set_global_seed(self.seed, bool(cfg["train"]["deterministic"]))
        self.actor, self.critic = build_networks(cfg)
        self.actor.to(self.device)
        self.critic.to(self.device)
        pc = cfg["ppo"]
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=float(pc["lr"]),
                                    eps=float(pc["adam_eps"]))
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(self.seed)
        self.sample_gen = torch.Generator(device=self.device)
        self.sample_gen.manual_seed(self.seed + 1)
        self.safety = bool(cfg["safety"]["enabled"])
        self.reward_key = "pen" if cfg["reward"]["type"] == "penalty" else "profit"
        groups = cfg["envs"]["groups"]
        self.E = sum(int(g["count"]) for g in groups)
        self.S = int(pc["rollout_len"])
        self.num_updates = max(1, int(float(pc["total_env_steps"]) // (self.E * self.S)))
        self.update = 0
        self.env_steps = 0
        self.wall = 0.0
        self.env_states = None
        if resume:
            self._load(self.out / "checkpoint_last.pt")
        save_config(cfg, self.out / "config.yaml")
        self.meta = RunMeta(self.out, bool(cfg.get("smoke", False)), str(self.device),
                            {"exp_id": self.exp_id, "method": cfg["method"], "seed": self.seed,
                             "config_hash8": self.h8, "train_N": self.train_Ns()})

    def train_Ns(self) -> str:
        return "+".join(str(n) for n in sorted({int(g["N"]) for g in self.cfg["envs"]["groups"]}))

    # ---------------------------------------------------------------------------- checkpoints
    def state(self) -> dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "optim": self.opt.state_dict(),
                "update": self.update, "env_steps": self.env_steps, "wall_clock_s": self.wall,
                "cfg": dict(self.cfg), "exp_id": self.exp_id, "seed": self.seed, "config_hash8": self.h8,
                "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
                        "gen": self.gen.get_state(), "sample_gen": self.sample_gen.get_state()},
                "env_states": self.env_states}

    def save(self, name: str) -> Path:
        path = self.out / name
        tmp = path.with_suffix(".tmp")
        torch.save(self.state(), tmp)
        os.replace(tmp, path)
        return path

    def _load(self, path: Path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        self.opt.load_state_dict(ck["optim"])
        self.update, self.env_steps, self.wall = ck["update"], ck["env_steps"], ck.get("wall_clock_s", 0.0)
        torch.set_rng_state(ck["rng"]["torch"])
        np.random.set_state(ck["rng"]["numpy"])
        random.setstate(ck["rng"]["python"])
        self.gen.set_state(ck["rng"]["gen"])
        self.sample_gen.set_state(ck["rng"]["sample_gen"])
        self.env_states = ck.get("env_states")

    # --------------------------------------------------------------------------------- envs
    def _make_vec(self, groups) -> VecEnv:
        backend = self.cfg["envs"]["backend"]
        return VecEnv(env_specs(groups, self.seed), train_mixture(self.cfg), wrapper_options(self.cfg, self.root),
                      backend=backend)

    def _groups_for(self, env_steps: int):
        cur = self.cfg["envs"]["curriculum"]
        if cur["enabled"] and env_steps < float(cur["frac"]) * float(self.cfg["ppo"]["total_env_steps"]):
            return cur["groups"], "phase1"
        return self.cfg["envs"]["groups"], "main"

    # --------------------------------------------------------------------------------- train
    @torch.no_grad()
    def _act(self, obs_list):
        b = batch_obs(obs_list, self.device)
        mu, log_std = self.actor(b["tokens"], b["glob"], b["occ"])
        v = self.critic(b["tokens"], b["glob"], b["occ"])
        eps = torch.randn(mu.shape, generator=self.sample_gen, device=self.device)
        a = mu + torch.exp(log_std) * eps
        logp = (gaussian_logp(a, mu, log_std) * b["infl"].to(torch.float32)).sum(-1)
        return a.cpu().numpy(), logp.cpu().numpy(), v.cpu().numpy()

    def train(self) -> Path:
        pc = self.cfg["ppo"]
        dt_h = float(self.cfg["scenario"]["dt_min"]) / 60.0
        r_ref = C_REF * P_REF * dt_h
        groups, phase = self._groups_for(self.env_steps)
        vec = self._make_vec(groups)
        if self.env_states is not None and len(self.env_states) == len(vec.specs):
            vec.set_states(self.env_states)
        obs = vec.reset()
        t_start = time.time() - self.wall
        try:
            while self.update < self.num_updates:
                g_now, ph = self._groups_for(self.env_steps)
                if ph != phase:
                    vec.close()
                    groups, phase = g_now, ph
                    vec = self._make_vec(groups)
                    obs = vec.reset()
                self.update += 1
                if pc["lr_decay"]:
                    frac = 1.0 - (self.update - 1.0) / self.num_updates
                    for gparam in self.opt.param_groups:
                        gparam["lr"] = frac * float(pc["lr"])
                lr_now = self.opt.param_groups[0]["lr"]
                Ns = vec.Ns
                buf = RolloutBuffer(self.S, Ns)
                episodes = []
                tr0 = time.time()
                for k in range(self.S):
                    a, logp, v = self._act(obs)
                    xs = [P_REF * a[e, :Ns[e]].astype(np.float64) for e in range(len(Ns))]
                    res = vec.step(xs)
                    for e, (out, nobs, ep) in enumerate(res):
                        r = out.reward_pen if self.reward_key == "pen" else out.reward
                        buf.envs[e].store(k, obs[e], a[e, :Ns[e]], logp[e], v[e], r / (Ns[e] * r_ref), out.done)
                        if ep is not None:
                            episodes.append(ep)
                    obs = [nobs for (_, nobs, _) in res]
                _, _, last_v = self._act(obs)
                for e, b in enumerate(buf.envs):
                    b.compute_gae(float(last_v[e]), float(pc["gamma"]), float(pc["gae_lambda"]))
                t_roll = time.time() - tr0
                self.env_steps += self.E * self.S
                tu0 = time.time()
                batch = buf.flatten(self.device, self.safety)
                st = ppo_update(self.actor, self.critic, self.opt, batch, pc, self.gen)
                t_upd = time.time() - tu0
                self.wall = time.time() - t_start
                feas = np.concatenate([b.feasible for b in buf.envs])
                self._log(episodes, st, lr_now, feas, t_roll, t_upd)
                if self.update % int(self.cfg["train"]["save_every"]) == 0 or self.update == self.num_updates:
                    self.env_states = vec.get_states()
                    self.save("checkpoint_last.pt")
                    if self.cfg["train"]["keep_checkpoints"]:
                        self.save(f"checkpoint_{self.update:06d}.pt")
        finally:
            vec.close()
        path = self.save("checkpoint_final.pt")
        self.meta.finish(updates=self.update, env_steps=self.env_steps)
        return path

    def _log(self, episodes, st, lr, feas, t_roll, t_upd) -> None:
        """Episode statistics are averaged over a running window of the last 2E completed episodes, so that every
        row has values even when an update completes no episode (T > rollout length, e.g. 5-min steps). Updates
        before the first completed episode go to train_log_warmup.csv (same columns, empty episode statistics)."""
        if not hasattr(self, "_ep_window"):
            from collections import deque
            self._ep_window = deque(maxlen=2 * self.E)
        n_new = len(episodes)
        self._ep_window.extend(episodes)
        episodes = list(self._ep_window)

        def m(key):
            vals = [e[key] for e in episodes if e.get(key) is not None and not
                    (isinstance(e.get(key), float) and math.isnan(e[key]))]
            return float(np.mean(vals)) if vals else float("nan")

        prof = [e["profit_eur"] for e in episodes]
        row = {"exp_id": self.exp_id, "env_steps": self.env_steps, "wall_clock_s": self.wall,
               "episode_return_eur_mean": float(np.mean(prof)) if prof else float("nan"),
               "episode_return_eur_std": float(np.std(prof)) if prof else float("nan"),
               "frac_unserved_mean": m("frac_evs_unserved"), "tr_violation_kwh_mean": m("tr_violation_kwh"),
               "proj_dist_mean": m("proj_dist_mean"), "lr": lr, "update": self.update, "n_episodes": n_new,
               "penalty_return_eur_mean": m("penalty_return_eur"), "infeasible_frac": float(np.mean(~feas)),
               "fallback_frac": float(np.mean(~feas)) if self.safety else 0.0,
               "n_evs_admission_reduced_mean": m("n_evs_admission_reduced"),
               "n_admission_not_restored_mean": m("n_admission_not_restored"),
               "sps": self.E * self.S / max(t_roll + t_upd, 1e-9), "update_time_s": t_upd, "rollout_time_s": t_roll,
               **st}
        name = "train_log.csv" if episodes else "train_log_warmup.csv"
        append_csv(self.out / name, [row], TRAIN_LOG_COLUMNS)


def load_actor(checkpoint: str | Path, device: str = "cpu"):
    """Load (actor, cfg, checkpoint dict) for evaluation."""
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = Cfg(ck["cfg"])
    actor, _ = build_networks(cfg)
    actor.load_state_dict(ck["actor"])
    actor.to(device).eval()
    return actor, cfg, ck
