"""Rollout storage with per-env widths and right-padding to N_max at update time (algorithm.md Alg. 2).

Each env writes into its own buffer of width N_env. :meth:`RolloutBuffer.flatten` pads all token tensors to
N_max (largest N in the batch) with occ = infl = False and x = l = u = 0 (padding changes neither actor outputs
of real tokens nor the critic value, Lemma N1(iv); masked sums ignore it).
"""
from __future__ import annotations

import numpy as np
import torch

from ..constants import F_C, F_G


class EnvBuffer:
    def __init__(self, S: int, N: int, n_cost: int = 0):
        self.S, self.N, self.n_cost = S, N, n_cost
        self.tokens = np.zeros((S, N, F_C), np.float32)
        self.glob = np.zeros((S, F_G), np.float32)
        self.occ = np.zeros((S, N), bool)
        self.infl = np.zeros((S, N), bool)
        self.l = np.zeros((S, N))
        self.u = np.zeros((S, N))
        self.Ctil = np.zeros(S)
        self.Crev = np.zeros(S)
        self.feasible = np.zeros(S, bool)
        self.a = np.zeros((S, N), np.float32)
        self.logp = np.zeros(S, np.float32)
        self.value = np.zeros(S, np.float32)
        self.reward = np.zeros(S, np.float32)
        self.done = np.zeros(S, bool)
        self.adv = np.zeros(S, np.float32)
        self.ret = np.zeros(S, np.float32)
        # PPO-Lagrangian (DD33(3)): normalised per-step costs, cost values, cost GAE and returns
        self.cost = np.zeros((S, n_cost), np.float32)
        self.cvalue = np.zeros((S, n_cost), np.float32)
        self.cadv = np.zeros((S, n_cost), np.float32)
        self.cret = np.zeros((S, n_cost), np.float32)

    def store(self, k: int, obs, a, logp, value, reward, done, cost=None, cvalue=None) -> None:
        self.tokens[k] = obs.tokens
        self.glob[k] = obs.glob
        self.occ[k] = obs.occ
        self.infl[k] = obs.infl
        self.l[k] = obs.l
        self.u[k] = obs.u
        self.Ctil[k] = obs.Ctil
        self.Crev[k] = obs.Crev
        self.feasible[k] = obs.feasible
        self.a[k] = a
        self.logp[k] = logp
        self.value[k] = value
        self.reward[k] = reward
        self.done[k] = done
        if self.n_cost:
            self.cost[k] = cost
            self.cvalue[k] = cvalue

    def _gae(self, rew, val, last_value, gamma, lam):
        adv = np.zeros(self.S, np.float32)
        last = 0.0
        for t in reversed(range(self.S)):
            nonterm = 1.0 - float(self.done[t])
            nextv = last_value if t == self.S - 1 else float(val[t + 1])
            delta = float(rew[t]) + gamma * nextv * nonterm - float(val[t])
            last = delta + gamma * lam * nonterm * last
            adv[t] = last
        return adv

    def compute_gae(self, last_value: float, gamma: float, lam: float, last_cvalue=None) -> None:
        """GAE with done[t] = episode terminated after step t (true terminal, V = 0; Alg. 2); the same recursion
        for every cost channel of PPO-Lagrangian."""
        self.adv[:] = self._gae(self.reward, self.value, last_value, gamma, lam)
        self.ret[:] = self.adv + self.value
        for j in range(self.n_cost):
            self.cadv[:, j] = self._gae(self.cost[:, j], self.cvalue[:, j], float(last_cvalue[j]), gamma, lam)
            self.cret[:, j] = self.cadv[:, j] + self.cvalue[:, j]


class RolloutBuffer:
    def __init__(self, S: int, Ns: list[int], n_cost: int = 0):
        self.S = S
        self.n_cost = n_cost
        self.envs = [EnvBuffer(S, N, n_cost) for N in Ns]

    def flatten(self, device="cpu", safety: bool = True) -> dict:
        """Concatenate all envs, padded to N_max. Returns a dict of tensors on ``device``."""
        Nmax = max(b.N for b in self.envs)
        B = self.S * len(self.envs)

        def pad(name, dtype, fill=0):
            arrs = []
            for b in self.envs:
                x = getattr(b, name)
                if x.ndim >= 2 and x.shape[1] == b.N and name not in ("glob",):
                    w = [(0, 0)] * x.ndim
                    w[1] = (0, Nmax - b.N)
                    x = np.pad(x, w, constant_values=fill)
                arrs.append(x)
            return torch.as_tensor(np.concatenate(arrs, 0), dtype=dtype, device=device)

        out = {
            "tokens": pad("tokens", torch.float32),
            "glob": torch.as_tensor(np.concatenate([b.glob for b in self.envs]), device=device),
            "occ": pad("occ", torch.bool),
            "infl": pad("infl", torch.bool),
            "l": pad("l", torch.float64),
            "u": pad("u", torch.float64),
            "a": pad("a", torch.float32),
            "Ctil": torch.as_tensor(np.concatenate([b.Ctil for b in self.envs]), dtype=torch.float64, device=device),
            "Crev": torch.as_tensor(np.concatenate([b.Crev for b in self.envs]), dtype=torch.float64, device=device),
            "feasible": torch.as_tensor(np.concatenate([b.feasible for b in self.envs]), device=device),
            "logp": torch.as_tensor(np.concatenate([b.logp for b in self.envs]), device=device),
            "value": torch.as_tensor(np.concatenate([b.value for b in self.envs]), device=device),
            "adv": torch.as_tensor(np.concatenate([b.adv for b in self.envs]), device=device),
            "ret": torch.as_tensor(np.concatenate([b.ret for b in self.envs]), device=device),
        }
        if self.n_cost:
            for k in ("cost", "cadv", "cret"):
                out[k] = torch.as_tensor(np.concatenate([getattr(b, k) for b in self.envs]), device=device)
        # actor-loss weight w (DD21): fallback steps masked for safety-layer methods; penalty methods have no
        # fallback, so every transition enters the actor loss. 'feasible' stays the A5 flag (rows the projection
        # of the beta regulariser may be applied to).
        out["w"] = out["feasible"] if safety else torch.ones_like(out["feasible"])
        assert out["tokens"].shape[0] == B
        return out
