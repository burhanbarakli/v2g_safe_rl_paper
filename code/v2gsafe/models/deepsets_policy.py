"""DeepSets baseline (DD33(4), method ``deepsets_safe``): the attention encoder replaced by a per-token MLP and a
mean-pooled context, no attention.

Actor: phi_i = MLP_phi([x_i; g]) (shared per-token MLP on the charger token and the global token), context
c = mean_{j occupied} phi_j (zero if no EV is connected), head_i = MLP_rho([phi_i; c]) -> (mu_i, log_std_i).
Critic: MLP_v([mean_j phi_j; MLP_g(g)]) -> value(s). Permutation-equivariant (actor) / invariant (critic) and
independent of N and of padding, like the attention policy (Lemma N1), but every charger sees the others only through
the mean context.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..constants import F_C, F_G
from .attention_policy import LOGSTD_MAX, LOGSTD_MIN


class _Phi(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(F_C + F_G, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, tokens, glob, occ):
        B, N, _ = tokens.shape
        x = torch.cat([tokens, glob[:, None, :].expand(B, N, glob.shape[-1])], -1)
        phi = self.net(x)
        m = occ.to(phi.dtype)[..., None]
        ctx = (phi * m).sum(1) / m.sum(1).clamp_min(1.0)
        return phi, ctx


class DeepSetsActor(nn.Module):
    def __init__(self, d_model: int = 128, head_hidden: int = 64):
        super().__init__()
        self.phi = _Phi(d_model)
        self.head = nn.Sequential(nn.Linear(2 * d_model, head_hidden), nn.GELU(), nn.Linear(head_hidden, 2))
        last = self.head[-1]
        with torch.no_grad():
            last.weight.mul_(0.01)
            last.bias.zero_()
            last.bias[1] = -0.5

    def forward(self, tokens, glob, occ):
        phi, ctx = self.phi(tokens, glob, occ)
        out = self.head(torch.cat([phi, ctx[:, None, :].expand_as(phi)], -1))
        return out[..., 0], torch.clamp(out[..., 1], LOGSTD_MIN, LOGSTD_MAX)


class DeepSetsCritic(nn.Module):
    def __init__(self, d_model: int = 128, head_hidden: int = 64, n_out: int = 1):
        super().__init__()
        self.n_out = n_out
        self.phi = _Phi(d_model)
        self.g = nn.Sequential(nn.Linear(F_G, d_model), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2 * d_model, head_hidden), nn.GELU(), nn.Linear(head_hidden, n_out))

    def forward(self, tokens, glob, occ):
        _, ctx = self.phi(tokens, glob, occ)
        out = self.head(torch.cat([ctx, self.g(glob)], -1))
        return out[:, 0] if self.n_out == 1 else out
