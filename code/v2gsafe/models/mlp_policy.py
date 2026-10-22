"""MLP baseline (algorithm.md Alg. 5 rows 1-2): actor and critic with two hidden layers of 256 units (tanh) on
the concatenation of the charger tokens in charger order, zero-padded to N_max = 200 slots, plus the global
token. The actor outputs 200 means and 200 log-stds; slots >= N are ignored. Runs at any N <= N_max with the
same weights; not permutation-equivariant. Parameters (actor): 614,440 at N_max = 20, hidden 640 (DD44; capacity matched to the attention actor).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..constants import F_C, F_G
from .attention_policy import LOGSTD_MAX, LOGSTD_MIN


def _flatten_padded(tokens, glob, occ, n_max: int) -> torch.Tensor:
    B, N, Fc = tokens.shape
    if N > n_max:
        raise ValueError(f"MLP policy supports at most N_max = {n_max} chargers, got {N}")
    x = tokens * occ.to(tokens.dtype)[..., None]          # empty chargers / padding contribute zeros
    if N < n_max:
        x = torch.cat([x, x.new_zeros(B, n_max - N, Fc)], 1)
    return torch.cat([x.reshape(B, n_max * Fc), glob], 1)


class MLPActor(nn.Module):
    def __init__(self, n_max: int = 200, hidden: int = 256):
        super().__init__()
        self.n_max = n_max
        self.net = nn.Sequential(nn.Linear(n_max * F_C + F_G, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 2 * n_max))
        last = self.net[-1]
        with torch.no_grad():
            last.weight.mul_(0.01)
            last.bias.zero_()
            last.bias[n_max:] = -0.5

    def forward(self, tokens, glob, occ):
        N = tokens.shape[1]
        out = self.net(_flatten_padded(tokens, glob, occ, self.n_max))
        mu = out[:, :N]
        log_std = torch.clamp(out[:, self.n_max:self.n_max + N], LOGSTD_MIN, LOGSTD_MAX)
        return mu, log_std


class MLPCritic(nn.Module):
    def __init__(self, n_max: int = 200, hidden: int = 256, n_out: int = 1):
        super().__init__()
        self.n_max = n_max
        self.n_out = n_out
        self.net = nn.Sequential(nn.Linear(n_max * F_C + F_G, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, n_out))

    def forward(self, tokens, glob, occ):
        out = self.net(_flatten_padded(tokens, glob, occ, self.n_max))
        return out[:, 0] if self.n_out == 1 else out
