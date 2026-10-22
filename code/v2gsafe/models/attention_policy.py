"""Attention actor and critic over charger tokens (formulation.md Sec. 5, Lemma N1).

* Embeddings e_i = W_c xi_i + b_c + v_c (chargers), e_0 = W_g xi_0 + b_g + v_g (global); v_c, v_g learned type
  embeddings; NO positional encoding.
* Sequence [e_0; e_1..e_N]; key-padding mask: charger key i masked iff o_i = 0; the global key is never masked.
* n_L pre-LayerNorm layers H <- H + MHA(LN(H)), H <- H + FFN(LN(H)), FFN = W2 GELU(W1 h + b1) + b2 (width 4 d_m),
  final LN. Masked logits are -inf (boolean mask in scaled_dot_product_attention).
* Actor head (shared over chargers) d_m -> 64 -> 2 (mu, ell), GELU; log sigma = clamp(ell, -5, 0.5);
  init: last-layer weights x0.01, mean bias 0, log-std bias -0.5.
* Critic: separate encoder, attention pooling with one learned query over all unmasked tokens (global token
  included), then d_m -> 64 -> 1.

Parameter counts at the defaults (d_m 128, n_L 3, n_H 4): actor 610,370, critic 676,481 (formulation.md Sec. 5).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import F_C, F_G

LOGSTD_MIN, LOGSTD_MAX = -5.0, 0.5


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention (packed in-projection with bias, output projection)."""

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        assert d % n_heads == 0
        self.d, self.h = d, n_heads
        self.in_proj = nn.Linear(d, 3 * d)
        self.out_proj = nn.Linear(d, d)

    def forward(self, xq: torch.Tensor, xkv: torch.Tensor, key_valid: torch.Tensor) -> torch.Tensor:
        """xq: (B, Lq, d); xkv: (B, Lk, d); key_valid: (B, Lk) bool (True = attend)."""
        B, Lq, d = xq.shape
        Lk = xkv.shape[1]
        w, b = self.in_proj.weight, self.in_proj.bias
        q = F.linear(xq, w[:d], b[:d])
        k = F.linear(xkv, w[d:2 * d], b[d:2 * d])
        v = F.linear(xkv, w[2 * d:], b[2 * d:])
        hd = d // self.h
        q = q.view(B, Lq, self.h, hd).transpose(1, 2)
        k = k.view(B, Lk, self.h, hd).transpose(1, 2)
        v = v.view(B, Lk, self.h, hd).transpose(1, 2)
        mask = key_valid[:, None, None, :]           # broadcast over heads and queries
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        o = o.transpose(1, 2).reshape(B, Lq, d)
        return self.out_proj(o)


class PreLNBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, ffn_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = MultiHeadAttention(d, n_heads)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))

    def forward(self, h: torch.Tensor, key_valid: torch.Tensor) -> torch.Tensor:
        x = self.ln1(h)
        h = h + self.attn(x, x, key_valid)
        return h + self.ffn(self.ln2(h))


class TokenEncoder(nn.Module):
    """Set encoder over [global; chargers] with type embeddings and key-padding mask (no positions)."""

    def __init__(self, d_model: int = 128, n_layers: int = 3, n_heads: int = 4, f_c: int = F_C, f_g: int = F_G):
        super().__init__()
        self.emb_c = nn.Linear(f_c, d_model)
        self.emb_g = nn.Linear(f_g, d_model)
        self.type_c = nn.Parameter(torch.zeros(d_model))
        self.type_g = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.type_c, std=0.02)
        nn.init.normal_(self.type_g, std=0.02)
        self.blocks = nn.ModuleList([PreLNBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, glob: torch.Tensor, occ: torch.Tensor):
        """tokens (B, N, F_c), glob (B, F_g), occ (B, N) bool -> H (B, N+1, d), key_valid (B, N+1)."""
        e_c = self.emb_c(tokens) + self.type_c
        e_g = (self.emb_g(glob) + self.type_g)[:, None, :]
        h = torch.cat([e_g, e_c], 1)
        valid = torch.cat([torch.ones_like(occ[:, :1], dtype=torch.bool), occ.to(torch.bool)], 1)
        for blk in self.blocks:
            h = blk(h, valid)
        return self.ln_f(h), valid


class AttentionActor(nn.Module):
    """Gaussian policy over raw per-charger actions a_i (formulation.md Sec. 5)."""

    def __init__(self, d_model: int = 128, n_layers: int = 3, n_heads: int = 4, head_hidden: int = 64):
        super().__init__()
        self.encoder = TokenEncoder(d_model, n_layers, n_heads)
        self.head = nn.Sequential(nn.Linear(d_model, head_hidden), nn.GELU(), nn.Linear(head_hidden, 2))
        last = self.head[-1]
        with torch.no_grad():
            last.weight.mul_(0.01)
            last.bias.zero_()
            last.bias[1] = -0.5

    def forward(self, tokens, glob, occ):
        """Returns mu (B, N), log_std (B, N) in normalised action units (x = P_ref a)."""
        h, _ = self.encoder(tokens, glob, occ)
        out = self.head(h[:, 1:, :])
        mu = out[..., 0]
        log_std = torch.clamp(out[..., 1], LOGSTD_MIN, LOGSTD_MAX)
        return mu, log_std


class AttentionCritic(nn.Module):
    """Value function with attention pooling (pooling by multi-head attention, one learned query)."""

    def __init__(self, d_model: int = 128, n_layers: int = 3, n_heads: int = 4, head_hidden: int = 64,
                 n_out: int = 1):
        super().__init__()
        self.n_out = n_out                  # 1: value; 3: value + two cost values (PPO-Lagrangian, DD33(3))
        self.encoder = TokenEncoder(d_model, n_layers, n_heads)
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.query, std=0.02)
        self.pool = MultiHeadAttention(d_model, n_heads)
        self.head = nn.Sequential(nn.Linear(d_model, head_hidden), nn.GELU(), nn.Linear(head_hidden, n_out))

    def forward(self, tokens, glob, occ):
        h, valid = self.encoder(tokens, glob, occ)
        q = self.query.expand(h.shape[0], 1, -1)
        pooled = self.pool(q, h, valid)[:, 0, :]
        out = self.head(pooled)
        return out[:, 0] if self.n_out == 1 else out


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
