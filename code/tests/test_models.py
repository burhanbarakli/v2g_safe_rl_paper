"""Attention actor/critic and MLP baseline (formulation.md Sec. 5, Lemma N1)."""
from __future__ import annotations

import pytest
import torch

from v2gsafe.constants import F_C, F_G
from v2gsafe.models.attention_policy import AttentionActor, AttentionCritic, count_params
from v2gsafe.models.mlp_policy import MLPActor, MLPCritic


def _obs(B, N, p_occ=0.7, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    tok = torch.randn(B, N, F_C, generator=g, dtype=dtype)
    occ = torch.rand(B, N, generator=g) < p_occ
    tok = tok * occ[..., None]
    glob = torch.randn(B, F_G, generator=g, dtype=dtype)
    return tok, glob, occ


def test_parameter_counts_match_formulation():
    assert count_params(AttentionActor()) == 610_370
    assert count_params(AttentionCritic()) == 676_481
    assert count_params(MLPActor()) == 792_976
    small = AttentionActor(64, 2, 4, 32), AttentionCritic(64, 2, 4, 32)   # small variant (I-21)
    assert sum(count_params(m) for m in small) == 228_067
    for N in (5, 200):  # independent of N (Lemma N1(v)) - parameters do not depend on the input
        a = AttentionActor()
        tok, glob, occ = _obs(2, N)
        mu, ls = a(tok, glob, occ)
        assert mu.shape == (2, N)


@pytest.mark.parametrize("N", [5, 10, 50, 100, 200])
def test_output_shapes(N):
    a, c = AttentionActor(), AttentionCritic()
    tok, glob, occ = _obs(3, N)
    mu, ls = a(tok, glob, occ)
    v = c(tok, glob, occ)
    assert mu.shape == (3, N) and ls.shape == (3, N) and v.shape == (3,)
    assert torch.all(ls >= -5) and torch.all(ls <= 0.5)
    ma, mc = MLPActor(), MLPCritic()
    mu2, ls2 = ma(tok, glob, occ)
    assert mu2.shape == (3, N) and mc(tok, glob, occ).shape == (3,)


def test_initialisation():
    a = AttentionActor()
    tok, glob, occ = _obs(4, 20)
    mu, ls = a(tok, glob, occ)
    assert torch.allclose(ls, torch.full_like(ls, -0.5), atol=0.05)   # std ~ 0.6 -> 6.7 kW
    assert mu.abs().max() < 0.1


def test_permutation_equivariance_and_invariance():
    torch.manual_seed(0)
    a, c = AttentionActor().double(), AttentionCritic().double()
    tok, glob, occ = _obs(2, 30, dtype=torch.float64)
    perm = torch.randperm(30)
    mu, ls = a(tok, glob, occ)
    mup, lsp = a(tok[:, perm], glob, occ[:, perm])
    torch.testing.assert_close(mup, mu[:, perm], atol=1e-10, rtol=0)
    torch.testing.assert_close(lsp, ls[:, perm], atol=1e-10, rtol=0)
    torch.testing.assert_close(c(tok[:, perm], glob, occ[:, perm]), c(tok, glob, occ), atol=1e-10, rtol=0)


@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-10), (torch.float32, 1e-5)])
def test_padding_invariance(dtype, tol):
    """Outputs of unmasked tokens and the value do not depend on masked tokens or their number (Lemma N1(iv))."""
    torch.manual_seed(1)
    a, c = AttentionActor().to(dtype), AttentionCritic().to(dtype)
    tok, glob, occ = _obs(4, 12, dtype=dtype)
    mu, _ = a(tok, glob, occ)
    v = c(tok, glob, occ)
    # 1) garbage in masked tokens
    tok2 = tok.clone()
    tok2[~occ] = 100 * torch.randn(int((~occ).sum()), F_C, dtype=dtype)
    mu2, _ = a(tok2, glob, occ)
    torch.testing.assert_close(mu2[occ], mu[occ], atol=tol, rtol=0)
    torch.testing.assert_close(c(tok2, glob, occ), v, atol=tol, rtol=0)
    # 2) extra padded tokens (N 12 -> 40)
    pad = 28
    tok3 = torch.cat([tok2, 50 * torch.randn(4, pad, F_C, dtype=dtype)], 1)
    occ3 = torch.cat([occ, torch.zeros(4, pad, dtype=torch.bool)], 1)
    mu3, _ = a(tok3, glob, occ3)
    torch.testing.assert_close(mu3[:, :12][occ], mu[occ], atol=tol, rtol=0)
    torch.testing.assert_close(c(tok3, glob, occ3), v, atol=tol, rtol=0)
    # 3) only the occupied tokens (station of size n)
    for b in range(4):
        idx = torch.nonzero(occ[b]).flatten()
        mu4, _ = a(tok[b:b + 1, idx], glob[b:b + 1], occ[b:b + 1, idx])
        torch.testing.assert_close(mu4[0], mu[b, idx], atol=tol, rtol=0)


def test_all_chargers_empty_is_well_defined():
    a, c = AttentionActor(), AttentionCritic()
    tok = torch.zeros(2, 7, F_C)
    occ = torch.zeros(2, 7, dtype=torch.bool)
    mu, ls = a(tok, torch.randn(2, F_G), occ)
    assert torch.isfinite(mu).all() and torch.isfinite(c(tok, torch.randn(2, F_G), occ)).all()


def test_mlp_zero_padding_and_limit():
    m = MLPActor(n_max=10)
    tok, glob, occ = _obs(2, 6)
    mu, _ = m(tok, glob, occ)
    tok2 = tok.clone()
    tok2[~occ] = 5.0                       # masked content is zeroed internally
    mu2, _ = m(tok2, glob, occ)
    torch.testing.assert_close(mu, mu2)
    with pytest.raises(ValueError):
        m(*_obs(1, 11))
