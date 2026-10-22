"""Robust Euclidean projection onto K(t) = {l <= p <= u, -C_rev <= 1'p <= C_tilde} (safety layer).

safety_layer.md Sec. 4 (Proposition 1, eqs. (S13)-(S16)) and algorithm.md Alg. 1.2.

* NumPy: :func:`project` (sorted breakpoints, batched, masked, M1-robust, post-condition + safe bisection),
  :func:`project_bisect` (single-row reference), :func:`jacobian` (S16), :func:`is_feasible` (A5), :func:`sanitize`.
  These reproduce ``method/projection_ref.py`` and are tested against it.
* PyTorch: :func:`project_batched` (float64, padded to N_max with an occupancy mask, no gradient),
  :func:`jacobian_batched` (S16).

Counters of non-finite replacements and post-condition hand-overs to the safe bisection are returned in
:class:`GuardCounters` (safety_layer.md Sec. 7.5 item 9); both should be 0 in normal operation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import EPS_NUM

try:  # torch is optional for the NumPy path
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class GuardCounters:
    nonfinite: int = 0      # non-finite raw actions replaced by 0
    bisect_handover: int = 0  # rows handed to the safe bisection by the post-condition

    def add(self, other: "GuardCounters") -> None:
        self.nonfinite += other.nonfinite
        self.bisect_handover += other.bisect_handover


# ============================================================================================ NumPy
def sanitize(x):
    """Replace non-finite raw actions by 0 (safety_layer.md Sec. 1). Returns (x, n_bad)."""
    x = np.asarray(x, float)
    bad = ~np.isfinite(x)
    return np.where(bad, 0.0, x), int(bad.sum())


def is_feasible(l, u, C, Crev, mask=None):
    """A5 per row (Proposition 1(i)): sum l <= C_tilde, sum u >= -C_rev, C_tilde >= -C_rev."""
    l = np.atleast_2d(np.asarray(l, float))
    u = np.atleast_2d(np.asarray(u, float))
    if mask is not None:
        mask = np.atleast_2d(mask)
        l = np.where(mask, l, 0.0)
        u = np.where(mask, u, 0.0)
    C = np.atleast_1d(np.asarray(C, float))
    Crev = np.atleast_1d(np.asarray(Crev, float))
    return (l.sum(1) <= C) & (u.sum(1) >= -Crev) & (C >= -Crev)


def _bisect_safe(a, l, u, C, Crev, upper, iters=200):
    """Feasible-end bisection for rows with an active aggregate constraint (safety_layer.md Sec. 4.2)."""
    s = lambda lam: np.clip(a - lam[:, None], l, u).sum(1)  # noqa: E731
    pad = 1.0 + 1e-6 * np.abs(a).max(1)
    lo = np.where(upper, 0.0, (a - u).min(1) - pad)
    hi = np.where(upper, (a - l).max(1) + pad, 0.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        sm = s(mid)
        ok_hi = upper & (sm <= C)
        ok_lo = (~upper) & (sm >= -Crev)
        hi = np.where(ok_hi | ((~upper) & ~ok_lo), mid, hi)
        lo = np.where(ok_lo | (upper & ~ok_hi), mid, lo)
    lam = np.where(upper, hi, lo)
    return np.clip(a - lam[:, None], l, u), lam


def project(a, l, u, C, Crev, mask=None, check=True, counters: GuardCounters | None = None):
    """Exact projection (S13)-(S15) by sorted breakpoints, batched over rows (safety_layer.md Sec. 4.2).

    a, l, u: (B, N) kW; C (= C_tilde), Crev: (B,); mask (B, N) True = occupied (masked entries are forced to
    a = l = u = 0 and returned as 0). Returns p (B, N), lam (B,), case (B,) in {0, +1 upper, -1 lower}.
    Precondition A5 (asserted if ``check``)."""
    a = np.atleast_2d(np.asarray(a, float)).copy()
    l = np.atleast_2d(np.asarray(l, float)).copy()
    u = np.atleast_2d(np.asarray(u, float)).copy()
    C = np.atleast_1d(np.asarray(C, float))
    Crev = np.atleast_1d(np.asarray(Crev, float))
    a, nbad = sanitize(a)
    if mask is not None:
        mask = np.atleast_2d(np.asarray(mask, bool))
        a = np.where(mask, a, 0.0)
        l = np.where(mask, l, 0.0)
        u = np.where(mask, u, 0.0)
    if check:
        assert np.all(l <= u) and np.all(is_feasible(l, u, C, Crev)), "projection called on an infeasible row"
    Bn, N = a.shape
    p0 = np.clip(a, l, u)
    s0 = p0.sum(1)
    case = np.where(s0 > C, 1, np.where(s0 < -Crev, -1, 0))
    target = np.where(case == 1, C, -Crev)
    beta = np.concatenate([a - u, a - l], 1)
    ev = np.concatenate([np.ones((Bn, N)), -np.ones((Bn, N))], 1)
    order = np.argsort(beta, 1, kind="stable")
    beta = np.take_along_axis(beta, order, 1)
    ev = np.take_along_axis(ev, order, 1)
    nfree = np.cumsum(ev, 1)
    dbeta = np.diff(beta, axis=1)
    S = np.empty_like(beta)
    S[:, 0] = u.sum(1)
    S[:, 1:] = u.sum(1, keepdims=True) - np.cumsum(nfree[:, :-1] * dbeta, 1)
    S[:, -1] = l.sum(1)                                   # exact end value (review M1)
    hit = S <= target[:, None]
    has = hit.any(1)
    j = np.where(has, np.argmax(hit, 1), 2 * N - 1)
    jm = np.maximum(j - 1, 0)
    b1 = np.take_along_axis(beta, jm[:, None], 1)[:, 0]
    b2 = np.take_along_axis(beta, j[:, None], 1)[:, 0]
    S1 = np.take_along_axis(S, jm[:, None], 1)[:, 0]
    S2 = np.take_along_axis(S, j[:, None], 1)[:, 0]
    den = S1 - S2
    lam_seg = np.where(den > 0, b1 + (S1 - target) * (b2 - b1) / np.where(den > 0, den, 1.0), b2)
    lam = np.where(j == 0, b2, lam_seg)
    lam = np.where(has, lam, beta[:, -1] + 1.0 + 1e-6 * np.abs(a).max(1))
    lam = np.where(case == 0, 0.0, lam)
    p = np.clip(a - lam[:, None], l, u)
    sp = p.sum(1)
    bad = (case != 0) & ((sp > C + EPS_NUM) | (sp < -Crev - EPS_NUM))
    if bad.any():
        pb, lb = _bisect_safe(a[bad], l[bad], u[bad], C[bad], Crev[bad], case[bad] == 1)
        p[bad], lam[bad] = pb, lb
    if mask is not None:
        p = np.where(mask, p, 0.0)
    if counters is not None:
        counters.nonfinite += nbad
        counters.bisect_handover += int(bad.sum())
    return p, lam, case


def project_bisect(a, l, u, C, Crev, iters=200):
    """Single-row reference: safe bisection + checked exact polish (safety_layer.md Sec. 4.2)."""
    a, l, u = (np.asarray(v, float) for v in (a, l, u))
    s = lambda lam: np.clip(a - lam, l, u).sum()  # noqa: E731
    if -Crev <= s(0.0) <= C:
        return np.clip(a, l, u), 0.0
    upper = s(0.0) > C
    target = C if upper else -Crev
    pb, lb = _bisect_safe(a[None], l[None], u[None], np.array([C]), np.array([Crev]), np.array([upper]), iters)
    lam = lb[0]
    F = (a - lam > l) & (a - lam < u)
    if F.any():
        fixed = np.clip(a - lam, l, u)[~F].sum()
        lam_p = (a[F].sum() + fixed - target) / F.sum()
        pp = np.clip(a - lam_p, l, u)
        if -Crev - EPS_NUM <= pp.sum() <= C + EPS_NUM:
            return pp, lam_p
    return pb[0], lam


def jacobian(a, l, u, lam, case, tol=1e-12):
    """dp/dx on a differentiable region, eq. (S16). Single row."""
    a, l, u = (np.asarray(v, float) for v in (a, l, u))
    F = ((a - lam) > l + tol) & ((a - lam) < u - tol)
    D = np.diag(F.astype(float))
    if case == 0 or F.sum() == 0:
        return D if case == 0 else np.zeros_like(D)
    d = F.astype(float)
    return D - np.outer(d, d) / d.sum()


# ============================================================================================ PyTorch
def _bisect_safe_torch(x, l, u, C, Crev, up, iters=200):
    """PyTorch port of the feasible-end bisection (algorithm.md Alg. 1.2, ``bisect_safe``)."""
    pad = 1.0 + 1e-6 * x.abs().amax(-1)
    zero = torch.zeros_like(C)
    lo = torch.where(up, zero, (x - u).amin(-1) - pad)
    hi = torch.where(up, (x - l).amax(-1) + pad, zero)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s = torch.maximum(torch.minimum(x - mid[:, None], u), l).sum(-1)
        ok_hi = up & (s <= C)
        ok_lo = (~up) & (s >= -Crev)
        hi = torch.where(ok_hi | ((~up) & ~ok_lo), mid, hi)
        lo = torch.where(ok_lo | (up & ~ok_hi), mid, lo)
    lam = torch.where(up, hi, lo)
    return torch.maximum(torch.minimum(x - lam[:, None], u), l), lam


def project_batched(x, l, u, C, Crev, occ, counters: GuardCounters | None = None):
    """Batched robust projection in PyTorch (algorithm.md Alg. 1.2).

    x, l, u: (B, N) [kW]; C (= C_tilde), Crev: (B,); occ: (B, N) bool. Rows must be A5-feasible.
    Computation in float64 under ``torch.no_grad()`` (DD11: no gradient through the projection).
    Returns (p, lam, case) with case in {0, 1, -1}."""
    with torch.no_grad():
        dtype = torch.float64
        x = x.to(dtype)
        l = l.to(dtype)
        u = u.to(dtype)
        C = C.to(dtype)
        Crev = Crev.to(dtype)
        occ = occ.to(torch.bool)
        z = torch.zeros_like(x)
        finite = torch.isfinite(x)
        nbad = int((~finite & occ).sum().item()) if counters is not None else 0
        x = torch.where(finite, x, z)
        x = torch.where(occ, x, z)
        l = torch.where(occ, l, z)
        u = torch.where(occ, u, z)
        n = x.shape[-1]
        p0 = torch.maximum(torch.minimum(x, u), l)
        s0 = p0.sum(-1)
        up = s0 > C
        lo = s0 < -Crev
        tgt = torch.where(up, C, -Crev)
        lam_all = torch.cat([x - u, x - l], -1)
        ev = torch.cat([torch.ones_like(x), -torch.ones_like(x)], -1)
        lam_all, idx = torch.sort(lam_all, dim=-1, stable=True)
        ev = torch.gather(ev, -1, idx)
        nfree = torch.cumsum(ev, -1)
        su = u.sum(-1, keepdim=True)
        S = torch.cat([su, su - torch.cumsum(nfree[:, :-1] * (lam_all[:, 1:] - lam_all[:, :-1]), -1)], -1)
        S[:, -1] = l.sum(-1)                                   # exact end value (review M1)
        hit = S <= tgt[:, None]
        has = hit.any(-1)
        first = torch.argmax(hit.to(torch.int64), -1)          # first maximal index
        j = torch.where(has, first, torch.full_like(first, 2 * n - 1))
        jm = torch.clamp(j - 1, min=0)
        b1 = lam_all.gather(-1, jm[:, None])[:, 0]
        b2 = lam_all.gather(-1, j[:, None])[:, 0]
        S1 = S.gather(-1, jm[:, None])[:, 0]
        S2 = S.gather(-1, j[:, None])[:, 0]
        den = S1 - S2
        lam = torch.where(den > 0, b1 + (S1 - tgt) * (b2 - b1) / den.clamp_min(1e-300), b2)
        lam = torch.where(j == 0, b2, lam)
        pad = 1.0 + 1e-6 * x.abs().amax(-1)
        lam = torch.where(has, lam, lam_all[:, -1] + pad)
        lam = torch.where(up | lo, lam, torch.zeros_like(lam))
        p = torch.maximum(torch.minimum(x - lam[:, None], u), l)
        sp = p.sum(-1)
        bad = (up | lo) & ((sp > C + EPS_NUM) | (sp < -Crev - EPS_NUM))
        nb = int(bad.sum().item())
        if nb > 0:
            pb, lb = _bisect_safe_torch(x[bad], l[bad], u[bad], C[bad], Crev[bad], up[bad])
            p[bad] = pb
            lam[bad] = lb
        if counters is not None:
            counters.nonfinite += nbad
            counters.bisect_handover += nb
        case = torch.where(up, torch.ones_like(j), torch.where(lo, -torch.ones_like(j), torch.zeros_like(j)))
        return torch.where(occ, p, z), lam, case


def jacobian_batched(x, l, u, lam, case, occ, tol=1e-12):
    """Batched Jacobian (S16): ``D - outer(d, d)/sum(d)`` when the aggregate constraint is active.

    Returns (B, N, N) float64. Used only by optional analytic-gradient variants (DD11)."""
    d = (occ & ((x - lam[:, None]) > l + tol) & ((x - lam[:, None]) < u - tol)).to(torch.float64)
    D = torch.diag_embed(d)
    active = (case != 0)
    cnt = d.sum(-1)
    outer = d[:, :, None] * d[:, None, :] / cnt.clamp_min(1.0)[:, None, None]
    J = torch.where(active[:, None, None], D - outer, D)
    J = torch.where((active & (cnt == 0))[:, None, None], torch.zeros_like(J), J)
    return J
