# VERBATIM COPY of method/projection_ref.py (A2, 2026-10-22), kept as the test oracle. Do not edit.
"""projection_ref.py -- NumPy reference implementation of the safety layer (safety_layer.md).

This file is the executable specification for the implementation agent. A PyTorch port must reproduce
these functions to 1e-9 (float64). Notation follows nomenclature.md.

Contents
  phi_ev2gym(...)            one-step maximum-charging map phi_i(E) of EV2Gym (closed form, eq. (S7); requires P_ev <= P_cs)
  thresholds(...)            backward-reachable thresholds B_i(k), k = 0..K   (eq. (S8), certified in the sense of (S10))
  bounds(...)                p_i^min, u_i, l_i and the individual-feasibility flag  (eqs. (S3), (S4), (S11))
  is_feasible(...)           A5 check incl. C_tilde >= -C_rev (Proposition 1(i))
  sanitize(...)              NaN/Inf guard for raw actions
  project(...)               exact projection by sorted breakpoints, batched, with M1-robust search and
                             aggregate post-condition (Sec. 4.2)
  project_bisect(...)        bisection (feasible-end) + checked polish (reference, 1 row)
  jacobian(...)              dp/da on a differentiable region (eq. (S16))
  fallback(...)              DD8 least-laxity-first fallback rule FB-0/1/2 (Sec. 7.2)
  actuation(...)             p (kW) -> EV2Gym action with upward rounding (Sec. 6.3, Lemma E3)
"""
import math
import numpy as np

# ----------------------------------------------------------------------------------------------
# EV model (EV2Gym instantiation: g(p) = p, i.e. eta^c = eta^d = 1 in the energy map, DD1)
# ----------------------------------------------------------------------------------------------


def phi_ev2gym(E, C, Pnom, s_tr, m, dt_h):
    """phi_i(E): energy (kWh) after one step at maximum charging power from energy E (kWh).
    C: capacity (kWh); Pnom: derated nominal max power zeta^c*min(P_cs, P_ev) (kW); s_tr: transition SoC;
    m: transition_soc_multiplier; dt_h: step (h). Vectorised over E (and broadcastable parameters)."""
    E = np.asarray(E, dtype=float)
    s = E / C
    d = Pnom * dt_h / C
    lin = s + d
    if np.all(np.asarray(s_tr) >= 1.0):
        return C * np.minimum(1.0, lin)
    s_tr = np.asarray(s_tr, dtype=float)
    w = np.maximum(1.0 - s_tr, 1e-12)
    mid = 1.0 - w * np.exp(-m * (s + d - s_tr) / w)
    up = 1.0 - (1.0 - s) * np.exp(-m * d / w)
    tap = np.where(s + d <= s_tr, lin, np.where(s < s_tr, mid, up))
    out = np.minimum(np.minimum(tap, lin), 1.0)
    out = np.where(s_tr >= 1.0, np.minimum(1.0, lin), out)
    return C * out


def thresholds(E_req, K, phi, E_min, E_cap, tol=1e-10):
    """B[k] = min{E in [E_min, E_cap] : phi^k(E) >= E_req}, k = 0..K (eq. (S8)).
    Computed backwards: B[0] = E_req, B[k] = min{E : phi(E) >= B[k-1]} by bisection.
    Returned values are rounded towards the feasible side (B_hat >= B), so the bound is conservative."""
    B = np.empty(K + 1)
    B[0] = E_req
    for k in range(1, K + 1):
        target = B[k - 1]
        if phi(E_min) >= target:
            B[k] = E_min
            continue
        lo, hi = E_min, B[k - 1]  # phi(E) >= E, so E = B[k-1] satisfies phi(E) >= B[k-1]
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if phi(mid) >= target:
                hi = mid
            else:
                lo = mid
        B[k] = hi
    return B


def cap_request(E_des, E_arr, K, phi, E_min, E_cap):
    """DD2 arrival cap, certified with the same threshold routine as the safety layer, so that A4
    (E_arr >= B_hat[K]) holds by construction for the returned table. K = tau at arrival.
    Candidate y = Psi_K(E_arr) - 1e-6 (forward reach); decreased geometrically until certified.
    Returns (E_req, B_table)."""
    B = thresholds(E_des, K, phi, E_min, E_cap)
    if E_arr >= B[K]:
        return E_des, B
    y, step = min(E_des, reach(E_arr, K, phi) - 1e-6), 1e-6
    while True:
        y = max(y, min(E_arr, E_des))       # y <= E_arr is always certified (B_hat[K](y) <= y)
        B = thresholds(y, K, phi, E_min, E_cap)
        if E_arr >= B[K]:
            return y, B
        y -= step
        step *= 10.0


def reach(E, k, phi):
    """Psi_k(E) = phi^k(E) = E + R(k, E)."""
    for _ in range(k):
        E = phi(E)
    return E


def bounds(E, tau, B, E_min, E_cap, Pdis, phi, dt_h):
    """Per-EV bounds at a decision step (EV2Gym instantiation, g(p) = p).
    tau: remaining actions incl. the current one (>= 1); B: threshold table of this EV.
    Returns (pmin, u, l, feasible)."""
    pmin = -min(Pdis, max(0.0, E - E_min) / dt_h)
    u = (phi(E) - E) / dt_h
    feasible = E >= B[tau] - EPS_NUM
    if not feasible:
        return pmin, u, u, False
    l = max(pmin, (B[tau - 1] - E) / dt_h)
    l = min(l, u)  # only float noise can make l > u here (feasible => l <= u, Lemma 1)
    return pmin, u, l, True


# ----------------------------------------------------------------------------------------------
# Projection onto {l <= p <= u, -Crev <= 1'p <= C}
# ----------------------------------------------------------------------------------------------


EPS_NUM = 1e-9   # kW / kWh: single numerical tolerance used in code and text (safety_layer.md Sec. 6.4)


def sanitize(x):
    """Replace non-finite raw actions (NaN/Inf from a diverging network) by 0. Returns (x, n_bad).
    Theorem 1 is stated for finite x; the count must be logged."""
    x = np.asarray(x, float)
    bad = ~np.isfinite(x)
    return np.where(bad, 0.0, x), int(bad.sum())


def is_feasible(l, u, C, Crev, mask=None):
    """A5 check (Proposition 1(i)) per row: sum l <= C, sum u >= -Crev, C >= -Crev (l <= u by construction)."""
    l = np.atleast_2d(np.asarray(l, float)); u = np.atleast_2d(np.asarray(u, float))
    if mask is not None:
        l = np.where(mask, l, 0.0); u = np.where(mask, u, 0.0)
    C = np.atleast_1d(np.asarray(C, float)); Crev = np.atleast_1d(np.asarray(Crev, float))
    return (l.sum(1) <= C) & (u.sum(1) >= -Crev) & (C >= -Crev)


def _bisect_safe(a, l, u, C, Crev, upper, iters=200):
    """Robust bisection for rows whose aggregate constraint is active. Keeps the *feasible* end of the
    bracket (upper case: s(hi) <= C; lower case: s(lo) >= -Crev) and returns clip(a - lambda) at that end,
    so the aggregate constraint holds as evaluated in floating point. a, l, u: (B, N); upper: (B,) bool."""
    s = lambda lam: np.clip(a - lam[:, None], l, u).sum(1)
    pad = 1.0 + 1e-6 * np.abs(a).max(1)       # strictly beyond the extreme breakpoint, so that in floating
    lo = np.where(upper, 0.0, (a - u).min(1) - pad)   # point clip(a - lo) == u and clip(a - hi) == l exactly
    hi = np.where(upper, (a - l).max(1) + pad, 0.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        sm = s(mid)
        ok_hi = upper & (sm <= C)            # upper: move hi down to mid when feasible
        ok_lo = (~upper) & (sm >= -Crev)     # lower: move lo up to mid when feasible
        hi = np.where(ok_hi | ((~upper) & ~ok_lo), mid, hi)
        lo = np.where(ok_lo | (upper & ~ok_hi), mid, lo)
    lam = np.where(upper, hi, lo)
    return np.clip(a - lam[:, None], l, u), lam


def project(a, l, u, C, Crev, mask=None, check=True):
    """Exact Euclidean projection, batched over rows. a, l, u: (B, N); C (= C_tilde), Crev: (B,).
    mask (B, N) bool, True = occupied; masked entries are forced to l = u = a = 0 and returned as 0.
    Returns p (B, N), lam (B,), case (B,) in {0: aggregate inactive, +1: upper active, -1: lower active}.
    Precondition (A5, Prop. 1(i)): is_feasible(l, u, C, Crev) for every row (asserted if check=True).
    Robustness (review M1): S at the last breakpoint is set to sum(l) exactly; if no breakpoint satisfies
    S <= target (A5 tight to rounding) lambda = last breakpoint (p = l); a post-condition on the aggregate
    sum (tolerance EPS_NUM) sends any failing row to _bisect_safe."""
    a = np.atleast_2d(np.asarray(a, float)).copy()
    l = np.atleast_2d(np.asarray(l, float)).copy()
    u = np.atleast_2d(np.asarray(u, float)).copy()
    C = np.atleast_1d(np.asarray(C, float))
    Crev = np.atleast_1d(np.asarray(Crev, float))
    a, _ = sanitize(a)
    if mask is not None:
        a = np.where(mask, a, 0.0); l = np.where(mask, l, 0.0); u = np.where(mask, u, 0.0)
    if check:
        assert np.all(l <= u) and np.all(is_feasible(l, u, C, Crev)), "projection called on an infeasible row"
    Bn, N = a.shape
    p0 = np.clip(a, l, u)
    s0 = p0.sum(1)
    case = np.where(s0 > C, 1, np.where(s0 < -Crev, -1, 0))
    target = np.where(case == 1, C, -Crev)
    # sorted breakpoints lambda_(j) and S_j = s(lambda_(j)) by slopes (O(N log N)):
    # for lambda below a_i-u_i entry i sits at u_i; between a_i-u_i and a_i-l_i it is free (slope -1);
    # above a_i-l_i it sits at l_i.
    beta = np.concatenate([a - u, a - l], 1)                       # (B, 2N) breakpoints
    ev = np.concatenate([np.ones((Bn, N)), -np.ones((Bn, N))], 1)  # +1 enters free set, -1 leaves
    order = np.argsort(beta, 1, kind="stable")
    beta = np.take_along_axis(beta, order, 1)
    ev = np.take_along_axis(ev, order, 1)
    nfree = np.cumsum(ev, 1)                        # free count on (lambda_(j), lambda_(j+1))
    dbeta = np.diff(beta, axis=1)                   # (B, 2N-1)
    S = np.empty_like(beta)
    S[:, 0] = u.sum(1)
    S[:, 1:] = u.sum(1, keepdims=True) - np.cumsum(nfree[:, :-1] * dbeta, 1)
    S[:, -1] = l.sum(1)                             # exact end value (M1 fix)
    hit = S <= target[:, None]
    has = hit.any(1)
    j = np.where(has, np.argmax(hit, 1), 2 * N - 1)  # no hit (tight A5): last breakpoint -> p = l
    jm = np.maximum(j - 1, 0)
    b1 = np.take_along_axis(beta, jm[:, None], 1)[:, 0]
    b2 = np.take_along_axis(beta, j[:, None], 1)[:, 0]
    S1 = np.take_along_axis(S, jm[:, None], 1)[:, 0]
    S2 = np.take_along_axis(S, j[:, None], 1)[:, 0]
    den = S1 - S2
    lam_seg = np.where(den > 0, b1 + (S1 - target) * (b2 - b1) / np.where(den > 0, den, 1.0), b2)
    lam = np.where(j == 0, b2, lam_seg)
    # no hit (A5 tight to rounding): lambda strictly beyond the last breakpoint, so p = l exactly
    lam = np.where(has, lam, beta[:, -1] + 1.0 + 1e-6 * np.abs(a).max(1))
    lam = np.where(case == 0, 0.0, lam)
    p = np.clip(a - lam[:, None], l, u)
    # post-condition (review M1): aggregate within tolerance, otherwise robust bisection for that row
    sp = p.sum(1)
    bad = (case != 0) & ((sp > C + EPS_NUM) | (sp < -Crev - EPS_NUM))
    if bad.any():
        pb, lb = _bisect_safe(a[bad], l[bad], u[bad], C[bad], Crev[bad], case[bad] == 1)
        p[bad], lam[bad] = pb, lb
    if mask is not None:
        p = np.where(mask, p, 0.0)
    return p, lam, case


def project_bisect(a, l, u, C, Crev, iters=200):
    """Single-row reference: bisection on s(lambda) + exact polish on the identified free set.
    The polished value is accepted only if it satisfies the aggregate constraint (tolerance EPS_NUM);
    otherwise the feasible bracket end is returned (_bisect_safe)."""
    a, l, u = map(lambda x: np.asarray(x, float), (a, l, u))
    s = lambda lam: np.clip(a - lam, l, u).sum()
    if -Crev <= s(0.0) <= C:
        return np.clip(a, l, u), 0.0
    upper = s(0.0) > C
    target = C if upper else -Crev
    pb, lb = _bisect_safe(a[None], l[None], u[None], np.array([C]), np.array([Crev]), np.array([upper]), iters)
    lam = lb[0]
    F = (a - lam > l) & (a - lam < u)
    if F.any():  # exact polish: lambda = (sum_F a + sum_{not F} p - target)/|F|
        fixed = np.clip(a - lam, l, u)[~F].sum()
        lam_p = (a[F].sum() + fixed - target) / F.sum()
        pp = np.clip(a - lam_p, l, u)
        if -Crev - EPS_NUM <= pp.sum() <= C + EPS_NUM:
            return pp, lam_p
    return pb[0], lam


def jacobian(a, l, u, lam, case, tol=1e-12):
    """dp/da at a point of differentiability (eq. (S16)). Single row."""
    F = ((a - lam) > l + tol) & ((a - lam) < u - tol)
    D = np.diag(F.astype(float))
    if case == 0 or F.sum() == 0:
        return D
    d = F.astype(float)
    return D - np.outer(d, d) / d.sum()


# ----------------------------------------------------------------------------------------------
# Fallback rule (DD8) and actuation
# ----------------------------------------------------------------------------------------------


def fallback(l, u, pmin, C, Crev, laxity, occupied):
    """DD8 least-laxity-first fallback for an infeasible step (safety_layer.md Sec. 7.2). Single row.
    C is the tightened forward limit C_tilde.
      FB-2 (sum u < -Crev, with C >= -Crev): p = u.
      FB-1 (sum l > C): start from p = pmin, raise towards l in order of (laxity ascending,
            l - pmin descending, index ascending) until the budget C - sum(pmin) is used.
      FB-0 (budget < 0): p = pmin.
      FB-1* (C < -Crev, limits contradictory after tightening): FB-1 loop with forward priority.
    Returns p, case string."""
    l, u, pmin, laxity = map(lambda x: np.asarray(x, float), (l, u, pmin, laxity))
    occupied = np.asarray(occupied, bool)
    contradictory = C < -Crev
    if not contradictory and u[occupied].sum() < -Crev:
        return np.where(occupied, u, 0.0), "FB-2"
    if not contradictory and l[occupied].sum() <= C:
        raise ValueError("fallback called on a feasible step")
    p = np.where(occupied, pmin, 0.0)
    budget = C - p.sum()
    if budget < 0:  # FB-0: even maximal discharge cannot meet C (only if C < sum pmin <= 0)
        return p, "FB-0"
    idx = np.arange(len(l))
    order = np.lexsort((idx, -(l - pmin), laxity))     # primary key: laxity (last key in lexsort)
    for i in order:
        if not occupied[i]:
            continue
        add = min(l[i] - pmin[i], budget)
        p[i] += add
        budget -= add
        if budget <= 0:
            break
    return p, ("FB-1*" if contradictory else "FB-1")


P_CS = 32.0 * 400.0 * math.sqrt(3) / 1000.0   # 22.170 kW
EPS_SAT = 1e-3                                  # kW
EPS_Q = 0.008                                   # kW, per-charger actuation margin (Sec. 6.3)


def actuation(p, u, zeta_c, zeta_d, occupied):
    """Map projected reported power p (kW) to EV2Gym's normalised action with upward rounding to the
    1e-5 grid that EV2Gym applies (round(action, 5)); saturated charging is sent as full current."""
    assert np.all(np.isfinite(p)), "non-finite commanded power"
    a = np.zeros_like(p)
    for i in range(len(p)):
        if not occupied[i]:
            continue
        if p[i] >= u[i] - EPS_SAT and u[i] > 0:
            a[i] = 1.0
        elif p[i] > 0:
            a[i] = min(1.0, math.ceil(p[i] / (zeta_c[i] * P_CS) * 1e5 - 1e-9) / 1e5)
        elif p[i] < 0:
            a[i] = max(-1.0, math.ceil(p[i] / (zeta_d[i] * P_CS) * 1e5 - 1e-9) / 1e5)
    return a
