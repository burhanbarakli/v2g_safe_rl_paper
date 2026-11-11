"""Capacity-aware ("joint") departure lower bound (safety.lower_bound: joint).

The per-EV lower bound (S11) looks only at each EV's own deadline. When several EVs must charge in the same final
steps and the transformer cannot serve all of them at full power, their bounds jump together and the aggregate test
A5 fails (safety_layer.md Sec. 7.1, counterexample). This module tightens the bound with a backward pass over the
forecast capacity:

1. Each connected EV i starts from its departure requirement at the end of its last step, M_i(tau_i) = B_i(0).
2. Going backward over the future steps k = K, ..., 1 (step t + k), the EVs still connected at that step may each
   lower their required energy by at most dt * Pnom_i (power limit) and not below their own threshold
   B_i(tau_i - k) (individual feasibility). The total lowering is limited by dt * C_hat(t + k), the forecast
   aggregate capacity. When the capacity binds, it is shared by water-filling on the residual excess measured in
   full-power steps (least-laxity order backward in time).
3. M_i(1) is the energy EV i should have after the current step so that the remaining requirements fit under the
   forecast capacity; the joint bound is l_i^J = (M_i(1) - E_i) / dt.

The layer then uses l_i = min{u_i, max{l_i^dep, l_i^J}}. If the sum exceeds C_tilde(t), the extra part
l^J - l^dep is shrunk uniformly so that A5 holds whenever it holds for l^dep. Since l^dep <= l <= u, Lemma 1 and
Theorem 1' are unchanged: the joint bound only removes actions; it never weakens the per-EV guarantee and never
turns an A5 step into an infeasible one. It is a heuristic in two respects, both stated openly: the capacity beyond
the forecast horizon is the last forecast held constant, and future arrivals are not reserved for.
"""
from __future__ import annotations

import numpy as np


def _waterfill(e: np.ndarray, w: np.ndarray, want: np.ndarray, cap: float, iters: int = 40) -> np.ndarray:
    """b = clip(e - theta * w, 0, want) with sum b = cap (theta >= 0), assuming sum want > cap >= 0."""
    if cap <= 0.0:
        return np.zeros_like(e)
    lo, hi = 0.0, float(np.max(e / w)) + 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if np.clip(e - mid * w, 0.0, want).sum() > cap:
            lo = mid
        else:
            hi = mid
    return np.clip(e - hi * w, 0.0, want)      # feasible end: sum <= cap


def joint_required_energy(tau: np.ndarray, B: list, pnom: np.ndarray, C_hat: np.ndarray, dt_h: float) -> np.ndarray:
    """Energy each EV should hold after the current step, M_i(1) (kWh).

    tau: (n,) steps remaining including the current one (tau_i = d_i - t + 1 >= 1); B: list of the certified
    threshold tables (B_i[k], k = 0..); pnom: (n,) nominal charging power (kW); C_hat: forecast aggregate capacity
    for steps t+1, t+2, ... (kW, at least max(tau) - 1 entries); dt_h: step length (h)."""
    tau = np.asarray(tau, int)
    n = tau.size
    M = np.array([b[0] for b in B], float)       # requirement at the end of each EV's last step
    K = int(tau.max()) - 1 if n else 0
    w = dt_h * np.asarray(pnom, float)
    for k in range(K, 0, -1):                     # future step t + k
        act = np.flatnonzero(tau > k)
        if act.size == 0:
            continue
        floor = np.array([B[i][tau[i] - k] for i in act])
        e = np.maximum(M[act] - floor, 0.0)        # excess above the individual threshold
        want = np.minimum(w[act], e)
        cap = dt_h * max(float(C_hat[k - 1]), 0.0)
        if want.sum() <= cap:
            b = want
        else:
            b = _waterfill(e, w[act], want, cap)
        M[act] = M[act] - b
    return M


def joint_lower_bound(E, tau, B, pnom, l_dep, u, C_hat, Ctil: float, dt_h: float) -> np.ndarray:
    """Layer lower bound min{u, max{l_dep, l^J}} with the A5-preserving shrink (see module docstring)."""
    E, l_dep, u = (np.asarray(v, float) for v in (E, l_dep, u))
    if E.size == 0:
        return l_dep.copy()
    M1 = joint_required_energy(tau, B, pnom, C_hat, dt_h)
    lJ = (M1 - E) / dt_h
    l_new = np.minimum(u, np.maximum(l_dep, lJ))
    extra = l_new - l_dep
    s_dep, s_extra = float(l_dep.sum()), float(extra.sum())
    if s_dep + s_extra <= Ctil or s_extra <= 0.0:
        return l_new
    alpha = 0.0 if s_dep >= Ctil else (1.0 - 1e-9) * (Ctil - s_dep) / s_extra
    return l_dep + alpha * extra
