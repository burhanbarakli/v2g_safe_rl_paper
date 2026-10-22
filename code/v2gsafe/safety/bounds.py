"""Per-EV model, reachable energy, certified thresholds, DD2 arrival cap and per-step bounds.

Implements safety_layer.md Sec. 2-3 for the EV2Gym instantiation g(p) = p (DD1):

* :func:`phi_ev2gym` - one-step maximum-charging map phi_i(E), closed form (S7) (requires C0);
* :func:`thresholds` - certified backward-reachable thresholds B_hat(k), (S8) with properties (S10);
* :func:`cap_request` - DD2 arrival cap (F6), certifies A4 by construction (Sec. 3.4);
* :func:`step_bounds` - p_min (S3), u (S4), l (S11), laxity chi (Sec. 3.3), vectorised over chargers;
* :func:`taper_minorant_rho` - linear taper minorant rho_i used by admission control (Sec. 7.4).

The scalar routines reproduce ``method/projection_ref.py`` (the executable specification) and are tested
against it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..constants import EPS_NUM, P_CS, THRESHOLD_TOL


# ------------------------------------------------------------------------------------------ phi (S7)
def phi_ev2gym(E, C, Pnom, s_tr, m, dt_h):
    """phi_i(E): energy (kWh) after one step at maximum charging power from E (kWh), eq. (S7).

    Vectorised over E and broadcastable parameters. C: capacity (kWh); Pnom: derated nominal max power
    zeta^c * min(P_cs, P_ev) (kW); s_tr: transition SoC; m: transition_soc_multiplier; dt_h: step (h).
    Identical to ``projection_ref.phi_ev2gym``."""
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


def phi_scalar(E: float, C: float, Pnom: float, s_tr: float, m: float, dt_h: float) -> float:
    """Scalar version of (S7) with the ``math`` module (fast inner loop of the threshold bisection)."""
    s = E / C
    d = Pnom * dt_h / C
    lin = s + d
    if s_tr >= 1.0:
        return C * min(1.0, lin)
    w = max(1.0 - s_tr, 1e-12)
    if lin <= s_tr:
        tap = lin
    elif s < s_tr:
        tap = 1.0 - w * math.exp(-m * (lin - s_tr) / w)
    else:
        tap = 1.0 - (1.0 - s) * math.exp(-m * d / w)
    return C * min(tap, lin, 1.0)


@dataclass
class EVParams:
    """Per-EV parameters read from the EV2Gym EV object at arrival (Alg. 3, step 1a)."""

    cap: float          # E_cap (kWh)
    e_min: float        # E_min (kWh)
    pnom: float         # zeta_c * min(P_cs, P_ev) (kW)
    pdis: float         # zeta_d * min(P_cs, P_ev_dis) (kW)
    s_tr: float         # transition SoC
    m: float            # taper multiplier
    zeta_c: float
    zeta_d: float
    dt_h: float

    def phi(self, E: float) -> float:
        return phi_scalar(E, self.cap, self.pnom, self.s_tr, self.m, self.dt_h)

    def phi_vec(self, E):
        return phi_ev2gym(E, self.cap, self.pnom, self.s_tr, self.m, self.dt_h)

    @property
    def kappa(self) -> float:
        if self.s_tr >= 1.0:
            return 0.0
        return self.m * (self.pnom * self.dt_h / self.cap) / (1.0 - self.s_tr)


def _eff(e) -> float:
    """Scalar efficiency of an EV2Gym EV (dict tables in percent, or a scalar fraction)."""
    if isinstance(e, dict):
        vals = {float(v) for v in e.values()}
        if len(vals) != 1:
            raise ValueError("EV efficiency table is not constant over current levels (DD4 violated); "
                             "use the spec file v2gsafe/specs/ev_specs_const_eta.json")
        return vals.pop() / 100.0
    return float(e)


def ev_params_from_ev2gym(ev, dt_h: float, p_cs: float = P_CS) -> EVParams:
    """Read EVParams from an ``ev2gym.models.ev.EV`` (formulation.md Sec. 1.2 'EV2Gym instantiation')."""
    zc, zd = _eff(ev.charge_efficiency), _eff(ev.discharge_efficiency)
    p_ev = float(ev.max_ac_charge_power)
    if p_ev > p_cs + 1e-9:
        raise ValueError(f"condition C0 violated: P_ev = {p_ev} kW > P_cs = {p_cs} kW (safety_layer.md Sec. 2.3)")
    return EVParams(cap=float(ev.battery_capacity), e_min=float(ev.min_battery_capacity),
                    pnom=zc * min(p_cs, p_ev), pdis=zd * min(p_cs, abs(float(ev.max_discharge_power))),
                    s_tr=float(ev.transition_soc), m=float(ev.transition_soc_multiplier), zeta_c=zc, zeta_d=zd,
                    dt_h=dt_h)


# ------------------------------------------------------------------------------- thresholds (S8)
def thresholds(E_req: float, K: int, phi: Callable[[float], float], E_min: float, E_cap: float,
               tol: float = THRESHOLD_TOL) -> np.ndarray:
    """Certified thresholds B_hat[k], k = 0..K (eq. (S8), properties (S10)).

    B_hat[0] = E_req; B_hat[k] = upper (feasible) end of a bisection for min{E : phi(E) >= B_hat[k-1]} on
    [E_min, B_hat[k-1]]. Identical to ``projection_ref.thresholds``; once a level equals E_min all further
    levels are E_min (the reference computes the same values one by one)."""
    K = int(K)
    B = np.empty(K + 1)
    B[0] = E_req
    for k in range(1, K + 1):
        target = B[k - 1]
        if phi(E_min) >= target:
            B[k:] = E_min
            break
        lo, hi = E_min, B[k - 1]
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if phi(mid) >= target:
                hi = mid
            else:
                lo = mid
        B[k] = hi
    return B


def reach(E: float, k: int, phi: Callable[[float], float]) -> float:
    """Psi_k(E) = phi^k(E) = E + R(k, E) (eq. (S6))."""
    for _ in range(int(k)):
        E = phi(E)
    return E


def cap_request(E_des: float, E_arr: float, K: int, phi: Callable[[float], float], E_min: float,
                E_cap: float) -> tuple[float, np.ndarray]:
    """DD2 arrival cap certified with the threshold routine (safety_layer.md Sec. 3.4, eq. (F6)).

    Returns (E_req, B_hat) with E_arr >= B_hat[K] (A4) by construction. K = tau at arrival (>= 1).
    Identical to ``projection_ref.cap_request``."""
    B = thresholds(E_des, K, phi, E_min, E_cap)
    if E_arr >= B[K]:
        return E_des, B
    y, step = min(E_des, reach(E_arr, K, phi) - 1e-6), 1e-6
    while True:
        y = max(y, min(E_arr, E_des))        # y <= E_arr is always certified
        B = thresholds(y, K, phi, E_min, E_cap)
        if E_arr >= B[K]:
            return y, B
        y -= step
        step *= 10.0


# --------------------------------------------------------------------------- per-step bounds (S11)
def step_bounds(E: np.ndarray, Bt: np.ndarray, Bt1: np.ndarray, e_min: np.ndarray, pdis: np.ndarray,
                pnom: np.ndarray, phiE: np.ndarray, dt_h: float):
    """Per-EV bounds at a decision step, vectorised over connected EVs (Alg. 1.1).

    E: energies; Bt = B_hat[tau], Bt1 = B_hat[tau-1]; phiE = phi_i(E_i). Returns a dict with
    pmin (S3), u (S4), l (S11), feas_i (individual feasibility (S9) with tolerance EPS_NUM),
    chi (laxity, Sec. 3.3, eta_c = 1), lg2v (ALAP bound: (S11) with p_min := 0, Alg. 5 row 5b)."""
    pmin = -np.minimum(pdis, np.maximum(0.0, E - e_min) / dt_h)
    u = (phiE - E) / dt_h
    feas = E >= Bt - EPS_NUM
    lraw = (Bt1 - E) / dt_h
    l = np.where(feas, np.minimum(np.maximum(pmin, lraw), u), u)
    lg2v = np.where(feas, np.minimum(np.maximum(0.0, lraw), u), u)
    chi = (E - Bt) / (pnom * dt_h)
    return {"pmin": pmin, "u": u, "l": l, "feas_i": feas, "chi": chi, "lg2v": lg2v}


def bounds_scalar(E, tau, B, E_min, E_cap, Pdis, phi, dt_h):
    """Scalar per-EV bounds, identical to ``projection_ref.bounds``. Returns (pmin, u, l, feasible)."""
    pmin = -min(Pdis, max(0.0, E - E_min) / dt_h)
    u = (phi(E) - E) / dt_h
    feasible = E >= B[tau] - EPS_NUM
    if not feasible:
        return pmin, u, u, False
    l = max(pmin, (B[tau - 1] - E) / dt_h)
    return pmin, u, min(l, u), True


# ------------------------------------------------------------------- admission-control minorant (S 7.4)
def taper_minorant_rho(params: EVParams, n_grid: int = 2000, safety: float = 0.99) -> float:
    """Largest slope rho_i [1/h] with Pbar_i(E) >= min{Pnom_i, rho_i (E_cap - E)} on a grid (Sec. 7.4).

    Computed on ``n_grid`` energies in [E_min, E_cap) with a 1 % safety factor. The LP constraint
    p <= rho (E_cap - E) is then an inner approximation of the taper."""
    E = np.linspace(params.e_min, params.cap, n_grid + 1)[:-1]
    pbar = (params.phi_vec(E) - E) / params.dt_h
    head = params.cap - E
    sel = (pbar < params.pnom - 1e-12) & (head > 1e-12)
    if not np.any(sel):
        return safety / params.dt_h
    return float(safety * np.min(pbar[sel] / head[sel]))


def taper_outer_rho(params: EVParams, n_grid: int = 20000, margin: float = 0.01) -> float:
    """Outer slope rho_out = sup_E Pbar_i(E) / (E_cap - E) [1/h] (DD32): the constraint p <= rho_out (E_cap - E) is
    satisfied by every power EV2Gym can deliver, so the oracle LP with it stays a relaxation (upper bound).

    EV2Gym's taper (ev.py ``_charge``, max pilot): above the transition SoC s >= s_tr the next SoC is
    1 - (1 - s) exp(-kappa) with kappa = m d / (1 - s_tr), i.e. Pbar(E) = (E_cap - E)(1 - e^{-kappa}) / dt exactly;
    without taper (s_tr >= 1) Pbar(E) = min(Pnom, (E_cap - E)/dt), sup ratio 1/dt. The supremum over the whole range
    (including the constant-power and transition regions) is evaluated on a grid, combined with the analytic
    upper-region value and inflated by ``margin``."""
    dt = params.dt_h
    if params.s_tr >= 1.0:
        return 1.0 / dt
    E = np.linspace(params.e_min, params.cap, n_grid + 1)[:-1]
    head = params.cap - E
    ratio = (params.phi_vec(E) - E) / dt / np.maximum(head, 1e-12)
    upper = (1.0 - math.exp(-params.kappa)) / dt
    return float(min(1.0 / dt, (1.0 + margin) * max(float(np.max(ratio)), upper)))
