"""Actuation map: commanded reported power p (kW) -> EV2Gym normalised action (safety_layer.md Sec. 6.3,
Lemma E3; algorithm.md Alg. 1.4).

Upward rounding to EV2Gym's 1e-5 action grid (``round(action, 5)`` in ``ev_charger.py``); saturated charging
(p >= u - eps_sat, u > 0) is sent as full current. Identical to ``projection_ref.actuation`` (vectorised).
"""
from __future__ import annotations

import numpy as np

from ..constants import EPS_SAT, P_CS


def actuation(p, u, zeta_c, zeta_d, occupied, p_cs: float = P_CS):
    p = np.asarray(p, float)
    u = np.asarray(u, float)
    zeta_c = np.asarray(zeta_c, float)
    zeta_d = np.asarray(zeta_d, float)
    occupied = np.asarray(occupied, bool)
    assert np.all(np.isfinite(p[occupied])), "non-finite commanded power"
    a = np.zeros_like(p)
    sat = occupied & (u > 0) & (p >= u - EPS_SAT)
    pos = occupied & ~sat & (p > 0)
    neg = occupied & ~sat & (p < 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        a_pos = np.minimum(1.0, np.ceil(p / (zeta_c * p_cs) * 1e5 - 1e-9) / 1e5)
        a_neg = np.maximum(-1.0, np.ceil(p / (zeta_d * p_cs) * 1e5 - 1e-9) / 1e5)
    a = np.where(sat, 1.0, a)
    a = np.where(pos, a_pos, a)
    a = np.where(neg, a_neg, a)
    return a
