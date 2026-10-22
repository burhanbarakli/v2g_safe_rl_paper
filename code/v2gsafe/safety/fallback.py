"""Fallback rule for infeasible steps (DD8; safety_layer.md Sec. 7.2, Lemma 2; algorithm.md Alg. 1.3).

Cases (C below is the tightened forward limit C_tilde):

* FB-1* : C < -C_rev (limits contradictory after tightening) -> FB-1 procedure with forward priority;
* FB-2  : C >= -C_rev and sum u < -C_rev -> p = u;
* FB-1  : C >= -C_rev and sum l > C -> start at p_min, raise towards l in the order
          (laxity ascending, l - p_min descending, charger index ascending) until the budget C - sum p_min is used;
* FB-0  : FB-1 with negative budget -> p = p_min (maximal discharge).

Identical to ``projection_ref.fallback``.
"""
from __future__ import annotations

import numpy as np

FB_CODES = {"FB-0": 0, "FB-1": 1, "FB-2": 2, "FB-1*": 3}


def fallback(l, u, pmin, C, Crev, laxity, occupied):
    """Return (p, case) for one infeasible row; raises ValueError on a feasible row."""
    l, u, pmin, laxity = (np.asarray(v, float) for v in (l, u, pmin, laxity))
    occupied = np.asarray(occupied, bool)
    contradictory = C < -Crev
    if not contradictory and u[occupied].sum() < -Crev:
        return np.where(occupied, u, 0.0), "FB-2"
    if not contradictory and l[occupied].sum() <= C:
        raise ValueError("fallback called on a feasible step")
    p = np.where(occupied, pmin, 0.0)
    budget = C - p.sum()
    if budget < 0:
        return p, "FB-0"
    idx = np.arange(len(l))
    order = np.lexsort((idx, -(l - pmin), laxity))     # primary key: laxity
    for i in order:
        if not occupied[i]:
            continue
        add = min(l[i] - pmin[i], budget)
        p[i] += add
        budget -= add
        if budget <= 0:
            break
    return p, ("FB-1*" if contradictory else "FB-1")
