"""EV specification file (DD4) and configuration checks of safety_layer.md Corollary 2."""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from ..constants import P_CS

SPEC_PATH = Path(__file__).resolve().parent / "ev_specs_const_eta.json"


@lru_cache(maxsize=None)
def load_specs(path: str | None = None) -> dict:
    return json.loads(Path(path or SPEC_PATH).read_text())


def spec_classes(path: str | None = None) -> list[tuple[float, float]]:
    """Distinct (battery_capacity, max_ac_charge_power) pairs, in file order.

    EV objects do not carry the model name; models with identical (capacity, power) have identical specs in
    the file (Tesla M3/MY, ID.3/Enyaq), so the pair identifies the spec class used as the 'EV model' feature
    of the conformal predictor (Alg. 4)."""
    out: list[tuple[float, float]] = []
    for sp in load_specs(path).values():
        key = (float(sp["battery_capacity"]), float(sp["max_ac_charge_power"]))
        if key not in out:
            out.append(key)
    return out


def z0(m: float) -> float:
    """Positive root of 1 - exp(-m z) = z for m > 1 (Lemma E2)."""
    lo, hi = 1e-9, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 1.0 - math.exp(-m * mid) - mid > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def kappa_star(m: float) -> float:
    """Exact-delivery threshold kappa*(m) = m z0(m) (safety_layer.md Lemma E2); requires m > 1."""
    if m <= 1:
        return 0.0
    return m * z0(m)


def check_spec_conditions(dt_min: float, m: float, path: str | None = None, s_tr_max: float = 0.9) -> dict:
    """Check C0 and the Lemma E2 condition kappa <= kappa*(m) for every model of the spec file.

    Returns a dict with max_kappa, kappa_star and ok flags; raises nothing (callers decide)."""
    dt_h = dt_min / 60.0
    ks = kappa_star(m)
    kmax, c0 = 0.0, True
    for sp in load_specs(path).values():
        eta = max(sp["3ph_ch_efficiency"]) / 100.0
        c0 &= sp["max_ac_charge_power"] <= P_CS
        pnom = eta * min(P_CS, sp["max_ac_charge_power"])
        delta = pnom * dt_h / sp["battery_capacity"]
        kmax = max(kmax, m * delta / (1.0 - s_tr_max))
    return {"C0": bool(c0), "max_kappa": kmax, "kappa_star": ks, "E2_ok": bool(m > 1 and kmax <= ks)}
