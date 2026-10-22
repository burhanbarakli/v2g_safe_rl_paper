"""Fallback rule (Lemma 2, Alg. 1.3) and actuation map (Lemma E3, Alg. 1.4)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import requires_ev2gym
from reference import projection_ref as R
from v2gsafe.constants import EPS_Q, P_CS
from v2gsafe.safety.actuation import actuation
from v2gsafe.safety.fallback import fallback


def _infeasible_row(rng, N=8):
    pmin = -rng.uniform(0, 11, N)
    l = pmin + rng.uniform(0, 15, N)
    u = l + rng.uniform(0, 5, N)
    lax = rng.normal(0, 3, N)
    occ = rng.random(N) > 0.2
    occ[0] = True
    return l, u, pmin, lax, occ


def test_fallback_identical_to_reference_and_lemma2(rng):
    n = {"FB-0": 0, "FB-1": 0, "FB-2": 0, "FB-1*": 0}
    for _ in range(3000):
        l, u, pmin, lax, occ = _infeasible_row(rng)
        sl, su, spmin = l[occ].sum(), u[occ].sum(), pmin[occ].sum()
        kind = rng.integers(4)
        if kind == 0:     # FB-1 with budget >= 0
            C = rng.uniform(spmin, sl - 1e-3); Crev = rng.uniform(max(-C, 0), 100)
        elif kind == 1:   # FB-0
            C = spmin - rng.uniform(0.1, 5); Crev = -C + rng.uniform(0, 50)
        elif kind == 2:   # FB-2
            C = rng.uniform(su + 1, su + 50); Crev = -su - rng.uniform(0.1, 5)
        else:             # FB-1*
            C = rng.uniform(-20, 0); Crev = -C - rng.uniform(0.1, 5)
        if not (C < -Crev) and sl <= C and su >= -Crev:
            continue
        p, case = fallback(l, u, pmin, C, Crev, lax, occ)
        pr, cr = R.fallback(l, u, pmin, C, Crev, lax, occ)
        np.testing.assert_array_equal(p, pr)
        assert case == cr
        n[case] += 1
        assert np.all(p[~occ] == 0)
        assert np.all(p[occ] >= pmin[occ] - 1e-12) and np.all(p[occ] <= u[occ] + 1e-12)
        if case == "FB-1":                                    # Lemma 2(a)
            assert abs(p.sum() - C) < 1e-9 and np.all(p[occ] <= l[occ] + 1e-12)
        if case == "FB-0":
            np.testing.assert_array_equal(p[occ], pmin[occ])
        if case == "FB-2":
            np.testing.assert_array_equal(p[occ], u[occ])
    assert all(v > 50 for v in n.values()), n


def test_fallback_priority_least_laxity_first():
    l = np.array([5.0, 5.0, 5.0]); u = l + 1; pmin = np.zeros(3)
    p, case = fallback(l, u, pmin, 6.0, 100.0, np.array([2.0, -1.0, 0.5]), np.ones(3, bool))
    assert case == "FB-1"
    np.testing.assert_allclose(p, [0.0, 5.0, 1.0])            # charger 1 (least laxity) first, then 2


def test_fallback_raises_on_feasible_row():
    with pytest.raises(ValueError):
        fallback(np.zeros(2), np.ones(2), -np.ones(2), 5.0, 5.0, np.zeros(2), np.ones(2, bool))


def test_actuation_identical_to_reference(rng):
    for _ in range(2000):
        N = 6
        u = rng.uniform(0, 11, N) * (rng.random(N) > 0.1)
        p = np.where(rng.random(N) < 0.3, u - rng.uniform(0, 2e-3, N), rng.uniform(-11, 11, N))
        p = np.minimum(p, u)
        zc, zd = rng.uniform(0.85, 1, N), rng.uniform(0.85, 1, N)
        occ = rng.random(N) > 0.2
        np.testing.assert_array_equal(actuation(p, u, zc, zd, occ), R.actuation(p, u, zc, zd, occ))


@requires_ev2gym
def test_actuation_margin_with_ev2gym_charger(rng):
    """Realised reported power within [p, p + eps_q] (Lemma E3) through EV2Gym's charger + EV logic."""
    import json
    from ev2gym.models.ev import EV
    from ev2gym.models.ev_charger import EV_Charger
    from v2gsafe.safety import bounds as Bd
    from v2gsafe.specs import SPEC_PATH
    specs = list(json.loads(SPEC_PATH.read_text()).values())
    worst_lo, worst_hi = 0.0, 0.0
    for _ in range(400):
        sp = specs[int(rng.integers(len(specs)))]
        eta = sp["3ph_ch_efficiency"][0]
        effd = {i: eta for i in range(0, 101)}
        cap = sp["battery_capacity"]
        E0 = float(round(rng.uniform(5.0, cap), 2))
        ev = EV(id=0, location=0, battery_capacity_at_arrival=E0, time_of_arrival=0, time_of_departure=20,
                battery_capacity=cap, min_battery_capacity=5.0, max_ac_charge_power=sp["max_ac_charge_power"],
                max_discharge_power=-sp["max_ac_discharge_power"], transition_soc=float(rng.uniform(0.7, 0.9)),
                transition_soc_multiplier=5, charge_efficiency=dict(effd), discharge_efficiency=dict(effd), timescale=15)
        prm = Bd.ev_params_from_ev2gym(ev, 0.25)
        pmin = -min(prm.pdis, max(0, E0 - 5.0) / 0.25)
        u = (prm.phi(E0) - E0) / 0.25
        p = float(rng.uniform(pmin, u)) if rng.random() < 0.8 else float(rng.choice([pmin, u, 0.003, -0.003]))
        p = min(max(p, pmin), u)
        cs = EV_Charger(id=0, connected_bus=0, connected_transformer=0, n_ports=1, max_charge_current=32,
                        min_charge_current=0, max_discharge_current=-32, min_discharge_current=0, voltage=400,
                        phases=3, timescale=15)
        cs.spawn_ev(ev)
        a = actuation(np.array([p]), np.array([u]), [prm.zeta_c], [prm.zeta_d], [True])
        cs.step(a, -0.2, 0.2)
        p_rep = cs.current_power_output
        worst_lo = min(worst_lo, p_rep - p)
        worst_hi = max(worst_hi, p_rep - p)
    assert worst_lo >= -1e-9 and worst_hi <= EPS_Q
