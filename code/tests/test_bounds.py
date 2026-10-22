"""Reach map, certified thresholds, DD2 arrival cap and per-step bounds (safety_layer.md Sec. 2-3)."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from conftest import requires_ev2gym
from reference import projection_ref as R
from v2gsafe.constants import P_CS
from v2gsafe.safety import bounds as Bd
from v2gsafe.specs import SPEC_PATH, check_spec_conditions

SPECS = json.loads(SPEC_PATH.read_text())


def _rand_ev(rng, dt_h=0.25):
    sp = list(SPECS.values())[int(rng.integers(len(SPECS)))]
    eta = sp["3ph_ch_efficiency"][0] / 100
    s_tr = float(rng.choice([rng.uniform(0.7, 0.9), 1.0], p=[0.85, 0.15]))
    return Bd.EVParams(cap=sp["battery_capacity"], e_min=5.0, pnom=eta * min(P_CS, sp["max_ac_charge_power"]),
                       pdis=eta * min(P_CS, sp["max_ac_discharge_power"]), s_tr=s_tr, m=5.0, zeta_c=eta, zeta_d=eta,
                       dt_h=dt_h)


def test_phi_vectorised_scalar_and_reference_agree(rng):
    for _ in range(200):
        p = _rand_ev(rng, float(rng.choice([5 / 60, 0.25])))
        E = rng.uniform(p.e_min, p.cap, 50)
        ref = R.phi_ev2gym(E, p.cap, p.pnom, p.s_tr, p.m, p.dt_h)
        np.testing.assert_allclose(p.phi_vec(E), ref, rtol=0, atol=1e-12)
        np.testing.assert_allclose([p.phi(e) for e in E], ref, rtol=0, atol=1e-12)


@requires_ev2gym
def test_phi_matches_ev2gym_full_current(rng):
    """(S7) equals EV2Gym's EV._charge at full current (check_A1.py, analytic vs EV2Gym)."""
    from ev2gym.models.ev import EV
    for _ in range(60):
        p = _rand_ev(rng)
        sp = next(s for s in SPECS.values() if s["battery_capacity"] == p.cap)
        for E0 in rng.uniform(p.e_min, p.cap, 5):
            ev = EV(id=0, location=0, battery_capacity_at_arrival=E0, time_of_arrival=0, time_of_departure=10,
                    battery_capacity=p.cap, min_battery_capacity=5.0, max_ac_charge_power=sp["max_ac_charge_power"],
                    max_discharge_power=-sp["max_ac_discharge_power"], transition_soc=p.s_tr,
                    transition_soc_multiplier=5, charge_efficiency=p.zeta_c, discharge_efficiency=p.zeta_d,
                    timescale=15)
            ev._charge(32.0, 400, phases=3)
            assert abs(ev.current_capacity - p.phi(E0)) < 1e-9


def test_thresholds_and_cap_identical_to_reference_and_certified(rng):
    for _ in range(150):
        p = _rand_ev(rng)
        K = int(rng.integers(1, 60))
        E_arr = float(rng.uniform(p.e_min, p.cap))
        E_des = float(rng.choice([p.cap, 0.8 * p.cap, rng.uniform(E_arr, p.cap)]))
        Bo = Bd.thresholds(E_des, K, p.phi, p.e_min, p.cap)
        Br = R.thresholds(E_des, K, lambda e: float(R.phi_ev2gym(e, p.cap, p.pnom, p.s_tr, p.m, p.dt_h)), p.e_min, p.cap)
        np.testing.assert_allclose(Bo, Br, atol=1e-9)
        # certified properties (S10)
        assert Bo[0] >= E_des
        for k in range(1, K + 1):
            assert p.phi(Bo[k]) >= Bo[k - 1] - 1e-12
            assert Bo[k] <= max(Bo[k - 1], p.e_min) + 1e-12
            assert p.e_min - 1e-12 <= Bo[k] <= p.cap + 1e-12
        Eo, Bc = Bd.cap_request(E_des, E_arr, K, p.phi, p.e_min, p.cap)
        Er, _ = R.cap_request(E_des, E_arr, K, lambda e: float(R.phi_ev2gym(e, p.cap, p.pnom, p.s_tr, p.m, p.dt_h)),
                              p.e_min, p.cap)
        assert abs(Eo - Er) < 1e-8
        assert E_arr >= Bc[K]                                     # A4 by construction
        assert Eo <= E_des + 1e-12
        reach = Bd.reach(E_arr, K, p.phi)
        if reach < E_des - 1e-3:                                  # cap active: close to the forward reach
            assert reach - 1e-3 <= Eo <= reach


def test_arrival_cap_counterexample_when_infeasible():
    p = Bd.EVParams(cap=60.0, e_min=5.0, pnom=10.0, pdis=10.0, s_tr=1.0, m=5.0, zeta_c=1, zeta_d=1, dt_h=0.25)
    E_req, B = Bd.cap_request(60.0, 10.0, 4, p.phi, p.e_min, p.cap)   # 4 steps of 2.5 kWh -> 20 kWh
    assert abs(E_req - 20.0) < 1e-5 and 10.0 >= B[4]
    # closed form (no taper): B(k) = max(E_min, E_req - k Pnom dt)
    np.testing.assert_allclose(B, [max(5.0, E_req - k * 2.5) for k in range(5)], atol=1e-9)


def test_step_bounds_match_reference_and_closed_form(rng):
    for _ in range(300):
        p = _rand_ev(rng)
        K = int(rng.integers(1, 40))
        E_req, B = Bd.cap_request(p.cap, float(rng.uniform(p.e_min, p.cap)), K, p.phi, p.e_min, p.cap)
        tau = int(rng.integers(1, K + 1))
        E = float(rng.uniform(p.e_min, p.cap))
        ref = R.bounds(E, tau, B, p.e_min, p.cap, p.pdis, lambda e: float(R.phi_ev2gym(e, p.cap, p.pnom, p.s_tr, p.m, p.dt_h)), p.dt_h)
        b = Bd.step_bounds(np.array([E]), np.array([B[tau]]), np.array([B[tau - 1]]), np.array([p.e_min]),
                           np.array([p.pdis]), np.array([p.pnom]), np.array([p.phi(E)]), p.dt_h)
        np.testing.assert_allclose([b["pmin"][0], b["u"][0], b["l"][0]], ref[:3], atol=1e-9)
        assert bool(b["feas_i"][0]) == ref[3]
        assert b["l"][0] <= b["u"][0] + 1e-12
        if ref[3]:
            assert b["l"][0] >= b["pmin"][0] - 1e-12 and b["chi"][0] >= -1e-6
            assert 0.0 <= b["lg2v"][0] <= b["u"][0] + 1e-12
    # (S12) closed form without taper
    p = Bd.EVParams(cap=50.0, e_min=5.0, pnom=11.0, pdis=11.0, s_tr=1.0, m=5.0, zeta_c=1, zeta_d=1, dt_h=0.25)
    E_req, B = Bd.cap_request(45.0, 20.0, 12, p.phi, p.e_min, p.cap)
    for tau in range(1, 13):
        for E in np.linspace(B[tau], 50, 7):
            b = Bd.step_bounds(np.array([E]), np.array([B[tau]]), np.array([B[tau - 1]]), np.array([5.0]),
                               np.array([11.0]), np.array([11.0]), np.array([p.phi(E)]), 0.25)
            l_cf = max(-min(11.0, (E - 5.0) / 0.25), (E_req - E) / 0.25 - (tau - 1) * 11.0)
            assert abs(b["l"][0] - min(l_cf, b["u"][0])) < 1e-6


@requires_ev2gym
def test_lemma1_one_step_invariance_with_ev2gym(rng):
    """For p in [l, u] commanded through the actuation map and EV2Gym's EV.step, E(t+1) >= B_hat(tau-1)."""
    from ev2gym.models.ev import EV
    from v2gsafe.safety.actuation import actuation
    for _ in range(300):
        p = _rand_ev(rng)
        sp = next(s for s in SPECS.values() if s["battery_capacity"] == p.cap)
        K = int(rng.integers(2, 40))
        E0 = float(round(rng.uniform(p.e_min, 0.9 * p.cap), 2))
        E_req, B = Bd.cap_request(p.cap, E0, K, p.phi, p.e_min, p.cap)
        tau = K
        b = Bd.step_bounds(np.array([E0]), np.array([B[tau]]), np.array([B[tau - 1]]), np.array([5.0]),
                           np.array([p.pdis]), np.array([p.pnom]), np.array([p.phi(E0)]), p.dt_h)
        cmd = float(rng.uniform(b["l"][0], b["u"][0]))
        a = actuation(np.array([cmd]), b["u"], [p.zeta_c], [p.zeta_d], [True])[0]
        ev = EV(id=0, location=0, battery_capacity_at_arrival=E0, time_of_arrival=0, time_of_departure=K,
                battery_capacity=p.cap, min_battery_capacity=5.0, max_ac_charge_power=sp["max_ac_charge_power"],
                max_discharge_power=-sp["max_ac_discharge_power"], transition_soc=p.s_tr, transition_soc_multiplier=5,
                charge_efficiency=p.zeta_c, discharge_efficiency=p.zeta_d, timescale=15)
        amps = round(a, 5) * 32.0
        if a < 0 and amps > -0.01:
            amps = 0.0
        ev.step(amps, 400, phases=3)
        assert ev.current_capacity >= B[tau - 1] - 1e-9


def test_taper_minorant_is_inner(rng):
    for _ in range(50):
        p = _rand_ev(rng)
        rho = Bd.taper_minorant_rho(p)
        E = np.linspace(p.e_min, p.cap, 20001)
        pbar = (p.phi_vec(E) - E) / p.dt_h
        assert np.all(pbar >= np.minimum(p.pnom, rho * (p.cap - E)) - 1e-9)


def test_spec_conditions_C0_and_E2():
    for dt in (5, 15):
        c = check_spec_conditions(dt, 5.0)
        assert c["C0"] and c["E2_ok"], c
    assert check_spec_conditions(15, 5.0)["max_kappa"] < 2.5
    assert not check_spec_conditions(15, 1.0)["E2_ok"]              # m <= 1: Lemma E2 fails (DD17)


def test_ev_params_rejects_current_dependent_eta():
    class FakeEV:
        charge_efficiency = {6: 87, 16: 90}
        discharge_efficiency = {6: 87, 16: 90}
        max_ac_charge_power = 11
        battery_capacity = 57.5
        min_battery_capacity = 5
        max_discharge_power = -11
        transition_soc = 0.8
        transition_soc_multiplier = 5
    with pytest.raises(ValueError):
        Bd.ev_params_from_ev2gym(FakeEV(), 0.25)
