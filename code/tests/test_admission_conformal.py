"""Admission-control LP (S 7.4, DD20) and split-conformal calibration (Alg. 4, Proposition 2)."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from v2gsafe.safety import bounds as Bd
from v2gsafe.safety.admission import AdmissionController
from v2gsafe.safety.conformal import ConformalBound, conformal_quantile, make_predictor


def _params(cap=60.0, pnom=10.0, s_tr=1.0):
    return Bd.EVParams(cap=cap, e_min=5.0, pnom=pnom, pdis=10.0, s_tr=s_tr, m=5.0, zeta_c=1, zeta_d=1, dt_h=1.0)


def _slot(E, d, E_req, p):
    return SimpleNamespace(ev=SimpleNamespace(current_capacity=E), d_ctrl=d, E_req=E_req, params=p,
                           rho=Bd.taper_minorant_rho(p))


def test_admission_small_case_feasible_and_reduced():
    """Two EVs, dt = 1 h, 10 kW each, constant capacity: the counterexample of S 7.1 at t0 = 0."""
    ac = AdmissionController(eps_q=0.0)
    p = _params()
    # existing EV: E = 0 -> needs 10 kWh within 2 steps; new EV: same. Capacity 10 kW per step.
    ex = [_slot(5.0, 1, 15.0, p)]
    C = np.array([10.0, 10.0]); Cr = np.array([100.0, 100.0])
    y, info = ac.admit(0, 5.0, 1, 15.0, p, Bd.taper_minorant_rho(p), ex, C, Cr, 2, 1.0)
    assert not info["reduced"] and y == 15.0                       # joint plan (5, 5), (5, 5) exists
    C2 = np.array([8.0, 8.0])                                      # only 16 kWh in total: new EV gets 6 kWh
    y2, info2 = ac.admit(0, 5.0, 1, 15.0, p, Bd.taper_minorant_rho(p), ex, C2, Cr, 2, 1.0)
    assert info2["reduced"] and not info2["not_restored"]
    assert abs(y2 - 11.0) < (15.0 - 5.0) / 2**12 + 1e-6 and y2 <= 11.0 + 1e-6
    # bisection with the exact predicate == 12 separate LP feasibility checks
    lo, hi = 5.0, 15.0
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        ok = ac.y_max(0, 5.0, 1, p, Bd.taper_minorant_rho(p), ex, C2, Cr, 2, 1.0, target_new=mid) is not None
        lo, hi = (mid, hi) if ok else (lo, mid)
    assert abs(lo - y2) < 1e-9


def test_admission_not_restored_and_lp_reachable_option():
    p = _params()
    ex = [_slot(5.0, 0, 20.0, p)]          # existing EV cannot reach 20 kWh in one 10 kW step
    C = np.array([50.0, 50.0]); Cr = np.array([50.0, 50.0])
    y, info = AdmissionController(eps_q=0.0, existing_targets="admitted").admit(0, 5.0, 1, 15.0, p, 1.0, ex, C, Cr,
                                                                                 2, 1.0)
    assert info["not_restored"] and y == 5.0                       # literal S 7.4: set to min(E_arr, E_req)
    y, info = AdmissionController(eps_q=0.0, existing_targets="lp_reachable").admit(0, 5.0, 1, 15.0, p, 1.0, ex, C,
                                                                                     Cr, 2, 1.0)
    assert not info["reduced"] and y == 15.0


def test_admission_full_battery_target_with_taper():
    """IMPLEMENTATION_NOTES I-3 / DD25: the linear minorant cannot reach E_cap exactly."""
    p = Bd.EVParams(cap=58.0, e_min=5.0, pnom=9.9, pdis=9.9, s_tr=0.8, m=5.0, zeta_c=0.9, zeta_d=0.9, dt_h=0.25)
    rho = Bd.taper_minorant_rho(p)
    C = np.full(40, 1000.0); Cr = np.full(40, 1000.0)
    ac = AdmissionController(eps_q=0.0)
    assert ac.existing_targets == "lp_reachable"                   # DD25 default
    # single EV with a full-battery target, unconstrained transformer: reduced only by < 0.01 kWh
    E_req, _ = Bd.cap_request(58.0, 30.0, 39, p.phi, p.e_min, p.cap)
    y, info = ac.admit(0, 30.0, 38, E_req, p, rho, [], C, Cr, 1, 0.25)
    assert 0.0 <= E_req - y < 0.01 and not info["not_restored"]
    assert ac.y_max(0, 30.0, 38, p, rho, [], C, Cr, 1, 0.25) < 58.0   # strictly below a full-battery target
    ex = [_slot(30.0, 39, 58.0, p)]
    y, info = ac.admit(0, 20.0, 30, 40.0, p, rho, ex, C, Cr, 2, 0.25)
    assert not info["reduced"] and y == 40.0                       # DD25 default: connected EV does not block
    _, info = AdmissionController(eps_q=0.0, existing_targets="admitted").admit(0, 20.0, 30, 40.0, p, rho, ex, C, Cr,
                                                                                 2, 0.25)
    assert info["not_restored"]                                    # literal S 7.4 variant (I-3)


# ------------------------------------------------------------------------------------------ conformal
def test_conformal_quantile_definition():
    s = np.arange(1, 10, dtype=float)            # n = 9
    assert conformal_quantile(s, 0.1) == 9.0     # k = ceil(10 * 0.9) = 9
    assert conformal_quantile(s, 0.2) == 8.0
    assert math.isinf(conformal_quantile(s, 0.01))  # k = 10 > n


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_conformal_coverage_on_synthetic_exchangeable_data(alpha):
    """Empirical coverage P(d >= d_lo) >= 1 - alpha - tolerance on i.i.d. synthetic sessions."""
    rng = np.random.default_rng(1)

    def draw(n):
        W = np.column_stack([rng.integers(0, 96, n), rng.integers(0, 7, n), rng.random(n), rng.random(n) * 0.5,
                             np.eye(6)[rng.integers(0, 6, n)]])
        t_arr = W[:, 0].astype(int)
        d = t_arr + np.maximum(1, np.round(20 + 10 * W[:, 2] + 8 * rng.standard_normal(n))).astype(int)
        return W, t_arr, d

    W, _, d = draw(3000)
    model = make_predictor("gbr", alpha)
    model.fit(W, d)
    Wc, _, dc = draw(1000)
    Q = conformal_quantile(model.predict(Wc) - dc, alpha)
    cb = ConformalBound(cfg_id="syn", alpha=alpha, model=model, Q=Q, dt_min=15)
    covs = []
    for _ in range(20):
        Wt, tt, dt = draw(500)
        dlo = np.array([cb.bound_from_features(w, t) for w, t in zip(Wt, tt)])
        assert np.all(dlo >= tt)
        covs.append(np.mean(dt >= dlo))
    assert np.mean(covs) >= 1 - alpha - 0.02
    # uncalibrated (Q = 0) bound is at least as optimistic
    cb0 = ConformalBound(cfg_id="syn", alpha=alpha, model=model, Q=Q, dt_min=15, uncalibrated=True)
    assert cb0.bound_from_features(Wt[0], tt[0]) >= cb.bound_from_features(Wt[0], tt[0]) or Q < 0


def test_conformal_save_load_roundtrip(tmp_path):
    model = make_predictor("gbr", 0.1)
    model.fit(np.random.default_rng(0).random((50, 3)), np.arange(50.0))
    cb = ConformalBound(cfg_id="x", alpha=0.1, model=model, Q=1.5, dt_min=15, classes=[(1.0, 2.0)])
    cb.save(tmp_path / "c.pkl")
    cb2 = ConformalBound.load(tmp_path / "c.pkl")
    w = np.random.default_rng(1).random(3)
    assert cb2.bound_from_features(w, 3) == cb.bound_from_features(w, 3) and cb2.Q == 1.5


@pytest.mark.skipif(not __import__("conftest").ev2gym_available(), reason="EV2Gym not installed")
def test_conformal_session_cache_reproduces_calibration(tmp_path):
    from v2gsafe.envs.scenarios import ScenarioConfig
    from v2gsafe.safety.conformal import calibrate_multi, sessions_cache_path
    s = ScenarioConfig(profile="public", N=4, rho_tr=4.0)
    a = calibrate_multi(s, [0.1, 0.2], 30, 60)
    b = calibrate_multi(s, [0.1, 0.2], 30, 60, cache_dir=tmp_path)
    assert sessions_cache_path(tmp_path, s.cfg_id, 30, 60).exists()
    c = calibrate_multi(s, [0.1], 30, 60, cache_dir=tmp_path)          # from the cache
    assert a[0].Q == b[0].Q == c[0].Q and a[1].Q == b[1].Q and a[0].n_cal == c[0].n_cal == 60
