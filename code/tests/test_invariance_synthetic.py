"""Feasibility invariance on synthetic episodes (Theorem 1, Corollary 1, Lemma 3; verify_projection.py T4 style).

Random sessions on N chargers, EV2Gym EV objects (taper m = 5, scalar efficiencies), tight random transformer
limits with loads and PV, random raw actions -> our bounds -> projection or fallback -> actuation -> EV.step
(with EV2Gym's charger rounding and dead-band). Checks: no transformer violation on A5 steps, every EV whose stay
had no infeasible step meets E_req, per-EV physical bounds always hold, Lemma 3 deficit bound for all EVs.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import requires_ev2gym
from v2gsafe.constants import EPS_Q, P_CS
from v2gsafe.safety import bounds as Bd
from v2gsafe.safety.actuation import actuation
from v2gsafe.safety.fallback import fallback
from v2gsafe.safety.projection import is_feasible, project
from v2gsafe.specs import SPEC_PATH

pytestmark = requires_ev2gym


@pytest.mark.parametrize("dt_min,T,ptr,seed", [(15, 96, (15, 45), 1), (15, 96, (40, 90), 2), (5, 288, (40, 90), 3)])
def test_theorem1_closed_loop(dt_min, T, ptr, seed):
    from ev2gym.models.ev import EV
    specs = list(json.loads(SPEC_PATH.read_text()).values())
    dt = dt_min / 60.0
    N = 8
    r = np.random.default_rng(seed)
    st = dict(evs=0, clean=0, clean_miss=0, viol_feasible=0, bound_viol=0, lemma3_viol=0, infeasible=0)
    for ep in range(12 if dt_min == 15 else 5):
        P_tr = r.uniform(*ptr)
        load = P_tr * r.uniform(0.0, 0.6, T)
        pv = P_tr * np.clip(np.sin(np.linspace(0, np.pi, T)) * r.uniform(0, 0.9), 0, None)
        Cf, Cr = P_tr - load + pv, P_tr + load - pv
        sessions = [[] for _ in range(N)]
        for i in range(N):
            t = int(r.integers(0, 10))
            while True:
                stay = int(r.integers(4, 40 if dt_min == 15 else 120))
                if t + stay > T - 2:
                    break
                sessions[i].append((t, t + stay))
                t += stay + 3 + int(r.integers(0, 10))
        evs = [None] * N
        for t in range(T):
            for i in range(N):
                for (a_i, d_i) in sessions[i]:
                    if a_i == t:
                        sp = specs[int(r.integers(len(specs)))]
                        eta = sp["3ph_ch_efficiency"][0]
                        eff = {k: eta for k in range(101)}
                        cap = sp["battery_capacity"]
                        ev = EV(id=0, location=i, battery_capacity_at_arrival=round(r.uniform(5.0, 0.8 * cap), 2),
                                time_of_arrival=a_i, time_of_departure=d_i, battery_capacity=cap,
                                min_battery_capacity=5.0, max_ac_charge_power=sp["max_ac_charge_power"],
                                max_discharge_power=-sp["max_ac_discharge_power"],
                                transition_soc=round(0.9 - r.random() / 5, 3), transition_soc_multiplier=5,
                                charge_efficiency=dict(eff), discharge_efficiency=dict(eff), timescale=dt_min,
                                desired_capacity=cap)
                        p = Bd.ev_params_from_ev2gym(ev, dt)
                        K = d_i - a_i + 1
                        E_req, B = Bd.cap_request(cap, ev.current_capacity, K, p.phi, 5.0, cap)
                        evs[i] = dict(ev=ev, p=p, d=d_i, E_req=E_req, B=B, clean=True, dsum=0.0)
                        st["evs"] += 1
            occ = np.array([e is not None for e in evs])
            l, u, pmin, chi = (np.zeros(N) for _ in range(4))
            for i in np.flatnonzero(occ):
                e = evs[i]
                E = e["ev"].current_capacity
                tau = e["d"] - t + 1
                b = Bd.step_bounds(np.array([E]), np.array([e["B"][tau]]), np.array([e["B"][tau - 1]]),
                                   np.array([5.0]), np.array([e["p"].pdis]), np.array([e["p"].pnom]),
                                   np.array([e["p"].phi(E)]), dt)
                pmin[i], u[i], l[i], chi[i] = b["pmin"][0], b["u"][0], b["l"][0], b["chi"][0]
                if not b["feas_i"][0]:
                    e["clean"] = False
            Ct = Cf[t] - occ.sum() * EPS_Q
            feasible = bool(is_feasible(l[None], u[None], Ct, Cr[t], mask=occ[None])[0])
            x = r.normal(0, 1.5, N) * 11.0
            if feasible:
                pc = project(x[None], l[None], u[None], np.array([Ct]), np.array([Cr[t]]), mask=occ[None])[0][0]
            else:
                st["infeasible"] += 1
                pc, _ = fallback(l, u, pmin, Ct, Cr[t], chi, occ)
                for i in np.flatnonzero(occ & (pc < l - 1e-12)):
                    evs[i]["clean"] = False
            for i in np.flatnonzero(occ):
                evs[i]["dsum"] += dt * max(0.0, l[i] - pc[i])
                st["bound_viol"] += int(pc[i] < pmin[i] - 1e-9 or pc[i] > u[i] + 1e-9)
            zc = [e["p"].zeta_c if e else 1 for e in evs]
            zd = [e["p"].zeta_d if e else 1 for e in evs]
            act = actuation(pc, u, zc, zd, occ)
            total = 0.0
            for i in np.flatnonzero(occ):
                ai = round(act[i], 5)
                amps = ai * 32.0
                if ai < 0 and amps > -0.01:
                    amps = 0.0
                en, _ = evs[i]["ev"].step(amps, 400, phases=3)
                total += en / dt
            if feasible and (total - Cf[t] > 1e-6 or -Cr[t] - total > 1e-6):
                st["viol_feasible"] += 1
            for i in np.flatnonzero(occ):
                e = evs[i]
                if e["d"] == t:
                    Ef = e["ev"].current_capacity
                    short = max(0.0, e["E_req"] - Ef)
                    st["lemma3_viol"] += int(short > e["dsum"] + 1e-6)
                    if e["clean"]:
                        st["clean"] += 1
                        st["clean_miss"] += int(Ef < e["E_req"] - 1e-9)
                    evs[i] = None
    print(st)
    assert st["evs"] > 100 and st["clean"] > 20
    assert st["viol_feasible"] == 0 and st["clean_miss"] == 0 and st["bound_viol"] == 0 and st["lemma3_viol"] == 0
