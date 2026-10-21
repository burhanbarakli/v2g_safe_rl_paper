"""check_A1.py -- numerical check of Assumptions A1 and A3 on EV2Gym's own battery model.

Run from anywhere with the EV2Gym venv:
    /home/claude/work/ext/venv/bin/python /home/claude/work/method/check_A1.py

What is checked (notation of formulation.md / safety_layer.md):
  phi_i(E)  = energy after one step of maximum-power charging from energy E (kWh), computed by
              calling EV2Gym's EV._charge (pinned commit 55ef368) with the charger at full current.
  Pbar_i(E) = (phi_i(E) - E)/dt  (step-average maximum charging power, kW).
  A1 : phi_i is non-decreasing in E on [E_min, E_cap].
       Equivalent: Pbar_i decreases no faster than 1/dt, i.e. dt * L_down <= 1 where
       L_down = max over E<E' of (Pbar(E) - Pbar(E'))/(E' - E).
  A3a: every command p in [0, Pbar_i(E)] is delivered exactly: E+ = E + p*dt (before EV2Gym's 0.01 kWh ceil).
  A3b: delivered energy is non-decreasing in the commanded power.
  A3c: EV.step (with the ceil) never delivers less than E + p*dt and at most +0.01 kWh more.
  A3d: discharge commands p in [-min(Pdis, (E-Emin)/dt), 0) are delivered exactly.
  Also: analytic closed form of phi (derived in safety_layer.md, Sec. 2.3) vs EV._charge.

EV specs: ev2gym/data/ev_specs_v2g_enabled2024.json (8 models). Under DD4 we use a scalar efficiency per
model; we test eta in {0.85, 0.90, 0.95, 1.00}. transition_soc ~ U[0.7, 0.9] in EV2Gym, tested on a grid
{0.70, 0.75, 0.80, 0.85, 0.90} plus 1.0 (no taper); multiplier m in {1, 5} (V2GProfitMax.yaml uses 5).
"""
import json
import math
import os
import sys
import itertools

import numpy as np

REPO = "/home/claude/work/ext/EV2Gym"
sys.path.insert(0, REPO)
from ev2gym.models.ev import EV  # noqa: E402

V_LL = 400.0          # V (V2GProfitMax.yaml)
PHASES = 3
I_MAX = 32.0          # A, charger max current
P_CS = I_MAX * V_LL * math.sqrt(PHASES) / 1000.0  # 22.17 kW
E_MIN = 5.0           # kWh, config ev.min_battery_capacity


def make_ev(spec, eta, ts, m, dt_min, E0):
    ev = EV(id=0, location=0, battery_capacity_at_arrival=E0, time_of_arrival=0, time_of_departure=10,
            battery_capacity=spec["battery_capacity"], min_battery_capacity=E_MIN,
            max_ac_charge_power=spec["max_ac_charge_power"],
            max_discharge_power=-spec["max_ac_discharge_power"],
            transition_soc=ts, transition_soc_multiplier=m,
            charge_efficiency=eta, discharge_efficiency=eta, timescale=dt_min)
    return ev


def charge_once(spec, eta, ts, m, dt_min, E0, p_cmd_kw, with_ceil=False):
    """Command grid-side power p_cmd_kw/eta (so that the *reported* power is p_cmd_kw in the linear region).
    Returns the energy after the step (kWh)."""
    ev = make_ev(spec, eta, ts, m, dt_min, E0)
    # reported power p = eta * P_cmd  =>  P_cmd = p/eta, amps = P_cmd*1000/(V*sqrt(3))
    amps = (p_cmd_kw / eta) * 1000.0 / (V_LL * math.sqrt(PHASES))
    amps = min(amps, I_MAX)
    if amps <= 0:
        return E0
    if with_ceil:
        ev.step(amps, V_LL, phases=PHASES)
    else:
        ev._charge(amps, V_LL, phases=PHASES)
    return float(ev.current_capacity)


def discharge_once(spec, eta, dt_min, E0, p_kw):
    ev = make_ev(spec, eta, 1.0, 1, dt_min, E0)
    amps = (p_kw / eta) * 1000.0 / (V_LL * math.sqrt(PHASES))   # negative
    amps = max(amps, -I_MAX)
    ev._discharge(amps, V_LL, phases=PHASES)
    return float(ev.current_capacity)


def phi_analytic(C, Pmax_rep, ts, m, dt_h, E):
    """Closed form of EV2Gym's full-pilot map (safety_layer.md eq. (phi-EV2Gym)). Pmax_rep = eta*min(P_ev, P_cs)."""
    s = E / C
    d = Pmax_rep * dt_h / C
    if ts >= 1.0:
        return C * min(1.0, s + d)
    if s + d <= ts:
        s_new = s + d
    elif s < ts:
        s_new = 1.0 - (1.0 - ts) * math.exp(-m * (s + d - ts) / (1.0 - ts))
    else:
        s_new = 1.0 - (1.0 - s) * math.exp(-m * d / (1.0 - ts))
    s_new = min(s_new, s + d)
    return C * s_new


def main():
    specs = json.load(open(os.path.join(REPO, "ev2gym/data/ev_specs_v2g_enabled2024.json")))
    etas = [0.85, 0.90, 0.95, 1.00]
    tss = [0.70, 0.75, 0.80, 0.85, 0.90, 1.0]
    ms = [1, 5]
    dts = [5, 15]
    n_grid = 600
    rows = []
    worst = dict(A1_min_increment=np.inf, dtL=0.0, A3a=0.0, A3b=np.inf, A3c_lo=np.inf, A3c_hi=0.0,
                 A3d=0.0, analytic=0.0)
    for (name, spec), eta, ts, m, dt_min in itertools.product(specs.items(), etas, tss, ms, dts):
        C = spec["battery_capacity"]
        dt_h = dt_min / 60.0
        Pmax_rep = eta * min(spec["max_ac_charge_power"], P_CS)
        E = np.linspace(E_MIN, C, n_grid)
        phi = np.array([charge_once(spec, eta, ts, m, dt_min, e, Pmax_rep) for e in E])
        # A1: monotone phi
        inc = np.diff(phi)
        a1_ok = bool(np.all(inc >= -1e-9))
        # A1': phi is 1-Lipschitz (Pbar non-increasing): phi(E') - phi(E) <= E' - E
        lip_excess = float(np.max(inc - np.diff(E)))
        kappa = (m * (Pmax_rep * dt_h / C) / (1.0 - ts)) if ts < 1 else 0.0
        # Lipschitz (downward) of Pbar
        Pbar = (phi - E) / dt_h
        dP = np.diff(Pbar) / np.diff(E)
        L_down = max(0.0, float(-dP.min()))
        dtL = dt_h * L_down
        # analytic vs EV2Gym
        phi_a = np.array([phi_analytic(C, Pmax_rep, ts, m, dt_h, e) for e in E])
        an_err = float(np.max(np.abs(phi_a - phi)))
        # A3a/A3b: sub-max commands are delivered exactly and monotonically
        a3a_err, a3b_min, a3c_lo, a3c_hi = 0.0, np.inf, np.inf, 0.0
        for e in E[:: max(1, n_grid // 60)]:
            pbar = (charge_once(spec, eta, ts, m, dt_min, e, Pmax_rep) - e) / dt_h
            if pbar <= 1e-9:
                continue
            ps = np.linspace(0.0, pbar, 41)[1:]
            out = np.array([charge_once(spec, eta, ts, m, dt_min, e, p) for p in ps])
            a3a_err = max(a3a_err, float(np.max(np.abs(out - (e + ps * dt_h)))))
            # beyond pbar: commanding more must not deliver less than pbar
            ps2 = np.linspace(0.0, Pmax_rep * 1.5, 61)[1:]
            out2 = np.array([charge_once(spec, eta, ts, m, dt_min, e, p) for p in ps2])
            a3b_min = min(a3b_min, float(np.min(np.diff(out2))))
            outc = np.array([charge_once(spec, eta, ts, m, dt_min, e, p, with_ceil=True) for p in ps])
            gap = outc - (e + ps * dt_h)
            a3c_lo = min(a3c_lo, float(gap.min()))
            a3c_hi = max(a3c_hi, float(gap.max()))
        # A3d: discharge exactness (only once per (spec, eta, dt))
        a3d_err = 0.0
        if ts == 1.0 and m == 1:
            Pdis = eta * min(spec["max_ac_discharge_power"], P_CS)
            for e in E[:: max(1, n_grid // 60)]:
                pmin = -min(Pdis, (e - E_MIN) / dt_h)
                for p in np.linspace(pmin, 0.0, 21)[:-1]:
                    if p >= -1e-12:
                        continue
                    out = discharge_once(spec, eta, dt_min, e, p)
                    a3d_err = max(a3d_err, abs(out - (e + p * dt_h)))
        rows.append(dict(model=name, eta=eta, ts=ts, m=m, dt=dt_min, A1=a1_ok, min_inc=float(inc.min()),
                         lip_excess=lip_excess, kappa=kappa,
                         dtL=dtL, analytic_err=an_err, A3a_err=a3a_err, A3b_min_diff=a3b_min,
                         ceil_gap=(a3c_lo, a3c_hi), A3d_err=a3d_err))
        worst["A1_min_increment"] = min(worst["A1_min_increment"], float(inc.min()))
        worst["dtL"] = max(worst["dtL"], dtL)
        worst["A3a"] = max(worst["A3a"], a3a_err)
        worst["A3b"] = min(worst["A3b"], a3b_min)
        worst["A3c_lo"] = min(worst["A3c_lo"], a3c_lo)
        worst["A3c_hi"] = max(worst["A3c_hi"], a3c_hi)
        worst["A3d"] = max(worst["A3d"], a3d_err)
        worst["analytic"] = max(worst["analytic"], an_err)

    n = len(rows)
    n_a1 = sum(r["A1"] for r in rows)
    print(f"configurations checked: {n} (8 models x {len(etas)} eta x {len(tss)} transition_soc x "
          f"{len(ms)} multipliers x {len(dts)} dt), grid {n_grid} energies each")
    print(f"A1 (phi non-decreasing) holds in {n_a1}/{n}; smallest grid increment of phi = "
          f"{worst['A1_min_increment']:.3e} kWh")
    for dt in dts:
        sub = [r for r in rows if r["dt"] == dt]
        print(f"  dt={dt:2d} min: max dt*L_down = {max(r['dtL'] for r in sub):.4f} "
              f"(A1 sufficient condition dt*L_down <= 1)")
        for m in ms:
            sub2 = [r for r in sub if r["m"] == m and r["ts"] < 1]
            print(f"     m={m}: max dt*L_down over taper configs = {max(r['dtL'] for r in sub2):.4f}")
    print(f"A1' (phi 1-Lipschitz, i.e. Pbar non-increasing): max of [phi(E')-phi(E)-(E'-E)] = "
          f"{max(r['lip_excess'] for r in rows):.3e} kWh")
    zs = {}
    for m in ms:
        if m > 1:
            from scipy.optimize import brentq
            z0 = brentq(lambda z: 1 - math.exp(-m * z) - z, 1e-6, 1.0)
            k2 = brentq(lambda k: k / math.expm1(k) - m * (1 - z0), 1e-9, 50.0)
            kstar = min(m * z0, k2)
            zs[m] = (z0, kstar)
            print(f"m={m}: z0={z0:.5f}, kappa*(m)=min(m z0={m*z0:.4f}, kappa_2={k2:.4f}) = {kstar:.4f}")
    for m in ms:
        for dt in dts:
            sub = [r for r in rows if r["m"] == m and r["dt"] == dt]
            print(f"  m={m} dt={dt:2d}: max kappa over configs = {max(r['kappa'] for r in sub):.4f}; "
                  f"A3a max err = {max(r['A3a_err'] for r in sub):.3e} kWh; "
                  f"ceil gap in [{min(r['ceil_gap'][0] for r in sub):.3e}, {max(r['ceil_gap'][1] for r in sub):.3e}]")
    print(f"closed-form phi vs EV._charge: max abs error = {worst['analytic']:.3e} kWh")
    print(f"A3a exact delivery of p in (0, Pbar(E)]: max abs error = {worst['A3a']:.3e} kWh")
    print(f"A3b delivered energy non-decreasing in command: min increment = {worst['A3b']:.3e} kWh")
    print(f"A3c EV.step ceil: realised - (E + p dt) in [{worst['A3c_lo']:.3e}, {worst['A3c_hi']:.3e}] kWh")
    print(f"A3d exact delivery of discharge p in [p_min, 0): max abs error = {worst['A3d']:.3e} kWh")
    bad = [r for r in rows if not r["A1"]]
    if bad:
        print("A1 FAILURES:")
        for r in bad[:20]:
            print("  ", r)
    return rows, worst


if __name__ == "__main__":
    main()
