"""Perfect-information optimum (formulation.md Sec. 3; DD10, DD19; Proposition O1), solved with HiGHS.

Sessions i in V with (t_arr, d, E_arr, E_req) where E_req is the DD2-capped value (F6) without admission
reductions; variables p_i^+(t), p_i^-(t) >= 0, E_i(t) for t = t_arr+1..d+1, z_i >= 0:

  (O1) E_i(t_arr) = E_arr;  (O2) E_i(t+1) = E_i(t) + dt (p^+ - p^-)  (eta = 1, F5);
  (O3) E_min <= E_i(t) <= E_cap;  (O4) p^+ <= P_nom, p^- <= P_dis (taper relaxed);
  (O5) E_i(d+1) + z_i >= E_req;  (O6) -C_rev(t) <= sum_i (p^+ - p^-) <= C(t)  (realised C, C_rev).

Stage 1: z* = min sum z.  Stage 2: J_LP = max sum_t dt sum_i (c_dis p^- - c_ch p^+) s.t. sum z <= z* + 1e-6.
Rounding-aware variant J_LP,ceil (Proposition O1(b)): E_req and E_min replaced by E_req - G_i and E_min - G_i with
G_i = 0.01 (d - t_arr + 1) kWh, stage 2 with the same bound sum z <= z* + 1e-6.

CLI (whole test grid, writes results/raw/oracle__{test_cfg_id}/episodes.csv)::

    python -m v2gsafe.baselines.oracle --grid configs/test_grid.yaml --test-sets results/test_sets --out results/raw
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ..safety.bounds import cap_request, ev_params_from_ev2gym
from ..safety.lp import LPBuilder


def oracle_sessions(env_e, dt_h: float) -> list[dict]:
    """All sessions of the episode from EV2Gym's pre-generated profile list (oracle information)."""
    out = []
    for ev in env_e.EVs_profiles:
        p = ev_params_from_ev2gym(ev, dt_h)
        ta, d = int(ev.time_of_arrival), int(ev.time_of_departure)
        E_arr = float(ev.battery_capacity_at_arrival)
        E_req, _ = cap_request(float(ev.desired_capacity), E_arr, d - ta + 1, p.phi, p.e_min, p.cap)
        out.append({"t_arr": ta, "d": d, "E_arr": E_arr, "E_des": float(ev.desired_capacity), "E_req": E_req,
                    "params": p})
    return out


def _build(sessions, T, dt, C, Crev, cch, cdis, gain: bool):
    lp = LPBuilder()
    agg = [[] for _ in range(T)]
    zc, prof_idx, prof_val, ech = [], [], [], []
    for s in sessions:
        p = s["params"]
        K = s["d"] - s["t_arr"] + 1
        G = 0.01 * K if gain else 0.0
        pp = lp.add_vars(K, 0.0, p.pnom)
        pm = lp.add_vars(K, 0.0, p.pdis)
        ec = lp.add_vars(K, p.e_min - G, p.cap)
        # (O2): E(1) - dt (p+(0) - p-(0)) = E_arr ; E(k+1) - E(k) - dt (p+(k) - p-(k)) = 0
        r = np.concatenate([[0, 0, 0], np.repeat(np.arange(1, K), 4)])
        c = np.concatenate([[ec[0], pp[0], pm[0]],
                            np.stack([ec[1:], ec[:-1], pp[1:], pm[1:]], 1).ravel()])
        v = np.concatenate([[1.0, -dt, dt], np.tile([1.0, -1.0, -dt, dt], K - 1)])
        rhs = np.zeros(K)
        rhs[0] = s["E_arr"]
        lp.add_rows_coo(K, r, c, v, rhs, rhs)
        z = lp.add_vars(1, 0.0, np.inf)[0]
        lp.add_row([ec[-1], z], [1.0, 1.0], s["E_req"] - G, np.inf)
        zc.append(z)
        for kk in range(K):
            tt = s["t_arr"] + kk
            agg[tt].append((pp[kk], pm[kk]))
            prof_idx += [pp[kk], pm[kk]]
            prof_val += [-dt * cch[tt], dt * cdis[tt]]
        ech.append((pp, pm))
    for tt in range(T):
        if agg[tt]:
            cols = [a for a, _ in agg[tt]] + [b for _, b in agg[tt]]
            vals = [1.0] * len(agg[tt]) + [-1.0] * len(agg[tt])
            lp.add_row(cols, vals, -Crev[tt], C[tt])
    prof = np.zeros(lp.n)
    np.add.at(prof, np.asarray(prof_idx, int), np.asarray(prof_val, float))
    zvec = np.zeros(lp.n)
    zvec[np.asarray(zc, int)] = 1.0
    return lp, prof, zvec, ech


def solve_oracle(env_e, dt_h: float, ceil_variant: bool = True) -> dict:
    """Lexicographic oracle LP for the current episode of an EV2Gym env (after ``reset``)."""
    t0 = time.perf_counter()
    T = int(env_e.simulation_length)
    tr = env_e.transformers[0]
    P_tr = np.asarray(tr.max_power, float)
    L = np.asarray(tr.inflexible_load, float)
    PV = -np.asarray(tr.solar_power, float)
    C, Crev = P_tr - L + PV, P_tr + L - PV
    cch = -np.asarray(env_e.charge_prices[0], float)
    cdis = np.asarray(env_e.discharge_prices[0], float)
    sessions = oracle_sessions(env_e, dt_h)
    out = {"n_evs": len(sessions),
           "n_evs_capped": int(sum(s["E_des"] - s["E_req"] > 0.01 for s in sessions)),
           "energy_capped_kwh": float(sum(max(0.0, s["E_des"] - s["E_req"]) for s in sessions))}
    lp, prof, zvec, ech = _build(sessions, T, dt_h, C, Crev, cch, cdis, gain=False)
    r1 = lp.solve(zvec, "min")
    if not r1.optimal:
        out.update(solver_status=f"stage1:{r1.status}", solve_time_s=time.perf_counter() - t0)
        return out
    zstar = max(0.0, r1.obj)
    r2 = lp.solve(prof, "max", extra_rows=[(zvec, -np.inf, zstar + 1e-6)])
    out.update(shortfall_stage_kwh=zstar, solver_status=r2.status)
    if r2.optimal:
        x = r2.x
        out["profit_eur"] = float(r2.obj)
        net = [x[pp] - x[pm] for pp, pm in ech]          # simultaneous p+ and p- carry no value (F 3)
        out["energy_charged_kwh"] = float(sum(np.maximum(n, 0).sum() for n in net) * dt_h)
        out["energy_discharged_kwh"] = float(sum(np.maximum(-n, 0).sum() for n in net) * dt_h)
        # departure energies of the LP plan (nominal, without EV2Gym's ceil gain) -> Sec. 3.1 shortfall columns
        E_dep = np.array([s["E_arr"] + dt_h * n.sum() for s, n in zip(sessions, net)])
        short = np.array([max(0.0, s["E_req"] - e) for s, e in zip(sessions, E_dep)])
        n_uns = int(np.sum(short > 0.01))
        out.update(n_evs_unserved=n_uns, energy_not_served_kwh=float(short.sum()),
                   frac_evs_unserved=n_uns / len(sessions) if sessions else 0.0,
                   user_satisfaction_mean=float(np.mean([min(1.0, e / s["E_des"]) if e < s["E_des"] - 1e-3 else 1.0
                                                         for s, e in zip(sessions, E_dep)])) if sessions else 1.0)
    if ceil_variant:
        lpc, profc, zvc, _ = _build(sessions, T, dt_h, C, Crev, cch, cdis, gain=True)
        r3 = lpc.solve(profc, "max", extra_rows=[(zvc, -np.inf, zstar + 1e-6)])
        out["oracle_profit_ceil_eur"] = float(r3.obj) if r3.optimal else float("nan")
    out["solve_time_s"] = time.perf_counter() - t0
    # Sec. 3.1 columns without meaning for the oracle (not a closed-loop controller): no admission control, no
    # violations (O6), no infeasible/fallback steps; inference time = total LP time spread over the T steps
    ms = 1000.0 * out["solve_time_s"] / T
    out.update(n_evs_admission_reduced=0, tr_violation_steps=0, tr_violation_kwh=0.0, tr_violation_max_kw=0.0,
               infeasible_steps=0, fb0_steps=0, fb1_steps=0, fb2_steps=0, fb1star_steps=0, a5_deficit_kwh=0.0,
               inference_ms_per_step_mean=ms, inference_ms_per_step_p95=ms)
    return out


def _main(argv=None):
    from ..envs.scenarios import ScenarioConfig, build_test_set, fingerprint, load_test_grid
    from ..envs.wrapper import SafeV2GEnv, WrapperOptions
    from ..eval.metrics import ORACLE_COLUMNS
    from ..utils import RunMeta, append_csv

    ap = argparse.ArgumentParser(description="perfect-information oracle LP on the test grid")
    ap.add_argument("--grid", required=True)
    ap.add_argument("--test-sets", required=True)
    ap.add_argument("--out", required=True, help="results/raw (one directory oracle__{test_cfg_id} per config)")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--max-replays", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    for scfg, n in load_test_grid(args.grid):
        if args.only and scfg.cfg_id not in args.only:
            continue
        n = min(n, args.max_replays) if args.max_replays else n
        run = Path(args.out) / f"oracle__{scfg.cfg_id}"
        csv = run / "episodes.csv"
        if csv.exists() and sum(1 for _ in csv.open()) - 1 >= n:
            print(f"exists: {csv}")
            continue
        if csv.exists():
            csv.unlink()
        meta = RunMeta(run, args.smoke, "cpu", {"exp_id": run.name, "method": "oracle"})
        man = build_test_set(scfg, n, args.test_sets)
        env = SafeV2GEnv(scfg, WrapperOptions(admission=False))
        rows = []
        for rep in man["replays"][:n]:
            env.reset(rep["seed"], scfg)
            fp = fingerprint(env.env)
            res = solve_oracle(env.env, scfg.dt_h)
            rows.append({"exp_id": run.name, "method": "oracle", "seed": 0, "train_N": None,
                         "test_cfg_id": scfg.cfg_id, "profile": scfg.profile, "N": scfg.N, "rho_tr": scfg.rho,
                         "dt_min": scfg.dt_min, "replay_id": rep["replay_id"], "smoke": args.smoke,
                         "replay_seed": rep["seed"],
                         "replay_fingerprint": fp if fp == rep["fingerprint"] else f"MISMATCH:{fp}", **res})
        append_csv(csv, rows, ORACLE_COLUMNS)
        meta.finish(n_episodes=len(rows))
        print(f"{scfg.cfg_id}: {len(rows)} oracle LPs, mean J_LP {np.nanmean([r.get('profit_eur', np.nan) for r in rows]):.2f} EUR, "
              f"mean solve {np.mean([r['solve_time_s'] for r in rows]):.2f}s", flush=True)


if __name__ == "__main__":
    _main()
