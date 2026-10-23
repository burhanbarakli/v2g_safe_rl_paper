"""Perfect-information optimum (formulation.md Sec. 3; DD10, DD19, DD26, DD31, DD32; Proposition O1), HiGHS.

Sessions i (t_arr, d, E_arr, E_req) where E_req is the **admitted** request of the controller-independent admission
pre-pass (DD31; the same request every method is scored against; without a table: the DD2-capped value F6);
variables p_i^+(t), p_i^-(t) >= 0, E_i(t) for t = t_arr+1..d+1, z_i >= 0:

  (O1) E_i(t_arr) = E_arr;  (O2) E_i(t+1) = E_i(t) + dt (p^+ - p^-)  (eta = 1, F5);
  (O3) E_min <= E_i(t) <= E_cap;  (O4) p^+ <= P_nom, p^- <= P_dis;
  (O4') taper (DD32): p^+(t_arr) <= Pbar_i(E_arr) and p^+(t) <= rho_out,i (E_cap - E_i(t)) for t > t_arr, with the
        outer slope rho_out = sup_E Pbar_i(E)/(E_cap - E) (:func:`~v2gsafe.safety.bounds.taper_outer_rho`; equal
        to (1 - e^{-kappa})/dt in EV2Gym's upper taper region, 1/dt without taper): a relaxation of the exact taper,
        so every bound below stays valid;
  (O5) E_i(d+1) + z_i >= E_req;  (O6) -C_rev(t) <= sum_i (p^+ - p^-) <= C(t)  (realised C, C_rev).

Profit per step: dt sum_i ((c_dis - c_thr) p^- - (c_ch + c_thr) p^+), c_thr = battery-throughput cost (DD39, 0 by
default); discharge prices come from EV2Gym (they include the discharge price factor, DD39).

* Hard lexicographic oracle: stage 1 z* = min sum z; stage 2 J_LP = max profit s.t. sum z <= z* + 1e-6.
  Rounding-aware J_LP,ceil (Prop. O1(b)): E_req, E_min -> E_req - G_i, E_min - G_i, G_i = 0.01 (d - t_arr + 1) kWh.
* Soft-penalty oracle (DD32): J^pen_LP,ceil = max profit - w_E sum_i z_i - w_P dt sum_t (o^+(t) + o^-(t)) with the
  aggregate band relaxed to -C_rev - o^- <= sum p <= C + o^+, rounding-aware (G_i as above), w = 10 EUR/kWh. It
  upper-bounds the episode value ``penalised_profit_eur`` (and ``penalised_profit_dd2_eur``) of every
  non-conformal method on the same replay -> column ``oracle_pen_profit_eur``.

CLI (whole test grid, writes results/raw/oracle__{test_cfg_id}/episodes.csv)::

    python -m v2gsafe.baselines.oracle --grid configs/test_grid.yaml --test-sets results/test_sets --out results/raw
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ..constants import REPORT_TOL_KWH, W_E_DEFAULT, W_P_DEFAULT
from ..safety.bounds import cap_request, taper_outer_rho
from ..safety.lp import LPBuilder


def oracle_sessions(env_e, dt_h: float, table: dict | None = None) -> list[dict]:
    """Sessions of the episode (oracle information) with the admitted request from the admission ``table`` (DD31;
    {(charger, t_arr): entry}) or the DD2-capped request without a table."""
    from ..safety.admission_prepass import arrival_records
    T = int(env_e.simulation_length)
    out = []
    for a in arrival_records(env_e, dt_h):
        p = a["params"]
        ta, d = a["t_arr"], min(a["d"], T - 1)
        E_dd2, _ = cap_request(a["E_des"], a["E_arr"], d - ta + 1, p.phi, p.e_min, p.cap)
        E_req = E_dd2
        if table is not None:
            e = table.get((a["loc"], ta))
            if e is None:
                raise RuntimeError(f"no admission entry for charger {a['loc']} at step {ta}")
            E_req = min(E_dd2, float(e["E_req"]))
        out.append({"t_arr": ta, "d": d, "E_arr": a["E_arr"], "E_des": a["E_des"], "E_req": E_req,
                    "E_req_dd2": E_dd2, "params": p})
    return out


def _build(sessions, T, dt, C, Crev, cch, cdis, gain: bool, taper: bool = True, thr: float = 0.0,
           soft: tuple[float, float] | None = None):
    """LP of the oracle. soft = (w_E, w_P): elastic aggregate band and shortfall/overload penalties in the profit
    vector (soft-penalty oracle)."""
    lp = LPBuilder()
    agg = [[] for _ in range(T)]
    zc, prof_idx, prof_val, ech = [], [], [], []
    for s in sessions:
        p = s["params"]
        K = s["d"] - s["t_arr"] + 1
        G = 0.01 * K if gain else 0.0
        hi = np.full(K, p.pnom)
        if taper:
            hi[0] = min(p.pnom, max(0.0, (p.phi(s["E_arr"]) - s["E_arr"]) / dt))
        pp = lp.add_vars(K, 0.0, hi)
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
        if taper and K > 1:                       # (O4') p+(k) + rho E(k) <= rho E_cap, E(k) = ec[k-1]
            rho = taper_outer_rho(p)
            lp.add_rows_coo(K - 1, np.repeat(np.arange(K - 1), 2), np.stack([pp[1:], ec[:-1]], 1).ravel(),
                            np.tile([1.0, rho], K - 1), -np.inf, rho * p.cap)
        z = lp.add_vars(1, 0.0, np.inf)[0]
        lp.add_row([ec[-1], z], [1.0, 1.0], s["E_req"] - G, np.inf)
        zc.append(z)
        for kk in range(K):
            tt = s["t_arr"] + kk
            agg[tt].append((pp[kk], pm[kk]))
            prof_idx += [pp[kk], pm[kk]]
            prof_val += [-dt * (cch[tt] + thr), dt * (cdis[tt] - thr)]
        ech.append((pp, pm))
    for tt in range(T):
        if agg[tt]:
            cols = [a for a, _ in agg[tt]] + [b for _, b in agg[tt]]
            vals = [1.0] * len(agg[tt]) + [-1.0] * len(agg[tt])
            if soft is not None:
                o = lp.add_vars(2, 0.0, np.inf)
                lp.add_row(cols + [o[0], o[1]], vals + [-1.0, 1.0], -Crev[tt], C[tt])
                prof_idx += [o[0], o[1]]
                prof_val += [-soft[1] * dt, -soft[1] * dt]
            else:
                lp.add_row(cols, vals, -Crev[tt], C[tt])
    if soft is not None:
        prof_idx += list(zc)
        prof_val += [-soft[0]] * len(zc)
    prof = np.zeros(lp.n)
    np.add.at(prof, np.asarray(prof_idx, int), np.asarray(prof_val, float))
    zvec = np.zeros(lp.n)
    zvec[np.asarray(zc, int)] = 1.0
    return lp, prof, zvec, ech


def _solve(lp, cost, sense, extra_rows=None):
    """HiGHS (dual simplex); on a non-optimal status ('Unknown' happens on the near-degenerate stage-2 LPs with
    full-battery targets and the taper rows) one retry with the interior-point solver and then one with the primal
    simplex (3 of 3,550 test replays needed it); status gets an '(ipm)' / '(primal)' tag."""
    r = lp.solve(cost, sense, extra_rows=extra_rows)
    if not r.optimal:
        for tag, opt in (("ipm", {"solver": "ipm"}), ("primal", {"solver": "simplex", "simplex_strategy": 4})):
            r2 = lp.solve(cost, sense, extra_rows=extra_rows, options=opt)
            if r2.optimal:
                r2.status = f"{r2.status}({tag})"
                return r2
    return r


def solve_oracle(env_e, dt_h: float, ceil_variant: bool = True, table: dict | None = None, thr: float = 0.0,
                 taper: bool = True, soft: bool = True, w_E: float = W_E_DEFAULT, w_P: float = W_P_DEFAULT) -> dict:
    """Hard lexicographic oracle (+ ceil variant) and soft-penalty oracle for the current episode of an EV2Gym env
    (after ``reset``). ``table``: admission table of the replay (DD31)."""
    t0 = time.perf_counter()
    T = int(env_e.simulation_length)
    tr = env_e.transformers[0]
    P_tr = np.asarray(tr.max_power, float)
    L = np.asarray(tr.inflexible_load, float)
    PV = -np.asarray(tr.solar_power, float)
    C, Crev = P_tr - L + PV, P_tr + L - PV
    cch = -np.asarray(env_e.charge_prices[0], float)
    cdis = np.asarray(env_e.discharge_prices[0], float)
    sessions = oracle_sessions(env_e, dt_h, table)
    red = [s["E_req_dd2"] - s["E_req"] for s in sessions]
    out = {"n_evs": len(sessions),
           "n_evs_capped": int(sum(s["E_des"] - s["E_req_dd2"] > REPORT_TOL_KWH for s in sessions)),
           "energy_capped_kwh": float(sum(max(0.0, s["E_des"] - s["E_req_dd2"]) for s in sessions)),
           "n_evs_admission_reduced": int(sum(r > REPORT_TOL_KWH for r in red)),
           "energy_admission_reduced_kwh": float(sum(max(0.0, r) for r in red)),
           "admission": table is not None}
    kw = dict(taper=taper, thr=thr)
    lp, prof, zvec, ech = _build(sessions, T, dt_h, C, Crev, cch, cdis, gain=False, **kw)
    r1 = _solve(lp, zvec, "min")
    if not r1.optimal:
        out.update(solver_status=f"stage1:{r1.status}", solve_time_s=time.perf_counter() - t0)
        return out
    zstar = max(0.0, r1.obj)
    r2 = _solve(lp, prof, "max", extra_rows=[(zvec, -np.inf, zstar + 1e-6)])
    if not r2.optimal:
        # The stage-2 value is very sensitive to this slack when stage 1 ends at zero shortfall with admitted
        # requests at their reachable maximum; if every algorithm fails, a tighter slack still gives a plan that is
        # feasible for the 1e-6 definition (so a slightly conservative J_LP; 1 of 3,550 test replays).
        r2b = _solve(lp, prof, "max", extra_rows=[(zvec, -np.inf, zstar + 1e-8)])
        if r2b.optimal:
            r2b.status = f"{r2b.status}(slack1e-8)"
            r2 = r2b
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
        short2 = np.array([max(0.0, s["E_req_dd2"] - e) for s, e in zip(sessions, E_dep)])
        n_uns = int(np.sum(short > REPORT_TOL_KWH))
        out.update(n_evs_unserved=n_uns, energy_not_served_kwh=float(short.sum()),
                   n_evs_unserved_shared=n_uns, energy_not_served_shared_kwh=float(short.sum()),
                   n_evs_unserved_dd2=int(np.sum(short2 > REPORT_TOL_KWH)),
                   energy_not_served_dd2_kwh=float(short2.sum()),
                   frac_evs_unserved=n_uns / len(sessions) if sessions else 0.0,
                   user_satisfaction_mean=float(np.mean([min(1.0, e / s["E_des"]) if e < s["E_des"] - 1e-3 else 1.0
                                                         for s, e in zip(sessions, E_dep)])) if sessions else 1.0,
                   penalised_profit_eur=float(r2.obj) - w_E * float(short.sum()),
                   penalised_profit_dd2_eur=float(r2.obj) - w_E * float(short2.sum()))
    if ceil_variant:
        lpc, profc, zvc, _ = _build(sessions, T, dt_h, C, Crev, cch, cdis, gain=True, **kw)
        r3 = _solve(lpc, profc, "max", extra_rows=[(zvc, -np.inf, zstar + 1e-6)])
        out["oracle_profit_ceil_eur"] = float(r3.obj) if r3.optimal else float("nan")
    if soft:
        lps, profs, _, _ = _build(sessions, T, dt_h, C, Crev, cch, cdis, gain=True, soft=(w_E, w_P), **kw)
        r4 = _solve(lps, profs, "max")
        out["oracle_pen_profit_eur"] = float(r4.obj) if r4.optimal else float("nan")
    out["solve_time_s"] = time.perf_counter() - t0
    # Sec. 3.1 columns without meaning for the oracle (not a closed-loop controller): no violations (O6), no
    # infeasible/fallback steps; inference time = total LP time spread over the T steps
    ms = 1000.0 * out["solve_time_s"] / T
    out.update(tr_violation_steps=0, tr_violation_kwh=0.0, tr_violation_max_kw=0.0,
               infeasible_steps=0, fb0_steps=0, fb1_steps=0, fb2_steps=0, fb1star_steps=0, a5_deficit_kwh=0.0,
               inference_ms_per_step_mean=ms, inference_ms_per_step_p95=ms)
    return out


def _main(argv=None):
    from ..envs.scenarios import build_test_set, fingerprint, load_test_grid
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
    ap.add_argument("--admission-cache", default=None, help="admission-table cache directory (DD31)")
    ap.add_argument("--no-admission", action="store_true", help="DD2-capped requests (no admission pre-pass)")
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
        # the wrapper's reset computes (or loads) the replay's admission table (DD31), shared with all methods
        env = SafeV2GEnv(scfg, WrapperOptions(admission=not args.no_admission, admission_cache=args.admission_cache))
        rows = []
        for rep in man["replays"][:n]:
            env.reset(rep["seed"], scfg)
            fp = fingerprint(env.env)
            res = solve_oracle(env.env, scfg.dt_h, table=env.adm_table,
                               thr=float(getattr(scfg, "throughput_cost", 0.0) or 0.0))
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
