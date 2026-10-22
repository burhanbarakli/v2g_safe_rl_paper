"""Controller-independent admission pre-pass (DD31; IMPLEMENTATION_NOTES I-22).

Admission control (safety_layer.md Sec. 7.4, DD9/DD20) reduces the request of an arriving EV when the joint
feasibility LP fails. In the original per-step form the LP starts from the *measured* energies of the connected EVs,
so the admitted requests depend on the controller that produced those energies, and every method would be scored
against a different request sequence. DD31 makes the request pipeline identical for all methods (and the oracle):

* arrivals of a replay are processed in time order (charger index as tie-break), using only exogenous
  information available at the arrival step t0: the arriving EV's arrival-time state (E_arr, E_des, d, EV
  parameters), the realised capacity at t0 and the wrapper's forecasts made at t0 (deterministic per (replay, t0),
  :class:`~v2gsafe.envs.forecasts.NoisyForecaster`), and the requests admitted earlier;
* the energies of EVs admitted earlier are taken from a **nominal plan**, not from the closed loop: after every
  admission decision an elastic LP over the connected EVs (same constraints as the admission LP, slacks on the
  targets and on the aggregate band, objective "reach the targets as early as possible",
  :meth:`AdmissionController.solve_records` with ``plan=True``) gives nominal trajectories E_i(t); the next
  arrival reads E_i(t0') from them;
* the admission rule itself is unchanged (DD2 cap, LP check, 12-step bisection on the arriving EV's request with
  the exact y_max predicate, existing requests never reduced, ``lp_reachable`` targets DD25, "not restored"
  fallback to min(E_arr, E_req)); the reduced request is re-certified with ``cap_request`` (A4).

The rule is causal (decision for EV j uses only information up to its arrival), so it can run online; it is
computed here once per replay ahead of the episode and cached (in memory and optionally on disk), and the wrapper
and the oracle look the admitted requests up by (charger, arrival step).

CLI (pre-computes the tables for a test grid; evaluation computes missing ones on the fly)::

    python -m v2gsafe.safety.admission_prepass --grid configs/test_grid.yaml --test-sets results/test_sets \\
        --out results/admission
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from ..constants import EPS_Q, H_F, REPORT_TOL_KWH
from .admission import AdmissionController
from .bounds import cap_request, ev_params_from_ev2gym, taper_minorant_rho

PREPASS_VERSION = 1
_MEM: dict = {}
_MEM_MAX = 256


def capacity_forecast(tr, t: int, horizon: int, fc_inf: np.ndarray, fc_pv: np.ndarray, T: int):
    """C_hat(t'), C_rev_hat(t') for t' = t .. t+horizon-1: realised at t, forecasts for t+1..t+H_f, then the last
    forecast held (safety_layer.md Sec. 7.4). The transformer limit P_tr(t') is known (exogenous schedule)."""
    k = np.arange(horizon)
    Ptr = np.asarray(tr.max_power, float)[np.minimum(t + k, T - 1)]
    inf = np.empty(horizon)
    pv = np.empty(horizon)
    inf[0], pv[0] = float(tr.inflexible_load[t]), -float(tr.solar_power[t])
    if horizon > 1:
        j = np.minimum(k[1:] - 1, len(fc_inf) - 1)
        inf[1:], pv[1:] = fc_inf[j], fc_pv[j]
    return Ptr - inf + pv, Ptr + inf - pv


def cp_key(cp) -> str:
    if cp is None:
        return "dtrue"
    return f"cp_a{getattr(cp, 'alpha', 'x')}_u{int(bool(getattr(cp, 'uncalibrated', False)))}_Q{getattr(cp, 'Q', 0):.6g}"


def table_key(cfg_id: str, seed: int, eps_q: float, existing_targets: str, forecast_rel_std: float, cp=None) -> str:
    h = hashlib.sha1(json.dumps([PREPASS_VERSION, eps_q, existing_targets, forecast_rel_std, cp_key(cp)]).encode())
    return f"{cfg_id}__s{int(seed)}__{h.hexdigest()[:8]}"


def arrival_records(env_e, dt_h: float, scfg=None, cp=None) -> list[dict]:
    """Arrivals of the current episode of an EV2Gym env (after ``reset``) in processing order. An EV with
    time_of_arrival t is spawned at the end of step t-1 and first observed by the wrapper at step t."""
    T = int(env_e.simulation_length)
    out = []
    for ev in env_e.EVs_profiles:
        ta = int(ev.time_of_arrival)
        if ta < 1 or ta >= T:
            continue
        loc = int(ev.location)
        p = ev_params_from_ev2gym(ev, dt_h, env_e.charging_stations[loc].get_max_power())
        d = int(ev.time_of_departure)
        if cp is not None:
            arr_dt = env_e.sim_starting_date + datetime.timedelta(minutes=round(dt_h * 60) * ta)
            d_ctrl = max(int(cp.lower_bound(ev, ta, arr_dt, scfg)), ta)
        else:
            d_ctrl = d
        out.append({"loc": loc, "t_arr": ta, "d": d, "d_ctrl": d_ctrl, "E_arr": float(ev.battery_capacity_at_arrival),
                    "E_des": float(ev.desired_capacity), "params": p})
    out.sort(key=lambda r: (r["t_arr"], r["loc"]))
    return out


def admission_prepass(env_e, dt_h: float, seed: int, eps_q: float = EPS_Q, existing_targets: str = "lp_reachable",
                      forecast_rel_std: float = 0.1, scfg=None, cp=None) -> dict:
    """Admission table of the current episode of ``env_e`` (after ``reset(seed)``).

    Returns {"entries": [{loc, t_arr, d, d_ctrl, E_arr, E_des, E_req_dd2, E_req, reduced_kwh, not_restored}],
    "n_lp": int, "lp_time_s": float}."""
    from ..envs.forecasts import NoisyForecaster
    t_start = time.perf_counter()
    T = int(env_e.simulation_length)
    tr = env_e.transformers[0]
    fc = NoisyForecaster(forecast_rel_std, H_F)
    fc.reseed(np.random.SeedSequence([int(seed) % (2**32), 7]).generate_state(1)[0])   # = SafeV2GEnv.reset
    load = np.asarray(tr.inflexible_load, float)
    pv_series = -np.asarray(tr.solar_power, float)
    ac = AdmissionController(eps_q=eps_q, existing_targets=existing_targets)
    active: list[dict] = []
    entries = []
    for a in arrival_records(env_e, dt_h, scfg, cp):
        t0 = a["t_arr"]
        active = [x for x in active if x["d"] >= t0]            # still connected at t0 (true departure)
        p = a["params"]
        K = a["d_ctrl"] - t0 + 1
        E_req_dd2, _ = cap_request(a["E_des"], a["E_arr"], K, p.phi, p.e_min, p.cap)
        rho = taper_minorant_rho(p)
        recs = []
        for x in active:
            j = t0 - x["t_plan"]
            E0 = float(x["traj"][min(max(j, 0), len(x["traj"]) - 1)])
            recs.append((E0, x["d_ctrl"], x["E_req"], x["params"], x["rho"]))
        recs.append((a["E_arr"], a["d_ctrl"], None, p, rho))
        H = max(1, max(r[1] for r in recs) - t0 + 1)
        Chat, Crhat = capacity_forecast(tr, t0, H, fc.forecast(load, t0, 0), fc.forecast(pv_series, t0, 1), T)
        n_now = len(recs)
        E_req, not_restored = E_req_dd2, False
        slack, traj = ac.solve_records(t0, recs, Chat, Crhat, n_now, dt_h, target_new=E_req_dd2, plan=True)
        if slack is None or slack > 1e-7:
            # a zero-slack plan proves feasibility; otherwise the exact admission rule (y_max + bisection)
            y, info = ac.admit_records(t0, recs, E_req_dd2, Chat, Crhat, n_now, dt_h)
            not_restored = bool(info.get("not_restored", False))
            if y < E_req_dd2 - 1e-12:
                E_req, _ = cap_request(y, a["E_arr"], K, p.phi, p.e_min, p.cap)     # re-certify A4 (DD20)
                slack, traj = ac.solve_records(t0, recs, Chat, Crhat, n_now, dt_h, target_new=E_req, plan=True)
        new = {"d": a["d"], "d_ctrl": a["d_ctrl"], "E_req": E_req, "params": p, "rho": rho, "t_plan": t0,
               "traj": np.array([a["E_arr"]])}
        group = active + [new]
        if traj is not None:
            for k, x in enumerate(group):
                if k in traj:
                    x["traj"], x["t_plan"] = traj[k], t0
        active = group
        entries.append({"loc": a["loc"], "t_arr": t0, "d": a["d"], "d_ctrl": a["d_ctrl"], "E_arr": a["E_arr"],
                        "E_des": a["E_des"], "E_req_dd2": float(E_req_dd2), "E_req": float(E_req),
                        "reduced_kwh": float(E_req_dd2 - E_req), "not_restored": not_restored})
    return {"entries": entries, "n_lp": ac.n_lp, "lp_time_s": ac.lp_time,
            "wall_s": time.perf_counter() - t_start, "version": PREPASS_VERSION}


def get_table(env_e, scfg, seed: int, eps_q: float, existing_targets: str, forecast_rel_std: float, cp=None,
              cache_dir: str | os.PathLike | None = None) -> dict:
    """Cached :func:`admission_prepass` (memory, then ``cache_dir``/<key>.json, else compute and store).
    Returns {(loc, t_arr): entry} plus the raw table under key None."""
    key = table_key(scfg.cfg_id, seed, eps_q, existing_targets, forecast_rel_std, cp)
    if key in _MEM:
        return _MEM[key]
    tab = None
    path = Path(cache_dir) / f"{key}.json" if cache_dir else None
    if path is not None and path.exists():
        try:
            tab = json.loads(path.read_text())
        except (OSError, ValueError):
            tab = None
    if tab is None:
        tab = admission_prepass(env_e, scfg.dt_h, seed, eps_q, existing_targets, forecast_rel_std, scfg, cp)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(tab))
            os.replace(tmp, path)
    out = {(int(e["loc"]), int(e["t_arr"])): e for e in tab["entries"]}
    out[None] = tab
    if len(_MEM) >= _MEM_MAX:
        _MEM.pop(next(iter(_MEM)))
    _MEM[key] = out
    return out


def summary(tab: dict) -> dict:
    ents = tab[None]["entries"] if None in tab else tab["entries"]
    return {"n_evs": len(ents), "n_reduced": int(sum(e["reduced_kwh"] > REPORT_TOL_KWH for e in ents)),
            "reduced_kwh": float(sum(e["reduced_kwh"] for e in ents)),
            "n_not_restored": int(sum(e["not_restored"] for e in ents))}


def _main(argv=None):
    from ..envs.scenarios import build_test_set, load_test_grid
    from ..envs.wrapper import make_ev2gym_env
    ap = argparse.ArgumentParser(description="controller-independent admission tables of a test grid (DD31)")
    ap.add_argument("--grid", required=True)
    ap.add_argument("--test-sets", required=True)
    ap.add_argument("--out", required=True, help="cache directory, e.g. results/admission")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--max-replays", type=int, default=None)
    ap.add_argument("--eps-q", type=float, default=EPS_Q)
    ap.add_argument("--existing-targets", default="lp_reachable", choices=["lp_reachable", "admitted"])
    ap.add_argument("--forecast-rel-std", type=float, default=0.1)
    args = ap.parse_args(argv)
    for scfg, n in load_test_grid(args.grid):
        if args.only and scfg.cfg_id not in args.only:
            continue
        n = min(n, args.max_replays) if args.max_replays else n
        man = build_test_set(scfg, n, args.test_sets)
        env = make_ev2gym_env(scfg)
        t0 = time.time()
        tot = {"n_evs": 0, "n_reduced": 0, "reduced_kwh": 0.0, "n_not_restored": 0}
        for rep in man["replays"][:n]:
            env.tr_seed = rep["seed"]
            env.reset(seed=rep["seed"])
            tab = get_table(env, scfg, rep["seed"], args.eps_q, args.existing_targets, args.forecast_rel_std,
                            cache_dir=args.out)
            for k, v in summary(tab).items():
                tot[k] += v
        print(f"{scfg.cfg_id}: {n} replays, {tot} ({time.time() - t0:.0f}s)", flush=True)
        if not args.max_replays:
            (Path(args.out) / f"{scfg.cfg_id}.done").write_text(json.dumps({"n": n, **tot}) + "\n")


if __name__ == "__main__":
    _main()
