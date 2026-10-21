"""verify_projection.py -- numerical checks of the safety layer (safety_layer.md).

    /home/claude/work/ext/venv/bin/python /home/claude/work/method/verify_projection.py

T1  closed-form projection (sorted breakpoints, batched) == bisection+polish == SciPy SLSQP QP; KKT residuals.
T1b (review M1) tight-A5 rows (sum l == C_tilde or sum u == -C_rev exactly in floating point) and
    large raw actions |x| in [1e4, 1e8]: aggregate post-condition, per-EV bounds, agreement with bisection.
T2  Jacobian (eq. (S16)) == central finite differences at random differentiable points.
T3  Moreau identity: grad of 0.5*dist^2(a, K) == a - Pi(a) (finite differences).
T4  closed loop with EV2Gym's own EV objects (pinned commit, taper m=5, heterogeneous specs, scalar zeta):
    random raw actions -> bounds -> projection or fallback -> actuation (upward rounding) -> EV.step.
    Checks Theorem 1 / Corollary 1 (departure energy and transformer limit on every A5-step),
    the shortfall bound of Lemma 3 (safety_layer.md), and counts infeasible steps.
"""
import json
import math
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, "/home/claude/work/method")
sys.path.insert(0, "/home/claude/work/ext/EV2Gym")
from projection_ref import (project, project_bisect, jacobian, fallback, actuation, phi_ev2gym,
                            thresholds, bounds, reach, cap_request, is_feasible, sanitize, P_CS, EPS_Q, EPS_NUM)

rng = np.random.default_rng(0)


def random_instance(N):
    l = rng.uniform(-11, 6, N)
    u = l + rng.uniform(0, 11, N) * (rng.random(N) > 0.1)  # 10 % with l = u
    a = rng.normal(0, 15, N)
    lo_s, hi_s = l.sum(), u.sum()
    # random feasible aggregate limits
    C = rng.uniform(lo_s, hi_s + 20)
    Crev = rng.uniform(-min(C, hi_s), -lo_s + 20) if rng.random() < 0.7 else rng.uniform(-hi_s, -lo_s)
    Crev = max(Crev, -hi_s)
    if -Crev > C:
        Crev = -C
    return a, l, u, C, Crev


def kkt_residual(a, l, u, C, Crev, p, lam):
    # primal feasibility
    r_p = max(0, (l - p).max(), (p - u).max(), p.sum() - C, -Crev - p.sum())
    # stationarity with box multipliers: nu = p - a + lam must be >=0 where p=l, <=0 where p=u, 0 inside
    nu = p - a + lam
    inside = (p > l + 1e-9) & (p < u - 1e-9)
    atl = (np.abs(p - l) <= 1e-9) & ~(np.abs(p - u) <= 1e-9)
    atu = (np.abs(p - u) <= 1e-9) & ~(np.abs(p - l) <= 1e-9)
    r_s = max(np.abs(nu[inside]).max(initial=0), (-nu[atl]).max(initial=0), nu[atu].max(initial=0))
    # sign / complementarity of the aggregate multiplier
    r_c = 0.0
    if lam > 1e-9:
        r_c = abs(p.sum() - C)
    elif lam < -1e-9:
        r_c = abs(p.sum() + Crev)
    return max(r_p, r_s, r_c)


def T1():
    worst = dict(sort_vs_bis=0.0, sort_vs_qp=0.0, kkt=0.0)
    cases = {0: 0, 1: 0, -1: 0}
    for trial in range(3000):
        N = int(rng.choice([1, 2, 5, 20, 60, 200]))
        a, l, u, C, Crev = random_instance(N)
        p, lam, case = project(a[None], l[None], u[None], np.array([C]), np.array([Crev]))
        p, lam, case = p[0], lam[0], int(case[0])
        cases[case] += 1
        pb, _ = project_bisect(a, l, u, C, Crev)
        worst["sort_vs_bis"] = max(worst["sort_vs_bis"], np.abs(p - pb).max())
        worst["kkt"] = max(worst["kkt"], kkt_residual(a, l, u, C, Crev, p, lam))
        if N <= 60 and trial % 10 == 0:
            cons = [{"type": "ineq", "fun": lambda x: C - x.sum(), "jac": lambda x: -np.ones_like(x)},
                    {"type": "ineq", "fun": lambda x: x.sum() + Crev, "jac": lambda x: np.ones_like(x)}]
            res = minimize(lambda x: 0.5 * np.sum((x - a) ** 2), np.clip(a, l, u), jac=lambda x: x - a,
                           bounds=list(zip(l, u)), constraints=cons, method="SLSQP",
                           options=dict(ftol=1e-14, maxiter=2000))
            worst["sort_vs_qp"] = max(worst["sort_vs_qp"], np.abs(p - res.x).max())
    # batched call with padding mask: 64 rows of mixed N padded to 200
    Bn, Nmax = 64, 200
    A = np.zeros((Bn, Nmax)); L = np.zeros((Bn, Nmax)); U = np.zeros((Bn, Nmax)); M = np.zeros((Bn, Nmax), bool)
    Cs = np.zeros(Bn); Cr = np.zeros(Bn); ref = []
    for b in range(Bn):
        N = int(rng.integers(1, Nmax + 1))
        a, l, u, C, Crev = random_instance(N)
        A[b, :N], L[b, :N], U[b, :N], M[b, :N], Cs[b], Cr[b] = a, l, u, True, C, Crev
        A[b, N:] = rng.normal(0, 15, Nmax - N)  # garbage in padded slots must be ignored
        ref.append(project_bisect(a, l, u, C, Crev)[0])
    P, _, _ = project(A, L, U, Cs, Cr, mask=M)
    batch_err = max(np.abs(P[b, :len(ref[b])] - ref[b]).max() for b in range(Bn))
    pad_err = np.abs(P[~M]).max()
    print(f"T1 projection: 3000 random instances (N in 1..200), cases inactive/upper/lower = "
          f"{cases[0]}/{cases[1]}/{cases[-1]}")
    print(f"   max |sorted - bisection| = {worst['sort_vs_bis']:.2e}; max |sorted - SLSQP| = "
          f"{worst['sort_vs_qp']:.2e} (300 instances, N<=60); max KKT residual = {worst['kkt']:.2e}")
    print(f"   batched+masked (64 rows padded to 200): max err = {batch_err:.2e}, max |p| on padding = {pad_err:.1e}")


def T1b(n_rows=20000):
    """Review M1: rows with A5 tight to rounding and huge raw actions."""
    res = {}
    for fam in ["tight_upper", "tight_lower", "tight_upper_huge", "huge_general", "hyperplane"]:
        Bn, N = n_rows, 20
        L = rng.uniform(-11, 6, (Bn, N)); U = L + rng.uniform(0, 11, (Bn, N))
        scale = 10.0 ** rng.uniform(4, 8, (Bn, 1)) if "huge" in fam else 15.0
        A = rng.normal(0, 1, (Bn, N)) * scale
        if fam.startswith("tight_upper"):
            Cs = L.sum(1); Cr = -L.sum(1) + rng.uniform(0, 50, Bn)          # sum l == C exactly
            A = np.abs(A) + U                                                 # force upper case
        elif fam == "tight_lower":
            Cr = -U.sum(1); Cs = U.sum(1) + rng.uniform(0, 50, Bn)           # sum u == -C_rev exactly
            A = -np.abs(A) + L
        elif fam == "hyperplane":
            Cs = rng.uniform(L.sum(1), U.sum(1)); Cr = -Cs                   # C == -C_rev
        else:
            Cs = rng.uniform(L.sum(1), U.sum(1) + 20); Cr = rng.uniform(-np.minimum(Cs, U.sum(1)), -L.sum(1) + 20)
        feas = is_feasible(L, U, Cs, Cr)
        L, U, A, Cs, Cr = L[feas], U[feas], A[feas], Cs[feas], Cr[feas]
        P, lam, case = project(A, L, U, Cs, Cr)
        sp = P.sum(1)
        agg_viol = np.maximum(sp - Cs, -Cr - sp).max()
        box_viol = max((L - P).max(), (P - U).max())
        k = min(300, len(A)); err = 0.0
        for b in range(k):
            pb, _ = project_bisect(A[b], L[b], U[b], Cs[b], Cr[b])
            err = max(err, np.abs(P[b] - pb).max() / (1 + np.abs(A[b]).max()))
        res[fam] = (len(A), int((case != 0).sum()), agg_viol, box_viol, err)
    print("T1b review-M1 families (N=20): rows / active / max aggregate violation [kW] / max box violation / "
          "max rel. |sorted - bisection| (300 rows)")
    for fam, r in res.items():
        print(f"   {fam:17s} {r[0]:6d} {r[1]:6d}  {r[2]: .2e}  {r[3]: .2e}  {r[4]:.2e}")
    # NaN guard
    x = np.array([np.nan, np.inf, -np.inf, 3.0]); xs, nb = sanitize(x)
    p, _, _ = project(x[None], np.zeros((1, 4)), np.ones((1, 4)), np.array([10.0]), np.array([10.0]))
    print(f"   NaN/Inf guard: {nb} non-finite entries replaced by 0; projected p = {p[0]}")


def T2():
    worst = 0.0; n = 0; nofree = 0
    while n < 500:
        N = int(rng.choice([2, 5, 20]))
        a, l, u, C, Crev = random_instance(N)
        p, lam, case = project(a[None], l[None], u[None], np.array([C]), np.array([Crev]))
        lam, case = lam[0], int(case[0])
        z = a - lam
        if np.min(np.abs(np.concatenate([z - l, z - u]))) < 1e-3:
            continue  # skip near-kink points
        if case != 0 and min(abs(p[0].sum() - C), abs(p[0].sum() + Crev)) > 1e-9:
            continue
        J = jacobian(a, l, u, lam, case)
        h = 1e-6
        Jfd = np.zeros((N, N))
        for j in range(N):
            e = np.zeros(N); e[j] = h
            pp = project((a + e)[None], l[None], u[None], np.array([C]), np.array([Crev]))[0][0]
            pm = project((a - e)[None], l[None], u[None], np.array([C]), np.array([Crev]))[0][0]
            Jfd[:, j] = (pp - pm) / (2 * h)
        worst = max(worst, np.abs(J - Jfd).max()); n += 1
        nofree += int(case != 0 and np.trace(J) == 0)
    print(f"T2 Jacobian vs finite differences: 500 points, max abs error = {worst:.2e} "
          f"(of which {nofree} with active aggregate and empty free set, J = 0)")


def T3():
    worst = 0.0
    for _ in range(200):
        N = 10
        a, l, u, C, Crev = random_instance(N)
        f = lambda x: 0.5 * np.sum((x - project(x[None], l[None], u[None], np.array([C]), np.array([Crev]))[0][0]) ** 2)
        g = a - project(a[None], l[None], u[None], np.array([C]), np.array([Crev]))[0][0]
        h = 1e-6
        gfd = np.array([(f(a + h * np.eye(N)[j]) - f(a - h * np.eye(N)[j])) / (2 * h) for j in range(N)])
        worst = max(worst, np.abs(g - gfd).max())
    print(f"T3 Moreau gradient of 0.5*dist^2: max |(a - Pi(a)) - finite diff| = {worst:.2e}")


# ---------------------------------------------------------------------------------------------- T4
def T4(n_episodes=60, N=8, T=96, dt_min=15, seed=1, ptr=(15, 45)):
    from ev2gym.models.ev import EV
    specs = json.load(open("/home/claude/work/ext/EV2Gym/ev2gym/data/ev_specs_v2g_enabled2024.json"))
    names = list(specs)
    dt = dt_min / 60.0
    E_MIN = 5.0
    r = np.random.default_rng(seed)
    stats = {"FB-1": 0, "FB-2": 0, "FB-0": 0, "FB-1*": 0}
    stats.update(steps=0, infeasible_steps=0, evs=0, evs_clean=0, clean_miss=0,
                 tr_viol_feasible=0, max_tr_excess=0.0, max_rev_excess=0.0, bound_viol=0, capped=0,
                 max_shortfall=0.0, shortfall_bound_viol=0, cap_amounts=[], tr_viol_any=0)
    for ep in range(n_episodes):
        P_tr = r.uniform(*ptr)  # kW for 8 chargers: from very tight to loose
        load = P_tr * r.uniform(0.0, 0.6, T)
        pv = P_tr * np.clip(np.sin(np.linspace(0, np.pi, T)) * r.uniform(0, 1.2), 0, None)
        Cfw = P_tr - load + pv
        Crev = P_tr + load - pv
        # sessions per charger (one port per charger, gap >= 3 steps, departure <= T-2)
        sessions = [[] for _ in range(N)]
        for i in range(N):
            t = int(r.integers(0, 10))
            while True:
                stay = int(r.integers(4, 40))
                if t + stay > T - 2:
                    break
                sessions[i].append((t, t + stay))  # (a_i, d_i): actions at a..d inclusive
                t = t + stay + 3 + int(r.integers(0, 10))
        evs = [None] * N
        for t in range(T):
            # arrivals
            for i in range(N):
                for (a_i, d_i) in sessions[i]:
                    if a_i == t:
                        name = names[r.integers(len(names))]
                        sp = specs[name]
                        Cap = sp["battery_capacity"]
                        zc, zd = r.uniform(0.85, 1.0), r.uniform(0.85, 1.0)
                        ts = round(0.9 - r.random() / 5, 3)
                        ev = EV(id=0, location=i, battery_capacity_at_arrival=round(r.uniform(E_MIN, 0.8 * Cap), 2),
                                time_of_arrival=a_i, time_of_departure=d_i, battery_capacity=Cap,
                                min_battery_capacity=E_MIN, max_ac_charge_power=sp["max_ac_charge_power"],
                                max_discharge_power=-sp["max_ac_discharge_power"], transition_soc=ts,
                                transition_soc_multiplier=5, charge_efficiency=zc, discharge_efficiency=zd,
                                timescale=dt_min, desired_capacity=Cap)
                        Pnom = zc * min(sp["max_ac_charge_power"], P_CS)
                        Pdis = zd * min(sp["max_ac_discharge_power"], P_CS)
                        phi = (lambda Pn, C_, s_: (lambda E: float(np.minimum(
                            phi_ev2gym(E, C_, Pn, s_, 5, dt), C_))))(Pnom, Cap, ts)
                        tau0 = d_i - a_i + 1
                        E0 = ev.current_capacity
                        Ereq, Btab = cap_request(ev.desired_capacity, E0, tau0, phi, E_MIN, Cap)  # DD2 cap
                        assert E0 >= Btab[tau0]  # A4 by construction
                        stats["capped"] += int(Ereq < ev.desired_capacity)
                        stats["cap_amounts"].append(ev.desired_capacity - Ereq)
                        evs[i] = dict(ev=ev, d=d_i, Ereq=Ereq, B=Btab, phi=phi, Pdis=Pdis, zc=zc, zd=zd,
                                      clean=True, Cap=Cap, Pnom=Pnom, dsum=0.0)
                        stats["evs"] += 1
            occ = np.array([e is not None for e in evs])
            l = np.zeros(N); u = np.zeros(N); pmin = np.zeros(N); lax = np.zeros(N); feas_i = np.ones(N, bool)
            for i in range(N):
                if evs[i] is None:
                    continue
                e = evs[i]; E = e["ev"].current_capacity; tau = e["d"] - t + 1
                pmin[i], u[i], l[i], feas_i[i] = bounds(E, tau, e["B"], E_MIN, e["Cap"], e["Pdis"], e["phi"], dt)
                lax[i] = (E - e["B"][tau]) / (e["Pnom"] * dt)
                if not feas_i[i]:
                    e["clean"] = False
            n_conn = int(occ.sum())
            Ct = Cfw[t] - n_conn * EPS_Q
            Crt = Crev[t]
            a_raw = r.normal(0, 1.5, N) * 11.0  # kW, wide exploration
            stats["steps"] += 1
            feasible = bool(is_feasible(l[None], u[None], Ct, Crt, mask=occ[None])[0])
            if feasible:
                p, _, _ = project(a_raw[None], l[None], u[None], np.array([Ct]), np.array([Crt]), mask=occ[None])
                p = p[0]
            else:
                stats["infeasible_steps"] += 1
                p, fc = fallback(l, u, pmin, Ct, Crt, lax, occ)
                stats[fc] += 1
                for i in range(N):
                    if occ[i] and p[i] < l[i] - 1e-12:
                        evs[i]["clean"] = False
            # bound bookkeeping for Proposition 5: l_tilde = l if feasible_i else u
            for i in range(N):
                if occ[i]:
                    lt = l[i] if feas_i[i] else u[i]
                    evs[i]["dsum"] += dt * max(0.0, lt - p[i])
                    if p[i] < pmin[i] - 1e-9 or p[i] > u[i] + 1e-9:
                        stats["bound_viol"] += 1
            # actuation through EV2Gym's charger logic (round(a,5), dead-band) and EV.step
            act = actuation(p, u, [e["zc"] if e else 1 for e in evs], [e["zd"] if e else 1 for e in evs], occ)
            total = 0.0
            for i in range(N):
                if not occ[i]:
                    continue
                ai = round(act[i], 5)
                amps = ai * 32.0
                if ai < 0 and amps > -0.01:
                    amps = 0.0
                en, _ = evs[i]["ev"].step(amps, 400, phases=3)
                total += en / dt
            if total - Cfw[t] > 1e-6 or -Crev[t] - total > 1e-6:
                stats["tr_viol_any"] += 1
            if feasible:
                ex = total - Cfw[t]; rx = -Crev[t] - total
                stats["max_tr_excess"] = max(stats["max_tr_excess"], ex)
                stats["max_rev_excess"] = max(stats["max_rev_excess"], rx)
                if ex > 1e-6 or rx > 1e-6:
                    stats["tr_viol_feasible"] += 1
            # departures
            for i in range(N):
                if evs[i] is not None and evs[i]["d"] == t:
                    e = evs[i]; Ef = e["ev"].current_capacity
                    short = max(0.0, e["Ereq"] - Ef)
                    stats["max_shortfall"] = max(stats["max_shortfall"], short)
                    if short > e["dsum"] + 1e-6:
                        stats["shortfall_bound_viol"] += 1
                    if e["clean"]:
                        stats["evs_clean"] += 1
                        if Ef < e["Ereq"] - 1e-9:
                            stats["clean_miss"] += 1
                    evs[i] = None
    print(f"T4 closed loop with EV2Gym EV objects: {n_episodes} episodes x {T} steps, N={N}, dt={dt_min} min, "
          f"random raw actions (std 16.5 kW)")
    ca = np.array(stats["cap_amounts"])
    print(f"   transformer limit P_tr ~ U{ptr} kW; DD2 cap: E_des - E_req > 0.01 kWh for {np.mean(ca > 0.01)*100:.1f} % "
          f"of EVs, > 1 kWh for {np.mean(ca > 1)*100:.1f} %, median cap {np.median(ca):.4f} kWh")
    print(f"   steps with any transformer violation (incl. infeasible steps) = {stats['tr_viol_any']}")
    print(f"   steps={stats['steps']}, infeasible steps={stats['infeasible_steps']} "
          f"(FB-1={stats['FB-1']}, FB-2={stats['FB-2']}, FB-0={stats['FB-0']}, FB-1*={stats['FB-1*']}); EVs={stats['evs']}, capped requests={stats['capped']}")
    print(f"   transformer: steps with violation among feasible steps = {stats['tr_viol_feasible']}; "
          f"max (sum p - C) = {stats['max_tr_excess']:.4f} kW, max (-C_rev - sum p) = {stats['max_rev_excess']:.4f} kW")
    print(f"   EVs never touched by an infeasible step: {stats['evs_clean']}, of which missed E_req: {stats['clean_miss']}")
    print(f"   per-EV bound violations (p outside [p_min,u]) = {stats['bound_viol']}; "
          f"max departure shortfall (all EVs) = {stats['max_shortfall']:.3f} kWh; "
          f"Lemma 3 shortfall-bound violations = {stats['shortfall_bound_viol']}")


if __name__ == "__main__":
    T1(); T1b(); T2(); T3()
    T4(dt_min=15, T=96, ptr=(15, 45))
    T4(dt_min=15, T=96, ptr=(40, 90), seed=2)
    T4(dt_min=5, T=288, ptr=(40, 90), seed=3, n_episodes=30)
