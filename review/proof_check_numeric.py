"""proof_check_numeric.py -- independent numerical checks of safety_layer.md (Internal Review A6).

Own code; projection_ref.py is imported ONLY for comparison (Part A).
Run: /home/claude/work/ext/venv/bin/python /home/claude/work/review/proof_check_numeric.py
Output: /home/claude/work/review/proof_check_numeric_results.txt

Parts
  A  projection: own exact solver (direct breakpoint evaluation, no cumulative sums) vs projection_ref.project,
     projection_ref.project_bisect, an LP optimality certificate (scipy linprog/HiGHS) and SLSQP (QP);
     degenerate instances (l=u, all tight, hyperplane slab, ties, N=1, padding, huge actions, infeasible);
     Jacobian (S16) vs finite differences.
  B  closed loop with a simple EV model (general eta, several tapers incl. one violating A1), random and
     adversarial raw actions, A3-type actuation noise (rounding, dead-band, 0.01 kWh ceil), fallback rule;
     checks Theorem 1 (a)-(c), Corollary 1, Lemma 1(a), Lemma 2(a), Lemma 3, conformal Prop. 2(ii).
  C  EV2Gym taper: A1/A1' on EV._charge for random configs, own closed form, Lemma E2 sharpness, P_ev > P_cs.
  D  gradient: score-function estimator with/without masking l=u tokens and fallback steps vs finite differences.
  E  conformal: coverage of the split-conformal lower bound (marginal vs per-group).
"""
import math
import sys
import time

import numpy as np
from scipy.optimize import linprog, minimize

sys.path.insert(0, "/home/claude/work/method")
import projection_ref as ref  # noqa: E402  (comparison only)

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


class Infeasible(Exception):
    pass


# =============================================================================================
# Part A: projection
# =============================================================================================

def my_project(x, l, u, Cup, Clo, tol=1e-12):
    """Euclidean projection onto {l<=p<=u, Clo <= 1'p <= Cup} (Clo = -C_rev).
    Method: evaluate s(lam)=sum clip(x-lam,l,u) DIRECTLY at every distinct breakpoint (O(N*M)), locate the
    crossing, interpolate linearly (s is affine between consecutive breakpoints). Raises Infeasible."""
    x, l, u = (np.asarray(v, float) for v in (x, l, u))
    sc = 1.0 + np.abs(l).sum() + np.abs(u).sum() + abs(Cup) + abs(Clo)
    if np.any(l > u + tol * sc) or l.sum() > Cup + tol * sc or u.sum() < Clo - tol * sc or Clo > Cup + tol * sc:
        raise Infeasible
    s = lambda lam: np.clip(x - lam, l, u).sum()
    s0 = s(0.0)
    if Clo <= s0 <= Cup:
        return np.clip(x, l, u), 0.0, 0
    target, case = (Cup, 1) if s0 > Cup else (Clo, -1)
    bp = np.unique(np.concatenate([x - u, x - l]))
    sv = np.clip(x[None, :] - bp[:, None], l[None, :], u[None, :]).sum(1)   # non-increasing in bp
    le = np.nonzero(sv <= target)[0]
    if len(le) == 0:          # only through rounding when sum(l) == target: p = l
        lam = bp[-1]
    elif le[0] == 0:
        lam = bp[0]
    else:
        k = le[0]
        lam = bp[k - 1] + (sv[k - 1] - target) * (bp[k] - bp[k - 1]) / (sv[k - 1] - sv[k])
    return np.clip(x - lam, l, u), lam, case


def lp_certificate(x, l, u, Cup, Clo, p):
    """max_{q in K} (x-p)'(q-p) via HiGHS; =0 iff p = Pi(x) (given p in K). Returns normalised gap."""
    c = -(x - p)
    A = np.vstack([np.ones(len(x)), -np.ones(len(x))])
    b = np.array([Cup, -Clo])
    res = linprog(c, A_ub=A, b_ub=b, bounds=list(zip(l, u)), method="highs")
    if res.status != 0:
        return np.nan
    val = -res.fun - (x - p) @ p
    return val / (1.0 + np.linalg.norm(x - p) * (1.0 + np.linalg.norm(u - l)))


def in_K(p, l, u, Cup, Clo, tol):
    return (np.all(p >= l - tol) and np.all(p <= u + tol) and p.sum() <= Cup + tol and p.sum() >= Clo - tol)


def gen_instance(rng, kind):
    N = int(rng.integers(1, 120)) if kind not in ("N1",) else 1
    l = rng.uniform(-11, 6, N)
    u = l + rng.uniform(0, 15, N)
    if kind in ("eq_lu_some", "padded"):
        m = rng.random(N) < 0.4
        u[m] = l[m]
    if kind == "eq_lu_all":
        u = l.copy()
    if kind == "padded":
        m = rng.random(N) < 0.3
        l[m] = 0.0
        u[m] = 0.0
    if kind == "ties":
        l = np.round(l)
        u = l + np.round(rng.uniform(0, 6, N))
    sl, su = l.sum(), u.sum()
    Cup = sl + rng.uniform(-0.1, 1.1) * (su - sl) + rng.uniform(0, 5)
    Clo = min(Cup, su) - rng.uniform(0, 1.2) * (su - sl) - rng.uniform(0, 5)
    Cup = max(Cup, sl)
    Clo = min(Clo, su, Cup)
    scale = 16.5
    x = rng.normal(0, scale, N)
    if kind == "ties":
        x = np.round(rng.normal(0, 5, N))
        x[: N // 2] = x[0]
    if kind == "huge":
        x = rng.normal(0, 1e6, N)
    if kind == "tight_upper":      # A5 holds with equality, aggregate upper active -> p = l
        Cup = float(np.sum(l[::-1]))   # summed in another order, as a wrapper would
        Clo = Cup - 50.0
        x = u + rng.uniform(1, 30, N)
    if kind == "tight_upper2":     # same, generic x
        Cup = float(np.sum(l[::-1]))
        Clo = Cup - 50.0
        x = rng.normal(0, 16.5, N) + 10
    if kind == "tight_lower":
        Clo = float(np.sum(u[::-1]))
        Cup = Clo + 50.0
        x = l - rng.uniform(1, 30, N)
    if kind == "hyperplane":
        Cup = Clo = sl + rng.uniform(0, 1) * (su - sl)
    if kind == "padded":
        x[(l == 0) & (u == 0)] = 0.0
    if kind == "infeasible":
        Cup = sl - rng.uniform(0.01, 5)
        Clo = Cup - 10
    return x, l, u, Cup, Clo


def part_A(rng):
    log("=" * 100)
    log("PART A  projection: own solver vs projection_ref.project / project_bisect / LP certificate / SLSQP")
    kinds = ["general"] * 6 + ["eq_lu_some", "eq_lu_all", "padded", "ties", "N1", "huge", "hyperplane",
                               "tight_upper", "tight_upper2", "tight_lower", "infeasible"]
    stats = {}
    n_total = 0
    t0 = time.time()
    for it in range(6800):
        kind = kinds[it % len(kinds)]
        x, l, u, Cup, Clo = gen_instance(rng, kind)
        st = stats.setdefault(kind, dict(n=0, d_ref=0.0, d_bis=0.0, lpgap=0.0, notinK_ref=0, notinK_my=0,
                                          ref_eq_u_fail=0, infeas_detected=0, ref_on_infeas_inK=0, cases=[0, 0, 0],
                                          ref_sum_excess=0.0, d_ref_ok=0.0))
        st["n"] += 1
        n_total += 1
        sc = 1.0 + np.abs(x).max() + np.abs(l).max() + np.abs(u).max()
        try:
            p, lam, case = my_project(x, l, u, Cup, Clo)
        except Infeasible:
            st["infeas_detected"] += 1
            pr, _, _ = ref.project(x[None], l[None], u[None], np.array([Cup]), np.array([-Clo]))
            st["ref_on_infeas_inK"] += int(in_K(pr[0], l, u, Cup, Clo, 1e-9))
            continue
        st["cases"][case + 1] += 1
        pr, lr, cr = ref.project(x[None], l[None], u[None], np.array([Cup]), np.array([-Clo]))
        pr = pr[0]
        pb, _ = ref.project_bisect(x, l, u, Cup, -Clo)
        st["d_ref"] = max(st["d_ref"], float(np.max(np.abs(pr - p))) / sc)
        st["d_bis"] = max(st["d_bis"], float(np.max(np.abs(pb - p))) / sc)
        st["ref_sum_excess"] = max(st["ref_sum_excess"], float(pr.sum() - Cup), float(Clo - pr.sum()))
        if in_K(pr, l, u, Cup, Clo, 1e-9 * sc):
            st["d_ref_ok"] = max(st["d_ref_ok"], float(np.max(np.abs(pr - p))) / sc)
        else:
            st["notinK_ref"] += 1
            if np.allclose(pr, u) and not np.allclose(u, l):
                st["ref_eq_u_fail"] += 1
        if not in_K(p, l, u, Cup, Clo, 1e-9 * sc):
            st["notinK_my"] += 1
        if kind != "huge" and len(x) <= 120:
            g = lp_certificate(x, l, u, Cup, Clo, p)
            st["lpgap"] = max(st["lpgap"], g)
    log(f"  {n_total} instances in {time.time() - t0:.1f}s; errors normalised by 1+max(|x|,|l|,|u|)")
    log(f"  {'kind':13s} {'n':>5s} {'inact/up/low':>14s} {'|my-ref|':>9s} {'(ref in K)':>10s} {'|my-bis|':>9s} {'LPgap':>9s} "
        f"{'ref!inK':>7s} {'ref=u':>6s} {'my!inK':>6s} {'max ref sum excess':>18s} {'infeas':>6s}")
    for k, st in stats.items():
        c = st["cases"]
        log(f"  {k:13s} {st['n']:5d} {c[1]:4d}/{c[2]:4d}/{c[0]:4d} {st['d_ref']:9.1e} {st['d_ref_ok']:10.1e} {st['d_bis']:9.1e} "
            f"{st['lpgap']:9.1e} {st['notinK_ref']:7d} {st['ref_eq_u_fail']:6d} {st['notinK_my']:6d} "
            f"{st['ref_sum_excess']:18.2e} {st['infeas_detected']:6d}")
    st = stats["infeasible"]
    log(f"  infeasible instances: detected by own solver {st['infeas_detected']}/{st['n']}; "
        f"projection_ref.project silently returns a point in K for {st['ref_on_infeas_inK']} (expected 0: "
        "it has no infeasibility check, caller must check A5)")

    # deterministic reproduction of the rounding failure in the sorted-breakpoint code
    log("  -- targeted test: A5 tight to rounding (sum(l) == C_tilde in floating point), aggregate upper active")
    fails, worst = 0, 0.0
    n_t = 20000
    for _ in range(n_t):
        N = int(rng.integers(2, 60))
        l = rng.uniform(-11, 8, N)
        u = l + rng.uniform(0, 12, N)
        x = rng.normal(5, 16.5, N)
        Cup = float(np.sum(l[rng.permutation(N)]))
        Clo = Cup - 100.0
        if np.clip(x, l, u).sum() <= Cup:
            continue
        pr, lr, cr = ref.project(x[None], l[None], u[None], np.array([Cup]), np.array([-Clo]))
        exc = pr[0].sum() - Cup
        if exc > 1e-6:
            fails += 1
            worst = max(worst, exc)
    log(f"     projection_ref.project: {fails}/{n_t} rows with sum(p) > C_tilde + 1e-6 (worst excess {worst:.2f} kW)")
    # SLSQP QP comparison on small instances
    d_qp = 0.0
    n_qp = 0
    for _ in range(400):
        x, l, u, Cup, Clo = gen_instance(rng, "general")
        if len(x) > 25:
            continue
        p, _, _ = my_project(x, l, u, Cup, Clo)
        cons = [dict(type="ineq", fun=lambda q, Cup=Cup: Cup - q.sum(), jac=lambda q: -np.ones_like(q)),
                dict(type="ineq", fun=lambda q, Clo=Clo: q.sum() - Clo, jac=lambda q: np.ones_like(q))]
        r = minimize(lambda q: 0.5 * np.sum((q - x) ** 2), np.clip(x, l, u), jac=lambda q: q - x,
                     bounds=list(zip(l, u)), constraints=cons, method="SLSQP",
                     options=dict(ftol=1e-14, maxiter=500))
        d_qp = max(d_qp, float(np.max(np.abs(r.x - p))))
        n_qp += 1
    log(f"  SLSQP (QP) on {n_qp} instances with N<=25: max |p_my - p_SLSQP| = {d_qp:.2e} kW")

    # Jacobian (S16) vs finite differences, own projection
    maxerr, nJ0, nF1, n = 0.0, 0, 0, 0
    for it in range(800):
        kind = ["general", "eq_lu_some", "ties"][it % 3]
        x, l, u, Cup, Clo = gen_instance(rng, kind)
        if kind == "ties":
            x = x + rng.normal(0, 1e-3, len(x))
        if len(x) > 60:
            continue
        p, lam, case = my_project(x, l, u, Cup, Clo)
        z = x - lam
        if np.min(np.abs(np.concatenate([z - l, z - u]))) < 1e-4:
            continue  # too close to a kink
        F = (z > l) & (z < u)
        D = np.diag(F.astype(float))
        if case != 0 and F.sum() > 0:
            J = D - np.outer(F, F) / F.sum()
        elif case != 0:
            J = np.zeros_like(D)
            nJ0 += 1
        else:
            J = D
        nF1 += int(case != 0 and F.sum() == 1)
        h = 1e-6
        Jfd = np.zeros_like(J)
        for j in range(len(x)):
            e = np.zeros(len(x))
            e[j] = h
            Jfd[:, j] = (my_project(x + e, l, u, Cup, Clo)[0] - my_project(x - e, l, u, Cup, Clo)[0]) / (2 * h)
        maxerr = max(maxerr, float(np.max(np.abs(J - Jfd))))
        n += 1
    log(f"  Jacobian (S16) vs central FD on {n} points ({nJ0} with active aggregate and empty free set, "
        f"{nF1} with |F|=1): max abs error = {maxerr:.2e}")
    return stats


# =============================================================================================
# Part B: closed loop with a simple EV model
# =============================================================================================

def ceil2(a):
    return math.ceil(a * 100 - 1e-9) / 100.0


class EVm:
    """Simple EV: E+ = E + dt*g(p), g(p)=etac p+ - p-/etad; max power Pbar(E) = min(Ptap(E), (Ecap-E)/(etac dt))."""

    def __init__(self, rng, cfg, dt):
        self.cfg = cfg
        self.dt = dt
        self.Ecap = float(rng.choice([46.3, 57.5, 58.0, 64.0, 64.8, 77.0]))
        self.Emin = 5.0
        self.Pnom = float(rng.choice([11.0, 7.4]))
        self.Pdis = self.Pnom * float(rng.uniform(0.85, 1.0))
        tp = cfg["taper"]
        if cfg["lossy"]:
            self.ec, self.ed = float(rng.uniform(0.85, 0.97)), float(rng.uniform(0.85, 0.97))
        else:
            self.ec = self.ed = 1.0
        self.str = float(rng.uniform(0.7, 0.9))
        if tp == "linear":   # Ptap linear from Pnom at s_tr*Ecap to 0 at Ecap; check eta dt L <= 1
            L = self.Pnom / ((1 - self.str) * self.Ecap)
            self.A1_suff = self.ec * dt * L <= 1.0
        if tp == "steep":    # drop from Pnom to 0.05 Pnom over a band w << eta dt Pnom  (A1 violated)
            self.Ek = float(rng.uniform(0.45, 0.8)) * self.Ecap
            self.w = 0.3
        self.m = 5.0

    def Ptap(self, E):
        tp = self.cfg["taper"]
        if tp == "none":
            return self.Pnom
        if tp == "linear":
            s0 = self.str * self.Ecap
            return self.Pnom * min(1.0, max(0.0, (self.Ecap - E) / (self.Ecap - s0)))
        if tp == "steep":
            if E <= self.Ek:
                return self.Pnom
            if E >= self.Ek + self.w:
                return 0.05 * self.Pnom
            return self.Pnom * (1 - 0.95 * (E - self.Ek) / self.w)
        if tp == "ev2gym":   # (S5) with g = id: Pbar = (phi - E)/dt
            return (self.phi_ev2gym(E) - E) / self.dt
        raise ValueError

    def phi_ev2gym(self, E):
        C = self.Ecap
        s = E / C
        d = self.Pnom * self.dt / C
        st, m = self.str, self.m
        if s + d <= st:
            v = s + d
        elif s < st:
            v = 1 - (1 - st) * math.exp(-m * (s + d - st) / (1 - st))
        else:
            v = 1 - (1 - s) * math.exp(-m * d / (1 - st))
        return C * min(v, s + d, 1.0)

    def Pbar(self, E):
        return max(0.0, min(self.Ptap(E), (self.Ecap - E) / (self.ec * self.dt)))

    def phi(self, E):
        return E + self.dt * self.ec * self.Pbar(E)

    def g(self, p):
        return self.ec * p if p >= 0 else p / self.ed

    def ginv(self, y):
        return y / self.ec if y >= 0 else self.ed * y

    def pmin(self, E):
        return -min(self.Pdis, self.ed * max(0.0, E - self.Emin) / self.dt)

    def reach(self, E, k):
        for _ in range(k):
            E = self.phi(E)
        return E

    def thresholds(self, Ereq, K, iters=50):
        """Own implementation of (S9): bisection, keeps the feasible (upper) end."""
        B = [Ereq]
        for k in range(1, K + 1):
            tgt = B[-1]
            if self.phi(self.Emin) >= tgt:
                B.append(self.Emin)
                continue
            lo, hi = self.Emin, max(B[-1], self.Emin)
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                if self.phi(mid) >= tgt:
                    hi = mid
                else:
                    lo = mid
            B.append(hi)
        return B

    def cap(self, Edes, Earr, K):
        B = self.thresholds(Edes, K)
        if Earr >= B[K]:
            return Edes, B
        y, step = min(Edes, self.reach(Earr, K) - 1e-6), 1e-6
        while True:
            y = max(y, min(Earr, Edes))
            B = self.thresholds(y, K)
            if Earr >= B[K]:
                return y, B
            y -= step
            step *= 10


def my_fallback(l, u, pmin, chi, Ct, Crev):
    n = len(l)
    if u.sum() < -Crev:
        return u.copy(), "F2"
    p = pmin.copy()
    b = Ct - p.sum()
    if b < 0:
        return p, "F0"
    order = sorted(range(n), key=lambda i: (chi[i], -(l[i] - pmin[i]), i))
    for i in order:
        add = min(l[i] - pmin[i], b)
        p[i] += add
        b -= add
        if b <= 0:
            break
    return p, "F1"


def actuate(rng, p, u, mode):
    """Realised (reported) grid power and whether the step was 'active' (for the ceil)."""
    if mode == "exact":
        return p, p != 0
    if u > 0 and p >= u - 1e-3:
        return u, True
    if p > 0:
        return p + rng.uniform(0, 2.217e-4), True
    if p < 0:
        c = min(0.0, p + rng.uniform(0, 2.217e-4))
        if abs(c) < 0.006928:
            return 0.0, False
        return c, True
    return 0.0, False


def run_episode(rng, cfg, ep_stats, ev_stats):
    dt = cfg["dt"]
    T = cfg["T"]
    N = int(rng.integers(3, 13))
    eps_q = 0.0 if cfg["act"] == "exact" else 0.008
    tight_margin = 0.0 if cfg.get("no_tightening") else eps_q
    Ptr = N * rng.uniform(2.0, 8.0)
    pv_peak = rng.uniform(0, 1.6) * Ptr
    inf_base = rng.uniform(0, 0.6) * Ptr
    amode = cfg.get("amode") or rng.choice(["gauss", "max", "min", "mixed", "heavy"])
    slots = [None] * N
    for t in range(T):
        # arrivals
        for i in range(N):
            if slots[i] is None and t <= T - 4 and rng.random() < 0.12:
                stay = int(rng.integers(1, 40))
                d = min(t + stay, T - 2)
                ev = EVm(rng, cfg, dt)
                Earr = float(rng.uniform(ev.Emin, 0.85 * ev.Ecap))
                Edes = ev.Ecap if rng.random() < 0.6 else 0.8 * ev.Ecap
                if cfg.get("conformal"):
                    if rng.random() < 0.85:
                        dlo = max(t, d - int(rng.integers(0, 6)))
                        covered = True
                    else:
                        dlo = d + int(rng.integers(1, 6))
                        covered = False
                else:
                    dlo, covered = d, True
                K = dlo - t + 1
                Ereq, B = ev.cap(Edes, Earr, K)
                ev.__dict__.update(E=Earr, d=d, dlo=dlo, covered=covered, Ereq=Ereq, B=B, tarr=t,
                                   allA5=True, allpl=True, bound=0.0, lem1a=0, invbreak=0)
                slots[i] = ev
        occ = [i for i in range(N) if slots[i] is not None]
        if not occ:
            continue
        evs = [slots[i] for i in occ]
        n = len(evs)
        # bounds
        pmin = np.array([ev.pmin(ev.E) for ev in evs])
        u = np.array([ev.Pbar(ev.E) for ev in evs])
        l = np.empty(n)
        chi = np.empty(n)
        feas_ind = np.empty(n, bool)
        for k, ev in enumerate(evs):
            tau = max(1, ev.dlo - t + 1)
            fe = ev.E >= ev.B[tau] - 1e-9          # float tolerance (as in projection_ref.bounds)
            if ev.E < ev.B[tau] - 1e-9:
                ev.invbreak += 1
            feas_ind[k] = fe
            if fe:
                lr = max(pmin[k], ev.ginv((ev.B[tau - 1] - ev.E) / dt))
                if lr > u[k] + 1e-9:
                    ev.lem1a += 1
                    ep_stats["lem1a_viol"] += 1
                l[k] = min(lr, u[k])
            else:
                l[k] = u[k]
            chi[k] = (ev.E - ev.B[tau]) / (ev.ec * ev.Pnom * dt)
        hour = (t * dt) % 24
        Ppv = pv_peak * max(0.0, math.sin(math.pi * (hour - 6) / 12)) if 6 < hour < 18 else 0.0
        Pinf = inf_base * (1 + 0.5 * math.sin(2 * math.pi * hour / 24)) * rng.uniform(0.8, 1.2)
        Pinf = min(Pinf, Ptr)
        C = Ptr - Pinf + Ppv
        Crev = Ptr + Pinf - Ppv
        Ct = C - n * tight_margin
        # raw action
        if amode == "gauss":
            x = rng.normal(0, 16.5, n)
        elif amode == "max":
            x = np.full(n, 1e4)
        elif amode == "min":
            x = np.full(n, -1e4)
        elif amode == "heavy":
            x = 10 * rng.standard_cauchy(n)
        else:
            x = rng.choice([-1e4, 0.0, 1e4], n) + rng.normal(0, 5, n)
        A5 = (l.sum() <= Ct) and (u.sum() >= -Crev) and (Ct >= -Crev)
        ep_stats["steps"] += 1
        if A5:
            p, lam, case = my_project(x, l, u, Ct, -Crev)
            fcase = None
        else:
            p, fcase = my_fallback(l, u, pmin, chi, Ct, Crev)
            ep_stats[fcase] += 1
        # (b) per-step physical limits
        if np.any(p < pmin - 1e-9) or np.any(p > u + 1e-9):
            ep_stats["b_viol"] += 1
        # actuation and energy update
        pt = np.empty(n)
        for k, ev in enumerate(evs):
            pt[k], active = actuate(rng, p[k], u[k], cfg["act"])
            Enew = ev.E + dt * ev.g(pt[k])
            if cfg["act"] != "exact" and active:
                Enew = min(ceil2(Enew), ev.Ecap)
            if Enew < ev.E + dt * ev.g(p[k]) - 1e-12:
                ep_stats["A3_viol"] += 1
            # bookkeeping for Cor. 1 / Lemma 3
            ltil = l[k]
            ev.bound += dt * max(0.0, ev.g(ltil) - ev.g(p[k]))
            if p[k] < l[k] - 1e-9:
                ev.allpl = False
            if not A5:
                ev.allA5 = False
            ev.E = Enew
        S = pt.sum()
        viol = max(0.0, S - C) + max(0.0, -Crev - S)
        if A5:
            ep_stats["A5_steps"] += 1
            if viol > 1e-9:
                ep_stats["a_viol_A5"] += 1
                ep_stats["a_viol_A5_max"] = max(ep_stats["a_viol_A5_max"], viol)
        elif fcase == "F1":
            if viol > 1e-9:
                ep_stats["F1_viol"] += 1
            ep_stats["fleet_bound"] += dt * (l.sum() - Ct)
        elif fcase == "F0":
            ep_stats["fleet_bound"] += dt * (l - pmin).sum()
        if viol > 1e-9:
            ep_stats["viol_any"] += 1
        # departures
        for k, i in enumerate(occ):
            ev = slots[i]
            if ev.d == t:
                sh = max(0.0, ev.Ereq - ev.E)
                ev_stats.append(dict(sh=sh, allA5=ev.allA5, allpl=ev.allpl, bound=ev.bound, covered=ev.covered,
                                     lem1a=ev.lem1a, invbreak=ev.invbreak,
                                     A1p=(cfg["taper"] != "steep")))
                ep_stats["total_sh"] += sh
                slots[i] = None


def part_B(rng):
    log("=" * 100)
    log("PART B  closed loop, simple EV model, random/adversarial raw actions (Theorem 1, Cor. 1, Lemmas 1-3)")
    base = dict(dt=0.25, T=96, act="ev2gym", lossy=False)
    cfgs = [
        ("ev2gym-taper g=id, A3 noise", dict(base, taper="ev2gym")),
        ("no taper, lossy eta, A3 noise", dict(base, taper="none", lossy=True)),
        ("linear taper (A1), lossy, exact", dict(base, taper="linear", lossy=True, act="exact")),
        ("linear taper (A1), lossy, A3", dict(base, taper="linear", lossy=True)),
        ("ev2gym-taper, dt=5min", dict(base, taper="ev2gym", dt=1 / 12, T=200)),
        ("ev2gym-taper, NO tightening", dict(base, taper="ev2gym", no_tightening=True, amode="max")),
        ("ev2gym-taper, conformal tau_lo", dict(base, taper="ev2gym", conformal=True)),
        ("STEEP taper (A1 violated)", dict(base, taper="steep", lossy=False)),
        ("STEEP taper (A1 violated), gauss", dict(base, taper="steep", lossy=False, amode="gauss")),
    ]
    n_eps = 120
    for name, cfg in cfgs:
        t0 = time.time()
        ep = dict(steps=0, A5_steps=0, F0=0, F1=0, F2=0, b_viol=0, a_viol_A5=0, a_viol_A5_max=0.0, F1_viol=0,
                  viol_any=0, A3_viol=0, lem1a_viol=0, fleet_bound=0.0, total_sh=0.0)
        evs = []
        fleet_fail = 0
        for e in range(n_eps):
            before_b, before_sh = ep["fleet_bound"], ep["total_sh"]
            run_episode(rng, cfg, ep, evs)
            if not cfg["lossy"] and cfg["taper"] != "steep" and not cfg.get("conformal"):
                if ep["total_sh"] - before_sh > ep["fleet_bound"] - before_b + 1e-6:
                    fleet_fail += 1
        nev = len(evs)
        cov = [v for v in evs if v["covered"]]
        thmA5 = [v for v in cov if v["allA5"]]
        cor1 = [v for v in cov if v["allpl"]]
        miss_thm = sum(v["sh"] > 1e-9 for v in thmA5)
        miss_cor = sum(v["sh"] > 1e-9 for v in cor1)
        lem3 = sum(v["sh"] > v["bound"] + 1e-9 for v in cov)
        inv = sum(v["invbreak"] > 0 for v in thmA5)
        log(f"-- {name}: {n_eps} episodes, {ep['steps']} steps, {nev} EVs ({time.time() - t0:.0f}s)")
        log(f"   A5 steps {ep['A5_steps']}, fallback F1/F0/F2 = {ep['F1']}/{ep['F0']}/{ep['F2']}; "
            f"A3 (energy >= E+dt g(p)) violations {ep['A3_viol']}")
        log(f"   Thm(a) realised transformer violation on A5 steps: {ep['a_viol_A5']} steps "
            f"(max {ep['a_viol_A5_max']:.4f} kW); Lemma 2(a) F1 steps with violation: {ep['F1_viol']}; "
            f"steps with any violation: {ep['viol_any']}")
        log(f"   Thm(b) commanded p outside [pmin,u]: {ep['b_viol']} steps; Lemma 1(a) l>u for 'feasible' EV: "
            f"{ep['lem1a_viol']} EV-steps")
        log(f"   Thm(c) EVs whose stay had only A5 steps: {len(thmA5)}, missed E_req (>1e-9 kWh): {miss_thm}; "
            f"invariant broken for {inv}")
        log(f"   Cor.1  EVs with p>=l at every step: {len(cor1)}, missed: {miss_cor}")
        log(f"   Lemma 3 per-EV bound violated: {lem3}/{len(cov)}"
            + (f"; fleet bound violated in {fleet_fail} episodes"
               if not cfg["lossy"] and cfg["taper"] != "steep" and not cfg.get("conformal") else ""))
        if cfg.get("conformal"):
            unc = [v for v in evs if not v["covered"]]
            log(f"   conformal: miscovered EVs {len(unc)}, of which missed {sum(v['sh'] > 1e-9 for v in unc)}")
        if cfg["taper"] == "steep":
            worst = max([v["sh"] for v in thmA5] + [0])
            log(f"   A1 violated: worst shortfall among 'guaranteed' EVs = {worst:.3f} kWh")


# =============================================================================================
# Part C: EV2Gym taper (A1, A1', closed form, Lemma E2 sharpness)
# =============================================================================================

def part_C(rng):
    log("=" * 100)
    log("PART C  EV2Gym EV._charge: A1/A1', own closed form, Lemma E2 sharpness, P_ev > P_cs")
    sys.path.insert(0, "/home/claude/work/ext/EV2Gym")
    from ev2gym.models.ev import EV
    V, I, PH = 400.0, 32.0, 3
    Pcs = I * V * math.sqrt(PH) / 1000

    def mk(C, Pev, zeta, st, m, dtmin, E0):
        return EV(id=0, location=0, battery_capacity_at_arrival=E0, time_of_arrival=0, time_of_departure=5,
                  battery_capacity=C, min_battery_capacity=5.0, max_ac_charge_power=Pev, max_discharge_power=-Pev,
                  transition_soc=st, transition_soc_multiplier=m, charge_efficiency=zeta,
                  discharge_efficiency=zeta, timescale=dtmin)

    def charge(C, Pev, zeta, st, m, dtmin, E0, amps):
        ev = mk(C, Pev, zeta, st, m, dtmin, E0)
        if amps <= 0:
            return E0
        ev._charge(amps, V, phases=PH)
        return ev.current_capacity

    def phi_own(C, Pev, zeta, st, m, dth, E):
        """Own derivation from ev.py at full current; handles Pev > Pcs (shifted transition)."""
        s = E / C
        D = zeta * Pev * dth / C                 # max_dsoc
        x = min(zeta * Pcs * dth / C, D)         # pilot_dsoc at full current, capped
        if st == 1:
            return C * min(1.0, s + x)
        sx = st + (x - D) / D * (st - 1)
        if s < sx:
            if (sx - s) / x >= 1:
                v = s + x
            else:
                v = 1 + math.exp(m * (x + s - sx) / (sx - 1)) * (sx - 1)
        else:
            v = 1 + math.exp(m * x / (sx - 1)) * (s - 1)
        return C * min(v, s + x)

    def phi_S5(C, Pnom, st, m, dth, E):
        s = E / C
        d = Pnom * dth / C
        if st >= 1:
            return C * min(1, s + d)
        if s + d <= st:
            v = s + d
        elif s < st:
            v = 1 - (1 - st) * math.exp(-m * (s + d - st) / (1 - st))
        else:
            v = 1 - (1 - s) * math.exp(-m * d / (1 - st))
        return C * min(v, s + d)

    specs = [(57.5, 11), (64.8, 11), (58, 11), (64, 11), (46.3, 7.4), (77, 11)]
    worst = dict(a1=0.0, lip=-1.0, own=0.0, s5=0.0)
    n = 0
    for _ in range(300):
        C, Pev = specs[rng.integers(len(specs))]
        zeta = float(rng.uniform(0.85, 1.0))
        st = 1.0 if rng.random() < 0.1 else float(rng.uniform(0.7, 0.9))
        m = float(rng.choice([0.5, 1, 2, 5, 10]))
        dtmin = int(rng.choice([5, 15, 30, 60]))
        dth = dtmin / 60
        E = np.linspace(5.0, C, 300)
        ph = np.array([charge(C, Pev, zeta, st, m, dtmin, e, I) for e in E])
        inc = np.diff(ph)
        worst["a1"] = min(worst["a1"], float(inc.min()))
        worst["lip"] = max(worst["lip"], float((inc - np.diff(E)).max()))
        po = np.array([phi_own(C, Pev, zeta, st, m, dth, e) for e in E])
        ps5 = np.array([phi_S5(C, zeta * min(Pcs, Pev), st, m, dth, e) for e in E])
        worst["own"] = max(worst["own"], float(np.abs(po - ph).max()))
        worst["s5"] = max(worst["s5"], float(np.abs(ps5 - ph).max()))
        n += 1
    log(f"  {n} random configs (m in {{0.5,1,2,5,10}}, dt in {{5,15,30,60}} min, zeta in [0.85,1]):")
    log(f"    min increment of phi on grid = {worst['a1']:.2e} kWh (A1 holds iff >= 0)")
    log(f"    max [phi(E')-phi(E)-(E'-E)] = {worst['lip']:.2e} kWh (A1' holds iff <= 0)")
    log(f"    own closed form vs EV._charge: {worst['own']:.2e} kWh; (S5) vs EV._charge: {worst['s5']:.2e} kWh")
    # P_ev > P_cs
    C, Pev, zeta, st, m, dtmin = 77.0, 40.0, 1.0, 0.8, 5.0, 15
    E = np.linspace(5, C, 400)
    ph = np.array([charge(C, Pev, zeta, st, m, dtmin, e, I) for e in E])
    ps5 = np.array([phi_S5(C, zeta * min(Pcs, Pev), st, m, dtmin / 60, e) for e in E])
    po = np.array([phi_own(C, Pev, zeta, st, m, dtmin / 60, e) for e in E])
    log(f"  hypothetical EV with P_ev=40 kW > P_cs=22.17 kW: |(S5)-EV._charge| = {np.abs(ps5 - ph).max():.3f} kWh, "
        f"|own-EV._charge| = {np.abs(po - ph).max():.1e} kWh  -> (S5) needs P_ev <= P_cs (true for all 8 models)")
    # Lemma E2 sharpness: exact delivery of sub-max commands as a function of kappa
    log("  Lemma E2 (exact delivery of p in (0,Pbar(E)]), m=5, zeta=1:")
    for (C, Pev, dtmin, st) in [(57.5, 11, 15, 0.9), (57.5, 11, 30, 0.9), (57.5, 11, 60, 0.9), (46.3, 7.4, 60, 0.7),
                                (57.5, 11, 60, 0.75)]:
        m = 5.0
        dth = dtmin / 60
        kappa = m * (Pev * dth / C) / (1 - st)
        err = 0.0
        for e in np.linspace(5, C - 1e-6, 120):
            pb = (charge(C, Pev, 1.0, st, m, dtmin, e, I) - e) / dth
            if pb <= 1e-9:
                continue
            for p in np.linspace(0, pb, 30)[1:]:
                amps = p * 1000 / (V * math.sqrt(PH))
                out = charge(C, Pev, 1.0, st, m, dtmin, e, amps)
                err = max(err, abs(out - (e + p * dth)))
        log(f"    C={C}, P_ev={Pev}, dt={dtmin} min, s_tr={st}: kappa={kappa:.3f} (kappa*=4.965) -> "
            f"max delivery error {err:.2e} kWh")


# =============================================================================================
# Part D: gradient estimators
# =============================================================================================

def part_D(rng):
    log("=" * 100)
    log("PART D  score-function gradient: full vs masked (l=u token; fallback step) vs finite differences")
    sig = 4.0
    l = np.array([0.0, 3.0])
    u = np.array([10.0, 3.0])        # token 2 has l = u
    Cup, Clo = 6.0, -100.0

    def proj_batch(X):
        # closed form for this instance (token 2 fixed at 3, so p1 = clip(x1, 0, Cup-3)); checked below
        return np.stack([np.clip(X[:, 0], 0.0, Cup - 3.0), np.full(len(X), 3.0)], 1)

    Xc = rng.normal(0, 8, (2000, 2))
    chk = max(float(np.abs(my_project(xx, l, u, Cup, Clo)[0] - proj_batch(xx[None])[0]).max()) for xx in Xc)
    log(f"  closed form of Pi for the bandit instance vs own projection: {chk:.1e}")

    def R(P):
        return 2.0 * P[:, 0] - 0.15 * P[:, 0] ** 2 + P[:, 1]

    theta = 1.5
    M = 400000
    eps = rng.normal(size=(M, 2))

    def J(th, e):
        mu = np.array([th, 2 * th])
        return R(proj_batch(mu + sig * e)).mean()

    # one-step bandit: true gradient by common-random-number central differences (reparameterised)
    # (pathwise FD is the exact derivative of the CRN estimator; fine since Pi is Lipschitz)
    Mfd = 100000
    h = 1e-3
    gfd = (J(theta + h, eps[:Mfd]) - J(theta - h, eps[:Mfd])) / (2 * h)
    from scipy.integrate import quad
    gtrue = quad(lambda a: (2 - 0.3 * a) * math.exp(-0.5 * ((a - theta) / sig) ** 2) / (sig * math.sqrt(2 * math.pi)),
                 0.0, Cup - 3.0)[0]
    log(f"  exact gradient (quadrature) {gtrue:.5f}; CRN finite difference {gfd:.5f}")
    mu = np.array([theta, 2 * theta])
    A = mu + sig * eps
    Rv = R(proj_batch(A))
    s1 = (A[:, 0] - mu[0]) / sig ** 2 * 1.0
    s2 = (A[:, 1] - mu[1]) / sig ** 2 * 2.0
    full = Rv * (s1 + s2)
    mask = Rv * s1
    log(f"  bandit (aggregate active, token 2 with l=u): exact gradient {gtrue:.4f}; "
        f"full score {full.mean():.4f} +- {3 * full.std() / math.sqrt(M):.4f}; "
        f"masked {mask.mean():.4f} +- {3 * mask.std() / math.sqrt(M):.4f}; variance ratio masked/full = "
        f"{mask.var() / full.var():.3f}")
    # two-step: step 0 is a fallback step (p independent of a0), step 1 normal; return = R(p1)
    a0 = theta + sig * rng.normal(size=M)
    a1 = A
    G = Rv  # return depends only on step-1 action
    sc0 = (a0 - theta) / sig ** 2
    full2 = G * (sc0 + s1 + s2)
    mask2 = G * (s1 + s2)
    log(f"  2-step (step 0 = fallback): full {full2.mean():.4f} +- {3 * full2.std() / math.sqrt(M):.4f}; "
        f"fallback-masked {mask2.mean():.4f} +- {3 * mask2.std() / math.sqrt(M):.4f}; variance ratio "
        f"{mask2.var() / full2.var():.3f}")


# =============================================================================================
# Part E: conformal coverage
# =============================================================================================

def part_E(rng):
    log("=" * 100)
    log("PART E  split-conformal lower bound on d: marginal vs per-group coverage")

    def episode():
        k = int(rng.integers(1, 9))
        z = rng.normal(0, 3)                       # episode effect (couples sessions of one episode)
        tarr = rng.integers(0, 60, k)
        grp = rng.integers(0, 3, k)
        stay = 8 + 0.2 * tarr + 6 * grp + z + rng.normal(0, 2 + 4 * (grp == 2), k)
        d = tarr + np.maximum(1, np.round(stay)).astype(int)
        return tarr, grp, d

    # crude predictor fitted on separate data: linear least squares on (1, tarr, grp) then shifted down
    X, Y = [], []
    for _ in range(2000):
        tarr, grp, d = episode()
        for a, g_, dd in zip(tarr, grp, d):
            X.append([1, a, g_])
            Y.append(dd)
    w = np.linalg.lstsq(np.array(X, float), np.array(Y, float), rcond=None)[0]
    qhat = lambda a, g_: w[0] + w[1] * a + w[2] * g_ - 3.0
    for alpha in (0.05, 0.1, 0.2):
        covs, covs_g2, pooled_num, pooled_den = [], [], 0, 0
        for trial in range(2000):
            sc = []
            for _ in range(300):
                tarr, grp, d = episode()
                j = rng.integers(len(d))
                sc.append(qhat(tarr[j], grp[j]) - d[j])
            sc = np.sort(sc)
            k = math.ceil((len(sc) + 1) * (1 - alpha))
            Q = sc[k - 1] if k <= len(sc) else np.inf
            tarr, grp, d = episode()
            j = rng.integers(len(d))
            dlo = max(tarr[j], math.ceil(qhat(tarr[j], grp[j]) - Q))
            covs.append(d[j] >= dlo)
            dl = np.maximum(tarr, np.ceil(qhat(tarr, grp) - Q))
            pooled_num += int((d >= dl).sum())
            pooled_den += len(d)
            m2 = grp == 2
            if m2.any():
                covs_g2.append(float((d[m2] >= dl[m2]).mean()))
        log(f"  alpha={alpha}: P(d >= dlo) for a uniformly chosen EV = {np.mean(covs):.3f} (>= {1 - alpha}); "
            f"pooled over all EVs {pooled_num / pooled_den:.3f}; group-2 EVs only {np.mean(covs_g2):.3f}")


if __name__ == "__main__":
    rng = np.random.default_rng(20261022)
    t0 = time.time()
    part_A(rng)
    part_C(rng)
    part_D(rng)
    part_E(rng)
    part_B(rng)
    log(f"total time {time.time() - t0:.0f}s")
    with open("/home/claude/work/review/proof_check_numeric_results.txt", "w") as f:
        f.write("\n".join(OUT) + "\n")
