"""Admission control on arrival (DD9, DD20; safety_layer.md Sec. 7.4; algorithm.md Alg. 3 step 1c).

When EV j arrives at t0 (after the DD2 cap), a feasibility LP over all connected EVs i (including j) and
steps t = t0..d_i checks whether every requirement can still be met under the forecast capacity profile:

    E_i(t+1) = E_i(t) + dt p_i(t),  E_min <= E_i(t+1) <= E_cap,  E_i(d_i+1) >= E_req_i,
    -P_dis <= p_i(t) <= P_nom,  p_i(t) <= rho_i (E_cap - E_i(t))  (t > t0, linear taper minorant),
    p_i(t0) in [p_min(E_i(t0)), Pbar(E_i(t0))],
    -C_rev_hat(t) <= sum_i p_i(t) <= C_hat(t) - n(t0) eps_q.

If infeasible, the largest feasible E_req_j in [min(E_arr, E_req_j), E_req_j] is searched by bisection with
12 iterations; requests of admitted EVs are never reduced; if even the lower end is infeasible E_req_j is set
to min(E_arr, E_req_j) and "admission did not restore feasibility" is logged. The caller re-certifies the
reduced request with ``cap_request`` (A4).

Implementation note (exactness, speed): feasibility is monotone in E_req_j, i.e. y is feasible iff
y <= y_max := max E_j(d_j+1) over the LP with j's target removed. One LP computes y_max; the 12-step bisection
is then evaluated with the exact predicate (y <= y_max + tol). This returns the same value as 12 separate LP
solves, at the cost of one LP.
"""
from __future__ import annotations

import numpy as np

from .bounds import EVParams, phi_scalar
from .lp import LPBuilder, add_ev_dynamics


class AdmissionController:
    def __init__(self, eps_q: float = 0.008, iters: int = 12, tol: float = 1e-6,
                 existing_targets: str = "lp_reachable"):
        """existing_targets: 'lp_reachable' (DD25, default: targets of connected EVs capped at what the LP model can
        reach individually) or 'admitted' (literal S 7.4: admitted E_req of connected EVs; IMPLEMENTATION_NOTES I-3)."""
        if existing_targets not in ("admitted", "lp_reachable"):
            raise ValueError(existing_targets)
        self.eps_q = eps_q
        self.iters = iters
        self.tol = tol
        self.existing_targets = existing_targets
        self.n_lp = 0
        self.lp_time = 0.0

    @staticmethod
    def _lp_reach(E0: float, K: int, p: EVParams, rho: float, dt_h: float) -> float:
        """Maximum energy after K steps in the LP model (exact first step, linear minorant afterwards)."""
        E = phi_scalar(E0, p.cap, p.pnom, p.s_tr, p.m, dt_h)
        for _ in range(K - 1):
            E = min(p.cap, E + dt_h * min(p.pnom, rho * (p.cap - E)))
        return E

    def _targets(self, recs, t0, dt_h):
        """Final-energy targets of the existing EVs (None for the arriving EV)."""
        out = []
        for E0, d, Ereq, p, rh in recs:
            if Ereq is None:
                out.append(None)
                continue
            tgt = Ereq
            if self.existing_targets == "lp_reachable":
                tgt = min(Ereq, self._lp_reach(E0, max(1, d - t0 + 1), p, rh, dt_h) - 1e-6)
            out.append(tgt)
        return out

    def solve_records(self, t0: int, recs: list, Chat: np.ndarray, Crhat: np.ndarray, n_now: int, dt_h: float,
                      target_new: float | None = None, plan: bool = False):
        """Admission LP over records (E0, d, E_req or None for the arriving EV, params, rho).

        plan=False, target_new=None: maximise the arriving EV's final energy (returns (y_max | None, None)).
        plan=False, target_new=y: feasibility with E_new(d+1) >= y (returns (y | None, None)).
        plan=True: elastic nominal plan (DD31): targets (arriving EV: ``target_new``) and the aggregate band get
        slacks with weight 1e5 EUR-free units; the objective maximises sum_k sum_i min(E_i(k), target_i) (reach the
        targets as early as possible). Returns (total slack, {record index: E trajectory E(t0..d+1)})."""
        H = len(Chat)
        lp = LPBuilder()
        agg_cols = [[] for _ in range(H)]
        obj_col = None
        tg = self._targets(recs, t0, dt_h)
        cost_idx, cost_val = [], []
        traj = {}
        M = 1e5
        for k, ((E0, d, Ereq, p, rh), tgt) in enumerate(zip(recs, tg)):
            K = d - t0 + 1
            if K < 1:
                continue
            E0 = float(min(max(E0, p.e_min), p.cap))
            pmin0 = -min(p.pdis, max(0.0, E0 - p.e_min) / dt_h)
            u0 = (phi_scalar(E0, p.cap, p.pnom, p.s_tr, p.m, dt_h) - E0) / dt_h
            plo = np.full(K, -p.pdis)
            phi_ = np.full(K, p.pnom)
            plo[0], phi_[0] = pmin0, max(u0, pmin0)
            pc = lp.add_vars(K, plo, phi_)
            elo = np.full(K, p.e_min)
            ehi = np.full(K, p.cap)
            is_new = Ereq is None
            if is_new:
                tgt = target_new
            if not plan and tgt is not None:
                elo[-1] = max(elo[-1], tgt)
            elo = np.minimum(elo, ehi)
            if not plan and elo[-1] > ehi[-1] + 1e-12:
                return None, None
            ec = lp.add_vars(K, elo, ehi)        # ec[k] = E(t0+k+1)
            add_ev_dynamics(lp, pc, ec, E0, dt_h, rho=rh, cap=p.cap)
            for kk in range(min(K, H)):
                agg_cols[kk].append(pc[kk])
            if is_new:
                obj_col = ec[-1]
            if plan:
                traj[k] = (E0, ec)
                if tgt is not None:
                    tgt = float(min(tgt, p.cap))
                    z = lp.add_vars(1, 0.0, np.inf)[0]
                    lp.add_row([ec[-1], z], [1.0, 1.0], tgt, np.inf)
                    cost_idx.append(z)
                    cost_val.append(-M)
                    mk = lp.add_vars(K, -np.inf, tgt)          # m_k <= min(E(k), target)
                    lp.add_rows_coo(K, np.repeat(np.arange(K), 2), np.stack([mk, ec], 1).ravel(),
                                    np.tile([1.0, -1.0], K), -np.inf, 0.0)
                    cost_idx += mk.tolist()
                    cost_val += [1.0] * K
        for tau in range(H):
            if agg_cols[tau]:
                hi = Chat[tau] - n_now * self.eps_q
                if plan:
                    o = lp.add_vars(2, 0.0, np.inf)
                    lp.add_row(list(agg_cols[tau]) + [o[0], o[1]], [1.0] * len(agg_cols[tau]) + [-1.0, 1.0],
                               -Crhat[tau], hi)
                    cost_idx += o.tolist()
                    cost_val += [-M, -M]
                else:
                    lp.add_row(agg_cols[tau], 1.0, -Crhat[tau], hi)
        cost = np.zeros(lp.n)
        sense = "min"
        if plan:
            np.add.at(cost, np.asarray(cost_idx, int), np.asarray(cost_val, float))
            sense = "max"
        elif target_new is None and obj_col is not None:
            cost[obj_col] = 1.0
            sense = "max"
        res = lp.solve(cost, sense)
        self.n_lp += 1
        self.lp_time += res.time_s
        if not res.optimal:
            return None, None
        if plan:
            slack = float(-sum(res.x[i] * v for i, v in zip(cost_idx, cost_val) if v < 0) / M)
            return slack, {k: np.concatenate([[E0], res.x[ec]]) for k, (E0, ec) in traj.items()}
        if target_new is None:
            return (float(res.x[obj_col]) if obj_col is not None else float(recs[-1][0])), None
        return float(target_new), None

    def y_max(self, t0: int, E_arr: float, d_new: int, params: EVParams, rho: float, existing: list,
              Chat: np.ndarray, Crhat: np.ndarray, n_now: int, dt_h: float, target_new: float | None = None):
        """Solve the admission LP for connected EV slots ``existing`` (objects with .ev.current_capacity, .d_ctrl,
        .E_req, .params, .rho) plus the arriving EV. With target_new None: maximise E_new(d_new+1) (returns y_max
        or None if infeasible). Otherwise: pure feasibility with E_new(d+1) >= target_new (returns target_new or
        None)."""
        recs = [(float(s.ev.current_capacity), int(s.d_ctrl), float(s.E_req), s.params, float(s.rho))
                for s in existing]
        recs.append((float(E_arr), int(d_new), None, params, float(rho)))
        return self.solve_records(t0, recs, Chat, Crhat, n_now, dt_h, target_new)[0]

    def admit_records(self, t0, recs, E_req, Chat, Crhat, n_now, dt_h):
        """``admit`` for record tuples (last record = the arriving EV with E_req None)."""
        ymax = self.solve_records(t0, recs, Chat, Crhat, n_now, dt_h)[0]
        return self._bisect(ymax, recs[-1][0], E_req)

    def admit(self, t0, E_arr, d_new, E_req, params, rho, existing, Chat, Crhat, n_now, dt_h):
        """Return (E_req_new, info). info: {'reduced': bool, 'not_restored': bool, 'y_max': float|None}."""
        ymax = self.y_max(t0, E_arr, d_new, params, rho, existing, Chat, Crhat, n_now, dt_h)
        return self._bisect(ymax, E_arr, E_req)

    def _bisect(self, ymax, E_arr, E_req):
        feas = (lambda y: ymax is not None and y <= ymax + self.tol)  # noqa: E731  exact monotone predicate
        if feas(E_req):
            return E_req, {"reduced": False, "not_restored": False, "y_max": ymax}
        lo = min(E_arr, E_req)
        if not feas(lo):
            return lo, {"reduced": True, "not_restored": True, "y_max": ymax}
        hi = E_req
        for _ in range(self.iters):          # bisection for the largest feasible value (keeps the feasible end)
            mid = 0.5 * (lo + hi)
            if feas(mid):
                lo = mid
            else:
                hi = mid
        return lo, {"reduced": True, "not_restored": False, "y_max": ymax}
