"""Economic MPC baselines (algorithm.md Alg. 5 row 6).

* :class:`EMPCGurobi` ('empc'): adapter to EV2Gym's ``eMPC_V2G`` (``baselines/mpc/eMPC.py``, Gurobi, horizon 10)
  as published. Needs ``gurobipy``; the restricted pip licence (2000 variables/constraints) limits it to
  N <= 25 at horizon 10; larger N needs a full/academic licence or the HiGHS port. Two adapter-level fixes
  (EV2Gym's files are not modified; IMPLEMENTATION_NOTES I-6): (i) the published ``get_action`` crashes under
  numpy 2 (solution vector of shape (n, 1)); our subclass repeats the identical formulation with a 1-D solution
  vector; (ii) ``eMPC`` reads the efficiency of the first EV from the dict table in PERCENT (e.g. 90) and uses
  it as a fraction; we divide it by 100 after construction.
* :class:`EMPCHighs` ('empc_highs'): our HiGHS receding-horizon LP implementing the same idea in reported-power
  units (formulation F5): horizon 10 (or 24 h, 'empc_highs_24h'), true prices, EV2Gym's own load/PV forecasts (unbiased, 10 % noise in our
  scenarios) and the realised values at t, future arrivals within the horizon read from ``env.EVs_profiles``
  (the same information advantage as eMPC, flagged), transformer constraint in both directions, departure
  requirement of EVs departing within the horizon. Differences to eMPC (documented in IMPLEMENTATION_NOTES I-6):
  charge/discharge variables without binaries (simultaneous p+ and p- is never strictly profitable for
  c_dis <= c_ch; throughput cost (DD39) included); per-EV efficiencies and limits (eMPC uses those of the first EV); the first step
  uses the exact per-step bounds [p_min, u] (taper); the departure requirement is the wrapper's E_req
  (DD2-capped/admitted) for connected EVs and min(E_des, E_arr + K P_nom dt) for future arrivals; it is elastic
  (slack penalised with 1000 EUR/kWh) instead of returning zero actions when infeasible.

Both return the raw action x (kW); the raw variant uses the physical clip, the '_safe' variant the safety layer.
"""
from __future__ import annotations

import numpy as np

from ..constants import P_CS
from ..safety.bounds import ev_params_from_ev2gym, phi_scalar
from ..safety.lp import LPBuilder, add_ev_dynamics_split
from .heuristics import Controller


def _empc_np2_class():
    """Subclass of EV2Gym's eMPC_V2G whose ``get_action`` is the published formulation (eMPC.py:32-163, MIT
    licence, Diaz-Londono & Orfanoudakis) with one numpy-2 fix: the solution vector is 1-D, so assigning
    ``a[i] / P`` to a scalar slot no longer raises "setting an array element with a sequence"."""
    import gurobipy as gp
    from gurobipy import GRB
    from ev2gym.baselines.mpc.eMPC import eMPC_V2G

    class eMPC_V2G_np2(eMPC_V2G):
        def get_action(self, env):
            t = env.current_step
            self.update_tr_power(t)
            self.reconstruct_state(t)
            self.calculate_XF_V2G(t)
            self.v2g_station_models(t)
            self.calculate_InequalityConstraints(t)
            self.set_power_limits_V2G(t)
            f = []
            for i in range(self.control_horizon):
                for j in range(self.n_ports):
                    f.append(self.T * self.ch_prices[t + i])
                    f.append(-self.T * self.disch_prices[t + i])
            f = np.array(f).reshape(-1)
            nb, n, h = self.nb, self.n_ports, self.control_horizon
            model = gp.Model("optimization_model")
            u = model.addMVar(nb * h, vtype=GRB.CONTINUOUS, name="u")
            Zbin = model.addMVar(n * h, vtype=GRB.BINARY, name="Zbin")
            model.addConstr((self.AU @ u) <= self.bU, name="constr1")
            model.addConstrs((0 <= u[j] for j in range(0, nb * h, 2)), name="constr3a")
            model.addConstrs((u[j] <= self.UB[j] * Zbin[j // 2] for j in range(0, nb * h, 2)), name="constr3b")
            model.addConstrs((0 <= u[j] for j in range(1, nb * h, 2)), name="constr4a")
            model.addConstrs((u[j] <= self.UB[j] * (1 - Zbin[j // 2]) for j in range(1, nb * h, 2)), name="constr4b")
            for tr_index in range(self.number_of_transformers):
                for i in range(self.control_horizon):
                    model.addConstr((gp.quicksum((u[j] - u[j + 1]) for index, j in enumerate(
                        range(i * self.nb, (i + 1) * self.nb, 2)) if self.cs_transformers[index] == tr_index)
                        + self.tr_loads[tr_index, i] + self.tr_pv[tr_index, i] <= self.tr_power_limit[tr_index, i]),
                        name=f"constr5_{tr_index}_t{i}")
            for tr_index in range(self.number_of_transformers):
                for i in range(self.control_horizon):
                    model.addConstr((gp.quicksum((u[j] - u[j + 1]) for index, j in enumerate(
                        range(i * self.nb, (i + 1) * self.nb, 2)) if self.cs_transformers[index] == tr_index)
                        + self.tr_loads[tr_index, i] + self.tr_pv[tr_index, i]
                        >= -self.tr_power_limit[tr_index, :].max()), name=f"constr5_{tr_index}_t{i}")
            model.setObjective(f @ u, GRB.MINIMIZE)
            model.setParam("OutputFlag", self.output_flag)
            model.params.NonConvex = 2
            if self.MIPGap is not None:
                model.params.MIPGap = self.MIPGap
            model.params.TimeLimit = self.time_limit
            model.optimize()
            self.total_exec_time += model.Runtime
            if model.status in (GRB.Status.INF_OR_UNBD, GRB.Status.INFEASIBLE):
                self.n_infeasible = getattr(self, "n_infeasible", 0) + 1
                return np.zeros(self.n_ports)          # published behaviour: default (zero) actions
            a = np.array([float(np.ravel(u[i].X)[0]) for i in range(2 * self.n_ports)])   # numpy-2 fix
            actions = np.zeros(self.n_ports)
            e = 0.001
            for i in range(0, 2 * self.n_ports, 2):
                if a[i] > e and a[i + 1] > e:
                    raise ValueError(f"Charging and discharging at the same time {i} {a[i]} {a[i + 1]}")
                elif a[i] > e:
                    actions[i // 2] = a[i] / self.max_ch_power[i // 2]
                elif a[i + 1] > e:
                    actions[i // 2] = -a[i + 1] / abs(self.max_disch_power[i // 2])
            return actions

    return eMPC_V2G_np2


class EMPCGurobi(Controller):
    name = "empc"
    uses_future_info = True

    def __init__(self, horizon: int = 10):
        self.horizon = horizon
        self.mpc = None
        self.n_fail = 0

    def reset(self, env):
        try:
            cls = _empc_np2_class()
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("EV2Gym eMPC needs gurobipy; use method empc_highs instead") from e
        self.mpc = cls(env.env, control_horizon=self.horizon, verbose=False)
        if self.mpc.ch_eff > 1.0:            # dict efficiency tables are in percent (adapter fix, see module doc)
            self.mpc.ch_eff /= 100.0
            self.mpc.disch_eff /= 100.0

    def act(self, obs, env):
        try:
            a = np.asarray(self.mpc.get_action(env.env), float)
        except Exception as e:  # gurobipy.GurobiError (licence size limit) and similar
            raise RuntimeError(f"EV2Gym eMPC (Gurobi) failed: {e}. The restricted pip licence supports only "
                               f"N <= 25 at horizon 10; use method empc_highs.") from e
        zeta = np.array([s.params.zeta_c if (s and a_i > 0) else (s.params.zeta_d if s else 1.0)
                         for s, a_i in zip(env.slots, a)])
        return np.where(obs.occ, a * P_CS * zeta, 0.0)


class EMPCHighs(Controller):
    """HiGHS receding-horizon economic MPC (see module doc). ``horizon`` in steps (10 = eMPC) or ``horizon_hours``
    (e.g. 24 h: 'empc_highs_24h', DD33(7)); horizons are truncated at the episode end."""

    name = "empc_highs"
    uses_future_info = True

    def __init__(self, horizon: int = 10, slack_penalty: float = 1000.0, horizon_hours: float | None = None):
        self.horizon = int(horizon)
        self.horizon_hours = horizon_hours
        self.M = float(slack_penalty)
        self.n_fail = 0
        self.solve_time = 0.0

    def act(self, obs, env):
        e = env.env
        t, T, dt = obs.t, env.T, env.dt_h
        Hs = int(round(self.horizon_hours / dt)) if self.horizon_hours else self.horizon
        H = min(Hs, T - t)
        tr = e.transformers[0]
        Ptr = np.asarray(tr.max_power, float)[t:t + H]
        L = np.asarray(tr.inflexible_load_forecast, float)[t:t + H].copy()
        PV = -np.asarray(tr.pv_generation_forecast, float)[t:t + H].copy()
        L[0], PV[0] = float(tr.inflexible_load[t]), -float(tr.solar_power[t])
        C, Crev = Ptr - L + PV, Ptr + L - PV
        cch = -np.asarray(e.charge_prices[0], float)[t:t + H]
        cdis = np.asarray(e.discharge_prices[0], float)[t:t + H]
        thr = float(getattr(env.scfg, "throughput_cost", 0.0) or 0.0)
        lp = LPBuilder()
        cost_idx, cost_val = [], []
        agg = [[] for _ in range(H)]
        first = {}

        def add_session(E0, d, t0, Ereq, p, plo0, phi0, key=None):
            K = min(d, t + H - 1) - t0 + 1
            if K < 1:
                return
            # net power p = p+ - p- with p+ <= P_nom, p- <= P_dis; first step within the exact [p_min, u]
            hi_p = np.full(K, p.pnom)
            hi_m = np.full(K, p.pdis)
            hi_p[0] = max(phi0, 0.0)
            hi_m[0] = max(-plo0, 0.0)
            pp = lp.add_vars(K, 0.0, hi_p)
            pm = lp.add_vars(K, 0.0, hi_m)
            ec = lp.add_vars(K, p.e_min, p.cap)
            add_ev_dynamics_split(lp, pp, pm, ec, E0, dt)
            if d <= t + H - 1:                         # departs within the horizon: elastic requirement
                z = lp.add_vars(1, 0.0, np.inf)[0]
                lp.add_row([ec[-1], z], [1.0, 1.0], min(Ereq, p.cap), np.inf)
                cost_idx.append(z)
                cost_val.append(-self.M)
            for k in range(K):
                tau = t0 - t + k
                agg[tau].append((pp[k], pm[k]))
                cost_idx += [pp[k], pm[k]]
                cost_val += [-dt * (cch[tau] + thr), dt * (cdis[tau] - thr)]
            if key is not None:
                first[key] = (pp[0], pm[0])

        for i in np.flatnonzero(obs.occ):
            s = env.slots[i]
            add_session(float(obs.E[i]), int(s.d), t, float(s.E_req), s.params, float(obs.pmin[i]),
                        float(obs.u[i]), key=int(i))
        for ev in e.EVs_profiles:                      # future arrivals inside the horizon (information advantage)
            ta = int(ev.time_of_arrival)
            if t < ta <= t + H - 1:
                p = ev_params_from_ev2gym(ev, dt)
                E0 = float(ev.battery_capacity_at_arrival)
                d = int(ev.time_of_departure)
                Ereq = min(float(ev.desired_capacity), E0 + (d - ta + 1) * p.pnom * dt)
                pmin0 = -min(p.pdis, max(0.0, E0 - p.e_min) / dt)
                u0 = (phi_scalar(E0, p.cap, p.pnom, p.s_tr, p.m, dt) - E0) / dt
                add_session(E0, d, ta, Ereq, p, pmin0, u0)
        for tau in range(H):
            if agg[tau]:
                cols = [a for a, _ in agg[tau]] + [b for _, b in agg[tau]]
                vals = [1.0] * len(agg[tau]) + [-1.0] * len(agg[tau])
                lp.add_row(cols, vals, -Crev[tau], C[tau])
        x = np.zeros(len(obs.u))
        if lp.n == 0:
            return x
        cost = np.zeros(lp.n)
        np.add.at(cost, np.asarray(cost_idx, int), np.asarray(cost_val, float))
        res = lp.solve(cost, "max")
        self.solve_time += res.time_s
        if not res.optimal:
            self.n_fail += 1
            return x
        for i, (a, b) in first.items():
            x[i] = res.x[a] - res.x[b]
        return x
