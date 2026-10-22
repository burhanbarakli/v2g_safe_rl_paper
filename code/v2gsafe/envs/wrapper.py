"""SafeV2GEnv: EV2Gym wrapper owning all safety-layer state (algorithm.md Alg. 1 and Alg. 3).

Per decision step t (= ``env.current_step``):

1. :meth:`SafeV2GEnv.observe` - processes arrivals (DD2 cap with certified threshold tables, optional conformal
   departure bound (Alg. 4) and the admitted request of the controller-independent admission pre-pass
   (DD9/DD20/DD31, :mod:`v2gsafe.safety.admission_prepass`)), computes the per-step bounds (Alg. 1.1),
   the A5 flag and the token observation (formulation.md Sec. 4.2: 12 charger features + 38 global features).
2. :meth:`SafeV2GEnv.step` - maps a raw action x (kW, one entry per charger) to the commanded power p:
   safety layer on: projection (Alg. 1.2) on A5 steps, fallback rule (Alg. 1.3) otherwise;
   safety layer off: per-EV physical clip p = clip(x, p_min, u) (Alg. 5). Then the actuation map with upward
   rounding (Alg. 1.4), ``env.step``, reported powers, transformer violation (DD7 metric), departures and
   shortfall, the profit reward r (F9) and the penalty reward r_pen (Alg. 5 / DD22).

The closed loop never reads ``env.EVs_profiles`` (future arrivals); the admission pre-pass does, but its rule is
causal (each decision uses only information up to that arrival, I-22) and controller-independent. EV2Gym is not patched; the wrapper only sets
instance attributes (``tr_seed`` per replay, scenario keys of ``env.config`` between episodes).
"""
from __future__ import annotations

import math
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..constants import EPS_NUM, EPS_Q, F_C, F_G, H_F, P_REF, E_REF, C_REF
from ..eval.metrics import EpisodeAccumulator
from ..safety.actuation import actuation
from ..safety.bounds import EVParams, cap_request, ev_params_from_ev2gym, phi_ev2gym, step_bounds, \
    taper_minorant_rho
from ..safety.fallback import FB_CODES, fallback
from ..safety.projection import GuardCounters, is_feasible, project, sanitize
from .forecasts import NoisyForecaster
from .scenarios import ScenarioConfig

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")


def _null_state(env):  # EV2Gym state function: our wrapper builds its own observation
    return np.zeros(1)


def _profit_reward(env, total_costs, user_satisfaction_list, *args):  # = V2G_costs_simple: r(t) of (F9)
    return total_costs


def make_ev2gym_env(scfg: ScenarioConfig, seed: int = 0):
    """Construct an EV2Gym env for a scenario (about 3.5 s: price CSV parsing)."""
    from ev2gym.models.ev2gym_env import EV2Gym

    path = scfg.write_ev2gym_yaml()
    try:
        env = EV2Gym(config_file=path, seed=seed, empty_ports_at_end_of_simulation=True,
                     state_function=_null_state, reward_function=_profit_reward, verbose=False)
    finally:
        os.remove(path)
    if any(cs.n_ports != 1 for cs in env.charging_stations):
        raise ValueError("one port per charger required (DD5)")
    return env


@dataclass
class WrapperOptions:
    safety: bool = True                 # projection + fallback (True) or physical clip only (False)
    lower_bound: bool = True            # False: the layer uses l_i = p_i^min (attn_proj_nolb, DD33(1))
    admission: bool = True              # admission control (DD9/DD20) as a replay pre-pass, all methods (DD31)
    admission_cache: str | None = None  # directory of cached admission tables (None: in-memory only)
    admission_existing_targets: str = "lp_reachable"   # DD25 (IMPLEMENTATION_NOTES I-3)
    eps_q: float = EPS_Q                # DD16
    p_ref: float = P_REF
    forecast_rel_std: float = 0.1       # DD6
    w_E: float = 10.0                   # DD22
    w_P: float = 10.0
    penalty_dense: bool = False         # Alg. 5 rule 5 ablation
    log_steps: bool = False
    conformal_provider: Callable | None = None   # ScenarioConfig -> ConformalBound | None


@dataclass
class Slot:
    """Per-charger state of a connected EV."""

    ev: object
    params: EVParams
    t_arr: int
    d: int                  # true departure step (logging only in conformal mode)
    d_ctrl: int             # departure step used by the controller (d, or the conformal bound d_lo)
    E_arr: float
    E_des: float
    E_req_dd2: float
    E_req: float            # admitted requirement (DD2 cap and admission control)
    E_req_shared: float     # shared admitted requirement of the replay (DD43(1); = E_req unless noadm/conformal)
    B: np.ndarray           # certified threshold table for E_req, length d_ctrl - t_arr + 2
    rho: float
    touched: bool = False
    stay_infeasible: bool = False   # the stay contains >= 1 infeasible (non-A5) step (DD43(2))


@dataclass
class Observation:
    t: int
    tokens: np.ndarray      # (N, 12) float32
    glob: np.ndarray        # (38,) float32
    occ: np.ndarray         # (N,) bool
    infl: np.ndarray        # (N,) bool: tokens whose action can influence p (DD21)
    E: np.ndarray
    l: np.ndarray           # lower bound used by the layer (= l_dep, or p_min without the departure bound)
    u: np.ndarray
    pmin: np.ndarray
    chi: np.ndarray
    lg2v: np.ndarray
    feas_i: np.ndarray
    C: float
    Ctil: float
    Crev: float
    feasible: bool          # A5 (Proposition 1(i)) for the layer's bounds
    l_dep: np.ndarray | None = None   # departure-aware lower bound (S11), for features and metrics


@dataclass
class StepOut:
    reward: float           # r(t), EUR (F9)
    reward_pen: float       # r_pen(t), EUR (Alg. 5)
    done: bool
    info: dict = field(default_factory=dict)


class SafeV2GEnv:
    """EV2Gym wrapper with the safety layer. One station with N chargers, one port each."""

    def __init__(self, scfg: ScenarioConfig, options: WrapperOptions | None = None, env_seed: int = 0):
        self.scfg = scfg
        self.opt = options or WrapperOptions()
        self.env = make_ev2gym_env(scfg, seed=env_seed)
        self.N = int(scfg.N)
        self.dt_h = scfg.dt_h
        self.T = scfg.horizon
        self.N_day = int(round(24 / self.dt_h))
        self.adm_table: dict | None = None
        self.forecaster = NoisyForecaster(self.opt.forecast_rel_std, H_F)
        self.counters = GuardCounters()
        self.slots: list[Slot | None] = [None] * self.N
        self._obs: Observation | None = None
        self.cp = None
        self.seed = None
        self.acc = EpisodeAccumulator(self.dt_h, self.N)
        self.step_log: list[dict] = []
        self.charger_log: list[dict] = []

    # ---------------------------------------------------------------------------------- scenario
    _IMMUTABLE = ("N", "dt_min", "horizon")

    def _apply_scenario(self, scfg: ScenarioConfig) -> None:
        for k in self._IMMUTABLE:
            if getattr(scfg, k) != getattr(self.scfg, k):
                raise ValueError(f"scenario field {k} differs from the env's ({getattr(scfg, k)} vs "
                                 f"{getattr(self.scfg, k)}); build a new SafeV2GEnv")
        c = scfg.ev2gym_config()
        cfg = self.env.config
        for k in ("hour", "minute", "simulation_days", "scenario", "spawn_multiplier", "random_day",
                  "random_hour", "discharge_price_factor"):
            cfg[k] = c[k]
        for k in ("transformer", "inflexible_loads", "solar_power", "demand_response", "ev"):
            cfg[k] = c[k]
        self.env.scenario = scfg.profile
        self.scfg = scfg

    # ------------------------------------------------------------------------------------- reset
    def reset(self, seed: int, scenario: ScenarioConfig | None = None) -> Observation:
        """Reset to the episode defined by (scenario, seed). Deterministic: same pair, same episode."""
        if scenario is not None:
            self._apply_scenario(scenario)
        seed = int(seed) % (2**32)
        self.seed = seed
        self.env.tr_seed = seed            # loads/PV randomness per replay (EV2Gym keeps its first seed otherwise)
        self.env.reset(seed=seed)
        self.env.done = False
        self.forecaster.reseed(np.random.SeedSequence([seed, 7]).generate_state(1)[0])
        self.slots = [None] * self.N
        self.acc = EpisodeAccumulator(self.dt_h, self.N)
        self.counters = GuardCounters()
        self.step_log, self.charger_log = [], []
        self.cp = self.opt.conformal_provider(self.scfg) if self.opt.conformal_provider else None
        # DD31/DD43(1): the shared admission table of the replay (true departures, controller-independent) is computed
        # for every pass; with admission on it gives the requirement, otherwise (noadm) it is the reference of the
        # *_shared shortfall columns. Conformal passes take the shared requirement, re-certified with their d_lo.
        from ..safety.admission_prepass import get_table
        self.adm_table = get_table(self.env, self.scfg, seed, self.opt.eps_q, self.opt.admission_existing_targets,
                                   self.opt.forecast_rel_std, None, self.opt.admission_cache)
        self._obs = None
        return self.observe()

    @property
    def t(self) -> int:
        return int(self.env.current_step)

    @property
    def done(self) -> bool:
        return self.env.current_step >= self.T

    def datetime_at(self, t: int):
        import datetime
        return self.env.sim_starting_date + datetime.timedelta(minutes=self.scfg.dt_min * t)

    # ----------------------------------------------------------------------------------- arrivals
    def _exogenous(self, t: int):
        tr = self.env.transformers[0]
        P_tr = float(tr.max_power[t])
        P_inf = float(tr.inflexible_load[t])
        P_pv = -float(tr.solar_power[t])
        return tr, P_tr, P_inf, P_pv

    def _arrive(self, i: int, ev, t: int) -> None:
        params = ev_params_from_ev2gym(ev, self.dt_h, self.env.charging_stations[i].get_max_power())
        E_arr = float(ev.current_capacity)
        E_des = float(ev.desired_capacity)
        d = int(ev.time_of_departure)
        if self.cp is not None:
            d_ctrl = int(self.cp.lower_bound(ev, t, self.datetime_at(t), self.scfg))
            d_ctrl = max(d_ctrl, t)
        else:
            d_ctrl = d
        K = d_ctrl - t + 1
        E_req, B = cap_request(E_des, E_arr, K, params.phi, params.e_min, params.cap)
        E_req_dd2 = E_req
        rho = taper_minorant_rho(params)
        adm_kwh, adm_failed = 0.0, False
        e = self.adm_table.get((i, t)) if self.adm_table is not None else None
        if self.adm_table is not None and (e is None or abs(e["E_arr"] - E_arr) > 1e-9 or int(e["d"]) != d):
            raise RuntimeError(f"admission table mismatch at charger {i}, step {t} (replay {self.seed})")
        E_shared = float(e["E_req"]) if e is not None else E_req_dd2
        if self.opt.admission and e is not None:
            y = E_shared
            if y < E_req - 1e-12:
                E_req, B = cap_request(y, E_arr, K, params.phi, params.e_min, params.cap)  # re-certify A4 (DD20)
                adm_kwh = E_req_dd2 - E_req
            adm_failed = bool(e.get("not_restored", False))
        # DD43(1): the requirement the method is given is also written into EV2Gym's EV object (spawned copy, not the
        # profile), so baselines that read EV2Gym state (EV-GNN, Gurobi eMPC) target the same requirement
        ev.desired_capacity = float(E_req)
        assert E_arr >= B[K] - 1e-12, "A4 violated after cap_request"
        self.slots[i] = Slot(ev=ev, params=params, t_arr=t, d=d, d_ctrl=d_ctrl, E_arr=E_arr, E_des=E_des,
                             E_req_dd2=E_req_dd2, E_req=E_req, E_req_shared=E_shared, B=B, rho=rho)
        self.acc.on_arrival(E_des - E_req_dd2, adm_kwh, adm_failed, params.kappa)

    # ------------------------------------------------------------------------------------ observe
    def observe(self) -> Observation:
        """Process arrivals at t and compute bounds, A5 flag and the token observation (Alg. 1.1, F 4.2)."""
        env = self.env
        t = self.t
        if t >= self.T:
            raise RuntimeError("episode finished; call reset()")
        tr, P_tr, P_inf, P_pv = self._exogenous(t)
        fc_inf = self.forecaster.forecast(np.asarray(tr.inflexible_load, float), t, 0)
        fc_pv = self.forecaster.forecast(-np.asarray(tr.solar_power, float), t, 1)
        for i, cs in enumerate(env.charging_stations):
            ev = cs.evs_connected[0]
            s = self.slots[i]
            if ev is None:
                if s is not None:          # defensive: departure not seen in step()
                    self.slots[i] = None
                continue
            if s is None or s.ev is not ev:
                self._arrive(i, ev, t)

        N = self.N
        occ = np.array([s is not None for s in self.slots])
        idx = np.flatnonzero(occ)
        z = np.zeros(N)
        E, l, u, pmin, chi, lg2v = z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), z.copy()
        feas_i = np.ones(N, bool)
        tau_feat = z.copy()
        cap, pnom, pdis = z.copy(), z.copy(), z.copy()
        Ereq = z.copy()
        if idx.size:
            sl = [self.slots[i] for i in idx]
            Ei = np.array([float(s.ev.current_capacity) for s in sl])
            tau = np.array([max(1, s.d_ctrl - t + 1) for s in sl])
            Bt = np.array([s.B[k] for s, k in zip(sl, tau)])
            Bt1 = np.array([s.B[k - 1] for s, k in zip(sl, tau)])
            ci = np.array([s.params.cap for s in sl])
            pn = np.array([s.params.pnom for s in sl])
            pdi = np.array([s.params.pdis for s in sl])
            emin = np.array([s.params.e_min for s in sl])
            phiE = phi_ev2gym(Ei, ci, pn, np.array([s.params.s_tr for s in sl]),
                              np.array([s.params.m for s in sl]), self.dt_h)
            b = step_bounds(Ei, Bt, Bt1, emin, pdi, pn, phiE, self.dt_h)
            E[idx], l[idx], u[idx], pmin[idx] = Ei, b["l"], b["u"], b["pmin"]
            chi[idx], lg2v[idx], feas_i[idx] = b["chi"], b["lg2v"], b["feas_i"]
            tau_feat[idx] = tau
            cap[idx], pnom[idx], pdis[idx] = ci, pn, pdi
            Ereq[idx] = [s.E_req for s in sl]
        n = int(occ.sum())
        C = P_tr - P_inf + P_pv
        Crev = P_tr + P_inf - P_pv
        Ctil = C - n * self.opt.eps_q
        l_dep = l
        l_lay = l if self.opt.lower_bound else pmin.copy()
        feasible = bool(is_feasible(l_lay[None], u[None], Ctil, Crev, mask=occ[None])[0])
        if not feasible:                                   # DD43(2): EVs whose stay contains an infeasible step
            for i in idx:
                if not self.slots[i].stay_infeasible:
                    self.slots[i].stay_infeasible = True
                    self.acc.n_stay_infeasible += 1
        lo = l_lay if self.opt.safety else pmin
        infl = occ & (u - lo > EPS_NUM)

        # ---- charger tokens (formulation.md Sec. 4.2)
        Nd = self.N_day
        tok = np.zeros((N, F_C), np.float64)
        if idx.size:
            tok[idx, 0] = 1.0
            tok[idx, 1] = E[idx] / E_REF
            tok[idx, 2] = cap[idx] / E_REF
            tok[idx, 3] = E[idx] / cap[idx]
            tok[idx, 4] = np.maximum(Ereq[idx] - E[idx], 0.0) / E_REF
            tok[idx, 5] = tau_feat[idx] / Nd
            tok[idx, 6] = np.clip(chi[idx], -Nd, 2 * Nd) / Nd
            tok[idx, 7] = pnom[idx] / P_REF
            tok[idx, 8] = pdis[idx] / P_REF
            tok[idx, 9] = pmin[idx] / P_REF
            tok[idx, 10] = l[idx] / P_REF
            tok[idx, 11] = u[idx] / P_REF
        # ---- global token
        Ptr_nom = float(np.max(tr.max_power))
        cch = -np.asarray(env.charge_prices[0], float)
        cdis = np.asarray(env.discharge_prices[0], float)
        g = np.zeros(F_G, np.float64)
        g[0] = cch[t] / C_REF
        g[1] = cdis[t] / C_REF
        g[2:10] = cch[np.minimum(np.arange(t + 1, t + 1 + H_F), self.T - 1)] / C_REF
        dtm = self.datetime_at(t)
        hour = dtm.hour + dtm.minute / 60.0
        g[10] = math.sin(2 * math.pi * hour / 24)
        g[11] = math.cos(2 * math.pi * hour / 24)
        g[12] = float(dtm.weekday() >= 5)
        g[13] = (self.T - t) / self.T
        g[14] = C / Ptr_nom
        g[15] = Crev / Ptr_nom
        g[16] = P_inf / Ptr_nom
        g[17] = P_pv / Ptr_nom
        g[18:26] = fc_inf / Ptr_nom
        g[26:34] = fc_pv / Ptr_nom
        g[34] = n / N
        g[35] = Ptr_nom / (N * P_REF)
        g[36] = l_dep[occ].sum() / Ptr_nom
        g[37] = u[occ].sum() / Ptr_nom
        self._obs = Observation(t=t, tokens=tok.astype(np.float32), glob=g.astype(np.float32), occ=occ, infl=infl,
                                E=E, l=l_lay, u=u, pmin=pmin, chi=chi, lg2v=lg2v, feas_i=feas_i, C=C, Ctil=Ctil,
                                Crev=Crev, feasible=feasible, l_dep=l_dep)
        return self._obs

    # ------------------------------------------------------------------------------------ command
    def command(self, x) -> tuple[np.ndarray, str | None, float | None]:
        """Raw action x (kW) -> commanded power p (safety layer or physical clip). Returns (p, fb_case, dist)."""
        obs = self._obs
        x = np.asarray(x, float).reshape(self.N)
        bad = ~np.isfinite(x) & obs.occ
        self.acc.nonfinite += int(bad.sum())
        x, _ = sanitize(x)
        x = np.where(obs.occ, x, 0.0)
        fb_case, dist = None, None
        if self.opt.safety:
            if obs.feasible:
                c = GuardCounters()
                p, _, _ = project(x[None], obs.l[None], obs.u[None], np.array([obs.Ctil]), np.array([obs.Crev]),
                                  mask=obs.occ[None], counters=c)
                self.acc.bisect += c.bisect_handover
                p = p[0]
                dist = float(np.linalg.norm(x - p)) / self.N
            else:
                p, fb_case = fallback(obs.l, obs.u, obs.pmin, obs.Ctil, obs.Crev, obs.chi, obs.occ)
        else:
            p = np.where(obs.occ, np.clip(x, obs.pmin, obs.u), 0.0)
        return p, fb_case, dist

    # --------------------------------------------------------------------------------------- step
    def step(self, x, inference_ms: float | None = None, p_override=None) -> StepOut:
        """Apply raw action x (kW per charger) for the current step. ``observe()`` must have been called."""
        if self._obs is None or self._obs.t != self.t:
            raise RuntimeError("call observe() before step()")
        obs = self._obs
        t = obs.t
        if p_override is None:
            p, fb_case, dist = self.command(x)
        else:
            p, fb_case, dist = p_override
        occ = obs.occ
        ldep = obs.l_dep if obs.l_dep is not None else obs.l
        deficit = float(np.sum(np.maximum(ldep - p, 0.0)[occ]))
        for i in np.flatnonzero(occ & (p < ldep - 1e-12)):
            self.slots[i].touched = True
        zc = np.array([s.params.zeta_c if s else 1.0 for s in self.slots])
        zd = np.array([s.params.zeta_d if s else 1.0 for s in self.slots])
        a_env = actuation(p, obs.u, zc, zd, occ)
        _, P_tr, P_inf, P_pv = self._exogenous(t)
        _, reward, done, _, _ = self.env.step(a_env)
        p_rep = np.array([cs.current_power_output for cs in self.env.charging_stations], float)
        sp = float(p_rep.sum())
        P_net = P_inf - P_pv + sp
        viol = max(0.0, P_net - P_tr) + max(0.0, -P_net - P_tr)
        deliv = math.inf
        for i in np.flatnonzero(occ):
            e_after = float(self.slots[i].ev.current_capacity)
            deliv = min(deliv, e_after - obs.E[i] - self.dt_h * p[i])
        if self.opt.log_steps:
            self._log_step(obs, p, p_rep, P_tr, P_inf, P_pv, fb_case)
        # departures
        dep_short_dd2 = 0.0
        dep_short_adm = 0.0
        # DD43(2) per-step diagnostics on the commanded powers: aggregate limit binding (sum p at C_tilde or -C_rev,
        # i.e. the projection's lambda > 0 / < 0 on A5 steps) and lower bound active (p_i = l_i > p_i^min)
        sp_cmd = float(p[occ].sum()) if occ.any() else 0.0
        tol = 1e-6 * max(1.0, abs(obs.Ctil), abs(obs.Crev))
        agg_up = bool(occ.any() and sp_cmd >= obs.Ctil - tol)
        agg_lo = bool(occ.any() and sp_cmd <= -obs.Crev + tol)
        lb_act = int(np.sum(occ & (np.abs(p - ldep) <= 1e-9) & (ldep > obs.pmin + 1e-9)))
        for ev in self.env.departing_evs:
            i = int(ev.location)
            s = self.slots[i]
            if s is None or s.ev is not ev:
                continue
            Ef = float(ev.current_capacity)
            short = max(0.0, s.E_req - Ef)
            short2 = max(0.0, s.E_req_dd2 - Ef)
            short3 = max(0.0, s.E_req_shared - Ef)
            dep_short_dd2 += short2
            dep_short_adm += short
            cov = slack = None
            if self.cp is not None:
                cov = bool(s.d >= s.d_ctrl)
                slack = float(s.d - s.d_ctrl)
            sat = 1.0 if Ef >= s.E_des - 0.001 else Ef / s.E_des      # vs the original E_des (EV2Gym's formula)
            self.acc.on_departure(short, short2, sat, s.touched, cov, slack, short_shared=short3)
            self.slots[i] = None
        r = float(reward)
        thr = float(getattr(self.scfg, "throughput_cost", 0.0) or 0.0)
        if thr:                                     # battery-throughput cost (DD39), reward and evaluation
            r -= thr * self.dt_h * float(np.abs(p_rep).sum())
        over = max(0.0, sp - obs.C) + max(0.0, -obs.Crev - sp)
        # penalty reward (Alg. 5, DD22) vs the admitted request (DD31: the same request pipeline for all methods;
        # without admission control the admitted request is the DD2-capped one)
        r_pen = r - self.opt.w_E * dep_short_adm - self.opt.w_P * self.dt_h * over
        if self.opt.penalty_dense:
            r_pen -= self.opt.w_E * self.dt_h * deficit
        self.acc.on_step(profit=r, pen_reward=r_pen, p_rep=p_rep, viol_kw=viol, feasible=obs.feasible,
                         fb_case=fb_case, deficit_kw=deficit, proj_dist=dist,
                         slack_kw=obs.Ctil - float(ldep[occ].sum()),
                         margin_kw=float(np.max(p_rep[occ] - p[occ])) if occ.any() else -math.inf,
                         deliv_min_kwh=deliv if np.isfinite(deliv) else math.inf,
                         agg_upper=agg_up, agg_lower=agg_lo, lb_active=lb_act, n_occ=int(occ.sum()))
        if inference_ms is not None:
            self.acc.on_inference(inference_ms)
        self._obs = None
        return StepOut(reward=r, reward_pen=r_pen, done=bool(done),
                       info={"feasible": obs.feasible, "fb_case": fb_case, "viol_kw": viol, "proj_dist": dist,
                             # constraint costs of PPO-Lagrangian (DD33(3)), kWh: departure shortfall vs the admitted
                             # request (DD31) and transformer overload energy (either direction)
                             "cost_short_kwh": dep_short_adm, "cost_over_kwh": self.dt_h * over})

    # -------------------------------------------------------------------------------------- logs
    def _log_step(self, obs, p, p_rep, P_tr, P_inf, P_pv, fb_case):
        t = obs.t
        self.step_log.append({
            "replay_seed": self.seed, "t": t,
            "price_ch": float(-self.env.charge_prices[0, t]), "price_dis": float(self.env.discharge_prices[0, t]),
            "P_tr": P_tr, "P_inf": P_inf, "P_pv": P_pv, "C": obs.C, "C_rev": obs.Crev, "C_tilde": obs.Ctil,
            "sum_p": float(p_rep.sum()), "sum_p_cmd": float(p.sum()), "sum_l": float(obs.l[obs.occ].sum()),
            "sum_u": float(obs.u[obs.occ].sum()), "n_connected": int(obs.occ.sum()),
            "sum_l_dep": float((obs.l_dep if obs.l_dep is not None else obs.l)[obs.occ].sum()),
            "fallback_code": -1 if fb_case is None else FB_CODES[fb_case], "feasible": obs.feasible})
        for i in np.flatnonzero(obs.occ):
            s = self.slots[i] if self.slots[i] is not None else None
            self.charger_log.append({"replay_seed": self.seed, "t": t, "i": int(i), "p": float(p[i]),
                                     "p_rep": float(p_rep[i]), "l": float(obs.l[i]), "u": float(obs.u[i]),
                                     "E": float(obs.E[i]),
                                     "E_req": float(s.E_req) if s is not None else float("nan"),
                                     "tau": int(s.d_ctrl - t + 1) if s is not None else -1})

    # ------------------------------------------------------------------------------------ result
    def episode_metrics(self) -> dict:
        m = self.acc.finalize()
        m.update(admission=bool(self.opt.admission), safety_layer=bool(self.opt.safety), replay_seed=self.seed)
        if self.cp is not None:
            m["alpha"] = getattr(self.cp, "alpha", float("nan"))
        return m
