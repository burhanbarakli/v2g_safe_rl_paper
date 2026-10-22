"""Per-episode metrics (code_spec.md Sec. 3.1; safety_layer.md Sec. 7.5).

:data:`EPISODE_COLUMNS` is the exact column list of ``episodes.csv`` that the analysis agent reads;
:data:`EXTRA_COLUMNS` are appended diagnostics (guard counters, DD2-referenced shortfall, conformal coverage,
oracle extras). :class:`EpisodeAccumulator` is filled by the environment wrapper step by step.
"""
from __future__ import annotations

import math

import numpy as np

from ..constants import REPORT_TOL_KWH, W_E_DEFAULT, W_P_DEFAULT

EPISODE_COLUMNS = [
    "exp_id", "method", "seed", "train_N", "test_cfg_id", "profile", "N", "rho_tr", "dt_min", "replay_id",
    "profit_eur", "energy_charged_kwh", "energy_discharged_kwh", "n_evs", "n_evs_capped",
    "n_evs_admission_reduced", "n_evs_unserved", "energy_not_served_kwh", "frac_evs_unserved",
    "user_satisfaction_mean", "tr_violation_steps", "tr_violation_kwh", "tr_violation_max_kw",
    "infeasible_steps", "fb0_steps", "fb1_steps", "fb2_steps", "fb1star_steps", "a5_deficit_kwh",
    "proj_dist_mean", "inference_ms_per_step_mean", "inference_ms_per_step_p95", "oracle_profit_eur", "smoke",
]

EXTRA_COLUMNS = [
    # requirement references and caps
    "energy_capped_kwh", "energy_admission_reduced_kwh", "n_admission_not_restored",
    "n_evs_unserved_dd2", "energy_not_served_dd2_kwh", "n_evs_touched_infeasible",
    # DD43: shortfall vs the shared admitted requirement of the replay (same reference for every pass), stay-level
    # feasibility (primary endpoint E4), aggregate-limit binding and lower-bound activity
    "n_evs_unserved_shared", "energy_not_served_shared_kwh", "n_evs_stay_infeasible", "share_evs_stay_feasible",
    "agg_upper_active_steps", "agg_lower_active_steps", "lb_active_frac",
    # safety-layer diagnostics
    "a5_slack_min_kw", "actuation_margin_max_kw", "energy_delivery_min_kwh", "n_nonfinite_actions",
    "n_bisect_handover", "max_kappa", "penalty_return_eur", "n_steps",
    # penalised profit J^pen = profit - w_E unserved - w_P overload, w = 10 EUR/kWh (DD32): vs the admitted request
    # (DD31) and vs the DD2-capped request; oracle_pen_profit_eur upper-bounds both for every non-conformal method
    "penalised_profit_eur", "penalised_profit_dd2_eur",
    # conformal (Alg. 4); empty unless the conformal bound is active. coverage_frac = conformal_coverage (alias
    # read by the analysis scripts)
    "alpha", "conformal_coverage", "n_evs_departed_before_bound", "coverage_frac", "mean_d_minus_dlo",
    "shortfall_departed_before_bound_kwh",
    # oracle (code_spec Sec. 3.4)
    "oracle_profit_ceil_eur", "oracle_pen_profit_eur", "solve_time_s", "solver_status", "shortfall_stage_kwh",
    # bookkeeping
    "admission", "safety_layer", "replay_seed", "replay_fingerprint", "train_N_set", "in_distribution",
]

ALL_COLUMNS = EPISODE_COLUMNS + EXTRA_COLUMNS
#: columns that only the oracle writes (code_spec Sec. 3.4); method files omit them (empty numeric columns would
#: be read as required-but-missing by the analysis loader)
ORACLE_ONLY = ["oracle_profit_ceil_eur", "oracle_pen_profit_eur", "solve_time_s", "solver_status",
               "shortfall_stage_kwh"]
METHOD_COLUMNS = [c for c in ALL_COLUMNS if c not in ORACLE_ONLY]
ORACLE_COLUMNS = ALL_COLUMNS


class EpisodeAccumulator:
    """Accumulates the per-episode metrics of one closed-loop episode."""

    def __init__(self, dt_h: float, N: int):
        self.dt_h = dt_h
        self.N = N
        self.profit = 0.0
        self.pen_return = 0.0
        self.e_ch = 0.0
        self.e_dis = 0.0
        self.n_evs = 0
        self.n_capped = 0
        self.cap_kwh = 0.0
        self.n_adm = 0
        self.adm_kwh = 0.0
        self.n_adm_fail = 0
        self.dep_short = []        # vs admitted E_req
        self.dep_short_dd2 = []    # vs DD2-capped E_req
        self.dep_short_shared = []  # vs the shared admitted E_req (DD43)
        self.n_stay_infeasible = 0
        self.agg_up = 0
        self.agg_lo = 0
        self.lb_act = 0
        self.occ_steps = 0
        self.sat = []
        self.touched = 0
        self.viol_steps = 0
        self.viol_kwh = 0.0
        self.viol_max = 0.0
        self.infeasible = 0
        self.fb = {"FB-0": 0, "FB-1": 0, "FB-2": 0, "FB-1*": 0}
        self.deficit = 0.0
        self.proj = []
        self.inf_ms = []
        self.slack_min = math.inf
        self.margin_max = -math.inf
        self.deliv_min = math.inf
        self.nonfinite = 0
        self.bisect = 0
        self.kappa_max = 0.0
        self.n_steps = 0
        self.cp = {"n": 0, "covered": 0, "slack": [], "short_unc": 0.0}

    # ----------------------------------------------------------------------------------- events
    def on_arrival(self, capped_kwh: float, adm_kwh: float, adm_failed: bool, kappa: float) -> None:
        self.n_evs += 1
        self.cap_kwh += max(0.0, capped_kwh)
        self.n_capped += int(capped_kwh > REPORT_TOL_KWH)
        self.adm_kwh += max(0.0, adm_kwh)
        self.n_adm += int(adm_kwh > REPORT_TOL_KWH)
        self.n_adm_fail += int(adm_failed)
        self.kappa_max = max(self.kappa_max, kappa)

    def on_departure(self, short: float, short_dd2: float, sat: float, touched: bool,
                     cp_covered: bool | None = None, cp_slack: float | None = None,
                     short_shared: float | None = None) -> None:
        self.dep_short.append(max(0.0, short))
        self.dep_short_shared.append(max(0.0, short if short_shared is None else short_shared))
        self.dep_short_dd2.append(max(0.0, short_dd2))
        self.sat.append(sat)
        self.touched += int(touched)
        if cp_covered is not None:
            self.cp["n"] += 1
            self.cp["covered"] += int(cp_covered)
            self.cp["slack"].append(cp_slack)
            if not cp_covered:
                self.cp["short_unc"] += max(0.0, short)

    def on_step(self, *, profit: float, pen_reward: float, p_rep: np.ndarray, viol_kw: float, feasible: bool,
                fb_case: str | None, deficit_kw: float, proj_dist: float | None, slack_kw: float,
                margin_kw: float, deliv_min_kwh: float, agg_upper: bool = False, agg_lower: bool = False,
                lb_active: int = 0, n_occ: int = 0) -> None:
        self.n_steps += 1
        self.agg_up += int(agg_upper)
        self.agg_lo += int(agg_lower)
        self.lb_act += int(lb_active)
        self.occ_steps += int(n_occ)
        self.profit += profit
        self.pen_return += pen_reward
        self.e_ch += float(np.sum(np.maximum(p_rep, 0.0))) * self.dt_h
        self.e_dis += float(np.sum(np.maximum(-p_rep, 0.0))) * self.dt_h
        if viol_kw > 1e-6:
            self.viol_steps += 1
        self.viol_kwh += viol_kw * self.dt_h
        self.viol_max = max(self.viol_max, viol_kw)
        self.infeasible += int(not feasible)
        if fb_case is not None:
            self.fb[fb_case] += 1
        self.deficit += deficit_kw * self.dt_h
        if proj_dist is not None:
            self.proj.append(proj_dist)
        self.slack_min = min(self.slack_min, slack_kw)
        self.margin_max = max(self.margin_max, margin_kw)
        self.deliv_min = min(self.deliv_min, deliv_min_kwh)

    def on_inference(self, ms: float) -> None:
        self.inf_ms.append(ms)

    # ---------------------------------------------------------------------------------- result
    def finalize(self) -> dict:
        short = np.asarray(self.dep_short)
        short2 = np.asarray(self.dep_short_dd2)
        n_uns = int(np.sum(short > REPORT_TOL_KWH))
        out = {
            "profit_eur": self.profit,
            "energy_charged_kwh": self.e_ch,
            "energy_discharged_kwh": self.e_dis,
            "n_evs": self.n_evs,
            "n_evs_capped": self.n_capped,
            "n_evs_admission_reduced": self.n_adm,
            "n_evs_unserved": n_uns,
            "energy_not_served_kwh": float(short.sum()),
            "frac_evs_unserved": n_uns / self.n_evs if self.n_evs else 0.0,
            "user_satisfaction_mean": float(np.mean(self.sat)) if self.sat else float("nan"),
            "tr_violation_steps": self.viol_steps,
            "tr_violation_kwh": self.viol_kwh,
            "tr_violation_max_kw": self.viol_max,
            "infeasible_steps": self.infeasible,
            "fb0_steps": self.fb["FB-0"],
            "fb1_steps": self.fb["FB-1"],
            "fb2_steps": self.fb["FB-2"],
            "fb1star_steps": self.fb["FB-1*"],
            "a5_deficit_kwh": self.deficit,
            "proj_dist_mean": float(np.mean(self.proj)) if self.proj else float("nan"),
            "inference_ms_per_step_mean": float(np.mean(self.inf_ms)) if self.inf_ms else float("nan"),
            "inference_ms_per_step_p95": float(np.percentile(self.inf_ms, 95)) if self.inf_ms else float("nan"),
            "energy_capped_kwh": self.cap_kwh,
            "energy_admission_reduced_kwh": self.adm_kwh,
            "n_admission_not_restored": self.n_adm_fail,
            "n_evs_unserved_dd2": int(np.sum(short2 > REPORT_TOL_KWH)),
            "energy_not_served_dd2_kwh": float(short2.sum()),
            "n_evs_touched_infeasible": self.touched,
            "n_evs_unserved_shared": int(np.sum(np.asarray(self.dep_short_shared) > REPORT_TOL_KWH)),
            "energy_not_served_shared_kwh": float(np.sum(self.dep_short_shared)),
            "n_evs_stay_infeasible": self.n_stay_infeasible,
            "share_evs_stay_feasible": 1.0 - self.n_stay_infeasible / self.n_evs if self.n_evs else float("nan"),
            "agg_upper_active_steps": self.agg_up,
            "agg_lower_active_steps": self.agg_lo,
            "lb_active_frac": self.lb_act / self.occ_steps if self.occ_steps else float("nan"),
            "a5_slack_min_kw": self.slack_min,
            "actuation_margin_max_kw": self.margin_max,
            "energy_delivery_min_kwh": self.deliv_min,
            "n_nonfinite_actions": self.nonfinite,
            "n_bisect_handover": self.bisect,
            "max_kappa": self.kappa_max,
            "penalty_return_eur": self.pen_return,
            "penalised_profit_eur": self.profit - W_E_DEFAULT * float(short.sum()) - W_P_DEFAULT * self.viol_kwh,
            "penalised_profit_dd2_eur": self.profit - W_E_DEFAULT * float(short2.sum()) - W_P_DEFAULT * self.viol_kwh,
            "n_steps": self.n_steps,
        }
        if self.cp["n"]:
            cov = self.cp["covered"] / self.cp["n"]
            out.update(conformal_coverage=cov, coverage_frac=cov,
                       n_evs_departed_before_bound=self.cp["n"] - self.cp["covered"],
                       mean_d_minus_dlo=float(np.mean(self.cp["slack"])),
                       shortfall_departed_before_bound_kwh=self.cp["short_unc"])
        return out
