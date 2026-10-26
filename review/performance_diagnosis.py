"""Diagnostics: why SA-Attn underperforms (gap decomposition, V2G use, bound riding, infeasible episodes)."""
import glob

import numpy as np
import pandas as pd

R = "/home/bob/v2g_safe_rl_paper/code/results/raw"
W = 10.0
M = ["attn_safe", "attn_safe_penrew", "deepsets_safe", "attn_proj_nolb", "attn_penalty_safedeploy", "attn_penalty", "mlp_safe",
     "price_llf_safe", "alap_safe", "afap_safe", "llf_safe", "empc_highs_safe"]
cols = ["method", "seed", "test_cfg_id", "profile", "N", "rho_tr", "replay_id", "profit_eur", "energy_charged_kwh",
        "energy_discharged_kwh", "energy_not_served_shared_kwh", "tr_violation_kwh", "infeasible_steps", "n_evs",
        "n_evs_unserved_shared", "lb_active_frac", "proj_dist_mean", "agg_upper_active_steps",
        "agg_lower_active_steps", "fb0_steps", "fb1_steps", "fb2_steps", "fb1star_steps", "n_steps"]
fs = [f for f in glob.glob(f"{R}/*/episodes.csv") if "/oracle__" not in f]
d = pd.concat([pd.read_csv(f, usecols=lambda c: c in cols) for f in fs])
d = d[d.method.isin(M) & d.rho_tr.isin([4.0, 7.0]) & ~d.test_cfg_id.str.startswith("literature")]
o = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/oracle__*/episodes.csv")])
o = o[["test_cfg_id", "replay_id", "oracle_pen_profit_eur", "profit_eur", "energy_charged_kwh", "energy_discharged_kwh"]]
o = o.rename(columns={"profit_eur": "or_profit", "energy_charged_kwh": "or_ch", "energy_discharged_kwh": "or_dis"})
d = d.merge(o, on=["test_cfg_id", "replay_id"], how="left")
d["jpen"] = d.profit_eur - W * d.energy_not_served_shared_kwh - W * d.tr_violation_kwh
d["ens_pen"] = W * d.energy_not_served_shared_kwh
d["ov_pen"] = W * d.tr_violation_kwh
d["gap"] = d.oracle_pen_profit_eur - d.jpen
d["infeas_ep"] = d.infeasible_steps > 0
for c in ["jpen", "profit_eur", "ens_pen", "ov_pen", "gap", "energy_charged_kwh", "energy_discharged_kwh",
          "oracle_pen_profit_eur", "or_profit", "or_ch", "or_dis"]:
    d[c + "_pc"] = d[c] / d.N

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
g = d.groupby("method")
t = pd.DataFrame({
    "gap/ch": g.gap_pc.mean(), "profit/ch": g.profit_eur_pc.mean(), "ENSpen/ch": g.ens_pen_pc.mean(),
    "OVpen/ch": g.ov_pen_pc.mean(), "chg kWh/ch": g.energy_charged_kwh_pc.mean(),
    "dis kWh/ch": g.energy_discharged_kwh_pc.mean(), "ep infeas %": 100 * g.infeas_ep.mean(),
    "infeas steps/ep": g.infeasible_steps.mean(), "lb_active": g.lb_active_frac.mean(),
    "proj_dist": g.proj_dist_mean.mean(), "aggUp/ep": g.agg_upper_active_steps.mean(),
    "aggLo/ep": g.agg_lower_active_steps.mean(), "unserved %": 100 * g.n_evs_unserved_shared.sum() / g.n_evs.sum(),
}).loc[M]
print("=== 1. Per method (main grid, rho 4/7, all N; EUR per charger per episode)")
print(t.round(3).to_string())
oo = d.drop_duplicates(["test_cfg_id", "replay_id"])
print(f"\noracle: soft J^pen/ch {oo.oracle_pen_profit_eur_pc.mean():.3f}, hard profit/ch {oo.or_profit_pc.mean():.3f}, "
      f"charged/ch {oo.or_ch_pc.mean():.2f} kWh, discharged/ch {oo.or_dis_pc.mean():.2f} kWh")

print("\n=== 2. Gap difference vs SA-Attn split into profit and ENS/overload parts (positive = SA-Attn worse)")
a = t.loc["attn_safe"]
for m in M[1:]:
    b = t.loc[m]
    print(f"{m:24s} dGap {a['gap/ch'] - b['gap/ch']:+.3f} = profit {b['profit/ch'] - a['profit/ch']:+.3f} "
          f"+ ENS {a['ENSpen/ch'] - b['ENSpen/ch']:+.3f} + overload {a['OVpen/ch'] - b['OVpen/ch']:+.3f}")

print("\n=== 3. SA-Attn: episodes with vs without infeasible steps")
s = d[d.method == "attn_safe"]
for k, x in s.groupby("infeas_ep"):
    print(f"infeasible={k}: share {100 * len(x) / len(s):.1f} %, gap/ch {x.gap_pc.mean():.3f}, profit/ch "
          f"{x.profit_eur_pc.mean():.3f}, ENSpen/ch {x.ens_pen_pc.mean():.3f}")
print(f"share of SA-Attn ENS penalty from infeasible episodes: "
      f"{100 * s[s.infeas_ep].ens_pen.sum() / s.ens_pen.sum():.1f} %")

print("\n=== 4. By profile and rho (gap per charger; infeasible-episode %)")
p = d[d.method.isin(["attn_safe", "attn_safe_penrew", "deepsets_safe", "attn_proj_nolb", "price_llf_safe"])]
print(p.pivot_table(index=["profile", "rho_tr"], columns="method", values="gap_pc", aggfunc="mean").round(3).to_string())
print(p.pivot_table(index=["profile", "rho_tr"], columns="method", values="infeas_ep", aggfunc="mean").mul(100).round(1).to_string())

print("\n=== 5. By N (gap per charger)")
print(p.pivot_table(index="N", columns="method", values="gap_pc", aggfunc="mean").round(3).to_string())

print("\n=== 6. Seed spread (gap per charger, per seed)")
print(d[d.method.isin(["attn_safe", "attn_safe_penrew", "deepsets_safe", "attn_proj_nolb", "mlp_safe"])]
      .pivot_table(index="method", columns="seed", values="gap_pc", aggfunc="mean").round(3).to_string())
