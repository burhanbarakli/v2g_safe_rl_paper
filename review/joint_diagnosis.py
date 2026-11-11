"""Sec. 13: does the capacity-aware (joint) lower bound + bound-relative action close SA-Attn's gap?

Compares, on the main grid (rho 4/7, no literature configs), the gap to the soft oracle per charger per episode:
  * per-EV layer (as in the paper): attn_safe_penrew, deepsets_safe_penrew, price_llf_safe
  * joint layer, zero-shot (old checkpoints / rule evaluated with lower_bound=joint): *_jlb
  * joint layer, retrained: attn_joint_safe_penrew (joint bound), attn_jbox_safe_penrew (joint bound + box action),
    attn_jres_safe_penrew (joint bound + residual around the price-LLF rule, r_rule clip 0.02),
    attn_jres2_safe_penrew (same, clip 0.15), attn_gjoint_safe_penrew (joint bound + global input, lr 1e-3, mb 512),
    attn_gres2_safe_penrew (jres2 + global input), attn_jres3_safe_penrew (jres2 + L2 prior on the residual logit)
Paired differences use the same (configuration, replay, seed) episodes; 95 % CI by bootstrap over replays.
"""
import glob

import numpy as np
import pandas as pd

R = "/home/bob/v2g_safe_rl_paper/code/results/raw"
W = 10.0
BASE = ["attn_safe_penrew", "attn_gin_safe_penrew", "attn_gctx_safe_penrew", "deepsets_safe_penrew", "price_llf_safe"]
JLB = ["attn_safe_penrew_jlb", "attn_gin_safe_penrew_jlb", "attn_gctx_safe_penrew_jlb", "deepsets_safe_penrew_jlb",
       "price_llf_safe_jlb"]
NEW = ["attn_joint_safe_penrew", "attn_jbox_safe_penrew", "attn_jres_safe_penrew", "attn_jres2_safe_penrew",
       "attn_gjoint_safe_penrew", "attn_gres2_safe_penrew", "attn_jres3_safe_penrew"]
M = BASE + JLB + NEW
cols = ["method", "seed", "test_cfg_id", "profile", "N", "rho_tr", "replay_id", "profit_eur",
        "energy_not_served_shared_kwh", "tr_violation_kwh", "infeasible_steps", "lb_active_frac",
        "energy_charged_kwh", "energy_discharged_kwh"]
fs = [f for f in glob.glob(f"{R}/*/episodes.csv") if "/oracle__" not in f]
d = pd.concat([pd.read_csv(f, usecols=lambda c: c in cols) for f in fs])
d = d[d.method.isin(M) & d.rho_tr.isin([4.0, 7.0]) & ~d.test_cfg_id.str.startswith("literature")]
d = d.drop_duplicates(["method", "seed", "test_cfg_id", "replay_id"])
o = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/oracle__*/episodes.csv")])
o = o[["test_cfg_id", "replay_id", "oracle_pen_profit_eur"]].drop_duplicates(["test_cfg_id", "replay_id"])
d = d.merge(o, on=["test_cfg_id", "replay_id"], how="left")
d["jpen"] = d.profit_eur - W * d.energy_not_served_shared_kwh - W * d.tr_violation_kwh
d["gap_pc"] = (d.oracle_pen_profit_eur - d.jpen) / d.N
d["profit_pc"] = d.profit_eur / d.N
d["ens_pc"] = W * d.energy_not_served_shared_kwh / d.N
d["ov_pc"] = W * d.tr_violation_kwh / d.N
d["dis_pc"] = d.energy_discharged_kwh / d.N
d["infeas_ep"] = 100.0 * (d.infeasible_steps > 0)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
present = [m for m in M if m in set(d.method)]
g = d.groupby("method")
t = pd.DataFrame({"episodes": g.size(), "seeds": g.seed.nunique(), "gap/ch": g.gap_pc.mean(),
                  "profit/ch": g.profit_pc.mean(), "ENSpen/ch": g.ens_pc.mean(), "OVpen/ch": g.ov_pc.mean(),
                  "dis kWh/ch": g.dis_pc.mean(), "ep infeas %": g.infeas_ep.mean(),
                  "lb_active": g.lb_active_frac.mean()}).loc[present]
print("=== 1. Per method (main grid, rho 4/7; EUR per charger per episode)")
print(t.round(3).to_string())

print("\n=== 2. Per seed (gap per charger)")
print(d.pivot_table(index="method", columns="seed", values="gap_pc", aggfunc="mean").reindex(present).round(3).to_string())


def paired(a, b, n_boot=2000, seed=0):
    """Mean gap(a) - gap(b) over common episodes (seed-agnostic for rules: rules have one seed, matched by cfg/replay;
    learned vs learned matched by seed too). Positive = a worse. CI by bootstrap over (cfg, replay) clusters."""
    x = d[d.method == a]
    y = d[d.method == b]
    key = ["test_cfg_id", "replay_id"]
    if x.seed.nunique() > 1 and y.seed.nunique() > 1:
        key = key + ["seed"]
    else:
        y = y.groupby(["test_cfg_id", "replay_id"], as_index=False).gap_pc.mean() if y.seed.nunique() > 1 else y
        x = x.groupby(["test_cfg_id", "replay_id"], as_index=False).gap_pc.mean() if x.seed.nunique() > 1 else x
    z = x[key + ["gap_pc"]].merge(y[key + ["gap_pc"]], on=key, suffixes=("_a", "_b"))
    if z.empty:
        return np.nan, np.nan, np.nan, 0
    z["diff"] = z.gap_pc_a - z.gap_pc_b
    cl = z.groupby(["test_cfg_id", "replay_id"])["diff"].mean().to_numpy()
    rng = np.random.default_rng(seed)
    bs = rng.choice(cl, (n_boot, cl.size)).mean(1)
    return cl.mean(), np.quantile(bs, 0.025), np.quantile(bs, 0.975), len(z)


print("\n=== 3. Paired gap differences (a - b, EUR/charger/episode; negative = a better), 95 % bootstrap CI")
pairs = [("attn_safe_penrew", "price_llf_safe"), ("attn_safe_penrew", "deepsets_safe_penrew"),
         ("attn_safe_penrew_jlb", "attn_safe_penrew"), ("price_llf_safe_jlb", "price_llf_safe"),
         ("deepsets_safe_penrew_jlb", "deepsets_safe_penrew")]
for a in ["attn_safe_penrew_jlb", "attn_gin_safe_penrew_jlb", "attn_gctx_safe_penrew_jlb"] + NEW:
    pairs += [(a, "price_llf_safe_jlb"), (a, "deepsets_safe_penrew_jlb"), (a, "price_llf_safe"),
              (a, "deepsets_safe_penrew")]
pairs += [("attn_jbox_safe_penrew", "attn_joint_safe_penrew"), ("attn_jres_safe_penrew", "attn_joint_safe_penrew"),
          ("attn_jres2_safe_penrew", "attn_joint_safe_penrew"), ("attn_gjoint_safe_penrew", "attn_joint_safe_penrew"),
          ("attn_gjoint_safe_penrew", "attn_gin_safe_penrew_jlb"), ("attn_gres2_safe_penrew", "attn_jres2_safe_penrew"),
          ("attn_jres3_safe_penrew", "attn_jres2_safe_penrew"),
          ("attn_gin_safe_penrew_jlb", "attn_gin_safe_penrew"), ("attn_gctx_safe_penrew_jlb", "attn_gctx_safe_penrew")]
for a, b in pairs:
    if a in present and b in present:
        m, lo, hi, n = paired(a, b)
        print(f"{a:26s} - {b:26s} {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]  n={n}")

p = d[d.method.isin(present)]
print("\n=== 4. By profile and rho (gap per charger)")
print(p.pivot_table(index=["profile", "rho_tr"], columns="method", values="gap_pc", aggfunc="mean")
      .reindex(columns=present).round(3).to_string())
print("\n=== 5. By N (gap per charger)")
print(p.pivot_table(index="N", columns="method", values="gap_pc", aggfunc="mean").reindex(columns=present)
      .round(3).to_string())
print("\n=== 6. Infeasible-episode % by profile and rho")
print(p.pivot_table(index=["profile", "rho_tr"], columns="method", values="infeas_ep", aggfunc="mean")
      .reindex(columns=present).round(1).to_string())
