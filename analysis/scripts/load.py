"""Load and validate raw result files (code_spec.md §2-§3).

Input contract
--------------
<raw>/<exp_id>/episodes.csv       one row per evaluated episode (§3.1)
<raw>/<exp_id>/train_log.csv      learned methods (§3.3)
<raw>/<exp_id>/meta.json          optional; `smoke`, `wall_clock_s`, ...
<raw>/<exp_id>/steps.parquet      optional (§3.2); steps.csv accepted as fallback
<raw>/oracle__<test_cfg_id>/episodes.csv   oracle (§3.4)

Processing
----------
1. read every episodes.csv / train_log.csv, validate columns (SchemaError on
   missing columns, with file name and the full list of missing columns);
2. coerce types; non-parsable values in required numeric columns -> SchemaError;
3. drop smoke rows (column `smoke` true) and whole runs whose meta.json says smoke;
4. check uniqueness of (exp_id, test_cfg_id, replay_id);
5. join the three oracle values per (test_cfg_id, replay_id): `oracle_pen_eur` = soft-penalty oracle
   J^pen_LP,ceil (`oracle_pen_profit_eur`, DD32), `oracle_ceil_eur` = hard oracle J_LP,ceil
   (`oracle_profit_ceil_eur`), `oracle_profit_eur` = J_LP (oracle `profit_eur`, supplementary, DD42-4);
6. DD46: shortfall columns are replaced by the shared-requirement versions (`energy_not_served_shared_kwh`,
   `n_evs_unserved_shared`; the pass's own values are kept as `*_own`), and J^pen = profit - 10*(energy not
   served vs the shared requirement + overload energy) is recomputed for every method and pass (`jpen_eur`;
   the file column `penalised_profit_eur` is not used);
7. bound check: no episode may exceed the soft oracle J^pen_LP,ceil by more than 1e-6 EUR (count reported);
8. DD47: primary optimality gap `gap_abs` = (J^pen_oracle - J^pen)/N in EUR per charger and episode;
   relative gaps only as ratios of sums (tables.py, supplementary).
Extra columns of SCHEMA_ADDITIONS.md are read when present (none of them is required).
"""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

EPISODE_COLS = [
    "exp_id", "method", "seed", "train_N", "test_cfg_id", "profile", "N", "rho_tr", "dt_min",
    "replay_id", "profit_eur", "energy_charged_kwh", "energy_discharged_kwh", "n_evs",
    "n_evs_capped", "n_evs_admission_reduced", "n_evs_unserved", "energy_not_served_kwh",
    "frac_evs_unserved", "user_satisfaction_mean", "tr_violation_steps", "tr_violation_kwh",
    "tr_violation_max_kw", "infeasible_steps", "fb0_steps", "fb1_steps", "fb2_steps",
    "fb1star_steps", "a5_deficit_kwh", "proj_dist_mean", "inference_ms_per_step_mean",
    "inference_ms_per_step_p95", "smoke",
]
# Sec. 3.1 column that stays empty in method rows (the join fills it); tolerated if absent
EPISODE_OPTIONAL = ["oracle_profit_eur"]
# columns that may be empty (NaN) by the schema
EPISODE_NULLABLE = {"train_N", "oracle_profit_eur", "proj_dist_mean"}
EPISODE_STR = {"exp_id", "method", "test_cfg_id", "profile"}
EPISODE_INT = {"seed", "N", "dt_min", "replay_id", "n_evs", "n_evs_capped", "n_evs_admission_reduced",
               "n_evs_unserved", "tr_violation_steps", "infeasible_steps", "fb0_steps", "fb1_steps",
               "fb2_steps", "fb1star_steps"}
ORACLE_EXTRA = ["solve_time_s", "solver_status", "shortfall_stage_kwh"]
ORACLE_CEIL = "oracle_profit_ceil_eur"        # hard oracle J_LP,ceil (DD26); warned about if absent
ORACLE_PEN = "oracle_pen_profit_eur"          # soft-penalty oracle J^pen_LP,ceil (DD32); main gap reference
PEN_COL = "jpen_eur"                          # J^pen recomputed by the loader (DD32, DD46)
FILE_PEN_COL = "penalised_profit_eur"         # written by the code; NOT used (DD46)
BOUND_TOL = 1e-6                              # EUR: tolerance of the soft-oracle bound check
W_PEN = 10.0                                  # EUR/kWh, w_E = w_P (DD22, DD32)
# optional numeric columns from SCHEMA_ADDITIONS.md used by T7/F-S3 (conformal)
OPTIONAL_COLS = ["alpha", "conformal_coverage", "n_evs_departed_before_bound", "coverage_frac",
                 "mean_d_minus_dlo", "shortfall_departed_before_bound_kwh", ORACLE_CEIL, ORACLE_PEN, FILE_PEN_COL,
                 "penalised_profit_dd2_eur", "energy_admission_reduced_kwh", "energy_capped_kwh",
                 "energy_not_served_dd2_kwh", "n_evs_touched_infeasible", "n_steps",
                 # DD43
                 "n_evs_stay_infeasible", "share_evs_stay_feasible", "agg_upper_active_steps",
                 "agg_lower_active_steps", "lb_active_frac", "n_evs_unserved_shared", "energy_not_served_shared_kwh"]
MAX_N = {"mlp_safe": 20, "mlp_penalty": 20}   # DD43(3): zero-padded MLP (N_max = 20) evaluated only at N <= 20

TRAIN_COLS = [
    "exp_id", "env_steps", "wall_clock_s", "episode_return_eur_mean", "episode_return_eur_std",
    "frac_unserved_mean", "tr_violation_kwh_mean", "policy_loss", "value_loss", "entropy",
    "approx_kl", "clip_frac", "beta_reg", "proj_dist_mean", "lr",
]
TRAIN_NULLABLE = {"beta_reg", "proj_dist_mean", "episode_return_eur_std"}

STEP_COLS = ["exp_id", "replay_id", "t", "price_ch", "price_dis", "P_tr", "P_inf", "P_pv", "C", "C_rev",
             "sum_p", "sum_l", "sum_u", "n_connected", "fallback_code"]
# fallback_code: -1 none, 0 FB-0, 1 FB-1, 2 FB-2, 3 FB-1* (SCHEMA_ADDITIONS.md §2)

GAP_EPS = 1.0           # EUR: only for the relative (ratio-of-sums) gap denominators; the primary gap is absolute (DD47)
SYNTH_SENTINEL = "SYNTHETIC_DATA"   # file in <raw> root written by synth.py


class SchemaError(ValueError):
    """Raised when a result file does not follow code_spec.md §3."""


@dataclass
class Dataset:
    episodes: pd.DataFrame            # all non-oracle methods, smoke removed, with oracle join + gap
    oracle: pd.DataFrame              # oracle episodes
    train: pd.DataFrame               # concatenated train logs
    meta: dict                        # exp_id -> meta.json dict
    run_dirs: dict                    # exp_id -> Path
    synthetic: bool
    smoke: bool = False               # smoke rows were kept (include_smoke=True)
    report: list = field(default_factory=list)
    gap_reference: str = "soft-penalty oracle J^pen_LP,ceil (DD32)"
    gap_excluded: dict = field(default_factory=dict)
    bound_violations: int = -1        # episodes with J^pen > soft oracle + BOUND_TOL (must be 0)
    bound_checked: int = 0

    def log(self, msg: str):
        self.report.append(msg)
        print(f"[load] {msg}", file=sys.stderr)


def _to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    m = s.astype(str).str.strip().str.lower()
    ok = m.isin(["true", "false", "1", "0", "1.0", "0.0", "yes", "no", "nan", ""])
    if not ok.all():
        bad = sorted(set(m[~ok]))[:5]
        raise SchemaError(f"column 'smoke' has non-boolean values {bad}")
    return m.isin(["true", "1", "1.0", "yes"])


def validate_columns(df: pd.DataFrame, required, path: Path | str, kind: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(
            f"{kind} file {path} is missing {len(missing)} required column(s): {', '.join(missing)}. "
            f"Expected schema: code_spec.md §3 ({kind}). Present columns: {', '.join(map(str, df.columns))}")


def _coerce_numeric(df: pd.DataFrame, cols, nullable, path, kind):
    for c in cols:
        if c not in df.columns:
            continue
        before_na = df[c].isna() | (df[c].astype(str).str.strip() == "")
        conv = pd.to_numeric(df[c], errors="coerce")
        bad = conv.isna() & ~before_na
        if bad.any():
            ex = df.loc[bad, c].astype(str).head(3).tolist()
            raise SchemaError(f"{kind} file {path}: column '{c}' has non-numeric values, e.g. {ex}")
        if c not in nullable and conv.isna().any():
            n = int(conv.isna().sum())
            raise SchemaError(f"{kind} file {path}: required column '{c}' has {n} empty value(s)")
        df[c] = conv
    return df


def read_episodes(path: Path, oracle: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"exp_id": str, "method": str, "test_cfg_id": str, "profile": str})
    req = EPISODE_COLS + (ORACLE_EXTRA if oracle else [])
    validate_columns(df, req, path, "oracle episodes.csv" if oracle else "episodes.csv")
    num = [c for c in EPISODE_COLS + EPISODE_OPTIONAL + ORACLE_EXTRA + OPTIONAL_COLS
           if c not in EPISODE_STR and c not in ("smoke", "solver_status")]
    df = _coerce_numeric(df, num, EPISODE_NULLABLE | set(OPTIONAL_COLS) | set(EPISODE_OPTIONAL), path,
                         "episodes.csv")
    for c in EPISODE_OPTIONAL:
        if c not in df.columns:
            df[c] = np.nan
    df["smoke"] = _to_bool(df["smoke"])
    for c in EPISODE_INT:
        if df[c].notna().all():
            df[c] = df[c].astype(np.int64)
    df["_source"] = str(path)
    return df


def read_train_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"exp_id": str})
    validate_columns(df, TRAIN_COLS, path, "train_log.csv")
    df = _coerce_numeric(df, [c for c in TRAIN_COLS if c != "exp_id"], TRAIN_NULLABLE, path, "train_log.csv")
    return df


def _read_meta(d: Path) -> dict:
    p = d / "meta.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise SchemaError(f"meta.json in {d} is not valid JSON: {e}") from e


def load_raw(raw: str | Path, require_oracle: bool = True, include_smoke: bool = False) -> Dataset:
    """include_smoke=True keeps smoke rows/runs (pipeline check on A3's smoke output only;
    make_all then watermarks everything and refuses to write into analysis/figures|tables)."""
    raw = Path(raw)
    if not raw.is_dir():
        raise FileNotFoundError(f"raw results directory not found: {raw}")
    synthetic = (raw / SYNTH_SENTINEL).exists()
    ds = Dataset(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {}, synthetic)
    if synthetic:
        ds.log(f"{raw} carries the {SYNTH_SENTINEL} sentinel: data are SYNTHETIC")

    ep_files = sorted(raw.glob("*/episodes.csv"))
    if not ep_files:
        raise SchemaError(f"no */episodes.csv found under {raw}")
    eps, ors, trs = [], [], []
    n_smoke_runs = 0
    for f in ep_files:
        d = f.parent
        meta = _read_meta(d)
        if meta.get("synthetic") and not synthetic:
            raise SchemaError(f"{d}/meta.json is flagged synthetic but {raw} has no {SYNTH_SENTINEL} sentinel")
        if meta.get("smoke") is True and not include_smoke:
            n_smoke_runs += 1
            continue
        is_oracle = d.name.startswith("oracle__")
        df = read_episodes(f, oracle=is_oracle)
        if is_oracle and not (df["method"] == "oracle").all():
            raise SchemaError(f"{f}: directory is oracle__* but method column is not 'oracle'")
        if not is_oracle and (df["method"] == "oracle").any():
            raise SchemaError(f"{f}: method 'oracle' found outside an oracle__<cfg> directory")
        (ors if is_oracle else eps).append(df)
        for e in df["exp_id"].unique():
            ds.meta[e] = meta
            ds.run_dirs[e] = d
        tl = d / "train_log.csv"
        if tl.exists():
            trs.append(read_train_log(tl))
    for tl in sorted(raw.glob("*/train_log.csv")):     # training dirs without episodes
        if not (tl.parent / "episodes.csv").exists():
            meta = _read_meta(tl.parent)
            if meta.get("smoke") is True and not include_smoke:
                n_smoke_runs += 1
                continue
            t = read_train_log(tl)
            trs.append(t)
            for e in t["exp_id"].unique():
                ds.meta.setdefault(e, meta)
                ds.run_dirs.setdefault(e, tl.parent)
    if n_smoke_runs:
        ds.log(f"skipped {n_smoke_runs} run directories flagged smoke in meta.json")
    if not eps:
        raise SchemaError(f"no non-oracle episodes.csv found under {raw}")

    ep = pd.concat(eps, ignore_index=True)
    orc = pd.concat(ors, ignore_index=True) if ors else pd.DataFrame(columns=EPISODE_COLS + ORACLE_EXTRA)
    n0 = len(ep)
    if include_smoke:
        ds.smoke = bool(ep["smoke"].any() or (len(orc) and orc["smoke"].astype(bool).any()))
        if ds.smoke:
            ds.log(f"SMOKE rows kept on request ({int(ep['smoke'].sum())} episodes): outputs are smoke-test only")
    else:
        ep = ep[~ep["smoke"]].copy()
        orc = orc[~orc["smoke"].astype(bool)].copy() if len(orc) else orc
        if n0 - len(ep):
            ds.log(f"dropped {n0 - len(ep)} smoke episode rows")
    if ep.empty:
        raise SchemaError(f"no non-smoke episodes under {raw} (all rows are smoke = true; "
                          f"use --include-smoke for a pipeline check)")

    # DD35: EV-GNN in-distribution rows (its training configuration) get their own method label
    if "in_distribution" in ep.columns:
        ind = _to_bool(ep["in_distribution"].astype(str).replace({"nan": "false", "": "false"}))
        ep["in_distribution"] = ind
        m_id = ep["method"].isin(["evgnn", "evgnn_safe"]) & ind
        ep.loc[m_id, "method"] = ep.loc[m_id, "method"] + "_id"
        if m_id.any():
            ds.log(f"EV-GNN: {int(m_id.sum())} in-distribution rows labelled evgnn_id / evgnn_safe_id")
    # DD43(3): MLP baseline only at N <= 20
    over = pd.Series(False, index=ep.index)
    for m, nmax in MAX_N.items():
        over |= (ep["method"] == m) & (ep["N"] > nmax)
    if over.any():
        ds.log(f"DD43(3): dropped {int(over.sum())} MLP episodes with N > 20")
        ep = ep[~over].copy()
    key = ["exp_id", "test_cfg_id", "replay_id"]
    dup = ep.duplicated(key, keep=False)
    if dup.any():
        ex = ep.loc[dup, key].head(3).to_dict("records")
        raise SchemaError(f"{int(dup.sum())} duplicate (exp_id, test_cfg_id, replay_id) rows, e.g. {ex}")
    dup = ep.duplicated(["method", "seed", "test_cfg_id", "replay_id"], keep=False)
    if dup.any():
        ex = ep.loc[dup, ["method", "seed", "test_cfg_id", "replay_id", "exp_id"]].head(4).to_dict("records")
        raise SchemaError(f"{int(dup.sum())} rows share (method, seed, test_cfg_id, replay_id) across "
                          f"different exp_ids (re-runs?) e.g. {ex}; remove the stale run directories")

    # ---- oracle join ----
    if len(orc):
        od = orc.duplicated(["test_cfg_id", "replay_id"], keep=False)
        if od.any():
            raise SchemaError(f"oracle has {int(od.sum())} duplicate (test_cfg_id, replay_id) rows")
        st = orc["solver_status"].astype(str).str.lower()
        bad = st.str.contains("infeasible|error|fail|timeout|unbounded|not_?solved", regex=True)
        if bad.any():
            ds.log(f"WARNING: {int(bad.sum())} oracle rows with non-optimal solver_status; their gap is NaN")
        if ORACLE_CEIL not in orc.columns or orc[ORACLE_CEIL].isna().all():
            ds.log(f"WARNING: oracle files have no {ORACLE_CEIL} (J_LP,ceil); the hard gap falls back to J_LP")
            orc[ORACLE_CEIL] = orc["profit_eur"]
        if ORACLE_PEN not in orc.columns or orc[ORACLE_PEN].isna().all():
            ds.log(f"WARNING: oracle files have no {ORACLE_PEN} (soft oracle, DD32); the main gap is empty")
            orc[ORACLE_PEN] = np.nan
            ds.gap_reference = "none (soft oracle missing)"
        else:
            ds.gap_reference = "soft-penalty oracle J^pen_LP,ceil (DD32)"
        omap = orc.loc[~bad, ["test_cfg_id", "replay_id", "profit_eur", ORACLE_CEIL, ORACLE_PEN]].rename(
            columns={"profit_eur": "_oracle_lp", ORACLE_CEIL: "_oracle_ceil", ORACLE_PEN: "_oracle_pen"})
        ep = ep.merge(omap, on=["test_cfg_id", "replay_id"], how="left")
        pre = ep["oracle_profit_eur"]
        mism = pre.notna() & ep["_oracle_lp"].notna() & ~np.isclose(pre, ep["_oracle_lp"], rtol=1e-6, atol=1e-6)
        if mism.any():
            ds.log(f"WARNING: {int(mism.sum())} rows had a pre-filled oracle_profit_eur that differs from the "
                   f"oracle file; the oracle file value is used")
        ep["oracle_profit_eur"] = ep["_oracle_lp"]
        ep["oracle_ceil_eur"] = ep["_oracle_ceil"]
        ep["oracle_pen_eur"] = ep["_oracle_pen"]
        ep = ep.drop(columns=["_oracle_lp", "_oracle_ceil", "_oracle_pen"])
    else:
        msg = "no oracle__* directories found: optimality gap cannot be computed"
        if require_oracle:
            raise SchemaError(msg + " (pass require_oracle=False to continue without)")
        ds.log("WARNING: " + msg)
        ep["oracle_profit_eur"] = np.nan
        ep["oracle_ceil_eur"] = np.nan
        ep["oracle_pen_eur"] = np.nan
        ds.gap_reference = "none"

    miss = ep["oracle_profit_eur"].isna()
    if miss.any():
        cfgs = sorted(ep.loc[miss, "test_cfg_id"].unique())[:5]
        ds.log(f"WARNING: {int(miss.sum())} episodes without oracle profit (configs e.g. {cfgs}); gap = NaN")

    _shared_shortfall(ep, ds)
    _ensure_pen(ep, ds)
    ep["gap_abs"] = (ep["oracle_pen_eur"] - ep[PEN_COL]) / ep["N"]              # DD47, EUR per charger
    chk = ep[ep["oracle_pen_eur"].notna()]
    exceed = chk[PEN_COL] > chk["oracle_pen_eur"] + BOUND_TOL
    ds.bound_checked, ds.bound_violations = int(len(chk)), int(exceed.sum())
    ds.log(f"soft-oracle bound check: {ds.bound_violations} of {ds.bound_checked} episodes exceed J^pen_oracle "
           f"by more than {BOUND_TOL:g} EUR" + ("" if ds.bound_violations == 0 else
                                               f" (methods: {sorted(chk.loc[exceed, 'method'].unique())})"))
    ep["gap_hard_abs"] = (ep["oracle_ceil_eur"] - ep["profit_eur"]) / ep["N"]   # raw profit vs J_LP,ceil
    add_derived(ep)
    if len(orc):
        _shared_shortfall(orc, None)
        _ensure_pen(orc, None)
        add_derived(orc)
        orc["gap_abs"] = (orc[ORACLE_PEN] - orc[PEN_COL]) / orc["N"]
    ds.episodes, ds.oracle = ep.reset_index(drop=True), orc.reset_index(drop=True)
    ds.train = pd.concat(trs, ignore_index=True) if trs else pd.DataFrame(columns=TRAIN_COLS)
    ds.log(f"loaded {len(ep)} episodes, {ep['method'].nunique()} methods, "
           f"{ep['test_cfg_id'].nunique()} configurations, {len(orc)} oracle episodes, "
           f"{ds.train['exp_id'].nunique() if len(ds.train) else 0} training logs")
    return ds


def gap_pct(profit, oracle):
    """Per-episode relative gap in % (kept for diagnostics only; not used in tables, DD47)."""
    profit = np.asarray(profit, dtype=float)
    oracle = np.asarray(oracle, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = 100.0 * (oracle - profit) / np.abs(oracle)
    g[~np.isfinite(oracle) | (np.abs(oracle) < GAP_EPS)] = np.nan
    return g


def _shared_shortfall(df: pd.DataFrame, ds=None):
    """DD46: shortfall against the replay's shared admitted requirement for every pass."""
    if "energy_not_served_shared_kwh" in df.columns and df["energy_not_served_shared_kwh"].notna().any():
        df["energy_not_served_own_kwh"] = df["energy_not_served_kwh"]
        df["n_evs_unserved_own"] = df["n_evs_unserved"]
        df["energy_not_served_kwh"] = df["energy_not_served_shared_kwh"].fillna(df["energy_not_served_kwh"])
        if "n_evs_unserved_shared" in df.columns:
            df["n_evs_unserved"] = df["n_evs_unserved_shared"].fillna(df["n_evs_unserved"])
            df["frac_evs_unserved"] = np.where(df["n_evs"] > 0, df["n_evs_unserved"] / df["n_evs"].where(df["n_evs"] > 0), 0.0)
        if ds is not None:
            ds.log("DD46: energy not served and unserved EVs measured against the shared admitted requirement "
                   "(own-requirement values kept as *_own)")
    elif ds is not None:
        ds.log("NOTE: no energy_not_served_shared_kwh column; shortfall uses energy_not_served_kwh")


def _ensure_pen(df: pd.DataFrame, ds=None):
    """J^pen = profit - w*(energy not served vs the shared requirement + overload energy), DD46.
    Always recomputed; the file column penalised_profit_eur is ignored."""
    df[PEN_COL] = df["profit_eur"] - W_PEN * (df["energy_not_served_kwh"] + df["tr_violation_kwh"])
    if ds is not None:
        ds.log(f"DD46: {PEN_COL} = profit - {W_PEN:g}*(energy not served, shared requirement + overload energy) "
               f"for every method and pass; file column {FILE_PEN_COL} not used")


def add_derived(df: pd.DataFrame):
    df["unserved_pct"] = 100.0 * df["frac_evs_unserved"]
    df["fallback_steps"] = df[["fb0_steps", "fb1_steps", "fb2_steps", "fb1star_steps"]].sum(axis=1)
    df["profit_per_charger"] = df["profit_eur"] / df["N"]
    if PEN_COL in df.columns:
        df["pen_per_charger"] = df[PEN_COL] / df["N"]
    # DD43 counts used in T3c/T3d
    df["unserved_no_infeasible"] = np.where(df["infeasible_steps"] == 0, df["n_evs_unserved"], 0)
    if "n_evs_stay_infeasible" not in df.columns:
        df["n_evs_stay_infeasible"] = np.nan
    df["profit_per_ev"] = np.where(df["n_evs"] > 0, df["profit_eur"] / df["n_evs"].where(df["n_evs"] > 0), np.nan)
    # DD43: primary endpoint E4 (share of EVs whose stay has no infeasible step) and binding rates
    if "share_evs_stay_feasible" in df.columns or "n_evs_stay_infeasible" in df.columns:
        if "share_evs_stay_feasible" not in df.columns:
            df["share_evs_stay_feasible"] = np.nan
        if "n_evs_stay_infeasible" in df.columns:
            alt = 1 - df["n_evs_stay_infeasible"] / df["n_evs"].where(df["n_evs"] > 0)
            df["share_evs_stay_feasible"] = df["share_evs_stay_feasible"].fillna(alt)
        df["e4_pct"] = 100.0 * df["share_evs_stay_feasible"]
    steps = df["n_steps"] if "n_steps" in df.columns else pd.Series(np.nan, index=df.index)
    for col, out in (("agg_upper_active_steps", "upper_bind_pct"), ("agg_lower_active_steps", "lower_bind_pct")):
        if col in df.columns:
            df[out] = 100.0 * df[col] / steps.where(steps > 0)
    if "lb_active_frac" in df.columns:
        df["lb_active_pct"] = 100.0 * df["lb_active_frac"]
    # Lemma 3 (paper: Lemma S4): episode shortfall relative to the feasibility-deficit bound (episodes with a deficit only)
    df["shortfall_deficit_ratio"] = np.where(df["a5_deficit_kwh"] > 1e-9,
                                             df["energy_not_served_kwh"] / df["a5_deficit_kwh"].where(df["a5_deficit_kwh"] > 1e-9),
                                             np.nan)
    df["any_unserved_pct"] = 100.0 * (df["n_evs_unserved"] > 0).astype(float)
    df["any_violation_pct"] = 100.0 * (df["tr_violation_steps"] > 0).astype(float)
    # test-grid configuration (profile_N<N>_rho<rho>) vs. extra configurations (e.g. literature)
    exp = df["profile"].astype(str) + "_N" + df["N"].astype(int).astype(str) + "_rho" + \
        df["rho_tr"].map(lambda r: f"{float(r):g}")
    cid = df["test_cfg_id"].astype(str)
    base = cid.str.replace(r"_T\d+(?=_|$)", "", regex=True)    # smoke ids carry _T16/_T48 (also before a tag)
    df["cfg_base"] = base
    plain = base == exp
    rho = df["rho_tr"].astype(float)
    df["is_grid"] = plain & rho.isin([4.0, 7.0])               # main grid (DD24)
    sens = base.str.contains(r"_(?:soc|dt|thr|dpf)[0-9.]+$", regex=True)
    df["sens_tag"] = base.str.extract(r"_((?:soc|dt|thr|dpf)[0-9.]+)$", expand=False)
    df["cfg_family"] = np.select([df["is_grid"], plain & (rho == 3.0), sens, base.str.startswith("literature")],
                                 ["grid", "stress", "sensitivity", "literature"], "other")
    df["sens_base"] = base.str.replace(r"_(?:soc|dt|thr|dpf)[0-9.]+$", "", regex=True)
    if "conformal_coverage" in df.columns:
        cov = df["coverage_frac"] if "coverage_frac" in df.columns else pd.Series(np.nan, index=df.index)
        df["coverage_frac"] = cov.fillna(df["conformal_coverage"])
    return df


def _read_steps_file(d: Path) -> pd.DataFrame | None:
    pq, csv = d / "steps.parquet", d / "steps.csv"
    df = None
    if pq.exists():
        try:
            df = pd.read_parquet(pq)
        except ImportError:
            if not csv.exists():
                warnings.warn(f"{pq} needs pyarrow or fastparquet; skipped")
                return None
    if df is None and csv.exists():
        df = pd.read_csv(csv)
    if df is None:
        return None
    validate_columns(df, STEP_COLS, pq if pq.exists() else csv, "steps")
    if "method" not in df.columns:
        df["method"] = df["exp_id"].astype(str).str.split("__").str[0]
    return df


def load_steps(ds: Dataset, method: str | None = None, test_cfg_id: str | None = None) -> pd.DataFrame | None:
    """All step logs (steps.parquet, or steps.csv when pyarrow is missing), filtered by method and
    test_cfg_id (columns from SCHEMA_ADDITIONS.md §2). None if nothing matches."""
    parts = []
    for d in sorted(set(ds.run_dirs.values())):
        df = _read_steps_file(d)
        if df is None:
            continue
        if method is not None:
            df = df[df["method"] == method]
        if test_cfg_id is not None and "test_cfg_id" in df.columns:
            df = df[df["test_cfg_id"] == test_cfg_id]
        if len(df):
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else None


def runs_with_steps(ds: Dataset) -> list[str]:
    return [e for e, d in ds.run_dirs.items() if (d / "steps.parquet").exists() or (d / "steps.csv").exists()]
