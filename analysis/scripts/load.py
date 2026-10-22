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
5. join both oracle profits per (test_cfg_id, replay_id) (DD26): `oracle_ceil_eur` = J_LP,ceil
   (oracle column `oracle_profit_ceil_eur`) and `oracle_profit_eur` = J_LP (oracle `profit_eur`);
6. optimality gap % per episode = 100 * (oracle - method) / |oracle|:
   `gap_pct` w.r.t. J_LP,ceil (main, DD26) and `gap_lp_pct` w.r.t. J_LP (supplementary).
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
ORACLE_CEIL = "oracle_profit_ceil_eur"        # J_LP,ceil (DD26); warned about if absent
# optional numeric columns from SCHEMA_ADDITIONS.md used by T7/F-S3 (conformal)
OPTIONAL_COLS = ["alpha", "conformal_coverage", "n_evs_departed_before_bound", "coverage_frac",
                 "mean_d_minus_dlo", "shortfall_departed_before_bound_kwh", ORACLE_CEIL]

TRAIN_COLS = [
    "exp_id", "env_steps", "wall_clock_s", "episode_return_eur_mean", "episode_return_eur_std",
    "frac_unserved_mean", "tr_violation_kwh_mean", "policy_loss", "value_loss", "entropy",
    "approx_kl", "clip_frac", "beta_reg", "proj_dist_mean", "lr",
]
TRAIN_NULLABLE = {"beta_reg", "proj_dist_mean", "episode_return_eur_std"}

STEP_COLS = ["exp_id", "replay_id", "t", "price_ch", "price_dis", "P_tr", "P_inf", "P_pv", "C", "C_rev",
             "sum_p", "sum_l", "sum_u", "n_connected", "fallback_code"]
# fallback_code: -1 none, 0 FB-0, 1 FB-1, 2 FB-2, 3 FB-1* (SCHEMA_ADDITIONS.md §2)

GAP_EPS = 1e-9          # |oracle| below this -> gap undefined (NaN)
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
    gap_reference: str = "J_LP,ceil (DD26)"

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
            ds.log(f"WARNING: oracle files have no {ORACLE_CEIL} (J_LP,ceil); the main gap falls back to J_LP")
            orc[ORACLE_CEIL] = orc["profit_eur"]
            ds.gap_reference = "J_LP (fallback: J_LP,ceil missing)"
        else:
            ds.gap_reference = "J_LP,ceil (DD26)"
        omap = orc.loc[~bad, ["test_cfg_id", "replay_id", "profit_eur", ORACLE_CEIL]].rename(
            columns={"profit_eur": "_oracle_lp", ORACLE_CEIL: "_oracle_ceil"})
        ep = ep.merge(omap, on=["test_cfg_id", "replay_id"], how="left")
        pre = ep["oracle_profit_eur"]
        mism = pre.notna() & ep["_oracle_lp"].notna() & ~np.isclose(pre, ep["_oracle_lp"], rtol=1e-6, atol=1e-6)
        if mism.any():
            ds.log(f"WARNING: {int(mism.sum())} rows had a pre-filled oracle_profit_eur that differs from the "
                   f"oracle file; the oracle file value is used")
        ep["oracle_profit_eur"] = ep["_oracle_lp"]
        ep["oracle_ceil_eur"] = ep["_oracle_ceil"]
        ep = ep.drop(columns=["_oracle_lp", "_oracle_ceil"])
    else:
        msg = "no oracle__* directories found: optimality gap cannot be computed"
        if require_oracle:
            raise SchemaError(msg + " (pass require_oracle=False to continue without)")
        ds.log("WARNING: " + msg)
        ep["oracle_profit_eur"] = np.nan
        ep["oracle_ceil_eur"] = np.nan
        ds.gap_reference = "none"

    miss = ep["oracle_profit_eur"].isna()
    if miss.any():
        cfgs = sorted(ep.loc[miss, "test_cfg_id"].unique())[:5]
        ds.log(f"WARNING: {int(miss.sum())} episodes without oracle profit (configs e.g. {cfgs}); gap = NaN")

    ep["gap_pct"] = gap_pct(ep["profit_eur"], ep["oracle_ceil_eur"])          # main (DD26)
    ep["gap_lp_pct"] = gap_pct(ep["profit_eur"], ep["oracle_profit_eur"])     # supplementary
    add_derived(ep)
    if len(orc):
        add_derived(orc)
        orc["gap_pct"] = gap_pct(orc["profit_eur"], orc[ORACLE_CEIL])
        orc["gap_lp_pct"] = 0.0
    ds.episodes, ds.oracle = ep.reset_index(drop=True), orc.reset_index(drop=True)
    ds.train = pd.concat(trs, ignore_index=True) if trs else pd.DataFrame(columns=TRAIN_COLS)
    ds.log(f"loaded {len(ep)} episodes, {ep['method'].nunique()} methods, "
           f"{ep['test_cfg_id'].nunique()} configurations, {len(orc)} oracle episodes, "
           f"{ds.train['exp_id'].nunique() if len(ds.train) else 0} training logs")
    return ds


def gap_pct(profit, oracle):
    """Optimality gap in %: 100*(oracle - method)/|oracle|; NaN if |oracle| < GAP_EPS or missing."""
    profit = np.asarray(profit, dtype=float)
    oracle = np.asarray(oracle, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = 100.0 * (oracle - profit) / np.abs(oracle)
    g[~np.isfinite(oracle) | (np.abs(oracle) < GAP_EPS)] = np.nan
    return g


def add_derived(df: pd.DataFrame):
    df["unserved_pct"] = 100.0 * df["frac_evs_unserved"]
    df["fallback_steps"] = df[["fb0_steps", "fb1_steps", "fb2_steps", "fb1star_steps"]].sum(axis=1)
    df["profit_per_charger"] = df["profit_eur"] / df["N"]
    df["any_unserved_pct"] = 100.0 * (df["n_evs_unserved"] > 0).astype(float)
    df["any_violation_pct"] = 100.0 * (df["tr_violation_steps"] > 0).astype(float)
    # test-grid configuration (profile_N<N>_rho<rho>) vs. extra configurations (e.g. literature)
    exp = df["profile"].astype(str) + "_N" + df["N"].astype(int).astype(str) + "_rho" + \
        df["rho_tr"].map(lambda r: f"{float(r):g}")
    cid = df["test_cfg_id"].astype(str)
    base = cid.str.replace(r"_T\d+$", "", regex=True)          # smoke ids carry _T16/_T48
    df["is_grid"] = base == exp
    df["cfg_family"] = np.where(df["is_grid"], "grid",
                                np.where(base.str.contains(r"_(?:soc|dt)[0-9.]+$", regex=True), "sensitivity", "other"))
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
