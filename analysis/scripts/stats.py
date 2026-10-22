"""Statistics: seed-level aggregation, confidence intervals, IQM, paired tests.

Exact definitions are written to tables/STAT_METHODS.md by `write_stat_methods()`;
keep that text and this code in sync.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

CONF_LEVEL = 0.95
DEFAULT_B = 10_000          # bootstrap resamples for means
DEFAULT_B_IQM = 2_000       # bootstrap resamples for IQM (more expensive)
RNG_SEED = 20261022
WILCOXON_ZERO = "wilcox"    # zero differences are discarded (Wilcoxon 1945)


@dataclass
class StatConfig:
    B: int = DEFAULT_B
    B_iqm: int = DEFAULT_B_IQM
    rng_seed: int = RNG_SEED
    conf: float = CONF_LEVEL

    def rng(self, *salt) -> np.random.Generator:
        # deterministic, independent stream per group (order-independent results)
        h = zlib.crc32("\x1f".join(str(s) for s in salt).encode())   # stable across processes
        return np.random.default_rng([self.rng_seed, h])


# ----------------------------------------------------------------------------- basics

def holm(pvals) -> np.ndarray:
    """Holm (1979) step-down adjusted p-values. NaNs are ignored (stay NaN)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    m = q.size
    if m == 0:
        return out
    order = np.argsort(q, kind="mergesort")
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * q[idx])
        adj[idx] = min(1.0, running)
    out[ok] = adj
    return out


def iqm(x, axis=None):
    """Interquartile mean = 25 % trimmed mean (Agarwal et al. 2021)."""
    return sps.trim_mean(np.asarray(x, dtype=float), 0.25, axis=axis)


def t_ci(values, conf=CONF_LEVEL):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 2:
        return np.nan, np.nan
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(n)
    h = sps.t.ppf(0.5 + conf / 2, n - 1) * se
    return m - h, m + h


def _pct(boot, conf):
    a = (1 - conf) / 2
    return float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a))


# ----------------------------------------------------------------------------- group summary

def _matrices(sub: pd.DataFrame, metric: str):
    """Per configuration: dict seed -> Series(replay_id -> value)."""
    out = {}
    for cfg, g in sub.groupby("test_cfg_id", sort=True):
        out[cfg] = {s: gs.set_index("replay_id")[metric] for s, gs in g.groupby("seed", sort=True)}
    return out


def summarize(sub: pd.DataFrame, metric: str, sc: StatConfig, salt=(), want_iqm=False) -> dict:
    """Summary of one method over one or more configurations (stratum = test_cfg_id).

    unit = 'seed'  if the method has >= 2 seeds: statistic = mean over configurations of the
                   mean over seeds of the seed-level mean over replays; std/t-CI over the
                   seed-level aggregates; bootstrap resamples seeds within each configuration.
    unit = 'replay' otherwise (deterministic single run): CI over test replays.
    """
    vals_all = sub[metric].to_numpy(dtype=float)
    ok = np.isfinite(vals_all)
    n_nan = int((~ok).sum())
    sub = sub.loc[ok]
    v = vals_all[ok]
    cfg_codes, cfg_names = pd.factorize(sub["test_cfg_id"], sort=True)
    seed_codes, seed_names = pd.factorize(sub["seed"], sort=True)
    res = dict(metric=metric, n_episodes=int(v.size), n_dropped_nan=n_nan,
               n_cfgs=int(len(cfg_names)), n_seeds=int(len(seed_names)),
               exp_ids=sorted(pd.unique(sub["exp_id"]).tolist()))
    keys = ["mean", "std", "t_lo", "t_hi", "b_lo", "b_hi", "iqm", "iqm_lo", "iqm_hi", "std_episodes"]
    res.update({k: np.nan for k in keys})
    if v.size == 0:
        res["unit"] = "none"
        return res
    rng = sc.rng(metric, *salt)
    K, S = len(cfg_names), len(seed_names)
    res["std_episodes"] = float(v.std(ddof=1)) if v.size > 1 else np.nan
    seed_unit = S >= 2
    res["unit"] = "seed" if seed_unit else "replay"

    if seed_unit:
        key = cfg_codes * S + seed_codes
        cnt = np.bincount(key, minlength=K * S).reshape(K, S)
        tot = np.bincount(key, weights=v, minlength=K * S).reshape(K, S)
        with np.errstate(invalid="ignore", divide="ignore"):
            sm = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)      # cfg x seed means
        cfg_mean = np.nanmean(sm, axis=1)
        agg = np.nanmean(sm, axis=0)                                       # seed-level aggregates
        res["mean"] = float(cfg_mean.mean())
        res["std"] = float(np.std(agg, ddof=1)) if agg.size > 1 else np.nan
        lo, hi = t_ci(agg, sc.conf)
        shift = res["mean"] - float(agg.mean())                            # 0 when balanced
        res["t_lo"], res["t_hi"] = lo + shift, hi + shift
        boot = np.zeros(sc.B)
        for c in range(K):
            x = sm[c][np.isfinite(sm[c])]
            boot += x[rng.integers(0, x.size, (sc.B, x.size))].mean(axis=1)
        res["b_lo"], res["b_hi"] = _pct(boot / K, sc.conf)
    else:
        order = np.argsort(cfg_codes, kind="stable")
        groups = np.split(v[order], np.cumsum(np.bincount(cfg_codes, minlength=K))[:-1])
        res["mean"] = float(np.mean([g.mean() for g in groups]))
        res["std"] = np.nan
        se2 = sum(g.var(ddof=1) / g.size for g in groups if g.size > 1)
        nt = v.size
        if nt - K >= 1 and se2 > 0:
            h = sps.t.ppf(0.5 + sc.conf / 2, nt - K) * np.sqrt(se2) / K
            res["t_lo"], res["t_hi"] = res["mean"] - h, res["mean"] + h
        elif se2 == 0:
            res["t_lo"] = res["t_hi"] = res["mean"]
        boot = np.zeros(sc.B)
        for g in groups:
            boot += g[rng.integers(0, g.size, (sc.B, g.size))].mean(axis=1)
        res["b_lo"], res["b_hi"] = _pct(boot / K, sc.conf)

    if want_iqm:
        res.update(_iqm_boot(_matrices(sub, metric), seed_unit, sc, rng))
    return res


def _iqm_boot(mats, seed_unit, sc: StatConfig, rng):
    """IQM over all pooled (configuration, seed, replay) values; stratified bootstrap
    (resample seeds within each configuration, or replays if single-seed)."""
    blocks = []
    for c, d in mats.items():
        if seed_unit:
            common = sorted(set.intersection(*[set(v.index) for v in d.values()]))
            X = np.stack([d[s].loc[common].values for s in d])        # seeds x replays
        else:
            X = next(iter(d.values())).values[:, None]                  # replays x 1
        blocks.append(X)
    pooled = np.concatenate([X.ravel() for X in blocks])
    point = float(iqm(pooled))
    out = np.empty(sc.B_iqm)
    chunk = max(1, min(sc.B_iqm, int(2e7 // max(1, pooled.size))))
    done = 0
    while done < sc.B_iqm:
        b = min(chunk, sc.B_iqm - done)
        parts = []
        for X in blocks:
            idx = rng.integers(0, X.shape[0], (b, X.shape[0]))
            parts.append(X[idx].reshape(b, -1))
        out[done:done + b] = iqm(np.concatenate(parts, axis=1), axis=1)
        done += b
    lo, hi = _pct(out, sc.conf)
    return dict(iqm=point, iqm_lo=lo, iqm_hi=hi)


def summary_table(ep: pd.DataFrame, metrics, by, sc: StatConfig, iqm_metrics=()) -> pd.DataFrame:
    """One row per (method, *by, metric). `by` are grouping columns besides method;
    configurations inside a group are strata."""
    rows = []
    for key, g in ep.groupby(["method"] + list(by), sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        for m in metrics:
            if m not in g.columns:
                continue
            r = summarize(g, m, sc, salt=key, want_iqm=m in iqm_metrics)
            r.update(dict(zip(["method"] + list(by), key)))
            rows.append(r)
    cols = ["method"] + list(by) + ["metric", "unit", "mean", "std", "t_lo", "t_hi", "b_lo", "b_hi",
                                    "iqm", "iqm_lo", "iqm_hi", "std_episodes", "n_seeds", "n_cfgs",
                                    "n_episodes", "n_dropped_nan", "exp_ids"]
    if not rows:                                   # e.g. no grid configuration in a smoke run
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols]


# ----------------------------------------------------------------------------- differences

def diff_summary(ep: pd.DataFrame, a: str, b: str, metric: str, sc: StatConfig, salt=()) -> dict:
    """Difference a - b of the summary statistic over the configurations in `ep`
    (same configurations for both). CI: bootstrap resampling seeds of a and b independently
    within each configuration (runs are independent); t-CI: Welch on seed-level aggregates."""
    A = ep[ep["method"] == a].dropna(subset=[metric])
    Bm = ep[ep["method"] == b].dropna(subset=[metric])
    cfgs = sorted(set(A["test_cfg_id"]) & set(Bm["test_cfg_id"]))
    res = dict(a=a, b=b, metric=metric, n_cfgs=len(cfgs), diff=np.nan, b_lo=np.nan, b_hi=np.nan,
               t_lo=np.nan, t_hi=np.nan, exp_ids=[])
    if not cfgs:
        return res
    A, Bm = A[A["test_cfg_id"].isin(cfgs)], Bm[Bm["test_cfg_id"].isin(cfgs)]
    res["exp_ids"] = sorted(set(A["exp_id"]) | set(Bm["exp_id"]))
    rng = sc.rng("diff", a, b, metric, *salt)
    K = len(cfgs)

    def units(df):
        seed_unit = df["seed"].nunique() >= 2
        col = "seed" if seed_unit else "replay_id"
        return {c: g.groupby(col)[metric].mean().values for c, g in df.groupby("test_cfg_id")}, seed_unit

    ua, sa = units(A)
    ub, sb = units(Bm)
    res["diff"] = float(np.mean([ua[c].mean() - ub[c].mean() for c in cfgs]))
    boot = np.zeros(sc.B)
    for c in cfgs:
        va, vb = ua[c], ub[c]
        boot += va[rng.integers(0, va.size, (sc.B, va.size))].mean(1) - vb[rng.integers(0, vb.size, (sc.B, vb.size))].mean(1)
    res["b_lo"], res["b_hi"] = _pct(boot / K, sc.conf)
    if sa and sb:
        aa = A.groupby(["seed", "test_cfg_id"])[metric].mean().groupby("seed").mean().values
        bb = Bm.groupby(["seed", "test_cfg_id"])[metric].mean().groupby("seed").mean().values
        lo, hi = welch_ci(aa, bb, sc.conf)
        shift = res["diff"] - (aa.mean() - bb.mean())
        res["t_lo"], res["t_hi"] = lo + shift, hi + shift
    res["unit"] = f"{'seed' if sa else 'replay'}/{'seed' if sb else 'replay'}"
    return res


def welch_ci(a, b, conf=CONF_LEVEL):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or b.size < 2:
        return np.nan, np.nan
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    se = np.sqrt(va + vb)
    if se == 0:
        d = a.mean() - b.mean()
        return d, d
    df = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))
    h = sps.t.ppf(0.5 + conf / 2, df) * se
    d = a.mean() - b.mean()
    return d - h, d + h


# ----------------------------------------------------------------------------- paired tests

def rank_biserial(d) -> float:
    """Matched-pairs rank-biserial correlation (Kerby 2014) over non-zero differences."""
    d = np.asarray(d, float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = sps.rankdata(np.abs(d))
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def hedges_g(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return np.nan
    J = 1 - 3 / (4 * (na + nb) - 9)
    return float(J * (a.mean() - b.mean()) / sp)


def paired_test(ep: pd.DataFrame, a: str, b: str, cfg: str, metric: str, parts: dict | None = None) -> dict:
    if parts is not None:
        A, Bm = parts[(a, cfg)], parts[(b, cfg)]
    else:
        A = ep[(ep["method"] == a) & (ep["test_cfg_id"] == cfg)]
        Bm = ep[(ep["method"] == b) & (ep["test_cfg_id"] == cfg)]
    ra = A.groupby("replay_id")[metric].mean()      # mean over seeds per replay
    rb = Bm.groupby("replay_id")[metric].mean()
    j = pd.concat([ra.rename("a"), rb.rename("b")], axis=1, join="inner").dropna()
    d = (j["a"] - j["b"]).values
    out = dict(a=a, b=b, test_cfg_id=cfg, metric=metric, n_pairs=int(d.size),
               mean_a=float(j["a"].mean()) if d.size else np.nan,
               mean_b=float(j["b"].mean()) if d.size else np.nan,
               mean_diff=float(d.mean()) if d.size else np.nan,
               median_diff=float(np.median(d)) if d.size else np.nan,
               n_nonzero=int((d != 0).sum()), wilcoxon_W=np.nan, p_wilcoxon=np.nan,
               rank_biserial=np.nan, d_z=np.nan, welch_unit="", welch_t=np.nan, p_welch=np.nan,
               hedges_g=np.nan, note="",
               exp_ids=sorted(set(A["exp_id"]) | set(Bm["exp_id"])))
    if d.size == 0:
        out["note"] = "no matched replays"
        return out
    if (d != 0).sum() == 0:
        out.update(p_wilcoxon=1.0, rank_biserial=0.0, note="all paired differences are zero")
    else:
        try:
            w = sps.wilcoxon(d, zero_method=WILCOXON_ZERO, alternative="two-sided")
            out["wilcoxon_W"], out["p_wilcoxon"] = float(w.statistic), float(w.pvalue)
        except ValueError as e:
            out["note"] = f"wilcoxon: {e}"
        out["rank_biserial"] = rank_biserial(d)
        sd = d.std(ddof=1) if d.size > 1 else 0.0
        out["d_z"] = float(d.mean() / sd) if sd > 0 else np.nan
    # secondary: Welch t-test on independent units
    if A["seed"].nunique() >= 2 and Bm["seed"].nunique() >= 2:
        xa, xb = A.groupby("seed")[metric].mean().values, Bm.groupby("seed")[metric].mean().values
        out["welch_unit"] = "seed means"
    else:
        xa, xb = j["a"].values, j["b"].values
        out["welch_unit"] = "replay values (unpaired)"
    if xa.size >= 2 and xb.size >= 2 and (np.var(xa) + np.var(xb)) > 0:
        t = sps.ttest_ind(xa, xb, equal_var=False)
        out["welch_t"], out["p_welch"] = float(t.statistic), float(t.pvalue)
        out["hedges_g"] = hedges_g(xa, xb)
    elif xa.size >= 2 and xb.size >= 2:
        out["p_welch"] = 1.0 if xa.mean() == xb.mean() else 0.0
    return out


def paired_tests(ep: pd.DataFrame, ref: str, others, metrics) -> pd.DataFrame:
    rows = []
    parts = {k: g for k, g in ep.groupby(["method", "test_cfg_id"], sort=False)}
    for b in others:
        if b == ref:
            continue
        cfgs = sorted(set(ep.loc[ep["method"] == ref, "test_cfg_id"]) & set(ep.loc[ep["method"] == b, "test_cfg_id"]))
        for m in metrics:
            for c in cfgs:
                rows.append(paired_test(ep, ref, b, c, m, parts))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    cfg = ep.drop_duplicates("test_cfg_id").set_index("test_cfg_id")[["profile", "N", "rho_tr", "is_grid"]]
    df = df.join(cfg, on="test_cfg_id")
    # Holm family 1 (Alg. 6): across N within (comparison, metric, profile, rho_tr)
    df["p_holm_N"] = np.nan
    for _, g in df.groupby(["a", "b", "metric", "is_grid", "profile", "rho_tr"]):
        df.loc[g.index, "p_holm_N"] = holm(g["p_wilcoxon"].values)
    # Holm family 2 (conservative): across all configurations of (comparison, metric)
    df["p_holm_all"] = np.nan
    for _, g in df.groupby(["a", "b", "metric"]):
        df.loc[g.index, "p_holm_all"] = holm(g["p_wilcoxon"].values)
    df["p_welch_holm_N"] = np.nan
    for _, g in df.groupby(["a", "b", "metric", "is_grid", "profile", "rho_tr"]):
        df.loc[g.index, "p_welch_holm_N"] = holm(g["p_welch"].values)
    return df


# ----------------------------------------------------------------------------- documentation

def write_stat_methods(path: Path, sc: StatConfig = StatConfig()):
    """Exact description of every statistic (written next to the tables)."""
    txt = f"""# Statistical methods used in the tables and figures

Generated by `analysis/scripts/stats.py` (keep in sync with the code). Confidence level: {int(sc.conf * 100)} %.
Random generator: NumPy PCG64 seeded with {sc.rng_seed} and a hash of the group key, so results are reproducible.

## Units of analysis
- **Episode**: one evaluation of one policy on one fixed test replay (`replay_id`) of one configuration
  (`test_cfg_id` = profile × N × rho_tr). All methods are evaluated on the same replays (common random numbers).
- **Seed**: one independent training run of a learned method (5 seeds, 0-4). Training seeds are the
  independent replicates for learned methods [Henderson et al. 2018].
- **Seed-level mean**: mean over the test replays of one configuration for one seed. All seed statistics
  start from seed-level means ("seed-level means first").
- Deterministic methods (heuristics, eMPC; one run, `seed` constant) have no seed variability.
  For them the unit is the **test replay**, and tables mark the value with †.
- **Optimality gap** per episode: 100·(J* − J^π)/|J*| [%] on the same (test_cfg_id, replay_id).
  Main text and all main tables/figures (column `gap_pct`): J* = **J_LP,ceil**, the rounding-aware
  perfect-information optimum (oracle column `oracle_profit_ceil_eur`, Prop. O1(b), DD26), because
  EV2Gym's unbilled 0.01 kWh ceil gain lets feasible controllers exceed J_LP slightly.
  Supplementary table TS1 (column `gap_lp_pct`): J* = **J_LP** (oracle `profit_eur`).
  If an oracle file lacks `oracle_profit_ceil_eur`, the main gap falls back to J_LP and `load_report.txt` says so.
  Episodes with |J*| < 1e-9 € or a non-optimal oracle status get no gap (NaN, counted in `n_dropped_nan`).
  The gap is not clipped; it can be negative (e.g. a penalty-based method that violates constraints, or
  J_LP with the ceil gain).

## Point estimates and dispersion (columns `mean`, `std`)
- Single configuration: `mean` = mean over seeds of the seed-level means; `std` = sample standard deviation
  (ddof = 1) of the seed-level means (n = number of seeds).
- Pooled over several configurations (e.g. all profiles and tightness levels at one N): configurations are
  strata. Per seed, the seed-level means are averaged over the configurations (seed-level aggregate);
  `mean` = mean of the configuration means; `std` = standard deviation of the seed-level aggregates.
- Deterministic methods: `mean` as above with replays in place of seeds; `std` over seeds is undefined (–);
  `std_episodes` (per-episode standard deviation) is given in the CSV files.

## Confidence intervals (both are reported in the CSV files; tables state which one they show)
1. **t-based** (`t_lo`, `t_hi`): mean ± t_(n−1, 0.975)·s/√n over the n seed-level aggregates.
   Deterministic methods: stratified standard error √(Σ_c s_c²/n_c)/K over replays, t with Σn_c − K d.f.
2. **Bootstrap** (`b_lo`, `b_hi`): stratified percentile bootstrap with B = {sc.B} resamples
   [Agarwal et al. 2021]: within every configuration, seeds are resampled with replacement
   (replays for deterministic methods); the statistic is recomputed; the 2.5 % and 97.5 % quantiles are taken.
   With 5 seeds the bootstrap distribution is coarse; the t-based interval is the primary interval in the
   tables unless stated otherwise.
- **Differences between two methods** (cost of the guarantee, conformal α vs known departure):
  difference of the two means; bootstrap CI by resampling the seeds of each method independently
  within each configuration (independent training runs), B = {sc.B}; t-based CI by Welch's interval on the
  seed-level aggregates (Welch–Satterthwaite d.f.).

## Interquartile mean (option, columns `iqm`, `iqm_lo`, `iqm_hi`)
- IQM = 25 % trimmed mean (`scipy.stats.trim_mean(x, 0.25)`) of all pooled per-episode values
  (configurations × seeds × replays) [Agarwal et al. 2021]. It is computed for the optimality gap
  (a normalised score, comparable across configurations) and for single-configuration profit.
- CI: stratified percentile bootstrap, B = {sc.B_iqm}: seeds (or replays for deterministic methods)
  resampled within each configuration, all their episodes pooled, IQM recomputed.
  Only replays present for every seed of a configuration enter the IQM.

## Paired tests across methods (file `paired_tests.csv`)
- Pairs: reference method SA-Attn (`attn_safe`) against every other method, per configuration,
  on the matched test replays (inner join on `replay_id`).
- Per replay, the per-episode value is averaged over the seeds of each method; the paired difference is
  d_r = value(SA-Attn) − value(other).
- Metrics tested: per-episode profit (€), per-episode energy not served (kWh, departure shortfall),
  per-episode transformer violation energy (kWh).
- **Primary test: Wilcoxon signed-rank test** [Wilcoxon 1945], two-sided, `scipy.stats.wilcoxon` with
  `zero_method='{WILCOXON_ZERO}'` (zero differences discarded) and `method='auto'` (exact distribution for
  n ≤ 50 without ties, normal approximation otherwise). If all differences are zero, p = 1 (no test possible;
  noted in the `note` column).
- **Multiple comparisons: Holm step-down correction** [Holm 1979]. Primary family (as pre-registered in
  algorithm.md Alg. 6): the six station sizes N of one (comparison, metric, profile, rho_tr) → `p_holm_N`.
  Conservative alternative: all configurations of one (comparison, metric) → `p_holm_all`.
- **Effect sizes**: matched-pairs rank-biserial correlation r = (W+ − W−)/(W+ + W−) over non-zero
  differences [Kerby 2014], in [−1, 1], positive = SA-Attn larger; Cohen's d_z = mean(d)/sd(d).
- **Secondary test: Welch's t-test** (`scipy.stats.ttest_ind(equal_var=False)`), two-sided, on independent
  units: the seed-level means of the two methods when both have ≥ 2 seeds; otherwise the seed-averaged
  replay values treated as unpaired samples (`welch_unit` column). Effect size: Hedges' g.
  Holm-corrected across N as above (`p_welch_holm_N`).

## What is not done
- No test statistic pools seeds and replays as if they were independent replicates of training.
- No outliers are removed. Smoke-test rows (`smoke = true`) are dropped before any statistic.
"""
    Path(path).write_text(txt)
    return txt
