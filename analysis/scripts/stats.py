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


def _replay_blocks(sub: pd.DataFrame, metric: str):
    """Per configuration: seeds x replays matrix (NaN where missing) and the seed labels."""
    out = []
    for cfg, g in sub.groupby("test_cfg_id", sort=True):
        pv = g.pivot_table(index="seed", columns="replay_id", values=metric, aggfunc="mean").sort_index()
        out.append((cfg, pv))
    return out


def summarize(sub: pd.DataFrame, metric: str, sc: StatConfig, salt=(), want_iqm=False) -> dict:
    """DD48 summary of one method over one or more configurations (strata = test_cfg_id).

    mean      : mean over configurations of the mean over (seeds x replays)  [= seed-averaged policy];
    seed_min/max, seeds : the same statistic computed for each training seed separately (min-max over seeds);
    b_lo/b_hi : stratified percentile bootstrap over test replays (resampled within each configuration) of the
                seed-averaged per-replay values; deterministic methods: the same with their single run.
    No seed-level t interval is computed (2 seeds, DD48)."""
    sub = sub[np.isfinite(sub[metric].to_numpy(dtype=float))] if len(sub) else sub
    n_nan = 0
    res = dict(metric=metric, n_episodes=int(len(sub)), n_dropped_nan=n_nan,
               n_cfgs=int(sub["test_cfg_id"].nunique()) if len(sub) else 0,
               n_seeds=int(sub["seed"].nunique()) if len(sub) else 0,
               exp_ids=sorted(pd.unique(sub["exp_id"]).tolist()) if len(sub) else [])
    for k in ["mean", "std", "seed_min", "seed_max", "b_lo", "b_hi", "iqm", "iqm_lo", "iqm_hi", "std_episodes"]:
        res[k] = np.nan
    res["seeds"] = {}
    if len(sub) == 0:
        res["unit"] = "none"
        return res
    rng = sc.rng(metric, *salt)
    blocks = _replay_blocks(sub, metric)
    K = len(blocks)
    res["unit"] = "seeds" if res["n_seeds"] >= 2 else "single run"
    v = sub[metric].to_numpy(dtype=float)
    res["std_episodes"] = float(v.std(ddof=1)) if v.size > 1 else np.nan
    cfg_means, seed_vals = [], {}
    boot = np.zeros(sc.B)
    for cfg, pv in blocks:
        X = pv.to_numpy(dtype=float)
        rep_avg = np.nanmean(X, axis=0)                  # seed-averaged per replay
        rep_avg = rep_avg[np.isfinite(rep_avg)]
        cfg_means.append(float(np.nanmean(X)))
        for s_, row in zip(pv.index, X):
            if np.isfinite(row).any():
                seed_vals.setdefault(s_, []).append(float(np.nanmean(row)))
        if rep_avg.size:
            boot += rep_avg[rng.integers(0, rep_avg.size, (sc.B, rep_avg.size))].mean(axis=1)
    res["mean"] = float(np.mean(cfg_means))
    res["b_lo"], res["b_hi"] = _pct(boot / K, sc.conf)
    per_seed = {int(k) if float(k).is_integer() else k: float(np.mean(vv)) for k, vv in seed_vals.items()
                if len(vv) == K}                         # only seeds present in every configuration
    res["seeds"] = per_seed
    if len(per_seed) >= 2:
        res["seed_min"], res["seed_max"] = min(per_seed.values()), max(per_seed.values())
    if want_iqm:
        res.update(_iqm_boot(blocks, sc, rng))
    return res


def _iqm_boot(blocks, sc: StatConfig, rng):
    """IQM over all pooled (configuration, seed, replay) values; stratified bootstrap resampling replays
    (all seeds of a resampled replay kept) within each configuration."""
    mats = []
    for _, pv in blocks:
        X = pv.to_numpy(dtype=float)
        X = X[:, np.isfinite(X).all(axis=0)]              # replays present for every seed
        if X.size:
            mats.append(X)
    if not mats:
        return dict(iqm=np.nan, iqm_lo=np.nan, iqm_hi=np.nan)
    pooled = np.concatenate([X.ravel() for X in mats])
    point = float(iqm(pooled))
    out = np.empty(sc.B_iqm)
    chunk = max(1, min(sc.B_iqm, int(2e7 // max(1, pooled.size))))
    done = 0
    while done < sc.B_iqm:
        b = min(chunk, sc.B_iqm - done)
        parts = []
        for X in mats:
            idx = rng.integers(0, X.shape[1], (b, X.shape[1]))
            parts.append(X[:, idx].transpose(1, 0, 2).reshape(b, -1))
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
    cols = ["method"] + list(by) + ["metric", "unit", "mean", "seed_min", "seed_max", "seeds", "b_lo", "b_hi",
                                    "iqm", "iqm_lo", "iqm_hi", "std_episodes", "n_seeds", "n_cfgs",
                                    "n_episodes", "exp_ids"]
    if not rows:                                   # e.g. no grid configuration in a smoke run
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols]


# ----------------------------------------------------------------------------- differences

def _seed_avg_per_replay(df: pd.DataFrame, metric: str) -> pd.Series:
    """Seed-averaged policy value per test replay (mean over seeds of each seed's episode value)."""
    return df.groupby(["seed", "replay_id"])[metric].mean().groupby("replay_id").mean()


def diff_summary(ep: pd.DataFrame, a: str, b: str, metric: str, sc: StatConfig, salt=()) -> dict:
    """DD48: difference a - b of the seed-averaged policies, paired on the shared test replays of every
    configuration (mean over configurations of the mean replay-level difference). CI: stratified bootstrap
    resampling replays within each configuration. Per-seed differences of the aggregates are also returned."""
    A = ep[ep["method"] == a].dropna(subset=[metric])
    Bm = ep[ep["method"] == b].dropna(subset=[metric])
    cfgs = sorted(set(A["test_cfg_id"]) & set(Bm["test_cfg_id"]))
    res = dict(a=a, b=b, metric=metric, n_cfgs=len(cfgs), diff=np.nan, b_lo=np.nan, b_hi=np.nan,
               n_pairs=0, exp_ids=[], unit="paired replays")
    if not cfgs:
        return res
    A, Bm = A[A["test_cfg_id"].isin(cfgs)], Bm[Bm["test_cfg_id"].isin(cfgs)]
    res["exp_ids"] = sorted(set(A["exp_id"]) | set(Bm["exp_id"]))
    rng = sc.rng("diff", a, b, metric, *salt)
    ds_, boot, K = [], np.zeros(sc.B), 0
    for c in cfgs:
        ra = _seed_avg_per_replay(A[A["test_cfg_id"] == c], metric)
        rb = _seed_avg_per_replay(Bm[Bm["test_cfg_id"] == c], metric)
        d = (ra - rb).dropna().to_numpy()
        if not d.size:
            continue
        K += 1
        ds_.append(d.mean())
        res["n_pairs"] += int(d.size)
        boot += d[rng.integers(0, d.size, (sc.B, d.size))].mean(axis=1)
    if K == 0:
        return res
    res["diff"] = float(np.mean(ds_))
    res["b_lo"], res["b_hi"] = _pct(boot / K, sc.conf)
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

def write_stat_methods(path: Path, sc: StatConfig = StatConfig(), gap_eps: float = 1.0):
    """Exact description of every statistic (written next to the tables). DD45-DD48 version."""
    txt = f"""# Statistical methods used in the tables and figures

Generated by `analysis/scripts/stats.py` (keep in sync with the code). Version for the core runs with 2 training
seeds per learned method (DD45-DD48). Confidence level: {int(sc.conf * 100)} %. Random generator: NumPy PCG64
seeded with {sc.rng_seed} and a CRC32 hash of the group key (reproducible).

## Data scope
- Core set as run: learned methods with 2 training seeds (0, 1), rule-based and MPC baselines (one deterministic
  run), the conformal and no-admission evaluation passes, the in-distribution reference trained at N = 50, and
  the perfect-information optima. Not run (no rows, columns or figures): extended ablations, penalty-weight and
  other sensitivities except the stress level rho_tr = 3 kW, EV-GNN, the example-day figure.
- The zero-padded MLP baseline is evaluated only for N <= 20 (DD43(3)).

## Metrics
- **Energy not served** and **unserved EVs** are measured against the replay's **shared admitted requirement**
  (`energy_not_served_shared_kwh`, `n_evs_unserved_shared`) for every method and pass, including the conformal and
  no-admission passes (DD46). The pass's own values are kept as `*_own` in the summaries.
- **Penalised profit** J^pen = profit - 10 €/kWh x (energy not served + transformer overload energy), recomputed by
  the analysis for every method and pass (DD46). The column `penalised_profit_eur` of the result files is not used.
- **Bound check** (DD46): after loading, the number of episodes with J^pen > J^pen of the soft-penalty
  perfect-information optimum + 1e-6 € is counted (`load_report.txt`, `findings_numbers.md`); the analysis stops
  if it is not 0.
- **Optimality gap (primary, DD47)**: (J^pen_oracle - J^pen_method) / N in € per charger and episode, per episode;
  J^pen_oracle = soft-penalty optimum (`oracle_pen_profit_eur`). Tables report its mean (and its IQM in T1).
- **Relative gaps** (supplementary table TS1 only): ratios of sums 100 x sum(J* - J) / sum|J*| over all episodes
  of the seed-averaged policy, for the soft optimum (J^pen) and the hard optima J^LP,ceil and J^LP (raw profit).
  Per-episode percentages are not used because J^pen of the optimum is often close to zero or negative.
- **E4**: share of admitted EVs whose stay contains no infeasible step (`share_evs_stay_feasible`).
- **Cost of the guarantee** (T5, Fig. 5): penalty-based policy minus the same architecture with the safety layer,
  in J^pen (primary) and raw profit (secondary), as a ratio of sums over the matched replays of the seed-averaged
  policies: per admitted EV (denominator: admitted EVs, shared) and in % of |J^LP,ceil|.
- **Counts** (T3c, T3d): sums over all evaluated episodes of both seeds (not averaged): infeasible steps, fallback
  activations, EVs whose stay had an infeasible step, unserved EVs (shared requirement), overload steps, and the
  episode-level check of Corollary 1: unserved EVs in episodes with zero infeasible steps.
- Fallback cases: (i) = FB-1* (`fb1star_steps`), (ii) = FB-2, (iii) = FB-1, (iv) = FB-0.

## Seeds, replays and summaries (DD48)
- **Seed-averaged policy**: for a learned method, the value on a test replay is the mean over its 2 training
  seeds. All methods are evaluated on the same fixed test replays (common random numbers).
- Table cells: mean over configurations (strata) of the mean over replays of the seed-averaged policy, followed
  by **(min-max over seeds)**: the same statistic computed separately for each training seed. Deterministic
  methods (one run) are marked with † and have no range.
- **95 % CI** (`_values.csv`): stratified percentile bootstrap, B = {sc.B}: test replays are resampled with
  replacement within each configuration, the seed-averaged statistic is recomputed. **Per-seed values** are listed
  in the `_values.csv` files (`| seed k` columns). No seed-level t interval is computed (1 degree of freedom).
- **IQM** (T1): 25 % trimmed mean of the pooled per-episode gaps (configurations x seeds x replays); CI by the
  same replay bootstrap (B = {sc.B_iqm}).
- **Differences** (T2 transfer gap to the N = 50 reference; T7, Fig. S3): mean over configurations of the
  replay-level differences of the seed-averaged policies on the shared replays; CI: stratified replay bootstrap.

## Tests (files `stats/tests_T1.csv`, `tests_T2.csv`, `tests_T3.csv`, `tests_T5.csv`; DD48)
- T1 (N in {{10, 20}}) and T2 (all N, main grid): SA-Attn vs every other method shown in the table, metrics:
  optimality gap (primary), energy not served, overload energy. T3: SA-Attn vs the T1 methods, metrics E4,
  overload energy, energy not served. T5: each safety-layer method vs its penalty-based twin, J^pen and raw profit.
  All comparisons per station size N, with the configurations of that N pooled.
- **Primary test: Wilcoxon signed-rank test** (two-sided, `scipy.stats.wilcoxon`, `zero_method='{WILCOXON_ZERO}'`)
  on the replay-level differences of the seed-averaged policies (reference minus other) over all shared replays of
  the group; effect size: matched-pairs rank-biserial correlation.
- **Multiple comparisons: Holm** correction over all comparisons of one test file (family = table):
  `p_wilcoxon_holm`.
- **CI of the difference**: stratified bootstrap over replays (within configurations), B = {sc.B}.
- **Per-seed values**: the group statistic of each seed of both methods (`per_seed_ref`, `per_seed_other`).
- With 2 seeds, between-seed variability is not estimated by any test; the seed count is a limitation and the
  (min-max over seeds) ranges are shown for this reason.

## What is not done
- No seed-level t-tests or t intervals (DD48). No adaptive seed rule. No outlier removal. Smoke-test rows are
  dropped before any statistic.
"""
    Path(path).write_text(txt)
    return txt


# ----------------------------------------------------------------------------- DD38: seed-level comparisons

def _seed_matrix(df: pd.DataFrame, metric: str, replays=None):
    """seeds x replays matrix (rows sorted by seed) and the seed labels; replays restricted if given."""
    pv = df.pivot_table(index="seed", columns="replay_id", values=metric, aggfunc="mean")
    if replays is not None:
        pv = pv.reindex(columns=replays)
    return pv.sort_index()


def _poi(a, b, higher_better=True):
    """P(X_a better than X_b) over all seed pairs (ties count 1/2) [Agarwal et al. 2021]."""
    a, b = np.asarray(a, float)[:, None], np.asarray(b, float)[None, :]
    gt = (a > b) if higher_better else (a < b)
    return float((gt + 0.5 * (a == b)).mean())


def seed_comparisons(ep: pd.DataFrame, ref: str, others, metrics, by, sc: StatConfig,
                     higher_better: dict | None = None, family: str = "") -> pd.DataFrame:
    """DD48 comparisons of seed-averaged policies on the shared test replays, per group of `by`:
      primary : Wilcoxon signed-rank test on the replay-level differences (ref - other) of the seed-averaged
                policies, all replays of the group's configurations pooled; Holm over all rows (family = table);
      CI      : stratified bootstrap of the mean difference, resampling replays within each configuration;
      per seed: the difference of the group statistic for every seed of `ref` (and of `other`) is listed."""
    higher_better = higher_better or {}
    rows = []
    parts = {k: g for k, g in ep.groupby(["method"] + list(by), sort=False)}
    groups = sorted({k[1:] for k in parts if k[0] == ref})
    for other in others:
        if other == ref:
            continue
        for gkey in groups:
            A_all, B_all = parts.get((ref,) + gkey), parts.get((other,) + gkey)
            if A_all is None or B_all is None:
                continue
            for metric in metrics:
                if metric not in A_all.columns:
                    continue
                rows.append(_one_comparison(A_all, B_all, ref, other, metric, higher_better.get(metric, True), sc,
                                            salt=(family, other, metric) + tuple(map(str, gkey)))
                            | dict(zip(by, gkey)))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.insert(0, "family", family)
    df["p_wilcoxon_holm"] = holm(df["p_wilcoxon"].values)
    return df


def _group_stat_by_seed(df, metric):
    """Mean over configurations of each seed's mean (seeds present in every configuration)."""
    g = df.groupby(["seed", "test_cfg_id"])[metric].mean().unstack("test_cfg_id")
    g = g.dropna(axis=0, how="any")
    return g.mean(axis=1)


def _one_comparison(A_all, B_all, ref, other, metric, hb, sc, salt):
    d = diff_summary(pd.concat([A_all, B_all]), ref, other, metric, sc, salt=salt)
    out = dict(ref=ref, other=other, metric=metric, higher_is_better=hb, n_cfgs=d["n_cfgs"],
               n_seeds_ref=int(A_all["seed"].nunique()), n_seeds_other=int(B_all["seed"].nunique()),
               mean_ref=np.nan, mean_other=np.nan, diff=d["diff"], b_lo=d["b_lo"], b_hi=d["b_hi"],
               n_pairs=d["n_pairs"], wilcoxon_W=np.nan, p_wilcoxon=np.nan, rank_biserial=np.nan,
               per_seed_ref="", per_seed_other="", exp_ids=";".join(d["exp_ids"]))
    cfgs = sorted(set(A_all["test_cfg_id"]) & set(B_all["test_cfg_id"]))
    if not cfgs:
        return out
    A, Bm = A_all[A_all["test_cfg_id"].isin(cfgs)], B_all[B_all["test_cfg_id"].isin(cfgs)]
    out["mean_ref"] = float(A.groupby("test_cfg_id")[metric].mean().mean())
    out["mean_other"] = float(Bm.groupby("test_cfg_id")[metric].mean().mean())
    sa, sb = _group_stat_by_seed(A, metric), _group_stat_by_seed(Bm, metric)
    out["per_seed_ref"] = "; ".join(f"s{int(k)}={v:.6g}" for k, v in sa.items())
    out["per_seed_other"] = "; ".join(f"s{int(k)}={v:.6g}" for k, v in sb.items())
    rdiff = []
    for c in cfgs:
        ra = _seed_avg_per_replay(A[A["test_cfg_id"] == c], metric)
        rb = _seed_avg_per_replay(Bm[Bm["test_cfg_id"] == c], metric)
        rdiff.append((ra - rb).dropna().to_numpy())
    dd = np.concatenate(rdiff) if rdiff else np.array([])
    dd = dd[np.isfinite(dd)]
    if dd.size and (dd != 0).any():
        try:
            w = sps.wilcoxon(dd, zero_method=WILCOXON_ZERO)
            out["wilcoxon_W"], out["p_wilcoxon"] = float(w.statistic), float(w.pvalue)
        except ValueError:
            pass
        out["rank_biserial"] = rank_biserial(dd)
    elif dd.size:
        out["p_wilcoxon"], out["rank_biserial"] = 1.0, 0.0
    return out


# ----------------------------------------------------------------------------- DD43(7): ratio of sums

def ratio_diff(ep: pd.DataFrame, a: str, b: str, num: str, den: str, sc: StatConfig, salt=(), abs_den=False) -> dict:
    """Ratio-of-sums difference  sum_r (a_r - b_r) / sum_r D_r  over the matched replays r of all configurations
    in `ep`; a_r, b_r = seed-averaged values on replay r; D_r = per-replay denominator (identical for both
    methods). CI: stratified bootstrap resampling replays within each configuration (DD48)."""
    res = dict(a=a, b=b, num=num, den=den, diff=np.nan, b_lo=np.nan, b_hi=np.nan, exp_ids=[], unit="ratio of sums")
    if num not in ep.columns or den not in ep.columns:
        return res
    A, Bm = ep[ep["method"] == a], ep[ep["method"] == b]
    cfgs = sorted(set(A["test_cfg_id"]) & set(Bm["test_cfg_id"]))
    blocks = []
    for c in cfgs:
        ac, bc = A[A["test_cfg_id"] == c].dropna(subset=[num]), Bm[Bm["test_cfg_id"] == c].dropna(subset=[num])
        dvec = (_seed_avg_per_replay(ac, num) - _seed_avg_per_replay(bc, num)).dropna()
        D = pd.concat([ac, bc]).groupby("replay_id")[den].mean().reindex(dvec.index).to_numpy(dtype=float)
        if abs_den:
            D = np.abs(D)
        ok = np.isfinite(D)
        if ok.any():
            blocks.append((dvec.to_numpy()[ok], D[ok]))
    if not blocks:
        return res
    res["exp_ids"] = sorted(set(A["exp_id"]) | set(Bm["exp_id"]))
    den_tot = sum(dd.sum() for _, dd in blocks)
    if den_tot == 0:
        return res
    res["diff"] = float(sum(x.sum() for x, _ in blocks) / den_tot)
    rng = sc.rng("ratio", a, b, num, den, *salt)
    bn, bd = np.zeros(sc.B), np.zeros(sc.B)
    for x, dd in blocks:
        ir = rng.integers(0, x.size, (sc.B, x.size))
        bn += x[ir].sum(axis=1)
        bd += dd[ir].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = bn / bd
    boot = boot[np.isfinite(boot)]
    if boot.size:
        res["b_lo"], res["b_hi"] = _pct(boot, sc.conf)
    return res


def rel_gap_ratio(df: pd.DataFrame, value: str, oracle: str, sc: StatConfig, salt=()) -> dict:
    """Relative gap as a ratio of sums (DD47, supplementary): 100 * sum_r (O_r - x_r) / sum_r |O_r| over the
    replays of all configurations in `df` (one method; x_r seed-averaged). CI: stratified replay bootstrap."""
    res = dict(diff=np.nan, b_lo=np.nan, b_hi=np.nan, exp_ids=sorted(df["exp_id"].unique().tolist()) if len(df) else [])
    blocks = []
    for c, g in df.groupby("test_cfg_id"):
        x = _seed_avg_per_replay(g.dropna(subset=[value]), value)
        o = g.groupby("replay_id")[oracle].mean().reindex(x.index)
        ok = o.notna() & x.notna()
        if ok.any():
            blocks.append(((o - x)[ok].to_numpy(), o[ok].abs().to_numpy()))
    if not blocks:
        return res
    den = sum(d.sum() for _, d in blocks)
    if den == 0:
        return res
    res["diff"] = 100 * float(sum(n.sum() for n, _ in blocks) / den)
    rng = sc.rng("relgap", value, oracle, *salt)
    bn, bd = np.zeros(sc.B), np.zeros(sc.B)
    for n, d in blocks:
        ir = rng.integers(0, n.size, (sc.B, n.size))
        bn += n[ir].sum(axis=1)
        bd += d[ir].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = 100 * bn / bd
    boot = boot[np.isfinite(boot)]
    if boot.size:
        res["b_lo"], res["b_hi"] = _pct(boot, sc.conf)
    return res
