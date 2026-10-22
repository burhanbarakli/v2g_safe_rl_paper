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

def write_stat_methods(path: Path, sc: StatConfig = StatConfig(), gap_eps: float = 1.0):
    """Exact description of every statistic (written next to the tables)."""
    txt = f"""# Statistical methods used in the tables and figures

Generated by `analysis/scripts/stats.py` (keep in sync with the code). Confidence level: {int(sc.conf * 100)} %.
Random generator: NumPy PCG64 seeded with {sc.rng_seed} and a CRC32 hash of the group key (reproducible).

## Metrics (DD32, DD43)
- **Penalised profit** J^pen = profit − w_E·(energy not served) − w_P·(transformer overload energy), w_E = w_P =
  10 €/kWh for every method (column `penalised_profit_eur`; if a file lacks it, it is computed from this formula).
  Energy not served is measured against the shared admitted requirement, which is the same for every method and
  pass (DD31, DD43(1)). J^pen is the **main comparison metric**. Raw profit, energy not served and overload are
  reported next to it.
- **Optimality gap** per episode = 100·(J* − J^π)/|J*| [%] on the same (test_cfg_id, replay_id):
  - main (`gap_pct`): J^π = J^pen of the method, J* = soft-penalty perfect-information optimum
    J^pen,LP,ceil (`oracle_pen_profit_eur`); computed for every method, including the conformal variants
    (DD43(1) supersedes DD42(2));
  - hard (`gap_hard_pct`): raw profit vs the hard lexicographic optimum J^LP,ceil (`oracle_profit_ceil_eur`);
  - supplementary (`gap_lp_pct`, table TS1): raw profit vs J^LP (oracle `profit_eur`, DD42-4).
  - Episodes with |J*| < {gap_eps:g} € or a non-optimal oracle status get no gap; their number per gap type is
    listed in `load_report.txt` and `findings_numbers.md`. Gaps are not clipped.
- **E4 (primary endpoint)**: `share_evs_stay_feasible` = 1 − (EVs whose stay contains at least one infeasible
  step)/(admitted EVs), per episode (T3 pooled, T3b and Fig. 4 per N; test file `tests_T3.csv`).
- **E3**: transformer overload energy per episode (`tr_violation_kwh`, kWh).
- **Binding rates** (T3b): steps with the aggregate upper / lower limit binding (`agg_upper_active_steps`,
  `agg_lower_active_steps`) in % of the episode's steps (`n_steps`); `lb_active_frac` = share of occupied
  charger-steps with p_i = l_i > p_i^min, in %.
- **Cost of the guarantee** (T5, main Fig. 5 = `fig6_cost_of_guarantee`; DD43(7)): penalty-based policy minus the
  same architecture with the safety layer, in J^pen (primary) and in raw profit (secondary), as a **ratio of
  sums** over the matched episodes of the station size: Δ = Σ_r (x̄_pen,r − x̄_safe,r) / Σ_r D_r, where x̄_·,r is
  the seed-averaged value on replay r (all configurations of the N pooled) and D_r = admitted EVs on the replay
  (identical for both methods; "€ per EV served") or |J^LP,ceil| ("% of |J^LP,ceil|", ×100).
- **Shortfall / deficit bound** (T3): per episode with a positive feasibility deficit (`a5_deficit_kwh` > 1e-9 kWh,
  bound of Lemma S4): energy not served / a5_deficit_kwh; mean over those episodes, safety-layer methods only.
- **Fallback cases** (T3b, SCHEMA_ADDITIONS.md §5): paper case (i) = FB-1* (`fb1star_steps`, code 3),
  (ii) = FB-2 (`fb2_steps`, code 2), (iii) = FB-1 (`fb1_steps`, code 1), (iv) = FB-0 (`fb0_steps`, code 0).
- The zero-padded MLP baseline (N_max = 20) is evaluated and reported only for N ≤ 20 (DD43(3)); rows with N > 20
  are dropped by the loader.

## Units of analysis (DD38, DD43(5))
- **Seed** = one independent training run of a learned method. The number of seeds is fixed in advance: 5 (3 for
  the N = 50 reference `attn_safe_trainN50`); there is no adaptive seed rule. The seed is the unit of analysis:
  replay-level values are averaged within each seed first (seed-level mean).
- Configurations (`test_cfg_id` = profile × N × rho_tr) are strata: pooled statistics average over
  configurations, and bootstrap resampling is done within each configuration.
- Deterministic methods (heuristics, MPC; one run) have no seed variability; for them the replay is the
  resampling unit and tables mark the value with †.

## Point estimates, dispersion and confidence intervals (tables T1–T7, TS1–TS3)
- `mean`: mean over configurations of the mean over seeds of the seed-level means; `std`: standard deviation of
  the seed-level aggregates (ddof = 1). Deterministic methods: std over seeds undefined (–).
- 95 % CIs in the `_values.csv` files: t-based over seed-level aggregates (deterministic methods: stratified
  standard error over replays) and stratified percentile bootstrap (B = {sc.B}).
- **IQM** (interquartile mean, 25 % trimmed mean) of the pooled per-episode main gap, with a stratified bootstrap
  CI (B = {sc.B_iqm}) [Agarwal et al. 2021].
- Differences (T2 transfer gap, T7 Δ profit): difference of means with a stratified bootstrap CI over the seeds of
  both methods. Ratio-of-sums differences (T5): stratified bootstrap over the seeds of both methods and the common
  replays (B = {sc.B}).

## Hypothesis tests (files `stats/tests_T1.csv`, `tests_T2.csv`, `tests_T3.csv`, `tests_T5.csv`; DD43(5))
- Comparisons, all per station size N (the configurations of that N pooled as strata):
  - T1 (N ∈ {{10, 20}}) and T2 (all N): SA-Attn vs every other non-conformal method; metrics J^pen (main),
    energy not served, overload energy;
  - T3: SA-Attn vs every other non-conformal method; metrics E4 (`share_evs_stay_feasible`), E3 (overload energy),
    energy not served;
  - T5: each safety-layer method vs its penalty-based twin (attention, MLP); metrics J^pen (primary) and raw profit
    (secondary), per episode.
- Difference = mean(reference) − mean(other) of the seed-level aggregates (column `diff`), with its stratified
  bootstrap 95 % CI (seeds of each method and the common replays resampled; `b_lo`, `b_hi`).
- **Primary test: paired t-test on seed means** (`scipy.stats.ttest_rel`, two-sided). The seed-level aggregates of
  the two methods are paired by seed index (training seed k of each method; both evaluated on the same fixed test
  replays). Against a deterministic baseline (one run), the seed-level differences to its value are tested with a
  one-sample t-test. If the seed indices do not match, Welch's t-test is used (column `test`). As a robustness
  check, the unpaired Welch t-test on the same seed means is reported (`p_welch_robustness`, not Holm-corrected).
- **Secondary test: Wilcoxon signed-rank test** (two-sided, `zero_method='{WILCOXON_ZERO}'`) on the replay-level
  differences of the seed-averaged policies (all replays of the N pooled); effect size: matched-pairs
  rank-biserial correlation.
- **Probability of improvement** P(reference better than other) on seed-level means (higher J^pen, E4 and profit
  are better; lower shortfall and overload are better; ties count 1/2), averaged over configurations
  [Agarwal et al. 2021]; CI by stratified seed bootstrap (B = {sc.B_iqm}).
- **Holm correction per table family** (DD43(5)): all comparisons in one test file (all methods × N × metrics of
  that table) form one family; columns `p_t_holm` (primary) and `p_wilcoxon_holm` (secondary).
- With 5 seeds the t-tests have low power; the confidence intervals are the primary evidence.

## What is not done
- No test treats replays as independent replicates of training.
- No outliers are removed. Smoke-test rows (`smoke = true`) are dropped before any statistic.
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
    """DD38. For every (other method, group of `by`, metric): difference ref - other of the mean over
    configurations (strata) of seed-averaged values.
      primary  : paired t-test over seeds on the seed-level aggregates (pairs by seed label; one-sample
                 t-test against a deterministic baseline); Welch if the seed labels do not match;
      CI       : stratified bootstrap (configurations as strata; seeds of each method and the common replays
                 resampled), percentile 95 %;
      secondary: Wilcoxon signed-rank on replay-level differences of the seed-averaged policies;
      PoI      : probability of improvement of ref over other on seed means, averaged over configurations;
      Holm     : over all rows of the returned frame (family = one table)."""
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
                hb = higher_better.get(metric, True)
                rows.append(_one_comparison(A_all, B_all, ref, other, metric, hb, sc,
                                            salt=(family, other, metric) + tuple(map(str, gkey)))
                            | dict(zip(by, gkey)))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.insert(0, "family", family)
    df["p_t_holm"] = holm(df["p_t"].values)
    df["p_wilcoxon_holm"] = holm(df["p_wilcoxon"].values)
    return df


def _one_comparison(A_all, B_all, ref, other, metric, hb, sc, salt):
    cfgs = sorted(set(A_all["test_cfg_id"]) & set(B_all["test_cfg_id"]))
    out = dict(ref=ref, other=other, metric=metric, higher_is_better=hb, n_cfgs=len(cfgs),
               n_seeds_ref=int(A_all["seed"].nunique()), n_seeds_other=int(B_all["seed"].nunique()),
               mean_ref=np.nan, mean_other=np.nan, diff=np.nan, b_lo=np.nan, b_hi=np.nan, test="",
               t=np.nan, p_t=np.nan, n_pairs=0, p_welch_robustness=np.nan, wilcoxon_W=np.nan, p_wilcoxon=np.nan, rank_biserial=np.nan,
               poi=np.nan, poi_lo=np.nan, poi_hi=np.nan,
               exp_ids=";".join(sorted(set(A_all["exp_id"]) | set(B_all["exp_id"]))))
    if not cfgs:
        return out
    rng = sc.rng("cmp", *salt)
    mats, aggA, aggB, rdiff, pois = [], {}, {}, [], []
    for c in cfgs:
        a, b = A_all[A_all["test_cfg_id"] == c].dropna(subset=[metric]), B_all[B_all["test_cfg_id"] == c].dropna(subset=[metric])
        reps = sorted(set(a["replay_id"]) & set(b["replay_id"]))
        if not reps:
            continue
        MA, MB = _seed_matrix(a, metric, reps), _seed_matrix(b, metric, reps)
        MA, MB = MA.dropna(axis=0, how="all"), MB.dropna(axis=0, how="all")
        mats.append((MA.to_numpy(), MB.to_numpy()))
        for s_, v in MA.mean(axis=1).items():
            aggA.setdefault(s_, []).append(v)
        for s_, v in MB.mean(axis=1).items():
            aggB.setdefault(s_, []).append(v)
        rdiff.append(np.nanmean(MA.to_numpy(), 0) - np.nanmean(MB.to_numpy(), 0))
        pois.append(_poi(MA.mean(axis=1).values, MB.mean(axis=1).values, hb))
    if not mats:
        return out
    K = len(mats)
    out["mean_ref"] = float(np.mean([np.nanmean(ma) for ma, _ in mats]))
    out["mean_other"] = float(np.mean([np.nanmean(mb) for _, mb in mats]))
    out["diff"] = out["mean_ref"] - out["mean_other"]
    # stratified bootstrap over seeds (each method) and common replays
    boot = np.zeros(sc.B)
    for ma, mb in mats:
        R = ma.shape[1]
        ir = rng.integers(0, R, (sc.B, R))
        ra = np.nanmean(ma[rng.integers(0, ma.shape[0], (sc.B, ma.shape[0]))], axis=1)   # (B, R)
        rb = np.nanmean(mb[rng.integers(0, mb.shape[0], (sc.B, mb.shape[0]))], axis=1)
        boot += np.take_along_axis(ra - rb, ir, axis=1).mean(axis=1)
    out["b_lo"], out["b_hi"] = _pct(boot / K, sc.conf)
    # primary: paired t-test over seeds on aggregates over configurations
    sa = {s_: np.mean(v) for s_, v in aggA.items()}
    sb = {s_: np.mean(v) for s_, v in aggB.items()}
    common = sorted(set(sa) & set(sb))
    if len(sa) >= 2 and len(sb) >= 2 and len(common) >= 2:
        x, y = np.array([sa[s_] for s_ in common]), np.array([sb[s_] for s_ in common])
        d = x - y
        out["test"], out["n_pairs"] = "paired t (seed label)", len(common)
        if np.ptp(d) > 0:
            r = sps.ttest_rel(x, y)
            out["t"], out["p_t"] = float(r.statistic), float(r.pvalue)
        else:
            out["p_t"] = 1.0 if d[0] == 0 else 0.0
    if len(sa) >= 2 and len(sb) >= 2:
        w_ = sps.ttest_ind(list(sa.values()), list(sb.values()), equal_var=False)
        out["p_welch_robustness"] = float(w_.pvalue) if np.isfinite(w_.pvalue) else np.nan
    if out["test"]:
        pass
    elif len(sa) >= 2 and len(sb) == 1:
        d = np.array(list(sa.values())) - next(iter(sb.values()))
        out["test"], out["n_pairs"] = "one-sample t vs deterministic", len(d)
        if np.ptp(d) > 0:
            r = sps.ttest_1samp(d, 0.0)
            out["t"], out["p_t"] = float(r.statistic), float(r.pvalue)
    elif len(sa) == 1 and len(sb) >= 2:
        d = next(iter(sa.values())) - np.array(list(sb.values()))
        out["test"], out["n_pairs"] = "one-sample t vs deterministic", len(d)
        if np.ptp(d) > 0:
            r = sps.ttest_1samp(d, 0.0)
            out["t"], out["p_t"] = float(r.statistic), float(r.pvalue)
    elif len(sa) >= 2 and len(sb) >= 2:
        r = sps.ttest_ind(list(sa.values()), list(sb.values()), equal_var=False)
        out["test"], out["t"], out["p_t"] = "Welch (seed labels differ)", float(r.statistic), float(r.pvalue)
    else:
        out["test"] = "none (single run each)"
    # secondary: Wilcoxon on replay-level differences of seed-averaged policies
    dd = np.concatenate(rdiff)
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
    # probability of improvement with a stratified seed bootstrap
    out["poi"] = float(np.mean(pois))
    bp = np.zeros(sc.B_iqm)
    for ma, mb in mats:
        xa, xb = np.nanmean(ma, 1), np.nanmean(mb, 1)
        ia = rng.integers(0, xa.size, (sc.B_iqm, xa.size))
        ib = rng.integers(0, xb.size, (sc.B_iqm, xb.size))
        A3, B3 = xa[ia][:, :, None], xb[ib][:, None, :]
        gt = (A3 > B3) if hb else (A3 < B3)
        bp += (gt + 0.5 * (A3 == B3)).mean(axis=(1, 2))
    out["poi_lo"], out["poi_hi"] = _pct(bp / K, sc.conf)
    return out


# ----------------------------------------------------------------------------- DD43(7): ratio of sums

def ratio_diff(ep: pd.DataFrame, a: str, b: str, num: str, den: str, sc: StatConfig, salt=(), abs_den=False) -> dict:
    """Ratio-of-sums difference  sum_r (mean_seeds a_r - mean_seeds b_r) / sum_r D_r  over the matched
    replays r of all configurations in `ep` (seed-averaged per replay first). D_r = per-replay denominator
    (e.g. admitted EVs, or |J_LP,ceil|), identical for both methods on a replay. CI: stratified bootstrap
    (configurations as strata; seeds of each method and the common replays resampled)."""
    res = dict(a=a, b=b, num=num, den=den, diff=np.nan, b_lo=np.nan, b_hi=np.nan, exp_ids=[], unit="ratio of sums")
    A, Bm = ep[ep["method"] == a], ep[ep["method"] == b]
    if num not in ep.columns or den not in ep.columns:
        return res
    cfgs = sorted(set(A["test_cfg_id"]) & set(Bm["test_cfg_id"]))
    blocks = []
    for c in cfgs:
        ac, bc = A[A["test_cfg_id"] == c].dropna(subset=[num]), Bm[Bm["test_cfg_id"] == c].dropna(subset=[num])
        reps = sorted(set(ac["replay_id"]) & set(bc["replay_id"]))
        if not reps:
            continue
        MA = _seed_matrix(ac, num, reps).dropna(axis=0, how="all").to_numpy()
        MB = _seed_matrix(bc, num, reps).dropna(axis=0, how="all").to_numpy()
        D = pd.concat([ac, bc]).groupby("replay_id")[den].mean().reindex(reps).to_numpy(dtype=float)
        if abs_den:
            D = np.abs(D)
        ok = np.isfinite(D)
        if not ok.any():
            continue
        blocks.append((MA[:, ok], MB[:, ok], D[ok]))
    if not blocks:
        return res
    res["exp_ids"] = sorted(set(A["exp_id"]) | set(Bm["exp_id"]))
    num_tot = sum(np.nansum(np.nanmean(ma, 0) - np.nanmean(mb, 0)) for ma, mb, _ in blocks)
    den_tot = sum(d.sum() for _, _, d in blocks)
    if den_tot == 0:
        return res
    res["diff"] = float(num_tot / den_tot)
    rng = sc.rng("ratio", a, b, num, den, *salt)
    bn, bd = np.zeros(sc.B), np.zeros(sc.B)
    for ma, mb, d in blocks:
        R = d.size
        ir = rng.integers(0, R, (sc.B, R))
        ra = np.nanmean(ma[rng.integers(0, ma.shape[0], (sc.B, ma.shape[0]))], axis=1)
        rb = np.nanmean(mb[rng.integers(0, mb.shape[0], (sc.B, mb.shape[0]))], axis=1)
        bn += np.take_along_axis(ra - rb, ir, axis=1).sum(axis=1)
        bd += d[ir].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = bn / bd
    boot = boot[np.isfinite(boot)]
    if boot.size:
        res["b_lo"], res["b_hi"] = _pct(boot, sc.conf)
    return res
