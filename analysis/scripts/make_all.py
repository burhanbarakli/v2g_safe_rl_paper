"""One command for all tables, figures, statistics and the numbers-only summary.

    python analysis/scripts/make_all.py --raw results/raw --out analysis
    python analysis/scripts/make_all.py --synthetic            # pipeline test on fake data

Outputs under --out: tables/ (T1-T7 as .tex/.md/.csv/_values.csv, provenance.json, STAT_METHODS.md),
figures/ (F2-F9 as .pdf/.eps/.svg/.png + data/*.csv), stats/ (summary and paired-test CSVs),
findings_numbers.md (numbers only, no interpretation), load_report.txt.
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common as C  # noqa: E402
import load  # noqa: E402
import stats  # noqa: E402
import tables  # noqa: E402
import figures  # noqa: E402

SUMMARY_METRICS = ["penalised_profit_eur", "pen_per_charger", "gap_pct", "gap_hard_pct", "gap_lp_pct", "profit_eur",
                   "profit_per_charger", "profit_per_ev", "unserved_pct", "energy_not_served_kwh",
                   "tr_violation_kwh", "tr_violation_steps", "tr_violation_max_kw", "infeasible_steps",
                   "fallback_steps", "fb1star_steps", "fb2_steps", "fb1_steps", "fb0_steps", "a5_deficit_kwh",
                   "shortfall_deficit_ratio", "n_evs_capped", "n_evs_admission_reduced", "inference_ms_per_step_mean",
                   "e4_pct", "upper_bind_pct", "lower_bind_pct", "lb_active_pct", "energy_not_served_shared_kwh"]
TEST_METRICS = ["penalised_profit_eur", "energy_not_served_kwh", "tr_violation_kwh"]       # J^pen = main (DD32)
T3_METRICS = ["share_evs_stay_feasible", "tr_violation_kwh", "energy_not_served_kwh"]      # E4, E3, shortfall
T5_METRICS = ["penalised_profit_eur", "profit_eur"]                                       # DD43(7): primary, secondary
HIGHER_BETTER = {"penalised_profit_eur": True, "profit_eur": True, "profit_per_ev": True,
                 "share_evs_stay_feasible": True, "energy_not_served_kwh": False, "tr_violation_kwh": False}
REPO = HERE.parent
SYNTH_DIR = REPO / "synthetic_test"


def _guard(out: Path, synthetic: bool, smoke: bool = False):
    o = out.resolve()
    if smoke and (o == REPO.resolve() or o.name == "analysis" or REPO.resolve() in o.parents
                  and "synthetic_test" not in o.parts):
        raise SystemExit(f"REFUSED: smoke-test data may not be written into the analysis folder ({out}); "
                         f"use a temporary directory, e.g. /tmp/smoke_analysis")
    inside = "synthetic_test" in o.parts
    if synthetic and not inside:
        raise SystemExit(f"REFUSED: the input data are SYNTHETIC; outputs may only be written inside a "
                         f"synthetic_test/ directory, not to {out}")
    if not synthetic and inside:
        print("[make_all] WARNING: real data written into a synthetic_test/ directory", file=sys.stderr)


def findings_numbers(out: Path, tabs: dict, figdir: Path, tests: pd.DataFrame, ds, synthetic: bool, raw: Path):
    L = []
    if synthetic:
        L += [f"# {C.active_mark()}", "", f"**{C.active_mark()}. Every number below comes from synthetic or "
              "smoke-test data and was produced only to test the pipeline.**", ""]
    L += ["# Measured values (numbers only)", "",
          "Factual listing of measured values with their source. It contains no interpretation, ranking "
          "statement or conclusion (SAUCIS AI policy: results are reported and interpreted by the authors).",
          f"Generated {dt.datetime.now().isoformat(timespec='seconds')} from `{raw}`: "
          f"{len(ds.episodes)} episodes, {ds.episodes['method'].nunique()} methods, "
          f"{ds.episodes['test_cfg_id'].nunique()} configurations, {len(ds.oracle)} oracle episodes. "
          f"Optimality-gap reference: {ds.gap_reference}.",
          "Cell ids: `table|[group / ]row|column` as in tables/provenance.json. Values: mean ± std over seeds "
          "(† = deterministic single run); [a, b] = 95 % CI as stated in the column header or STAT_METHODS.md.", ""]
    ref = C.label("attn_safe")
    for name, t in tabs.items():
        L.append(f"## {name}")
        L.append(f"Caption: {t.caption}")
        small = len(t.rows) <= 12
        for k, h in t.columns:
            hh = tables.flat(h)
            cells = [(r, r["cells"].get(k)) for r in t.rows if r["cells"].get(k) is not None and not r["cells"][k].empty]
            if not cells:
                continue
            L.append(f"- **{hh}**")
            sel = cells if small else [(r, c) for r, c in cells if r["label"] in (ref, "SA-Attn, known departure step")]
            for r, c in sel:
                gid = (r["group"] + " / ") if r["group"] else ""
                ci = f" (95 % {c.ci} CI [{tables.fmt_num(c.lo, 3)}, {tables.fmt_num(c.hi, 3)}])" if c.ci == "t" and np.isfinite(c.lo) else ""
                L.append(f"  - `{name}|{gid}{r['label']}|{hh}`: {c.text}{ci}")
            if not small:
                vals = [(c.value, r, c) for r, c in cells]
                lo = min(vals, key=lambda v: v[0])
                hi = max(vals, key=lambda v: v[0])
                for tag, (v, r, c) in (("minimum over rows", lo), ("maximum over rows", hi)):
                    gid = (r["group"] + " / ") if r["group"] else ""
                    L.append(f"  - {tag}: `{name}|{gid}{r['label']}|{hh}`: {c.text}")
        L.append("")
    # figures: from their data files
    L.append("## Figures (values from figures/data/*.csv)")
    for f in sorted((figdir / "data").glob("*.csv")):
        d = pd.read_csv(f)
        L.append(f"### {f.stem}")
        if "N" in d and "method" in d and "mean" in d:
            for (m, met), g in d.groupby(["method", "metric"]):
                g = g.dropna(subset=["mean"]).sort_values("N")
                if g.empty:
                    continue
                ex = "" if "rho_tr" not in g or g["rho_tr"].isna().all() else None
                for keys, gg in ([((), g)] if ex == "" else list(g.groupby("rho_tr"))):
                    a, b = gg.iloc[0], gg.iloc[-1]
                    tag = "" if ex == "" else f", rho_tr = {keys:g}"
                    L.append(f"- {C.label(m)}, {met}{tag}: N = {int(a['N'])}: {a['mean']:.4g}; N = {int(b['N'])}: {b['mean']:.4g}")
        elif f.stem.startswith("F2") and "env_steps" in d:
            for (m, met), g in d.groupby(["method", "metric"]):
                last = g.sort_values("env_steps").iloc[-1]
                L.append(f"- {C.label(m)}, {met} at {last['env_steps']:.4g} env steps: {last['mean']:.4g} "
                         f"(95 % t-CI [{last['ci_lo']:.4g}, {last['ci_hi']:.4g}], {int(last['n_seeds'])} seeds)")
        elif f.stem.startswith("F5"):
            over = int((d["sum_p"] > d["C"] + 1e-6).sum())
            under = int((d["sum_p"] < -d["C_rev"] - 1e-6).sum())
            fb = int((d["fallback_code"].fillna(-1) >= 0).sum())
            L.append(f"- source: {d['source_exp_id'].iat[0]}, replay {int(d['replay_id'].iat[0])}, {len(d)} steps; "
                     f"steps with sum_p > C: {over}; steps with sum_p < -C_rev: {under}; fallback steps: {fb}")
            for c in [c for c in d.columns if c.startswith("sum_p_")]:
                L.append(f"- {c}: steps above C: {int((d[c] > d['C'] + 1e-6).sum())}")
        elif "alpha" in d:
            for _, r in d.iterrows():
                L.append(f"- {r['metric']}, alpha = {r['alpha']:g}: {r['mean']:.4g} [{r['ci_lo']:.4g}, {r['ci_hi']:.4g}]")
        elif "variant" in d:
            for _, r in d.iterrows():
                L.append(f"- {r['variant']}, {r['regime']}, {r['metric']}: {r['mean']:.4g}")
        elif "rho_tr" in d and "method" in d:
            for _, r in d.iterrows():
                L.append(f"- {C.label(r['method'])}, rho_tr = {r['rho_tr']:g}, {r['metric']}: {r['mean']:.4g}")
        L.append("")
    # gap exclusions (M2)
    if ds.gap_excluded:
        L.append(f"## Episodes excluded from the gap (|oracle| < {load.GAP_EPS:g} EUR)")
        for k, v in ds.gap_excluded.items():
            L.append(f"- {k}: {v}")
        L.append("")
    # seed-level comparisons (DD38)
    for name, df in (tests or {}).items():
        if df is None or df.empty:
            continue
        L.append(f"## Comparisons {name} (stats/tests_{name}.csv; paired t over seeds, Holm per table)")
        for _, r in df.iterrows():
            grp = ", ".join(f"{k} = {r[k]}" for k in ("N",) if k in r)
            L.append(f"- {C.label(r['ref'])} vs {C.label(r['other'])}, {r['metric']}, {grp}: difference "
                     f"{r['diff']:.4g} [{r['b_lo']:.4g}, {r['b_hi']:.4g}]; {r['test']}: p_holm = {r['p_t_holm']:.3g}; "
                     f"Wilcoxon (replay level) p_holm = {r['p_wilcoxon_holm']:.3g}; PoI = {r['poi']:.3g} "
                     f"[{r['poi_lo']:.3g}, {r['poi_hi']:.3g}]")
        L.append("")
    if synthetic:
        L += [f"**{C.active_mark()}**", ""]
    (out / "findings_numbers.md").write_text("\n".join(L))


CONCEPTUAL = {"fig2_concepts": "fig_concepts", "figS5_pipeline": "figS5_pipeline"}


def copy_conceptual(fdir: Path):
    """Copy the (data-free) conceptual figures into the figure folder under their paper names."""
    src = REPO / "figures_conceptual"
    for dst, name in CONCEPTUAL.items():
        for ext in ("svg", "pdf", "png", "eps"):
            f = src / f"{name}.{ext}"
            if f.exists():
                shutil.copyfile(f, fdir / f"{dst}.{ext}")


def run(raw: Path, out: Path, sc: stats.StatConfig, require_oracle=True, include_smoke=False):
    t0 = time.time()
    ds = load.load_raw(raw, require_oracle=require_oracle, include_smoke=include_smoke)
    _guard(out, ds.synthetic, ds.smoke)
    C.set_mark(C.SMOKE_MARK if ds.smoke and not ds.synthetic else C.SYNTH_MARK)
    synthetic = ds.synthetic or ds.smoke          # = "watermark every output"
    out.mkdir(parents=True, exist_ok=True)
    (out / "load_report.txt").write_text("\n".join(ds.report) + "\n")
    tdir, fdir, sdir = out / "tables", out / "figures", out / "stats"
    for d in (tdir, fdir, sdir):
        d.mkdir(parents=True, exist_ok=True)
    stats.write_stat_methods(tdir / "STAT_METHODS.md", sc, gap_eps=load.GAP_EPS)

    ep = ds.episodes
    print("[make_all] summaries ...", file=sys.stderr)
    metrics = [m for m in SUMMARY_METRICS if m in ep.columns]
    s_cfg = stats.summary_table(ep, metrics, ["test_cfg_id", "profile", "N", "rho_tr"], sc, iqm_metrics=["gap_pct"])
    s_N = stats.summary_table(ep[ep["is_grid"]], metrics, ["N"], sc, iqm_metrics=["gap_pct"])
    s_Nr = stats.summary_table(ep[ep["is_grid"]], metrics, ["N", "rho_tr"], sc)
    print("[make_all] comparisons (DD38) ...", file=sys.stderr)
    grid = ep[ep["is_grid"]]
    tests = {}
    if C.REFERENCE in set(ep["method"]):
        others = [m for m in C.method_order(grid["method"].unique())
                  if m != C.REFERENCE and C.info(m).group not in ("conformal", "oracle", "sensitivity", "other")]
        tests["T1"] = stats.seed_comparisons(grid[grid["N"].isin(C.TRAIN_SIZES)], C.REFERENCE, others, TEST_METRICS,
                                             ["N"], sc, HIGHER_BETTER, family="T1")
        tests["T2"] = stats.seed_comparisons(grid, C.REFERENCE, others, TEST_METRICS, ["N"], sc, HIGHER_BETTER,
                                             family="T2")
        t3m = [m for m in T3_METRICS if m in grid.columns and grid[m].notna().any()]
        tests["T3"] = stats.seed_comparisons(grid, C.REFERENCE, others, t3m, ["N"], sc, HIGHER_BETTER, family="T3")
    t5 = [stats.seed_comparisons(grid, ms, [mp], T5_METRICS, ["N"], sc, HIGHER_BETTER, family="T5")
          for _, ms, mp in C.GUARANTEE_PAIRS if {ms, mp} <= set(grid["method"])]
    t5 = [x for x in t5 if not x.empty]
    if t5:
        t5 = pd.concat(t5, ignore_index=True)
        t5["p_t_holm"], t5["p_wilcoxon_holm"] = stats.holm(t5["p_t"].values), stats.holm(t5["p_wilcoxon"].values)
        tests["T5"] = t5
    outs = [("summary_by_config", s_cfg), ("summary_by_N", s_N), ("summary_by_N_rho", s_Nr)] + \
        [(f"tests_{k}", v) for k, v in tests.items()]
    for name, df in outs:
        df = df.copy()
        if "exp_ids" in df:
            df["exp_ids"] = df["exp_ids"].map(lambda x: ";".join(x) if isinstance(x, list) else x)
        if synthetic:
            df.insert(0, "WATERMARK", C.active_mark())
        df.to_csv(sdir / f"{name}.csv", index=False)

    print("[make_all] tables ...", file=sys.stderr)
    tabs = tables.build_all(ds, sc, tdir, synthetic)
    print("[make_all] figures ...", file=sys.stderr)
    figures.Figures(ds, sc, fdir, synthetic).all()
    copy_conceptual(fdir)
    findings_numbers(out, tabs, fdir, tests, ds, synthetic, raw)
    if synthetic:
        (out / ("README_SYNTHETIC.txt" if ds.synthetic else "README_SMOKE.txt")).write_text(
            f"{C.active_mark()}\nEverything in this directory was produced from synthetic or smoke-test data to test "
            "the analysis pipeline. Nothing here may be used in the paper.\n")
    print(f"[make_all] done in {time.time() - t0:.0f} s -> {out}", file=sys.stderr)
    return ds, tabs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, help="results/raw directory")
    ap.add_argument("--out", type=Path, help="output directory (the project's analysis/)")
    ap.add_argument("--synthetic", action="store_true",
                    help="generate synthetic data into synthetic_test/raw and write all outputs into synthetic_test/")
    ap.add_argument("--replay-scale", type=float, default=0.3, help="synthetic only: fraction of the real replay counts")
    ap.add_argument("--B", type=int, default=stats.DEFAULT_B, help="bootstrap resamples (means)")
    ap.add_argument("--B-iqm", type=int, default=stats.DEFAULT_B_IQM, help="bootstrap resamples (IQM)")
    ap.add_argument("--no-oracle", action="store_true", help="continue without oracle results (gap = NaN)")
    ap.add_argument("--include-smoke", action="store_true",
                    help="keep smoke rows (pipeline check on smoke output; outputs watermarked; not into analysis/)")
    a = ap.parse_args(argv)
    sc = stats.StatConfig(B=a.B, B_iqm=a.B_iqm)
    if a.synthetic:
        import synth
        out = a.out or SYNTH_DIR
        if "synthetic_test" not in out.resolve().parts:
            raise SystemExit("--synthetic writes only into a synthetic_test/ directory")
        raw = out / "raw"
        synth.generate(raw, replay_scale=a.replay_scale)
        return run(raw, out, sc)
    if not a.raw or not a.out:
        ap.error("--raw and --out are required (or use --synthetic)")
    return run(a.raw, a.out, sc, require_oracle=not a.no_oracle, include_smoke=a.include_smoke)


if __name__ == "__main__":
    main()
