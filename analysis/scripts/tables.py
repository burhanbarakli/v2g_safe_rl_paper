"""Tables T1-T7: LaTeX (booktabs, IEEE-compatible), CSV and Markdown, plus provenance.

Every cell records the exp_ids of all runs whose episodes entered it (oracle runs included for
optimality-gap cells); tables/provenance.json maps "<table>|<row>|<column>" -> exp_ids.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import common as C
from stats import StatConfig, summarize, diff_summary

DAGGER = "\u2020"


# ----------------------------------------------------------------------------- formatting

def flat(h: str) -> str:
    """Header without the LaTeX line-break marker."""
    return h.replace("\n", " ")


def fmt_num(v, d=2) -> str:
    if v is None or not np.isfinite(v):
        return "\u2013"
    if v == 0:
        return "0"
    if abs(v) < 0.5 * 10 ** (-d):
        return f"{v:.1e}"
    return f"{v:.{d}f}"


def fmt_ci(v, d=2) -> str:
    """CI bounds: fixed decimals (no scientific notation)."""
    if v is None or not np.isfinite(v):
        return "\u2013"
    return f"{v:.{d}f}"


@dataclass
class Cell:
    text: str = "\u2013"
    value: float = np.nan
    std: float = np.nan
    lo: float = np.nan
    hi: float = np.nan
    ci: str = ""             # 't' | 'bootstrap' | ''
    unit: str = ""           # 'seed' | 'replay'
    n: int = 0
    exp_ids: list = field(default_factory=list)

    @property
    def empty(self):
        return not np.isfinite(self.value)


def cell_from_summary(s: dict, d=2, show="mean_std", extra_ids=()) -> Cell:
    if s is None or s.get("unit") in (None, "none") or not np.isfinite(s.get("mean", np.nan)):
        return Cell()
    if show == "iqm":
        txt = f"{fmt_num(s['iqm'], d)} [{fmt_ci(s['iqm_lo'], d)}, {fmt_ci(s['iqm_hi'], d)}]"
        return Cell(txt, s["iqm"], np.nan, s["iqm_lo"], s["iqm_hi"], "bootstrap", s["unit"], s["n_episodes"],
                    sorted(set(s["exp_ids"]) | set(extra_ids)))
    if s["unit"] == "seed":
        txt = f"{fmt_num(s['mean'], d)} \u00b1 {fmt_num(s['std'], d)}"
    else:
        txt = f"{fmt_num(s['mean'], d)}{DAGGER}"
    if show == "mean":
        txt = fmt_num(s["mean"], d) + ("" if s["unit"] == "seed" else DAGGER)
    return Cell(txt, s["mean"], s["std"], s["t_lo"], s["t_hi"], "t", s["unit"], s["n_episodes"],
                sorted(set(s["exp_ids"]) | set(extra_ids)))


def cell_from_diff(r: dict, d=2) -> Cell:
    if not np.isfinite(r.get("diff", np.nan)):
        return Cell()
    txt = f"{fmt_num(r['diff'], d)} [{fmt_ci(r['b_lo'], d)}, {fmt_ci(r['b_hi'], d)}]"
    return Cell(txt, r["diff"], np.nan, r["b_lo"], r["b_hi"], "bootstrap", r.get("unit", ""), 0, r["exp_ids"])


@dataclass
class Table:
    name: str
    caption: str
    columns: list                   # list of (key, header)
    rows: list = field(default_factory=list)   # dicts: {"label": str, "group": str|None, "cells": {key: Cell}}
    notes: list = field(default_factory=list)
    stub: str = "Method"
    size: str = "footnotesize"      # LaTeX font size of the tabular
    colsep: str = "3.5pt"

    def add(self, label, cells, group=None):
        self.rows.append(dict(label=label, group=group, cells=cells))

    # --- provenance
    def provenance(self) -> dict:
        out = {}
        for r in self.rows:
            for k, h in self.columns:
                c = r["cells"].get(k)
                if c is not None and not c.empty:
                    key = f"{self.name}|{(r['group'] + ' / ') if r['group'] else ''}{r['label']}|{flat(h)}"
                    out[key] = c.exp_ids
        return out

    # --- outputs
    def to_frame(self, numeric=False) -> pd.DataFrame:
        recs = []
        for r in self.rows:
            rec = {"group": r["group"] or "", self.stub: r["label"]}
            for k, h in self.columns:
                c = r["cells"].get(k, Cell())
                h = flat(h)
                if numeric:
                    rec.update({f"{h} | value": c.value, f"{h} | std": c.std, f"{h} | ci_lo": c.lo,
                                f"{h} | ci_hi": c.hi, f"{h} | ci": c.ci, f"{h} | unit": c.unit,
                                f"{h} | n_episodes": c.n})
                else:
                    rec[h] = c.text
            recs.append(rec)
        return pd.DataFrame(recs)

    def to_markdown(self, synthetic=False) -> str:
        cap = (f"**{C.active_mark()}.** " if synthetic else "") + self.caption
        hdr = ["", self.stub] if any(r["group"] for r in self.rows) else [self.stub]
        lines = [f"**{self.name}.** {cap}", "", "| " + " | ".join(hdr + [flat(h) for _, h in self.columns]) + " |",
                 "|" + "---|" * len(hdr) + "---:|" * len(self.columns)]
        for r in self.rows:
            first = [r["group"] or ""] if len(hdr) == 2 else []
            lines.append("| " + " | ".join(first + [r["label"]] + [r["cells"].get(k, Cell()).text for k, _ in self.columns]) + " |")
        if self.notes:
            lines += [""] + [f"*{n}*" for n in self.notes]
        return "\n".join(lines) + "\n"

    def to_latex(self, synthetic=False) -> str:
        def esc(s: str) -> str:
            s = s.replace("\\", "\x00")
            for a, b in [("{", r"\{"), ("}", r"\}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")]:
                s = s.replace(a, b)
            s = s.replace("\u00b1", r"$\pm$").replace(DAGGER, r"$^{\dagger}$").replace("\u2013", "--")
            s = s.replace("\u03b1", r"$\alpha$").replace("\u03b2", r"$\beta$").replace("\u0394", r"$\Delta$")
            s = s.replace("\u2264", r"$\le$").replace("\u2265", r"$\ge$").replace("\u20ac", r"\euro{}")
            s = s.replace("\u2212", "--").replace("\u00b7", r"$\cdot$")
            s = s.replace("\u2208", r"$\in$").replace("\u00d7", r"$\times$").replace("\x00", r"\textbackslash{}")
            s = re.sub(r"\^([0-9lo]+)", r"\\textsuperscript{\1}", s)
            s = re.sub(r"(^|[\s\[(])-(?=\d)", r"\1$-$", s)          # numeric minus sign
            return s
        def head(h: str) -> str:
            parts = [esc(x) for x in h.split("\n")]
            if len(parts) == 1:
                return parts[0]
            return r"\begin{tabular}[b]{@{}r@{}}" + r" \\{} ".join(parts) + r"\end{tabular}"
        cap = (r"\textbf{" + esc(C.active_mark()) + r".} " if synthetic else "") + esc(self.caption)
        n = len(self.columns)
        lab = "tab:" + self.name.lower()
        out = [r"% generated by analysis/scripts/tables.py -- do not edit by hand",
               r"% requires \usepackage{booktabs} and \usepackage{eurosym} (or define \euro)",
               r"\begin{table}[!t]", r"\centering", rf"\caption{{{cap}}}", rf"\label{{{lab}}}",
               "\\" + self.size, r"\setlength{\tabcolsep}{" + self.colsep + "}",
               r"\begin{tabular}{l" + "r" * n + "}", r"\toprule",
               esc(self.stub) + " & " + " & ".join(head(h) for _, h in self.columns) + r" \\", r"\midrule"]
        cur = None
        for i, r in enumerate(self.rows):
            if r["group"] and r["group"] != cur:
                if cur is not None:
                    out.append(r"\midrule")
                out.append(rf"\multicolumn{{{n + 1}}}{{l}}{{\textit{{{esc(r['group'])}}}}} \\")
                cur = r["group"]
            out.append(esc(r["label"]) + " & " + " & ".join(esc(r["cells"].get(k, Cell()).text) for k, _ in self.columns) + r" \\")
        out += [r"\bottomrule", r"\end{tabular}"]
        if self.notes:
            out.append(r"\par\smallskip{\scriptsize " + r" \\{} ".join(esc(x) for x in self.notes) + "}")
        out.append(r"\end{table}")
        return "\n".join(out) + "\n"

    def write(self, outdir: Path, synthetic=False):
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / f"{self.name}.tex").write_text(self.to_latex(synthetic))
        (outdir / f"{self.name}.md").write_text(self.to_markdown(synthetic))
        df = self.to_frame()
        if synthetic:
            df.insert(0, "WATERMARK", C.active_mark())
        df.to_csv(outdir / f"{self.name}.csv", index=False, encoding="utf-8-sig")
        dn = self.to_frame(numeric=True)
        if synthetic:
            dn.insert(0, "WATERMARK", C.active_mark())
        dn.to_csv(outdir / f"{self.name}_values.csv", index=False, encoding="utf-8-sig")


# ----------------------------------------------------------------------------- builders

COMMON_NOTE = ("Values: mean \u00b1 standard deviation over training seeds of the seed-level means "
               "(configurations pooled as strata; n per cell in the _values.csv files). \u2020 deterministic method, single run "
               "(no seed variability). 95 % confidence intervals (t-based and bootstrap) are in the "
               "_values.csv files; definitions in STAT_METHODS.md.")


class Builder:
    def __init__(self, ds, sc: StatConfig, synthetic=False):
        self.ds = ds
        self.ep = ds.episodes
        self.sc = sc
        self.synthetic = synthetic
        self._cache = {}
        self.grid = self.ep[self.ep["is_grid"]]
        self.methods = C.method_order(self.ep["method"].unique())

    def present(self, names):
        have = set(self.ep["method"])
        return [m for m in names if m in have]

    def S(self, method, mask_key, metric, want_iqm=False) -> dict | None:
        """Cached summary of `method` over the grid configurations selected by mask_key
        (tuple of (column, values))."""
        key = (method, mask_key, metric, want_iqm)
        if key in self._cache:
            return self._cache[key]
        df = self.ep[self.ep["method"] == method]
        for col, vals in mask_key:
            df = df[df[col].isin(vals)]
        r = summarize(df, metric, self.sc, salt=(method, str(mask_key)), want_iqm=want_iqm) if len(df) else None
        if r is not None:
            r["cfgs"] = sorted(df["test_cfg_id"].unique())
        self._cache[key] = r
        return r

    def oracle_ids(self, s):
        return [f"oracle__{c}" for c in (s or {}).get("cfgs", [])]

    def cell(self, method, mk, metric, d=2, show="mean_std"):
        s = self.S(method, mk, metric, want_iqm=(show == "iqm"))
        extra = self.oracle_ids(s) if metric in ("gap_pct", "gap_lp_pct") else ()
        return cell_from_summary(s, d, show, extra)

    @staticmethod
    def grid_mask(**kw):
        return (("is_grid", (True,)),) + tuple((k, tuple(v)) for k, v in kw.items())

    def method_rows(self, exclude_groups=("conformal", "ablation", "oracle", "other")):
        return [m for m in self.methods if C.info(m).group not in exclude_groups]

    # ---------------- T1
    def t1(self) -> Table:
        cols = [("profit_eur", "Profit\n(\u20ac/episode)"), ("gap_pct", "Optimality\ngap (%)"),
                ("gap_iqm", "Gap IQM (%)\n[95 % CI]"), ("unserved_pct", "Unserved\nEVs (%)"),
                ("tr_violation_kwh", "Transformer\nviolation (kWh)"),
                ("infeasible_steps", "Infeasible\nsteps")]
        t = Table("T1", "Main comparison at the training station sizes N \u2208 {10, 20} "
                        "(three arrival profiles and both tightness levels pooled).", cols, notes=[COMMON_NOTE,
                  "Gap IQM: interquartile mean of the per-episode optimality gap, stratified bootstrap CI."])
        for n in C.TRAIN_SIZES:
            mk = self.grid_mask(N=[n])
            for m in self.method_rows():
                if self.S(m, mk, "profit_eur") is None:
                    continue
                t.add(C.label(m), {"profit_eur": self.cell(m, mk, "profit_eur", 2),
                                   "gap_pct": self.cell(m, mk, "gap_pct", 1),
                                   "gap_iqm": self.cell(m, mk, "gap_pct", 1, show="iqm"),
                                   "unserved_pct": self.cell(m, mk, "unserved_pct", 2),
                                   "tr_violation_kwh": self.cell(m, mk, "tr_violation_kwh", 2),
                                   "infeasible_steps": self.cell(m, mk, "infeasible_steps", 1)}, group=f"N = {n}")
        return t

    # ---------------- T2
    def t2(self) -> tuple[Table, pd.DataFrame]:
        cols = [("profit_per_charger", "Profit\n(\u20ac/charger)"), ("gap_pct", "Optimality\ngap (%)"),
                ("unserved_pct", "Unserved\nEVs (%)"), ("tr_violation_kwh", "Transformer\nviolation (kWh)"),
                ("infeasible_steps", "Infeasible\nsteps"), ("fallback_steps", "Fallback\nactivations")]
        t = Table("T2", "Zero-shot transfer across station sizes N (policies trained at N \u2208 {10, 20}; "
                        "profiles and tightness levels pooled; infeasible steps and fallback activations per episode).",
                  cols, notes=[COMMON_NOTE])
        meths = self.present(C.SCALING_METHODS)
        digits = dict(profit_per_charger=2, gap_pct=1, unserved_pct=2, tr_violation_kwh=2, infeasible_steps=1, fallback_steps=1)
        for n in sorted(self.grid["N"].unique()):
            mk = self.grid_mask(N=[n])
            for m in meths:
                if self.S(m, mk, "profit_eur") is None:
                    continue
                t.add(C.label(m), {k: self.cell(m, mk, k, digits[k]) for k, _ in cols},
                      group=f"N = {n}" + (" (training size)" if n in C.TRAIN_SIZES else ""))
        # full per-configuration table (supplementary, CSV only)
        recs = []
        for m in self.methods:
            for cfg, g in self.ep[self.ep["method"] == m].groupby("test_cfg_id"):
                rec = dict(method=m, label=C.label(m), test_cfg_id=cfg, profile=g["profile"].iat[0],
                           N=int(g["N"].iat[0]), rho_tr=float(g["rho_tr"].iat[0]))
                for k, _ in cols:
                    s = summarize(g, k, self.sc, salt=(m, cfg))
                    rec.update({f"{k}_mean": s["mean"], f"{k}_std": s["std"], f"{k}_t_lo": s["t_lo"],
                                f"{k}_t_hi": s["t_hi"], f"{k}_b_lo": s["b_lo"], f"{k}_b_hi": s["b_hi"]})
                rec["unit"] = s["unit"]
                rec["n_episodes"] = s["n_episodes"]
                rec["exp_ids"] = ";".join(sorted(set(g["exp_id"])))
                recs.append(rec)
        return t, pd.DataFrame(recs)

    # ---------------- T3
    def t3(self) -> Table:
        cols = [("unserved_pct", "Unserved\nEVs (%)"), ("energy_not_served_kwh", "Energy not\nserved (kWh)"),
                ("any_unserved_pct", "Episodes with\nunserved EV (%)"),
                ("tr_violation_kwh", "Violation\nenergy (kWh)"), ("tr_violation_max_kw", "Max.\nviolation (kW)"),
                ("any_violation_pct", "Episodes with\nviolation (%)"), ("fallback_steps", "Fallback\nactivations")]
        t = Table("T3", f"Constraint satisfaction per method over all {self.grid['test_cfg_id'].nunique()} test-grid configurations "
                        "(per episode; departure energy requirement and transformer limit).", cols, notes=[COMMON_NOTE])
        mk = self.grid_mask()
        d = dict(unserved_pct=2, energy_not_served_kwh=2, any_unserved_pct=1, tr_violation_kwh=2,
                 tr_violation_max_kw=2, any_violation_pct=1, fallback_steps=2)
        for m in self.method_rows():
            if self.S(m, mk, "unserved_pct") is None:
                continue
            t.add(C.label(m), {k: self.cell(m, mk, k, d[k]) for k, _ in cols},
                  group="with safety layer" if C.info(m).safety_layer else "without safety layer")
        t.rows.sort(key=lambda r: (r["group"] != "with safety layer"))
        t.size, t.colsep = "scriptsize", "2.5pt"
        return t

    # ---------------- T4
    def t4(self) -> Table:
        cols = [("gap_tr", "Gap (%)\nN \u2208 {10, 20}"), ("uns_tr", "Unserved (%)\nN \u2208 {10, 20}"),
                ("vio_tr", "Violation (kWh)\nN \u2208 {10, 20}"), ("gap_zs", "Gap (%)\nN \u2265 50"),
                ("uns_zs", "Unserved (%)\nN \u2265 50"), ("vio_zs", "Violation (kWh)\nN \u2265 50")]
        t = Table("T4", "Ablation: architecture (attention vs MLP) \u00d7 constraint handling (safety layer vs penalty), "
                        "and further ablations of SA-Attn, at training sizes and in zero-shot transfer (N \u2265 50).",
                  cols, stub="Variant", notes=[COMMON_NOTE])
        sizes = sorted(self.grid["N"].unique())
        mtr = self.grid_mask(N=[n for n in sizes if n in C.TRAIN_SIZES])
        mzs = self.grid_mask(N=[n for n in sizes if n >= 50])

        def cells(m):
            return {"gap_tr": self.cell(m, mtr, "gap_pct", 1), "uns_tr": self.cell(m, mtr, "unserved_pct", 2),
                    "vio_tr": self.cell(m, mtr, "tr_violation_kwh", 2), "gap_zs": self.cell(m, mzs, "gap_pct", 1),
                    "uns_zs": self.cell(m, mzs, "unserved_pct", 2), "vio_zs": self.cell(m, mzs, "tr_violation_kwh", 2)}
        for (arch, cons), m in C.ABLATION_2X2.items():
            if m in self.ep["method"].values:
                t.add(f"{arch} + {cons}", cells(m), group="2 \u00d7 2 design")
        for m in self.present(C.EXTRA_ABLATIONS):
            t.add(C.label(m), cells(m), group="further ablations")
        t.size, t.colsep = "scriptsize", "2.5pt"
        return t

    # ---------------- T5
    def t5(self) -> Table:
        cols = [("gap_safe", "Gap, safety\nlayer (%)"), ("gap_pen", "Gap,\npenalty (%)"),
                ("dprofit", "\u0394 profit (\u20ac/charger)\npenalty \u2212 safety layer\n[95 % CI]"),
                ("dgap", "\u0394 gap (pp)\n[95 % CI]"),
                ("uns_pen", "Unserved EVs,\npenalty (%)"), ("vio_pen", "Violation,\npenalty (kWh)")]
        t = Table("T5", "Cost of the guarantee: profit of the penalty-based policy minus profit of the same "
                        "architecture with the safety layer, per station size N (profiles and tightness levels pooled), "
                        "together with the violations the penalty-based policy incurs.", cols, stub="N",
                  notes=[COMMON_NOTE, "\u0394: difference of means; CI: bootstrap over the seeds of both methods "
                                      "(independent runs), stratified by configuration. \u0394 gap = gap(safety layer) \u2212 gap(penalty)."])
        pairs = [("Attention", "attn_safe", "attn_penalty"), ("MLP", "mlp_safe", "mlp_penalty")]
        for arch, ms, mp in pairs:
            if ms not in self.ep["method"].values or mp not in self.ep["method"].values:
                continue
            for n in sorted(self.grid["N"].unique()):
                sub = self.grid[self.grid["N"] == n]
                dprof = diff_summary(sub, mp, ms, "profit_per_charger", self.sc, salt=(n,))
                dgap = diff_summary(sub, ms, mp, "gap_pct", self.sc, salt=(n,))
                mk = self.grid_mask(N=[n])
                c_dgap = cell_from_diff(dgap, 1)
                c_dgap.exp_ids = sorted(set(c_dgap.exp_ids) | {f"oracle__{c}" for c in sub["test_cfg_id"].unique()})
                t.add(f"{n}", {"gap_safe": self.cell(ms, mk, "gap_pct", 1), "gap_pen": self.cell(mp, mk, "gap_pct", 1),
                               "dprofit": cell_from_diff(dprof, 2), "dgap": c_dgap,
                               "uns_pen": self.cell(mp, mk, "unserved_pct", 2),
                               "vio_pen": self.cell(mp, mk, "tr_violation_kwh", 2)}, group=arch)
        t.size, t.colsep = "scriptsize", "2.5pt"
        return t

    # ---------------- T6
    def t6(self) -> Table:
        sizes = sorted(self.grid["N"].unique())
        cols = [("train_h", "Training\ntime (h)")] + [(f"inf{n}", f"N = {n}\n(ms/step)") for n in sizes]
        t = Table("T6", "Computation: wall-clock training time per run and inference time per control step "
                        "(policy + safety layer, mean over episodes; columns N = 5 ... 200) versus station size N.", cols,
                  notes=[COMMON_NOTE, "Training time: last wall_clock_s of train_log.csv per run. Hardware: see meta.json."])
        tr = self.ds.train
        for m in self.method_rows(exclude_groups=("oracle", "other", "conformal")):
            cells = {}
            ids = sorted(self.ep.loc[self.ep["method"] == m, "exp_id"].unique())
            if len(tr):
                tt = tr[tr["exp_id"].isin(ids)].groupby("exp_id")["wall_clock_s"].max() / 3600
                if len(tt):
                    std = tt.std(ddof=1) if len(tt) > 1 else np.nan
                    cells["train_h"] = Cell(f"{fmt_num(tt.mean(), 2)} \u00b1 {fmt_num(std, 2)}" if len(tt) > 1 else fmt_num(tt.mean(), 2),
                                            float(tt.mean()), float(std), np.nan, np.nan, "", "seed", len(tt), sorted(tt.index))
            for n in sizes:
                cells[f"inf{n}"] = self.cell(m, self.grid_mask(N=[n]), "inference_ms_per_step_mean", 2, show="mean")
            if any(not c.empty for c in cells.values()):
                t.add(C.label(m), cells)
        return t

    # ---------------- T7
    def t7(self) -> Table | None:
        conf = sorted([m for m in self.ep["method"].unique() if C.conf_alpha(m) is not None], key=C.conf_alpha)
        if not conf and "attn_safe_conf_point" not in set(self.ep["method"]):
            return None
        cols = [("gap_pct", "Optimality\ngap (%)"), ("unserved_pct", "Unserved\nEVs (%)"),
                ("energy_not_served_kwh", "Energy not\nserved (kWh)")]
        has_cov = "coverage_frac" in self.ep and self.ep["coverage_frac"].notna().any()
        has_dep = "n_evs_departed_before_bound" in self.ep and self.ep["n_evs_departed_before_bound"].notna().any()
        has_dd = "mean_d_minus_dlo" in self.ep and self.ep["mean_d_minus_dlo"].notna().any()
        if has_cov:
            cols += [("coverage_frac", "Coverage\n(episode mean)")]
        if has_dep:
            cols += [("n_evs_departed_before_bound", "EVs departed\nbefore bound")]
        if has_dd:
            cols += [("mean_d_minus_dlo", "Mean d \u2212 d^lo\n(steps)")]
        cols += [("dprofit", "\u0394 profit vs known d\n(\u20ac/charger) [95 % CI]")]
        t = Table("T7", "Conformal departure bound: miscoverage level \u03b1 versus profit and departure shortfall "
                        f"(all {self.grid['test_cfg_id'].nunique()} test-grid configurations pooled).", cols, stub="Variant", notes=[COMMON_NOTE,
                  "\u0394 profit: SA-Attn-CP(\u03b1) minus SA-Attn with known departure step; bootstrap CI over seeds."])
        mk = self.grid_mask()
        rows = (["attn_safe"] if "attn_safe" in self.ep["method"].values else []) + conf + self.present(["attn_safe_conf_point"])
        for m in rows:
            cells = {"gap_pct": self.cell(m, mk, "gap_pct", 1), "unserved_pct": self.cell(m, mk, "unserved_pct", 2),
                     "energy_not_served_kwh": self.cell(m, mk, "energy_not_served_kwh", 2)}
            if m != "attn_safe":
                if has_cov:
                    cells["coverage_frac"] = self.cell(m, mk, "coverage_frac", 3)
                if has_dep:
                    cells["n_evs_departed_before_bound"] = self.cell(m, mk, "n_evs_departed_before_bound", 2)
                if has_dd:
                    cells["mean_d_minus_dlo"] = self.cell(m, mk, "mean_d_minus_dlo", 2)
            if m != "attn_safe" and "attn_safe" in self.ep["method"].values:
                cells["dprofit"] = cell_from_diff(diff_summary(self.grid, m, "attn_safe", "profit_per_charger", self.sc), 2)
            t.add("SA-Attn, known departure step" if m == "attn_safe" else C.label(m), cells)
        return t


def _ts1(self) -> Table:
    cols = [("gap_pct", "Gap vs J_LP,ceil (%)\n(main, DD26)"), ("gap_lp_pct", "Gap vs J_LP (%)\n(supplementary)")]
    t = Table("TS1", "Optimality gap with respect to the rounding-aware perfect-information optimum J_LP,ceil "
                     "(main text) and with respect to J_LP, per station size N (profiles and tightness levels pooled).",
              cols, notes=[COMMON_NOTE])
    for n in sorted(self.grid["N"].unique()):
        mk = self.grid_mask(N=[n])
        for m in self.present(C.SCALING_METHODS):
            if self.S(m, mk, "profit_eur") is None:
                continue
            t.add(C.label(m), {k: self.cell(m, mk, k, 2) for k, _ in cols}, group=f"N = {n}")
    return t


Builder.ts1 = _ts1


def build_all(ds, sc: StatConfig, outdir: Path, synthetic=False) -> dict:
    b = Builder(ds, sc, synthetic)
    tabs = {}
    tabs["T1"] = b.t1()
    tabs["T2"], t2full = b.t2()
    tabs["T3"] = b.t3()
    tabs["T4"] = b.t4()
    tabs["T5"] = b.t5()
    tabs["T6"] = b.t6()
    t7 = b.t7()
    if t7 is not None:
        tabs["T7"] = t7
    tabs["TS1"] = b.ts1()
    outdir.mkdir(parents=True, exist_ok=True)
    prov = {}
    for t in tabs.values():
        t.write(outdir, synthetic)
        prov.update(t.provenance())
    if synthetic:
        t2full.insert(0, "WATERMARK", C.active_mark())
    t2full.to_csv(outdir / "T2_full_per_configuration.csv", index=False, encoding="utf-8-sig")
    meta = {"_synthetic": synthetic, "_note": "key = table|[group / ]row|column -> exp_ids whose episodes entered the cell"}
    (outdir / "provenance.json").write_text(json.dumps({**meta, **prov}, indent=1, ensure_ascii=False))
    return tabs
