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
from stats import StatConfig, summarize, diff_summary, ratio_diff

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
        math = [("J^pen", r"$J^{\mathrm{pen}}$"), ("J^LP,ceil", r"$J^{\mathrm{LP,ceil}}$"), ("J^LP", r"$J^{\mathrm{LP}}$"),
                ("\u03c1_tr", r"$\rho_{\mathrm{tr}}$"), ("d^lo", r"$\hat d^{\mathrm{lo}}$"), ("p_i = l_i", r"$p_i = l_i$")]

        def esc(s: str) -> str:
            for k, (tok, _) in enumerate(math):
                s = s.replace(tok, f"\x02{k}\x03")
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
            for k, (_, tex) in enumerate(math):
                s = s.replace(f"\x02{k}\x03", tex)
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
               r"% requires \usepackage{booktabs}, \usepackage{adjustbox} and \usepackage{eurosym} (or define \euro)",
               r"\begin{table}[!t]", r"\centering", rf"\caption{{{cap}}}", rf"\label{{{lab}}}",
               "\\" + self.size, r"\setlength{\tabcolsep}{" + self.colsep + "}",
               r"\begin{adjustbox}{max width=\linewidth}",       # shrinks only if wider than the text
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
        out += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]
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
        if metric not in self.ep.columns:
            return Cell()
        s = self.S(method, mk, metric, want_iqm=(show == "iqm"))
        extra = self.oracle_ids(s) if metric.startswith("gap") else ()
        return cell_from_summary(s, d, show, extra)

    def dcell(self, sub, a, b, metric, d=2, salt=()):
        """Difference a - b (cell with bootstrap CI); oracle ids added for gap metrics."""
        if metric not in sub.columns:
            return Cell()
        c = cell_from_diff(diff_summary(sub, a, b, metric, self.sc, salt=salt), d)
        if metric.startswith("gap") and not c.empty:
            c.exp_ids = sorted(set(c.exp_ids) | {f"oracle__{x}" for x in sub["test_cfg_id"].unique()})
        return c

    @staticmethod
    def grid_mask(**kw):
        return (("is_grid", (True,)),) + tuple((k, tuple(v)) for k, v in kw.items())

    @staticmethod
    def cfg_mask(cfgs):
        return (("test_cfg_id", tuple(sorted(cfgs))),)

    def method_rows(self, exclude_groups=("conformal", "ablation", "oracle", "other", "sensitivity")):
        return [m for m in self.methods if C.info(m).group not in exclude_groups]

    def has(self, col):
        return col in self.ep.columns and self.ep[col].notna().any()

    # ---------------- T1
    def t1(self) -> Table:
        cols = [("pen", "J^pen\n(\u20ac/episode)"), ("gap_pct", "Optimality\ngap (%)"),
                ("gap_iqm", "Gap IQM (%)\n[95 % CI]"), ("profit_eur", "Profit\n(\u20ac/episode)"),
                ("energy_not_served_kwh", "Energy not\nserved (kWh)"), ("tr_violation_kwh", "Overload\n(kWh)")]
        t = Table("T1", "Main comparison at the training station sizes N \u2208 {10, 20} (three arrival profiles and "
                        "both tightness levels pooled). J^pen: penalised profit (10 \u20ac/kWh for energy not served "
                        "and transformer overload); optimality gap of J^pen w.r.t. the soft-penalty perfect-information "
                        "optimum.", cols, notes=[COMMON_NOTE,
                  "Gap IQM: interquartile mean of the per-episode optimality gap, stratified bootstrap CI."])
        for n in C.TRAIN_SIZES:
            mk = self.grid_mask(N=[n])
            for m in self.method_rows():
                if self.S(m, mk, "profit_eur") is None:
                    continue
                t.add(C.label(m), {"pen": self.cell(m, mk, PEN, 2), "gap_pct": self.cell(m, mk, "gap_pct", 1),
                                   "gap_iqm": self.cell(m, mk, "gap_pct", 1, show="iqm"),
                                   "profit_eur": self.cell(m, mk, "profit_eur", 2),
                                   "energy_not_served_kwh": self.cell(m, mk, "energy_not_served_kwh", 2),
                                   "tr_violation_kwh": self.cell(m, mk, "tr_violation_kwh", 2)}, group=f"N = {n}")
        t.size, t.colsep = "scriptsize", "3pt"
        return t

    # ---------------- T2
    def t2(self) -> tuple[Table, pd.DataFrame]:
        cols = [("pen_per_charger", "J^pen\n(\u20ac/charger)"), ("gap_pct", "Optimality\ngap (%)"),
                ("profit_per_charger", "Profit\n(\u20ac/charger)"), ("unserved_pct", "Unserved\nEVs (%)"),
                ("tr_violation_kwh", "Overload\n(kWh)"), ("fallback_steps", "Fallback\nactivations"),
                ("transfer", "Transfer gap\nto N = 50 ref. (pp)")]
        t = Table("T2", "Zero-shot transfer across station sizes N (policies trained at N \u2208 {10, 20}; profiles and "
                        "tightness levels \u03c1_tr \u2208 {4, 7} kW pooled; per episode). Transfer gap: optimality gap of "
                        "SA-Attn minus that of SA-Attn trained at N = 50 (in-distribution reference, DD36).",
                  cols, notes=[COMMON_NOTE, "Transfer gap: difference of means, bootstrap CI over seeds of both runs."])
        meths = self.present(C.SCALING_METHODS)
        digits = dict(pen_per_charger=2, gap_pct=1, profit_per_charger=2, unserved_pct=2, tr_violation_kwh=2,
                      fallback_steps=1)
        for n in sorted(self.grid["N"].unique()):
            mk = self.grid_mask(N=[n])
            sub = self.grid[self.grid["N"] == n]
            for m in meths:
                if self.S(m, mk, "profit_eur") is None:
                    continue
                cells = {k: self.cell(m, mk, k, digits[k]) for k in digits}
                ref = C.TRANSFER_REF.get(m)
                if ref and ref in set(sub["method"]):
                    cells["transfer"] = self.dcell(sub, m, ref, "gap_pct", 1, salt=("transfer", n))
                t.add(C.label(m), cells, group=f"N = {n}" + (" (training size)" if n in C.TRAIN_SIZES else ""))
        t.size, t.colsep = "scriptsize", "2.5pt"
        recs = []
        full_cols = ["pen_per_charger", "gap_pct", "gap_hard_pct", "profit_per_charger", "unserved_pct",
                     "energy_not_served_kwh", "tr_violation_kwh", "fallback_steps"]
        for m in self.methods:
            for cfg, g in self.ep[self.ep["method"] == m].groupby("test_cfg_id"):
                rec = dict(method=m, label=C.label(m), test_cfg_id=cfg, family=g["cfg_family"].iat[0],
                           profile=g["profile"].iat[0], N=int(g["N"].iat[0]), rho_tr=float(g["rho_tr"].iat[0]))
                for k in full_cols:
                    if k not in g.columns:
                        continue
                    s = summarize(g, k, self.sc, salt=(m, cfg))
                    rec.update({f"{k}_mean": s["mean"], f"{k}_std": s["std"], f"{k}_t_lo": s["t_lo"],
                                f"{k}_t_hi": s["t_hi"], f"{k}_b_lo": s["b_lo"], f"{k}_b_hi": s["b_hi"]})
                rec["unit"], rec["n_episodes"] = s["unit"], s["n_episodes"]
                rec["exp_ids"] = ";".join(sorted(set(g["exp_id"])))
                recs.append(rec)
        return t, pd.DataFrame(recs)

    # ---------------- T3 / T3b
    def t3(self) -> Table:
        cols = []
        if self.has("e4_pct"):
            cols.append(("e4_pct", "EVs without infeasible\nstep, E4 (%)"))
        cols += [("unserved_pct", "Unserved\nEVs (%)"), ("energy_not_served_kwh", "Energy not\nserved (kWh)"),
                 ("tr_violation_kwh", "Overload\nenergy (kWh)"), ("any_violation_pct", "Episodes with\noverload (%)")]
        ratio = self.has("a5_deficit_kwh") and self.ep["shortfall_deficit_ratio"].notna().any()
        if ratio:
            cols.append(("shortfall_deficit_ratio", "Shortfall /\ndeficit bound"))
        cols += [("n_evs_capped", "Capped\nrequests"), ("n_evs_admission_reduced", "Admission-\nreduced")]
        n = self.grid["test_cfg_id"].nunique()
        t = Table("T3", f"Constraint satisfaction per method over all {n} test-grid configurations (per episode): share of "
                        "EVs whose stay contains no infeasible step (E4), departure energy requirement, transformer "
                        "limit and request pipeline.", cols, notes=[COMMON_NOTE,
                  "Shortfall / deficit bound: energy not served divided by the feasibility deficit bound (Lemma S4), mean "
                  "over episodes with a positive deficit (safety-layer methods). Capped: requests reduced by the "
                  "arrival cap by more than 0.01 kWh; admission-reduced: requests reduced by admission control "
                  "(identical for all methods, DD31). Per-N values: Table T3b."])
        mk = self.grid_mask()
        dig = dict(e4_pct=2, unserved_pct=2, energy_not_served_kwh=2, tr_violation_kwh=2, any_violation_pct=1,
                   shortfall_deficit_ratio=3, n_evs_capped=2, n_evs_admission_reduced=2)
        for m in self.method_rows():
            if self.S(m, mk, "unserved_pct") is None:
                continue
            safe = C.info(m).safety_layer
            cells = {k: self.cell(m, mk, k, dig[k]) for k, _ in cols
                     if not (k == "shortfall_deficit_ratio" and not safe)}
            t.add(C.label(m), cells, group="with safety layer" if safe else "without safety layer")
        t.rows.sort(key=lambda r: (r["group"] != "with safety layer"))
        t.size, t.colsep = "scriptsize", "2.5pt"
        return t

    def t3b(self) -> Table:
        cols = []
        if self.has("e4_pct"):
            cols.append(("e4_pct", "E4: EVs without\ninfeasible step (%)"))
        cols.append(("infeasible_steps", "Infeasible\nsteps"))
        cols += [(col, f"{case} {lab}\n(steps)") for case, lab, col, _ in C.FALLBACK_CASES]
        for c, h in (("upper_bind_pct", "Upper limit\nbinding (% steps)"), ("lower_bind_pct", "Lower limit\nbinding (% steps)"),
                     ("lb_active_pct", "p_i = l_i\n(% token-steps)")):
            if self.has(c):
                cols.append((c, h))
        t = Table("T3b", "Primary endpoint E4 (share of EVs whose stay contains no infeasible step), infeasible steps, "
                         "fallback activations per case and binding rates of the aggregate limits and of the departure "
                         "lower bound, per station size N (profiles and tightness levels pooled; per episode).", cols,
                  notes=["Fallback cases: (i) = FB-1* (contradictory limits), (ii) = FB-2, (iii) = FB-1, (iv) = FB-0 "
                         "(code labels in the logs). Upper/lower limit binding: steps on which the aggregate sum is at "
                         "the forward / reverse limit (\u03bb > 0 / \u03bb < 0 on projected steps; raw methods: at or "
                         "beyond the limit), in % of the episode's steps. p_i = l_i: share of occupied charger-steps "
                         "whose commanded power equals a departure lower bound above the physical minimum.",
                         "Mean per episode; \u2020 deterministic method. Standard deviations and CIs in the _values.csv files."])
        dig = dict(e4_pct=2, infeasible_steps=2, fb1star_steps=2, fb2_steps=2, fb1_steps=2, fb0_steps=2,
                   upper_bind_pct=1, lower_bind_pct=1, lb_active_pct=1)
        for n in sorted(self.grid["N"].unique()):
            mk = self.grid_mask(N=[n])
            for m in self.present(C.SCALING_METHODS):
                if self.S(m, mk, "infeasible_steps") is None:
                    continue
                t.add(C.label(m), {k: self.cell(m, mk, k, dig[k], show="mean") for k, _ in cols}, group=f"N = {n}")
        t.size, t.colsep = "scriptsize", "2.5pt"
        return t

    # ---------------- T4
    def t4(self) -> Table:
        cols = [("gap_tr", "Gap (%)\nN \u2208 {10, 20}"), ("uns_tr", "Unserved (%)\nN \u2208 {10, 20}"),
                ("vio_tr", "Overload (kWh)\nN \u2208 {10, 20}"), ("gap_zs", "Gap (%)\nN \u2265 50"),
                ("uns_zs", "Unserved (%)\nN \u2265 50"), ("vio_zs", "Overload (kWh)\nN \u2265 50")]
        t = Table("T4", "Ablation: architecture (attention vs MLP) \u00d7 constraint handling (safety layer vs penalty), "
                        "further baselines that isolate one component, evaluation variants and ablations of SA-Attn, at "
                        "training sizes and in zero-shot transfer (N \u2265 50). Gap: optimality gap of J^pen.",
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
            t.add(C.label(m), cells(m), group="further baselines, variants and ablations")
        t.size, t.colsep = "scriptsize", "2.5pt"
        return t

    # ---------------- T5 (DD43(7))
    def rcell(self, sub, a, b, num, den, d, salt, pct=False, abs_den=False):
        r = ratio_diff(sub, a, b, num, den, self.sc, salt=salt, abs_den=abs_den)
        if pct:
            r = {**r, "diff": 100 * r["diff"], "b_lo": 100 * r["b_lo"], "b_hi": 100 * r["b_hi"]}
        c = cell_from_diff(r, d)
        if den == "oracle_ceil_eur" and not c.empty:
            c.exp_ids = sorted(set(c.exp_ids) | {f"oracle__{x}" for x in sub["test_cfg_id"].unique()})
        return c

    def t5(self) -> Table:
        cols = [("dpen_ev", "\u0394 J^pen\n(\u20ac per EV)\n[95 % CI]"), ("dpen_pct", "\u0394 J^pen\n(% of |J^LP,ceil|)\n[95 % CI]"),
                ("dprof_ev", "\u0394 profit\n(\u20ac per EV)\n[95 % CI]"), ("dprof_pct", "\u0394 profit\n(% of |J^LP,ceil|)\n[95 % CI]"),
                ("uns_pen", "Unserved EVs,\npenalty (%)"), ("vio_pen", "Overload,\npenalty (kWh)")]
        t = Table("T5", "Cost of the guarantee per station size N (profiles and tightness levels pooled): penalty-based "
                        "policy minus the same architecture with the safety layer, in penalised profit J^pen (primary) "
                        "and in raw profit (secondary), as ratios of sums over all matched episodes, per EV served and "
                        "in % of the hard perfect-information optimum |J^LP,ceil|, together with the violations of the "
                        "penalty-based policy.", cols, stub="N",
                  notes=[COMMON_NOTE, "\u0394 = \u03a3 (penalty \u2212 safety layer) / \u03a3 denominator over the "
                                      "matched episodes (seed-averaged per replay); per EV: denominator = admitted EVs "
                                      "(shared requests, DD31); %: denominator = |J^LP,ceil|. CI: stratified bootstrap "
                                      "over the seeds of both methods and the common replays. Paired tests: "
                                      "stats/tests_T5.csv."])
        for arch, ms, mp in C.GUARANTEE_PAIRS:
            if ms not in self.ep["method"].values or mp not in self.ep["method"].values:
                continue
            for n in sorted(self.grid["N"].unique()):
                sub = self.grid[self.grid["N"] == n]
                if not ({ms, mp} <= set(sub["method"])):
                    continue
                mk = self.grid_mask(N=[n])
                t.add(f"{n}", {"dpen_ev": self.rcell(sub, mp, ms, PEN, "n_evs", 3, ("T5a", n)),
                               "dpen_pct": self.rcell(sub, mp, ms, PEN, "oracle_ceil_eur", 2, ("T5b", n), pct=True, abs_den=True),
                               "dprof_ev": self.rcell(sub, mp, ms, "profit_eur", "n_evs", 3, ("T5c", n)),
                               "dprof_pct": self.rcell(sub, mp, ms, "profit_eur", "oracle_ceil_eur", 2, ("T5d", n), pct=True, abs_den=True),
                               "uns_pen": self.cell(mp, mk, "unserved_pct", 2),
                               "vio_pen": self.cell(mp, mk, "tr_violation_kwh", 2)}, group=arch)
        t.size, t.colsep = "scriptsize", "2.5pt"
        return t

    # ---------------- T6
    def t6(self) -> Table:
        sizes = sorted(self.grid["N"].unique())
        cols = [("train_h", "Training\ntime (h)")] + [(f"inf{n}", f"N = {n}\n(ms/step)") for n in sizes]
        t = Table("T6", "Computation: wall-clock training time per run and inference time per control step "
                        "(policy + safety layer, mean over episodes) versus station size N.", cols,
                  notes=[COMMON_NOTE, "Training time: last wall_clock_s of train_log.csv per run. Hardware: see meta.json."])
        tr = self.ds.train
        for m in self.method_rows(exclude_groups=("oracle", "other", "conformal", "sensitivity")):
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

    # ---------------- T7 (conformal; DD43(1): same soft-oracle gap as the other methods)
    def t7(self) -> Table | None:
        conf = sorted([m for m in self.ep["method"].unique() if C.conf_alpha(m) is not None], key=C.conf_alpha)
        if not conf and "attn_safe_conf_point" not in set(self.ep["method"]):
            return None
        cols = [("gap_pct", "Optimality\ngap (%)"), ("profit_per_ev", "Profit\n(\u20ac per EV)"),
                ("pen", "J^pen\n(\u20ac/episode)"), ("unserved_pct", "Unserved\nEVs (%)"),
                ("energy_not_served_kwh", "Energy not\nserved (kWh)")]
        has_cov, has_dep, has_dd = self.has("coverage_frac"), self.has("n_evs_departed_before_bound"), self.has("mean_d_minus_dlo")
        if has_cov:
            cols += [("coverage_frac", "Coverage\n(episode mean)")]
        if has_dep:
            cols += [("n_evs_departed_before_bound", "EVs departed\nbefore bound")]
        if has_dd:
            cols += [("mean_d_minus_dlo", "Mean d \u2212 d^lo\n(steps)")]
        cols += [("dprofit", "\u0394 profit vs known d\n(\u20ac per EV) [95 % CI]")]
        t = Table("T7", "Conformal departure bound: miscoverage level \u03b1 versus profit, departure shortfall and "
                        f"coverage (all {self.grid['test_cfg_id'].nunique()} test-grid configurations pooled). The conformal "
                        "variants use the shared requirement re-certified with their departure bound (DD43); the "
                        "optimality gap is w.r.t. the soft-penalty optimum, as for the other methods.", cols, stub="Variant", notes=[COMMON_NOTE,
                  "\u0394 profit: SA-Attn-CP(\u03b1) minus SA-Attn with known departure step; bootstrap CI over seeds."])
        mk = self.grid_mask()
        rows = (["attn_safe"] if "attn_safe" in self.ep["method"].values else []) + conf + self.present(["attn_safe_conf_point"])
        for m in rows:
            cells = {"gap_pct": self.cell(m, mk, "gap_pct", 1),
                     "profit_per_ev": self.cell(m, mk, "profit_per_ev", 3), "pen": self.cell(m, mk, PEN, 2),
                     "unserved_pct": self.cell(m, mk, "unserved_pct", 2),
                     "energy_not_served_kwh": self.cell(m, mk, "energy_not_served_kwh", 2)}
            if m != "attn_safe":
                if has_cov:
                    cells["coverage_frac"] = self.cell(m, mk, "coverage_frac", 3)
                if has_dep:
                    cells["n_evs_departed_before_bound"] = self.cell(m, mk, "n_evs_departed_before_bound", 2)
                if has_dd:
                    cells["mean_d_minus_dlo"] = self.cell(m, mk, "mean_d_minus_dlo", 2)
                if "attn_safe" in self.ep["method"].values:
                    cells["dprofit"] = self.dcell(self.grid, m, "attn_safe", "profit_per_ev", 3, salt=("T7", m))
            t.add("SA-Attn, known departure step" if m == "attn_safe" else C.label(m), cells)
        t.size, t.colsep = "scriptsize", "2.5pt"
        return t

    # ---------------- TS1: hard gaps (supplementary)
    def ts1(self) -> Table:
        cols = [("gap_hard_pct", "Gap vs J^LP,ceil (%)\n(hard optimum)"), ("gap_lp_pct", "Gap vs J^LP (%)\n(no ceil gain)")]
        t = Table("TS1", "Raw-profit optimality gap with respect to the hard perfect-information optimum J^LP,ceil and to "
                         "J^LP (without EV2Gym's ceil gain), per station size N (profiles and tightness levels pooled).",
                  cols, notes=[COMMON_NOTE])
        for n in sorted(self.grid["N"].unique()):
            mk = self.grid_mask(N=[n])
            for m in self.present(C.SCALING_METHODS):
                if self.S(m, mk, "profit_eur") is None:
                    continue
                t.add(C.label(m), {k: self.cell(m, mk, k, 2) for k, _ in cols}, group=f"N = {n}")
        return t

    # ---------------- TS2: sensitivities (DD28, DD36, DD39, penalty weights)
    SENS_NAMES = {"soc": "desired SoC {v}", "dt": "\u0394t = {v} min", "thr": "throughput cost {v} \u20ac/kWh",
                  "dpf": "discharge price factor {v}"}

    def ts2(self) -> Table | None:
        cols = [("pen_per_charger", "J^pen\n(\u20ac/charger)"), ("gap_pct", "Optimality\ngap (%)"),
                ("profit_per_charger", "Profit\n(\u20ac/charger)"), ("unserved_pct", "Unserved\nEVs (%)"),
                ("tr_violation_kwh", "Overload\n(kWh)")]
        t = Table("TS2", "Sensitivity analyses (per episode; profiles pooled): reference configuration, desired SoC 0.8, "
                         "\u0394t = 5 min (separately trained policy), battery throughput cost, discharge price factor, "
                         "penalty weights of the penalty-based policy, and the out-of-training tightness "
                         "\u03c1_tr = 3 kW.", cols, stub="Method", notes=[COMMON_NOTE])
        ep = self.ep
        sens = ep[ep["cfg_family"] == "sensitivity"]
        dig = dict(pen_per_charger=2, gap_pct=1, profit_per_charger=2, unserved_pct=2, tr_violation_kwh=2)

        def add_group(name, sub, methods):
            for m in methods:
                g = sub[sub["method"] == m]
                if g.empty:
                    continue
                mk = self.cfg_mask(g["test_cfg_id"].unique())
                t.add(C.label(m), {k: self.cell(m, mk, k, dig[k]) for k in dig}, group=name)
        bases = set(sens["sens_base"])
        base = ep[ep["is_grid"] & ep["cfg_base"].isin(bases)]
        if len(base):
            ns = ", ".join(f"{n:g}" for n in sorted(base["N"].unique()))
            rs = ", ".join(f"{r:g}" for r in sorted(base["rho_tr"].unique()))
            add_group(f"reference (N = {ns}, \u03c1_tr = {rs} kW)", base, self.present(C.SENS_METHODS))
        for tag in sorted(sens["sens_tag"].dropna().unique()):
            m = re.match(r"([a-z]+)([0-9.]+)", tag)
            name = self.SENS_NAMES.get(m.group(1), "{v}").format(v=m.group(2)) if m else tag
            add_group(name, sens[sens["sens_tag"] == tag], self.present(C.SENS_METHODS))
        wsub = ep[ep["is_grid"] & ep["method"].isin(C.SENS_WEIGHT_METHODS[1:])]
        if len(wsub):
            cf = set(wsub["test_cfg_id"])
            add_group("penalty weight w (\u20ac/kWh)", ep[ep["test_cfg_id"].isin(cf)], self.present(C.SENS_WEIGHT_METHODS))
        stress = ep[ep["cfg_family"] == "stress"]
        for n in sorted(stress["N"].unique()):
            add_group(f"\u03c1_tr = 3 kW (stress), N = {n}", stress[stress["N"] == n], self.present(C.SCALING_METHODS))
        if not t.rows:
            return None
        t.size, t.colsep = "scriptsize", "2.5pt"
        return t

    # ---------------- TS3: literature configuration
    def ts3(self) -> Table | None:
        lit = self.ep[self.ep["cfg_family"] == "literature"]
        if lit.empty:
            return None
        cols = [("pen", "J^pen\n(\u20ac/episode)"), ("gap_pct", "Optimality\ngap (%)"), ("profit_eur", "Profit\n(\u20ac/episode)"),
                ("energy_not_served_kwh", "Energy not\nserved (kWh)"), ("tr_violation_kwh", "Overload\n(kWh)"),
                ("inference_ms_per_step_mean", "Inference\n(ms/step)")]
        cfgs = sorted(lit["test_cfg_id"].unique())
        t = Table("TS3", f"Literature configuration ({', '.join(cfgs)}: EV2Gym V2GProfitMax, 25 chargers, 100 kW, "
                         "workplace profile), for comparison with published EV2Gym results.", cols, notes=[COMMON_NOTE])
        mk = self.cfg_mask(cfgs)
        dig = dict(pen=2, gap_pct=1, profit_eur=2, energy_not_served_kwh=2, tr_violation_kwh=2, inference_ms_per_step_mean=2)
        for m in C.method_order(lit["method"].unique()):
            if C.info(m).group in ("conformal", "oracle", "sensitivity", "ablation"):
                continue
            t.add(C.label(m), {k: self.cell(m, mk, PEN if k == "pen" else k, d) for k, d in dig.items()})
        return t


PEN = "penalised_profit_eur"


def build_all(ds, sc: StatConfig, outdir: Path, synthetic=False) -> dict:
    b = Builder(ds, sc, synthetic)
    tabs = {}
    tabs["T1"] = b.t1()
    tabs["T2"], t2full = b.t2()
    tabs["T3"] = b.t3()
    tabs["T3b"] = b.t3b()
    tabs["T4"] = b.t4()
    tabs["T5"] = b.t5()
    tabs["T6"] = b.t6()
    for name, fn in (("T7", b.t7), ("TS1", b.ts1), ("TS2", b.ts2), ("TS3", b.ts3)):
        tt = fn()
        if tt is not None:
            tabs[name] = tt
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
