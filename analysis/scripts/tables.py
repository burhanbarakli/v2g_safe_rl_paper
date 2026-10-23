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
from stats import StatConfig, summarize, diff_summary, ratio_diff, rel_gap_ratio

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
    std: float = np.nan      # unused since DD48 (kept for the frame layout)
    lo: float = np.nan
    hi: float = np.nan
    ci: str = ""             # 'bootstrap (replays)' | ''
    unit: str = ""           # 'seeds' | 'single run' | 'paired replays' | 'count'
    n: int = 0
    exp_ids: list = field(default_factory=list)
    seeds: dict = field(default_factory=dict)   # per-seed values (DD48)

    @property
    def empty(self):
        return not np.isfinite(self.value)


def cell_from_summary(s: dict, d=2, show="mean_std", extra_ids=()) -> Cell:
    """DD48 cell: mean of the seed-averaged policy, followed by (min\u2013max over seeds) when the method has
    two or more seeds; \u2020 for single-run (deterministic) methods. The bootstrap CI over replays and the
    per-seed values go to the _values.csv file."""
    if s is None or s.get("unit") in (None, "none") or not np.isfinite(s.get("mean", np.nan)):
        return Cell()
    ids = sorted(set(s["exp_ids"]) | set(extra_ids))
    if show == "iqm":
        txt = f"{fmt_num(s['iqm'], d)} [{fmt_ci(s['iqm_lo'], d)}, {fmt_ci(s['iqm_hi'], d)}]"
        return Cell(txt, s["iqm"], np.nan, s["iqm_lo"], s["iqm_hi"], "bootstrap (replays)", s["unit"],
                    s["n_episodes"], ids, s.get("seeds", {}))
    multi = np.isfinite(s.get("seed_min", np.nan))            # range over seeds available
    single = s.get("n_seeds", 0) < 2                          # deterministic method (one run): dagger
    if show == "mean" or (not multi and not single):
        txt = fmt_num(s["mean"], d) + (DAGGER if single else "")
    elif multi:
        sep = " to " if s["seed_min"] < 0 else "\u2013"          # avoid "-3.70--3.28"
        txt = f"{fmt_num(s['mean'], d)} ({fmt_num(s['seed_min'], d)}{sep}{fmt_num(s['seed_max'], d)})"
    else:
        txt = f"{fmt_num(s['mean'], d)}{DAGGER}"
    return Cell(txt, s["mean"], np.nan, s["b_lo"], s["b_hi"], "bootstrap (replays)", s["unit"], s["n_episodes"],
                ids, s.get("seeds", {}))


def cell_count(value, exp_ids, d=0) -> Cell:
    if value is None or not np.isfinite(value):
        return Cell()
    txt = f"{int(round(value)):,}" if d == 0 else fmt_num(value, d)
    return Cell(txt, float(value), np.nan, np.nan, np.nan, "", "count", 0, sorted(exp_ids))


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
    range_cols: tuple | None = None  # LaTeX: columns that keep "(min-max over seeds)"; others show the mean only
    max_lines: int = 64             # LaTeX: body lines per float; longer tables continue in a second float
    stub_width: str = "2.6cm"       # LaTeX: wrapped stub column if a row label is longer than ~26 characters

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
                    rec.update({f"{h} | value": c.value, f"{h} | ci_lo": c.lo, f"{h} | ci_hi": c.hi,
                                f"{h} | ci": c.ci, f"{h} | unit": c.unit, f"{h} | n_episodes": c.n})
                    for sd, v in sorted(c.seeds.items(), key=lambda kv: str(kv[0])):
                        rec[f"{h} | seed {sd}"] = v
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
            s = s.replace("\u03b1", r"$\alpha$").replace("\u03b2", r"$\beta$").replace("\u0394", r"$\Delta$").replace("\u03a3", r"$\Sigma$")
            s = s.replace("\u2264", r"$\le$").replace("\u2265", r"$\ge$").replace("\u20ac", r"\euro{}")
            s = s.replace("\u2212", "--").replace("\u00b7", r"$\cdot$")
            s = s.replace("|", r"$|$").replace("\u2208", r"$\in$").replace("\u00d7", r"$\times$").replace("\x00", r"\textbackslash{}")
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
        wrap = any(len(r["label"]) > 26 for r in self.rows)
        stubcol = (r">{\raggedright\arraybackslash}p{" + self.stub_width + "}") if wrap else "l"

        def txt(c, k):
            if c.empty:
                return c.text
            if self.range_cols is not None and k not in self.range_cols and c.unit == "seeds" and " (" in c.text:
                return c.text.split(" (")[0]
            return c.text

        # body lines, grouped; split into several floats if longer than max_lines (estimated wrapped lines)
        blocks, cur = [], None
        for r in self.rows:
            if not blocks or r["group"] != cur:
                blocks.append((r["group"], []))
                cur = r["group"]
            blocks[-1][1].append(r)
        wpl = 19 if wrap else 10 ** 6                          # characters per stub line (scriptsize, 2.9 cm)

        def weight(r):
            return max(1, -(-len(r["label"]) // wpl))
        parts, curp, w = [], [], 0
        for g, rows in blocks:
            for i, r in enumerate(rows):
                need = weight(r) + (1 if i == 0 and g else 0)
                if curp and w + need > self.max_lines:
                    parts.append(curp)
                    curp, w = [], 0
                    need = weight(r) + (1 if g else 0)
                    if g:
                        curp.append(("group", g + (" (continued)" if i > 0 else "")))
                elif i == 0 and g:
                    curp.append(("group", g))
                curp.append(("row", r))
                w += need
        if curp:
            parts.append(curp)

        head_line = esc(self.stub) + " & " + " & ".join(head(h) for _, h in self.columns) + r" \\"
        out = [r"% generated by analysis/scripts/tables.py -- do not edit by hand",
               r"% requires \usepackage{booktabs}, \usepackage{array}, \usepackage{adjustbox}, \usepackage{eurosym}",
               r"% (or define \euro) and, for tables split over two floats, \usepackage{caption} (\ContinuedFloat)"]
        for pi, part in enumerate(parts):
            out += [r"\begin{table}[!t]", r"\centering"]
            if pi == 0:
                out += [rf"\caption{{{cap}}}", rf"\label{{{lab}}}"]
            else:
                out += [r"\ifdefined\ContinuedFloat\ContinuedFloat\fi",
                        r"\caption[]{(continued)}"]
            out += ["\\" + self.size, r"\setlength{\tabcolsep}{" + self.colsep + "}",
                    r"\begin{adjustbox}{max width=\linewidth}",       # safety net; tables are sized to fit
                    r"\begin{tabular}{" + stubcol + "r" * n + "}", r"\toprule", head_line, r"\midrule"]
            first_group = True
            for kind, x in part:
                if kind == "group":
                    if not first_group:
                        out.append(r"\midrule")
                    first_group = False
                    out.append(rf"\multicolumn{{{n + 1}}}{{l}}{{\textit{{{esc(x)}}}}} \\")
                else:
                    out.append(esc(x["label"]).replace("+", r"+\allowbreak{}") + " & " +
                               " & ".join(esc(txt(x["cells"].get(k, Cell()), k)) for k, _ in self.columns) + r" \\")
            out += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]
            if self.notes and pi == len(parts) - 1:
                out.append(r"\par\smallskip{\raggedright\scriptsize " + r" \\{} ".join(esc(x) for x in self.notes) + r"\par}")
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

COMMON_NOTE = ("Values: mean of the seed-averaged policy over all test episodes; a range in parentheses is the "
               "min\u2013max over the 2 training seeds; \u2020 deterministic method (single run). "
               "Ranges for all columns, bootstrap confidence intervals and per-seed values are provided with the result "
               "data.")
# LaTeX: columns that keep the (min-max over seeds) range; the other columns show the mean only
RANGE_COLS = {"T1": ("gap_abs",), "T2": ("gap_abs",), "T3": ("e4_pct",), "T4": ("gap_tr", "gap_zs"), "T5": (),
              "T7": ("energy_not_served_kwh", "gap_abs"), "TS2": ("gap_abs",), "TS3": ("gap_abs",)}
GAP_NOTE = ("Optimality gap: J^pen of the soft-penalty perfect-information optimum minus J^pen of the method, "
            "divided by N (\u20ac per charger and episode).")
GAP_H = "Optimality gap\n(\u20ac per charger\nand episode)"
MLP_NOTE = "The zero-padded MLP is evaluated only for N \u2264 20."


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
        if s is not None:          # dagger = deterministic method, judged from the method's runs, not from this metric
            s = {**s, "n_seeds": max(s.get("n_seeds", 0), self.n_method_seeds(method))}
        extra = self.oracle_ids(s) if metric.startswith("gap") else ()
        return cell_from_summary(s, d, show, extra)

    def n_method_seeds(self, method):
        return int(self.ep.loc[self.ep["method"] == method, "seed"].nunique())

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

    def counts(self, df, col):
        return float(df[col].sum()) if col in df.columns and df[col].notna().any() else np.nan

    def best_baselines(self, k=3):
        """The k rule-based / MPC methods with the safety layer that have the lowest mean gap over the grid."""
        cand = [m for m in self.methods if C.info(m).group in ("heuristic", "mpc") and C.info(m).safety_layer]
        sc = {m: self.grid.loc[self.grid["method"] == m, "gap_abs"].mean() for m in cand}
        sc = {m: v for m, v in sc.items() if np.isfinite(v)}
        return sorted(sc, key=sc.get)[:k]

    def t2_methods(self):
        learned = self.present(["attn_safe", "deepsets_safe", "attn_safe_trainN50", "attn_proj_nolb",
                                "attn_penalty_safedeploy", "attn_penalty", "attn_lagrangian", "mlp_safe", "mlp_penalty"])
        return learned + [m for m in self.best_baselines() if m not in learned]

    # ---------------- T1
    def t1(self) -> Table:
        cols = [("gap_abs", GAP_H), ("gap_iqm", "Optimality gap,\nIQM [95 % CI]"),
                ("pen", "J^pen\n(\u20ac/episode)"), ("profit_eur", "Profit\n(\u20ac/episode)"),
                ("energy_not_served_kwh", "Energy not\nserved (kWh)"), ("tr_violation_kwh", "Overload\n(kWh)")]
        t = Table("T1", "Main comparison at the training station sizes N \u2208 {10, 20} (three arrival profiles and "
                        "both tightness levels pooled; per episode). J^pen: penalised profit with 10 \u20ac/kWh for energy "
                        "not served against the shared requirement and for transformer overload energy.", cols,
                  notes=[COMMON_NOTE, GAP_NOTE, "IQM: interquartile mean of the per-episode optimality gap; bootstrap CI "
                                                "over test replays. " + MLP_NOTE])
        for n in C.TRAIN_SIZES:
            mk = self.grid_mask(N=[n])
            for m in self.method_rows():
                if self.S(m, mk, "profit_eur") is None:
                    continue
                t.add(C.label(m), {"gap_abs": self.cell(m, mk, "gap_abs", 3),
                                   "gap_iqm": self.cell(m, mk, "gap_abs", 3, show="iqm"),
                                   "pen": self.cell(m, mk, PEN, 2), "profit_eur": self.cell(m, mk, "profit_eur", 2),
                                   "energy_not_served_kwh": self.cell(m, mk, "energy_not_served_kwh", 2),
                                   "tr_violation_kwh": self.cell(m, mk, "tr_violation_kwh", 2)}, group=f"N = {n}")
        t.size, t.colsep = "scriptsize", "3pt"
        return t

    # ---------------- T2 (zero-shot)
    def t2(self) -> tuple[Table, pd.DataFrame]:
        cols = [("gap_abs", GAP_H), ("pen_per_charger", "J^pen\n(\u20ac/charger)"),
                ("profit_per_charger", "Profit\n(\u20ac/charger)"), ("unserved_pct", "Unserved\nEVs (%)"),
                ("tr_violation_kwh", "Overload\n(kWh)"),
                ("transfer", "Transfer gap to the\nN = 50 reference\n(\u20ac/charger) [95 % CI]")]
        t = Table("T2", "Zero-shot transfer across station sizes N for policies trained at N \u2208 {10, 20} (profiles and "
                        "\u03c1_tr \u2208 {4, 7} kW pooled; per episode): learned methods, SA-Attn trained at N = 50 "
                        "(reference) and the three baselines with the safety layer that have the lowest mean optimality gap.",
                  cols, notes=[COMMON_NOTE, GAP_NOTE, "Transfer gap: optimality gap of the method minus that of SA-Attn "
                                                      "trained at N = 50, paired on the shared test replays; bootstrap CI "
                                                      "over replays. " + MLP_NOTE])
        digits = dict(gap_abs=3, pen_per_charger=2, profit_per_charger=2, unserved_pct=2, tr_violation_kwh=2)
        meths = self.t2_methods()
        for n in sorted(self.grid["N"].unique()):
            mk = self.grid_mask(N=[n])
            sub = self.grid[self.grid["N"] == n]
            for m in meths:
                if self.S(m, mk, "profit_eur") is None:
                    continue
                cells = {k: self.cell(m, mk, k, digits[k]) for k in digits}
                if m in ("attn_safe", "deepsets_safe") and "attn_safe_trainN50" in set(sub["method"]):
                    cells["transfer"] = self.dcell(sub, m, "attn_safe_trainN50", "gap_abs", 3, salt=("transfer", n))
                t.add(C.label(m), cells, group=f"N = {n}" + (" (training size)" if n in C.TRAIN_SIZES else ""))
        t.size, t.colsep = "scriptsize", "2.2pt"
        recs = []
        full_cols = ["gap_abs", "pen_per_charger", "profit_per_charger", "unserved_pct", "energy_not_served_kwh",
                     "tr_violation_kwh", "fallback_steps", "e4_pct"]
        for m in self.methods:
            for cfg, g in self.ep[self.ep["method"] == m].groupby("test_cfg_id"):
                rec = dict(method=m, label=C.label(m), test_cfg_id=cfg, family=g["cfg_family"].iat[0],
                           profile=g["profile"].iat[0], N=int(g["N"].iat[0]), rho_tr=float(g["rho_tr"].iat[0]))
                for k in full_cols:
                    if k not in g.columns:
                        continue
                    s_ = summarize(g, k, self.sc, salt=(m, cfg))
                    rec.update({f"{k}_mean": s_["mean"], f"{k}_seed_min": s_["seed_min"], f"{k}_seed_max": s_["seed_max"],
                                f"{k}_b_lo": s_["b_lo"], f"{k}_b_hi": s_["b_hi"]})
                rec["n_seeds"], rec["n_episodes"] = int(g["seed"].nunique()), int(len(g))
                rec["exp_ids"] = ";".join(sorted(set(g["exp_id"])))
                recs.append(rec)
        return t, pd.DataFrame(recs)

    # ---------------- T3 / T3b / T3c / T3d
    def t3(self) -> Table:
        cols = []
        if self.has("e4_pct"):
            cols.append(("e4_pct", "EVs without\ninfeasible\nstep, E4 (%)"))
        cols += [("unserved_pct", "Unserved\nEVs (%)"), ("energy_not_served_kwh", "Energy\nnot served\n(kWh)"),
                 ("tr_violation_kwh", "Overload\nenergy\n(kWh)"), ("any_violation_pct", "Episodes\nwith\noverload (%)")]
        if self.has("a5_deficit_kwh") and self.ep["shortfall_deficit_ratio"].notna().any():
            cols.append(("shortfall_deficit_ratio", "Shortfall /\ndeficit\nbound"))
        cols += [("n_evs_capped", "Capped\nrequests"), ("n_evs_admission_reduced", "Admission-\nreduced\nrequests")]
        n = self.grid["test_cfg_id"].nunique()
        t = Table("T3", f"Constraint satisfaction per method over all {n} test-grid configurations (per episode): share of "
                        "EVs whose stay contains no infeasible step (E4), energy not served against the shared admitted "
                        "requirement, transformer limit and request pipeline.", cols, notes=[COMMON_NOTE,
                  "Shortfall / deficit bound: energy not served divided by the a-priori deficit bound of the fallback rule, "
                  "averaged over the episodes with a positive deficit within each configuration and then over the "
                  "configurations (methods with the safety layer). Capped: requests reduced by "
                  "the arrival cap by more than 0.01 kWh. Admission-reduced: requests reduced by admission control "
                  "(identical for all methods)."])
        mk = self.grid_mask()
        dig = dict(e4_pct=2, unserved_pct=2, energy_not_served_kwh=2, tr_violation_kwh=2, any_violation_pct=1,
                   shortfall_deficit_ratio=3, n_evs_capped=2, n_evs_admission_reduced=2)
        rows = self.method_rows() + self.present(["attn_safe_noadm", "attn_penalty_noadm", "attn_penalty_safedeploy"])
        for m in rows:
            if self.S(m, mk, "unserved_pct") is None:
                continue
            safe = C.info(m).safety_layer
            cells = {k: self.cell(m, mk, k, dig[k]) for k, _ in cols if not (k == "shortfall_deficit_ratio" and not safe)}
            t.add(C.label(m), cells, group="with safety layer" if safe else "without safety layer")
        t.rows.sort(key=lambda r: (r["group"] != "with safety layer"))
        t.size, t.colsep = "scriptsize", "2.2pt"
        return t

    def t3b(self) -> Table:
        cols = []
        if self.has("e4_pct"):
            cols.append(("e4_pct", "E4\n(%)"))
        cols.append(("infeasible_steps", "Infeasible\nsteps"))
        cols += [(col, f"Case {case}\n(steps)") for case, lab, col, _ in C.FALLBACK_CASES]
        for c, h in (("upper_bind_pct", "Upper limit\nbinding\n(% steps)"), ("lower_bind_pct", "Lower limit\nbinding\n(% steps)"),
                     ("lb_active_pct", "p_i = l_i\n(% charger-\nsteps)")):
            if self.has(c):
                cols.append((c, h))
        t = Table("T3b", "E4 (share of EVs whose stay contains no infeasible step), infeasible steps, fallback activations per case and binding rates of the aggregate limits and "
                         "of the departure lower bound per station size N (profiles and tightness levels pooled; mean per "
                         "episode).", cols,
                  notes=["Fallback cases: (i) contradictory limits, (ii) excess PV export, (iii) excess required power, "
                         "(iv) negative budget. Limit binding: steps on which the aggregate power is at the forward or "
                         "reverse limit, in % of steps. p_i = l_i: share of occupied charger-steps whose power equals a "
                         "departure lower bound above the physical minimum.",
                         "Mean per episode of the seed-averaged policy; \u2020 deterministic method (single run)."])
        dig = dict(e4_pct=2, infeasible_steps=2, fb1star_steps=2, fb2_steps=2, fb1_steps=2, fb0_steps=2,
                   upper_bind_pct=1, lower_bind_pct=1, lb_active_pct=1)
        for n in sorted(self.grid["N"].unique()):
            mk = self.grid_mask(N=[n])
            for m in self.t2_methods():
                if self.S(m, mk, "infeasible_steps") is None:
                    continue
                t.add(C.label(m), {k: self.cell(m, mk, k, dig[k], show="mean") for k, _ in cols}, group=f"N = {n}")
        t.size, t.colsep = "scriptsize", "2.2pt"
        return t

    COUNT_COLS = [("n_episodes", "Episodes"), ("n_evs", "Admitted\nEVs"), ("infeasible_steps", "Infeasible\nsteps"),
                  ("fallback_steps", "Fallback\nactivations"), ("n_evs_stay_infeasible", "EVs with an\ninfeasible\nstep"),
                  ("n_evs_unserved", "Unserved\nEVs"), ("tr_violation_steps", "Overload\nsteps"),
                  ("unserved_no_infeasible", "Unserved EVs\nin episodes\nwithout infea-\nsible step")]

    def _count_cells(self, df):
        ids = df["exp_id"].unique().tolist()
        cells = {}
        for k, _ in self.COUNT_COLS:
            v = float(len(df)) if k == "n_episodes" else self.counts(df, k)
            cells[k] = cell_count(v, ids)
        return cells

    def t3c(self) -> Table:
        t = Table("T3c", "Event counts per method over all evaluated episodes (all test configurations, both training "
                         "seeds).", self.COUNT_COLS,
                  notes=["Sums over episodes. Unserved EVs: against the shared admitted requirement. Last column: "
                         "episode-level check of Theorem 1(b); it is 0 whenever every unserved EV lies in an episode with an "
                         "infeasible step. Theorem 1(b) covers the methods with the safety layer and the departure lower "
                         "bound, not the conformal variants or the methods without the bound or the layer."])
        for m in self.methods:
            if C.info(m).group == "oracle":
                continue
            df = self.ep[self.ep["method"] == m]
            if df.empty:
                continue
            safe, conf = C.info(m).safety_layer, C.info(m).group == "conformal"
            grp = ("conformal departure bound" if conf else
                   "with safety layer" if safe and m != "attn_proj_nolb" else
                   "safety layer without departure bound" if m == "attn_proj_nolb" else "without safety layer")
            t.add(C.label(m), self._count_cells(df), group=grp)
        order = ["with safety layer", "safety layer without departure bound", "conformal departure bound", "without safety layer"]
        t.rows.sort(key=lambda r: order.index(r["group"]))
        t.size, t.colsep = "scriptsize", "2.2pt"
        return t

    def t3d(self) -> Table:
        t = Table("T3d", "Event counts per method and station size N (all test configurations of that N, both training "
                         "seeds); columns as in the table of event counts over all episodes.", self.COUNT_COLS,
                  notes=["Sums over episodes. The stress configurations (\u03c1_tr = 3 kW; N = 20, 200) and the literature "
                         "configuration (N = 25) are included at their N."])
        for n in sorted(self.ep["N"].unique()):
            sub = self.ep[self.ep["N"] == n]
            for m in self.t2_methods():
                df = sub[sub["method"] == m]
                if df.empty:
                    continue
                t.add(C.label(m), self._count_cells(df), group=f"N = {n}")
        t.size, t.colsep = "scriptsize", "2.2pt"
        return t

    # ---------------- T4 (architecture / constraint-handling ablation from the runs that exist)
    def t4(self) -> Table:
        cols = [("gap_tr", "Optimality gap\nN \u2208 {10, 20}"), ("uns_tr", "Unserved (%)\nN \u2208 {10, 20}"),
                ("vio_tr", "Overload (kWh)\nN \u2208 {10, 20}"), ("gap_zs", "Optimality gap\nN \u2265 50"),
                ("uns_zs", "Unserved (%)\nN \u2265 50"), ("vio_zs", "Overload (kWh)\nN \u2265 50")]
        t = Table("T4", "Architecture and constraint handling at the training sizes N \u2208 {10, 20} and in zero-shot "
                        "transfer (N \u2265 50): attention, Deep Sets and zero-padded MLP policies with the safety layer, "
                        "with a penalty or PPO-Lagrangian, with the layer but without the departure lower bound, and the "
                        "penalty-trained policy with the layer added at deployment.",
                  cols, stub="Variant", notes=[COMMON_NOTE, GAP_NOTE, MLP_NOTE])
        sizes = sorted(self.grid["N"].unique())
        mtr = self.grid_mask(N=[n for n in sizes if n in C.TRAIN_SIZES])
        mzs = self.grid_mask(N=[n for n in sizes if n >= 50])
        for m in self.present(C.T4_ROWS):
            t.add(C.label(m), {"gap_tr": self.cell(m, mtr, "gap_abs", 3), "uns_tr": self.cell(m, mtr, "unserved_pct", 2),
                               "vio_tr": self.cell(m, mtr, "tr_violation_kwh", 2), "gap_zs": self.cell(m, mzs, "gap_abs", 3),
                               "uns_zs": self.cell(m, mzs, "unserved_pct", 2), "vio_zs": self.cell(m, mzs, "tr_violation_kwh", 2)})
        t.size, t.colsep = "scriptsize", "2.2pt"
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
        cols = [("dpen_ev", "\u0394 J^pen\n(\u20ac per EV)\n[95 % CI]"), ("dpen_pct", "\u0394 J^pen (% of\n|J^LP,ceil|)\n[95 % CI]"),
                ("dprof_ev", "\u0394 profit\n(\u20ac per EV)\n[95 % CI]"), ("dprof_pct", "\u0394 profit (% of\n|J^LP,ceil|)\n[95 % CI]"),
                ("uns_pen", "Unserved\nEVs, pen-\nalty (%)"), ("vio_pen", "Overload,\npenalty\n(kWh)")]
        t = Table("T5", "Cost of the guarantee per station size N (profiles and tightness levels pooled): penalty-based "
                        "policy minus the same architecture with the safety layer, in J^pen and in raw profit, as ratios of "
                        "sums over matched episodes, per admitted EV and in % of |J^LP,ceil|, with the violations of the "
                        "penalty-based policy.", cols, stub="N",
                  notes=[COMMON_NOTE, "\u0394 = \u03a3 (penalty \u2212 safety layer) / \u03a3 denominator over the matched "
                                      "test replays of the seed-averaged policies; denominator: admitted EVs (identical for "
                                      "both policies) or |J^LP,ceil| (hard perfect-information optimum). CI: bootstrap over "
                                      "replays, stratified by configuration. " + MLP_NOTE])
        for arch, ms, mp in C.GUARANTEE_PAIRS:
            if ms not in self.ep["method"].values or mp not in self.ep["method"].values:
                continue
            for n in sorted(self.grid["N"].unique()):
                sub = self.grid[self.grid["N"] == n]
                if not ({ms, mp} <= set(sub["method"])):
                    continue
                mk = self.grid_mask(N=[n])
                t.add(f"{n}", {"dpen_ev": self.rcell(sub, mp, ms, PEN, "n_evs", 3, ("T5a", n)),
                               "dpen_pct": self.rcell(sub, mp, ms, PEN, "oracle_ceil_eur", 1, ("T5b", n), pct=True, abs_den=True),
                               "dprof_ev": self.rcell(sub, mp, ms, "profit_eur", "n_evs", 3, ("T5c", n)),
                               "dprof_pct": self.rcell(sub, mp, ms, "profit_eur", "oracle_ceil_eur", 1, ("T5d", n), pct=True, abs_den=True),
                               "uns_pen": self.cell(mp, mk, "unserved_pct", 2),
                               "vio_pen": self.cell(mp, mk, "tr_violation_kwh", 2)}, group=arch)
        t.size, t.colsep = "scriptsize", "2.2pt"
        return t

    # ---------------- T6
    def t6(self) -> Table:
        sizes = sorted(self.grid["N"].unique())
        cols = [("train_h", "Training\ntime (h)")] + [(f"inf{n}", f"N = {n}\n(ms/step)") for n in sizes]
        t = Table("T6", "Computation: training time per run and inference time per control step (policy and safety layer) "
                        "versus station size N.", cols,
                  notes=["Training time: wall-clock time of each training run, mean (min\u2013max) over the 2 runs. "
                         "Inference: mean over episodes. Hardware: see the experimental setup."])
        tr = self.ds.train
        for m in dict.fromkeys(self.method_rows(exclude_groups=("oracle", "other", "conformal", "sensitivity")) +
                               self.present(["attn_penalty_safedeploy"])):          # no duplicate rows
            cells = {}
            ids = sorted(self.ep.loc[self.ep["method"] == m, "exp_id"].unique())
            if len(tr):
                tt = tr[tr["exp_id"].isin(ids)].groupby("exp_id")["wall_clock_s"].max() / 3600
                if len(tt):
                    txt = (f"{fmt_num(tt.mean(), 2)} ({fmt_num(tt.min(), 2)}\u2013{fmt_num(tt.max(), 2)})"
                           if len(tt) > 1 else fmt_num(tt.mean(), 2))
                    cells["train_h"] = Cell(txt, float(tt.mean()), np.nan, np.nan, np.nan, "", "runs", len(tt),
                                            sorted(tt.index), {k: float(v) for k, v in tt.items()})
            for n in sizes:
                cells[f"inf{n}"] = self.cell(m, self.grid_mask(N=[n]), "inference_ms_per_step_mean", 2, show="mean")
            if any(not c.empty for c in cells.values()):
                t.add(C.label(m), cells)
        return t

    # ---------------- T7 (conformal)
    def t7(self) -> Table | None:
        conf = sorted([m for m in self.ep["method"].unique() if C.conf_alpha(m) is not None], key=C.conf_alpha)
        if not conf and "attn_safe_conf_point" not in set(self.ep["method"]):
            return None
        cols = [("target", "Target\ncoverage\n1 \u2212 \u03b1"), ("coverage_frac", "Coverage\n(episode\nmean)"),
                ("energy_not_served_kwh", "Shortfall vs\nshared requi-\nrement (kWh)"),
                ("ens_total", "Total\nshortfall\n(kWh, all\nepisodes)"), ("gap_abs", GAP_H),
                ("energy_charged_kwh", "Energy\ncharged\n(kWh)")]
        t = Table("T7", "Conformal departure bound per miscoverage level \u03b1: target and achieved coverage, shortfall "
                        "against the shared admitted requirement, optimality gap and charged energy "
                        f"(all {self.grid['test_cfg_id'].nunique()} test-grid configurations pooled; per episode unless "
                        "stated).", cols, stub="Variant", notes=[COMMON_NOTE, GAP_NOTE,
                  "Coverage: episode mean of the fraction of EVs whose actual departure is not earlier than the conformal "
                  "bound. Total shortfall: sum over all episodes of both seeds."])
        mk = self.grid_mask()
        rows = (["attn_safe"] if "attn_safe" in self.ep["method"].values else []) + conf + self.present(["attn_safe_conf_point"])
        for m in rows:
            a = C.conf_alpha(m)
            df = self.grid[self.grid["method"] == m]
            cells = {"energy_not_served_kwh": self.cell(m, mk, "energy_not_served_kwh", 2),
                     "ens_total": cell_count(self.counts(df, "energy_not_served_kwh"), df["exp_id"].unique()),
                     "gap_abs": self.cell(m, mk, "gap_abs", 3),
                     "energy_charged_kwh": self.cell(m, mk, "energy_charged_kwh", 1)}
            if a is not None:
                cells["target"] = Cell(f"{1 - a:.2f}", 1 - a, np.nan, np.nan, np.nan, "", "target", 0,
                                       sorted(df["exp_id"].unique().tolist()))   # target of these runs
            if m != "attn_safe" and self.has("coverage_frac"):
                cells["coverage_frac"] = self.cell(m, mk, "coverage_frac", 3)
            t.add("SA-Attn, known departure step" if m == "attn_safe" else C.label(m), cells)
        t.size, t.colsep = "scriptsize", "2.2pt"
        return t

    # ---------------- TS1: relative gaps as ratios of sums (supplementary, DD47)
    def ts1(self) -> Table:
        cols = [("soft", "J^pen gap vs soft\noptimum (%) [95 % CI]"), ("hard", "Profit gap vs\nJ^LP,ceil (%) [95 % CI]"),
                ("lp", "Profit gap vs\nJ^LP (%) [95 % CI]")]
        t = Table("TS1", "Relative optimality gaps as ratios of sums, per station size N (profiles and tightness levels "
                         "pooled): 100 \u00b7 \u03a3(J* \u2212 J) / \u03a3|J*| over all test episodes of the seed-averaged "
                         "policy, for the soft-penalty optimum (J^pen) and for the hard optima J^LP,ceil and J^LP (raw "
                         "profit).", cols, notes=["CI: bootstrap over replays, stratified by configuration. Per-episode "
                                                  "percentages are not reported because the optimum's J^pen is often "
                                                  "close to zero or negative."])
        for n in sorted(self.grid["N"].unique()):
            sub = self.grid[self.grid["N"] == n]
            for m in self.t2_methods():
                g = sub[sub["method"] == m]
                if g.empty:
                    continue
                cells = {}
                for k, val, orc in (("soft", PEN, "oracle_pen_eur"), ("hard", "profit_eur", "oracle_ceil_eur"),
                                    ("lp", "profit_eur", "oracle_profit_eur")):
                    r = rel_gap_ratio(g, val, orc, self.sc, salt=(m, n, k))
                    c = cell_from_diff(r, 1)
                    if not c.empty:
                        c.exp_ids = sorted(set(c.exp_ids) | {f"oracle__{x}" for x in g["test_cfg_id"].unique()})
                    cells[k] = c
                t.add(C.label(m), cells, group=f"N = {n}")
        t.size, t.colsep = "scriptsize", "2.2pt"
        return t

    # ---------------- TS2: stress level only (sensitivities were not run)
    def ts2(self) -> Table | None:
        stress = self.ep[self.ep["cfg_family"] == "stress"]
        sens = self.ep[self.ep["cfg_family"] == "sensitivity"]
        if stress.empty and sens.empty:
            return None
        cols = [("gap_abs", GAP_H), ("pen_per_charger", "J^pen\n(\u20ac/charger)"),
                ("profit_per_charger", "Profit\n(\u20ac/charger)"), ("unserved_pct", "Unserved\nEVs (%)"),
                ("tr_violation_kwh", "Overload\n(kWh)"), ("e4_pct", "E4\n(%)")]
        t = Table("TS2", "Out-of-training transformer rating \u03c1_tr = 3 kW per charger (stress level), per station "
                         "size N (profiles pooled; per episode).", cols, notes=[COMMON_NOTE, GAP_NOTE, MLP_NOTE])
        dig = dict(gap_abs=3, pen_per_charger=2, profit_per_charger=2, unserved_pct=2, tr_violation_kwh=2, e4_pct=2)
        for n in sorted(stress["N"].unique()):
            g = stress[stress["N"] == n]
            for m in self.t2_methods():
                gm = g[g["method"] == m]
                if gm.empty:
                    continue
                mk = self.cfg_mask(gm["test_cfg_id"].unique())
                t.add(C.label(m), {k: self.cell(m, mk, k, d) for k, d in dig.items()},
                      group=f"\u03c1_tr = 3 kW, N = {n}")
        if not t.rows:
            return None
        t.size, t.colsep = "scriptsize", "2.2pt"
        return t

    # ---------------- TS3: literature configuration
    def ts3(self) -> Table | None:
        lit = self.ep[self.ep["cfg_family"] == "literature"]
        if lit.empty:
            return None
        cols = [("gap_abs", GAP_H), ("pen", "J^pen\n(\u20ac/episode)"), ("profit_eur", "Profit\n(\u20ac/episode)"),
                ("energy_not_served_kwh", "Energy not\nserved (kWh)"), ("tr_violation_kwh", "Overload\n(kWh)"),
                ("inference_ms_per_step_mean", "Inference\n(ms/step)")]
        cfgs = sorted(lit["test_cfg_id"].unique())
        t = Table("TS3", "Literature configuration (EV2Gym V2GProfitMax layout, 25 chargers, workplace profile), for "
                         "comparison with published EV2Gym results.", cols, notes=[COMMON_NOTE, GAP_NOTE])
        mk = self.cfg_mask(cfgs)
        dig = dict(gap_abs=3, pen=2, profit_eur=2, energy_not_served_kwh=2, tr_violation_kwh=2, inference_ms_per_step_mean=2)
        for m in C.method_order(lit["method"].unique()):
            if C.info(m).group in ("conformal", "oracle", "sensitivity", "ablation"):
                continue
            t.add(C.label(m), {k: self.cell(m, mk, PEN if k == "pen" else k, d) for k, d in dig.items()})
        return t


PEN = "jpen_eur"


def build_all(ds, sc: StatConfig, outdir: Path, synthetic=False) -> dict:
    b = Builder(ds, sc, synthetic)
    tabs = {}
    tabs["T1"] = b.t1()
    tabs["T2"], t2full = b.t2()
    for name, fn in (("T3", b.t3), ("T3b", b.t3b), ("T3c", b.t3c), ("T3d", b.t3d), ("T4", b.t4), ("T5", b.t5),
                     ("T6", b.t6), ("T7", b.t7), ("TS1", b.ts1), ("TS2", b.ts2), ("TS3", b.ts3)):
        tt = fn()
        if tt is not None:
            tabs[name] = tt
    for name, t in tabs.items():
        if name in RANGE_COLS:
            t.range_cols = RANGE_COLS[name]
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
