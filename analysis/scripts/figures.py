"""Publication figures F2-F9 (vector PDF + EPS + SVG + 300 dpi PNG).

Style: sans-serif (Arial/Helvetica, metric-compatible Liberation Sans as fallback), 8 pt text,
full width 6.5 in (SAUCIS is single-column) or half width 3.5 in; Okabe-Ito colours plus distinct
markers and line styles (solid = safety layer, dashed = penalty / raw), so that every figure is
readable in greyscale. Confidence bands use opaque lightened colours (EPS has no transparency).
Every figure also writes the plotted numbers to figures/data/<name>.csv (with exp_ids).
"""
from __future__ import annotations

import sys
from pathlib import Path

import logging
import matplotlib
matplotlib.use("Agg")
logging.getLogger("fontTools").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
import numpy as np
import pandas as pd

import common as C
from stats import StatConfig, summarize, diff_summary, t_ci

FULL_W = 6.5
HALF_W = 3.5
F3_METHODS = C.FIG_SCALING          # names.py
F9_METHODS = C.FIG_INFERENCE


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7, "legend.frameon": False,
        "lines.linewidth": 1.2, "lines.markersize": 4, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e0e0e0", "grid.linewidth": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "errorbar.capsize": 1.5, "mathtext.fontset": "custom", "mathtext.rm": "sans",
        "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold", "mathtext.sf": "sans", "mathtext.cal": "sans", "axes.unicode_minus": True, "hatch.linewidth": 0.5, "axes.titlelocation": "left",
    })


class FigWriter:
    def __init__(self, outdir: Path, synthetic: bool):
        self.outdir = Path(outdir)
        self.synthetic = synthetic
        (self.outdir / "data").mkdir(parents=True, exist_ok=True)
        self.written = {}

    def save(self, fig, name: str, data: pd.DataFrame | None = None):
        if self.synthetic:
            fig.suptitle(C.active_mark(), color="#C00000", fontweight="bold", fontsize=9, y=1.02)
            fig.text(0.5, 0.5, C.active_mark().split()[0], fontsize=44, color="#e8a0a0", ha="center", va="center",
                     rotation=25, zorder=1000, alpha=0.35, fontweight="bold")
        paths = []
        for ext in ("pdf", "eps", "svg", "png"):
            p = self.outdir / f"{name}.{ext}"
            fig.savefig(p, dpi=300 if ext == "png" else None)
            paths.append(p)
        plt.close(fig)
        if data is not None:
            if self.synthetic:
                data = data.copy()
                data.insert(0, "WATERMARK", C.active_mark())
            data.to_csv(self.outdir / "data" / f"{name}.csv", index=False)
        self.written[name] = paths
        print(f"[figures] wrote {name}", file=sys.stderr)


def _style(m):
    i = C.info(m)
    return dict(color=i.color, marker=i.marker, linestyle=i.linestyle,
                markerfacecolor=i.color if i.filled else "white", markeredgecolor=i.color, label=i.label)


def _logN(ax, sizes):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(sizes))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Station size N (chargers)")


def _train_span(ax, sizes):
    """Shade the training sizes only if the plotted N range contains them (partial/smoke grids)."""
    if not sizes or min(sizes) > 10 or max(sizes) < 20:
        return False
    ax.axvspan(10, 20, color="#efefef", zorder=0, lw=0)
    return True


def _legend_below(fig, axes, ncol=4, y=-0.02):
    h, l = [], []
    for ax in np.atleast_1d(axes).ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l and not ll.startswith("_"):
                h.append(hh)
                l.append(ll)
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, handlelength=2.6, columnspacing=1.2)


def _panel(ax, letter):
    ax.set_title(f"({letter})", fontweight="bold", loc="left", fontsize=8.5)


class Figures:
    def __init__(self, ds, sc: StatConfig, outdir: Path, synthetic=False):
        setup_style()
        self.ds, self.sc = ds, sc
        self.ep = ds.episodes
        self.grid = self.ep[self.ep["is_grid"]]
        self.fw = FigWriter(outdir, synthetic)
        self.sizes = sorted(self.grid["N"].unique().tolist())

    def present(self, ms):
        have = set(self.ep["method"])
        return [m for m in ms if m in have]

    def _byN(self, methods, metric, extra=None):
        recs = []
        for m in methods:
            sub = self.grid[self.grid["method"] == m]
            if extra:
                for k, v in extra.items():
                    sub = sub[sub[k] == v]
            for n, g in sub.groupby("N"):
                s = summarize(g, metric, self.sc, salt=(m, n, str(extra)))
                recs.append(dict(method=m, label=C.label(m), N=n, metric=metric, mean=s["mean"], std=s["std"],
                                 ci_lo=s["t_lo"], ci_hi=s["t_hi"], b_lo=s["b_lo"], b_hi=s["b_hi"], unit=s["unit"],
                                 n_seeds=s["n_seeds"], n_episodes=s["n_episodes"], **(extra or {}),
                                 exp_ids=";".join(s["exp_ids"])))
        return pd.DataFrame(recs)

    @staticmethod
    def _err(d):
        lo = np.where(np.isfinite(d["ci_lo"]), d["mean"] - d["ci_lo"], 0)
        hi = np.where(np.isfinite(d["ci_hi"]), d["ci_hi"] - d["mean"], 0)
        return np.vstack([np.clip(lo, 0, None), np.clip(hi, 0, None)])

    # ------------------------------------------------------------------ F2
    def f2_training(self):
        tr = self.ds.train
        if tr is None or tr.empty:
            print("[figures] F2 skipped: no train_log.csv", file=sys.stderr)
            return
        tr = tr.copy()
        tr["method"] = tr["exp_id"].str.split("__").str[0]
        methods = self.present([m for m in C.MAIN_LEARNED]) or C.method_order(tr["method"].unique())
        methods = [m for m in methods if m in set(tr["method"])]
        panels = [("episode_return_eur_mean", "Episode return (€)"),
                  ("frac_unserved_mean", "Fraction of EVs unserved"),
                  ("tr_violation_kwh_mean", "Transformer violation (kWh)")]
        fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.15), constrained_layout=True)
        recs = []
        for ax, (col, ylab), let in zip(axes, panels, "abc"):
            for m in methods:
                runs = tr[tr["method"] == m]
                ends = runs.groupby("exp_id")["env_steps"].max()
                grid = np.linspace(0, ends.min(), 120)
                Y = np.vstack([np.interp(grid, g["env_steps"].values, g[col].values)
                               for _, g in runs.sort_values("env_steps").groupby("exp_id")])
                mu = Y.mean(0)
                if Y.shape[0] > 1:
                    from scipy import stats as sps
                    h = sps.t.ppf(0.975, Y.shape[0] - 1) * Y.std(0, ddof=1) / np.sqrt(Y.shape[0])
                    st = _style(m)
                    ax.fill_between(grid / 1e6, mu - h, mu + h, color=C.lighten(st["color"], 0.78), lw=0, zorder=1)
                else:
                    h = np.full_like(mu, np.nan)
                st = _style(m)
                ax.plot(grid / 1e6, mu, color=st["color"], ls=st["linestyle"], label=st["label"], zorder=3)
                idx = np.linspace(0, grid.size - 1, 7).astype(int)
                ax.plot(grid[idx] / 1e6, mu[idx], ls="none", marker=st["marker"], color=st["color"],
                        markerfacecolor=st["markerfacecolor"], zorder=4)
                recs.append(pd.DataFrame(dict(method=m, metric=col, env_steps=grid, mean=mu, ci_lo=mu - h, ci_hi=mu + h,
                                              n_seeds=Y.shape[0], exp_ids=";".join(sorted(runs["exp_id"].unique())))))
            ax.set_xlabel("Environment steps (million)")
            ax.set_ylabel(ylab)
            _panel(ax, let)
        # legend proxies with markers
        from matplotlib.lines import Line2D
        hs = [Line2D([], [], **{k: v for k, v in _style(m).items()}) for m in methods]
        fig.legend(hs, [C.label(m) for m in methods], loc="upper center", bbox_to_anchor=(0.5, -0.02),
                   ncol=min(4, len(methods)), handlelength=2.6)
        self.fw.save(fig, "figS1_training_curves", pd.concat(recs, ignore_index=True))

    # ------------------------------------------------------------------ F3
    def f3_scaling(self):
        methods = self.present(F3_METHODS)
        rhos = sorted(self.grid["rho_tr"].unique())
        fig, axes = plt.subplots(2, len(rhos), figsize=(FULL_W, 4.3), sharex=True, constrained_layout=True, squeeze=False)
        recs = []
        for j, rho in enumerate(rhos):
            for i, (metric, ylab) in enumerate([("profit_per_charger", "Profit per charger (€/episode)"),
                                                 ("gap_pct", "Optimality gap (%)")]):
                ax = axes[i, j]
                _train_span(ax, self.sizes)
                d = self._byN(methods, metric, extra={"rho_tr": rho})
                recs.append(d)
                for k, m in enumerate(methods):
                    dm = d[d["method"] == m].sort_values("N")
                    if dm.empty:
                        continue
                    jitter = 1 + 0.025 * (k - len(methods) / 2)
                    st = _style(m)
                    ax.errorbar(dm["N"] * jitter, dm["mean"], yerr=self._err(dm), color=st["color"], ls=st["linestyle"],
                                marker=st["marker"], mfc=st["markerfacecolor"], mec=st["color"], label=st["label"],
                                elinewidth=0.7, zorder=3)
                _logN(ax, self.sizes)
                if i == 0:
                    ax.set_xlabel("")
                ax.set_ylabel(ylab)
                _panel(ax, "abcd"[i * len(rhos) + j])
                ax.set_title(f"({'abcd'[i * len(rhos) + j]}) {C.tightness(rho)}, " + r"$\rho_{\mathrm{tr}}$" + f" = {rho:g} kW per charger",
                             fontweight="normal", loc="left", fontsize=8)
        if self.sizes and min(self.sizes) <= 10 and max(self.sizes) >= 20:
            for ax in axes.ravel():
                ax.text(14.1, 0.985, "training\nsizes", transform=ax.get_xaxis_transform(), ha="center",
                        va="top", fontsize=6.3, color="#555555", zorder=2)
        _legend_below(fig, axes, ncol=4)
        self.fw.save(fig, "fig3_zero_shot_scaling", pd.concat(recs, ignore_index=True))

    # ------------------------------------------------------------------ F4
    def f4_constraints(self):
        methods = [m for m in C.method_order(self.ep["method"].unique())
                   if C.info(m).group not in ("conformal", "ablation", "oracle", "other")]
        rhos = sorted(self.grid["rho_tr"].unique())
        metrics = [("energy_not_served_kwh", "Energy not served (kWh/episode)"),
                   ("tr_violation_kwh", "Transformer violation (kWh/episode)")]
        fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 0.24 * len(methods) * len(rhos) * 0.6 + 1.2),
                                 sharey=True, constrained_layout=True)
        recs = []
        ypos = np.arange(len(methods))[::-1]
        bh = 0.8 / len(rhos)
        for ax, (metric, xl), let in zip(axes, metrics, "ab"):
            vmax = 0
            for r, rho in enumerate(rhos):
                for y, m in zip(ypos, methods):
                    g = self.grid[(self.grid["method"] == m) & (self.grid["rho_tr"] == rho)]
                    if g.empty:
                        continue
                    s = summarize(g, metric, self.sc, salt=("F4", m, rho))
                    recs.append(dict(method=m, label=C.label(m), rho_tr=rho, metric=metric, mean=s["mean"], std=s["std"],
                                     ci_lo=s["t_lo"], ci_hi=s["t_hi"], unit=s["unit"], exp_ids=";".join(s["exp_ids"])))
                    st = _style(m)
                    yy = y + (r - (len(rhos) - 1) / 2) * bh
                    face = st["color"] if r == 0 else C.lighten(st["color"], 0.55)
                    ax.barh(yy, s["mean"], height=bh * 0.92, color=face, edgecolor="#222222", lw=0.5,
                            hatch=None if r == 0 else "////", zorder=3)
                    if np.isfinite(s["t_hi"]) and s["mean"] > 0:
                        ax.errorbar(s["mean"], yy, xerr=[[max(0, s["mean"] - s["t_lo"])], [max(0, s["t_hi"] - s["mean"])]],
                                    color="#222222", elinewidth=0.6, capsize=1.2, zorder=4, ls="none")
                    if s["mean"] == 0:
                        ax.text(0.0, yy, " 0", va="center", ha="left", fontsize=6.5, zorder=5)
                    vmax = max(vmax, s["t_hi"] if np.isfinite(s["t_hi"]) else s["mean"])
            ax.set_xscale("symlog", linthresh=0.01, linscale=0.5)
            ax.set_xlim(0, max(1, vmax) * 2)
            ax.set_xlabel(xl + ", symlog scale")
            ax.grid(axis="y", visible=False)
            _panel(ax, let)
        axes[0].set_yticks(ypos)
        axes[0].set_yticklabels([C.label(m) for m in methods])
        from matplotlib.patches import Patch
        hs = [Patch(facecolor="#888888", edgecolor="#222222", lw=0.5, label=f"{C.tightness(rhos[0])} (" + r"$\rho_{\mathrm{tr}}$" + f" = {rhos[0]:g} kW)")]
        if len(rhos) > 1:
            hs.append(Patch(facecolor=C.lighten("#888888", 0.55), edgecolor="#222222", lw=0.5, hatch="////",
                            label=f"{C.tightness(rhos[1])} (" + r"$\rho_{\mathrm{tr}}$" + f" = {rhos[1]:g} kW)"))
        fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.0), ncol=2)
        self.fw.save(fig, "fig4_constraint_satisfaction", pd.DataFrame(recs))

    # ------------------------------------------------------------------ F5
    def f5_example_day(self, method: str = "attn_safe", test_cfg_id: str | None = None,
                       replay_id: int | None = None, preferred_cfg: str = "workplace_N20_rho4"):
        from load import load_steps
        st = load_steps(self.ds, method)
        if st is None:
            st = load_steps(self.ds)
            if st is None:
                print("[figures] F5 skipped: no steps.parquet/steps.csv", file=sys.stderr)
                return
            method = sorted(st["method"].unique(), key=lambda m: C.info(m).order)[0]
            st = st[st["method"] == method]
        m = method
        if "test_cfg_id" in st.columns:
            cfgs = list(st["test_cfg_id"].unique())
            if test_cfg_id is None:
                grid = set(self.ep.loc[self.ep["is_grid"], "test_cfg_id"])
                test_cfg_id = preferred_cfg if preferred_cfg in cfgs else sorted(cfgs, key=lambda c: (c not in grid, c))[0]
            st = st[st["test_cfg_id"] == test_cfg_id]
        if replay_id is None:
            replay_id = int(st["replay_id"].min())
        d = st[st["replay_id"] == replay_id]
        if "exp_id" in d.columns and d["exp_id"].nunique() > 1:      # several seeds: take the first run
            d = d[d["exp_id"] == sorted(d["exp_id"].unique())[0]]
        d = d.sort_values("t")
        exp_id = d["exp_id"].iat[0]
        cfg = test_cfg_id
        dt_h = 0.25
        if cfg is not None:
            dd = self.ep.loc[self.ep["test_cfg_id"] == cfg, "dt_min"]
            if len(dd):
                dt_h = float(dd.iat[0]) / 60
        x = d["t"].values * dt_h
        # optional overlay: the penalty-based counterpart on the same replay (same seed if available)
        other = None
        cp = {"attn_safe": "attn_penalty", "mlp_safe": "mlp_penalty"}.get(m)
        if cp:
            so = load_steps(self.ds, cp, cfg)
            if so is not None:
                so = so[so["replay_id"] == replay_id]
                if len(so):
                    e2 = sorted(so["exp_id"].unique())[0]
                    other = (cp, e2, so[so["exp_id"] == e2].sort_values("t"))
        fig, (ax, axp) = plt.subplots(2, 1, figsize=(FULL_W, 3.6), sharex=True, constrained_layout=True,
                                      gridspec_kw=dict(height_ratios=[2.2, 1]))
        ax.fill_between(x, d["sum_l"], d["sum_u"], step="post", color="#dddddd", lw=0, label=r"$[\Sigma_i\, l_i,\ \Sigma_i\, u_i]$ (per-EV bounds)")
        ax.step(x, d["C"], where="post", color="#000000", ls="--", lw=1.1, label=r"transformer limit $C(t)$")
        ax.step(x, -d["C_rev"], where="post", color="#000000", ls="-.", lw=1.1, label=r"$-C_{\mathrm{rev}}(t)$")
        sm = _style(m)
        ax.step(x, d["sum_p"], where="post", color=sm["color"], lw=1.5, label=r"$\Sigma_i\, p_i$, " + C.label(m))
        if other is not None:
            so = other[2]
            so_style = _style(other[0])
            ax.step(so["t"].values * dt_h, so["sum_p"], where="post", color=so_style["color"], ls=":", lw=1.3,
                    label=r"$\Sigma_i\, p_i$, " + C.label(other[0]))
        fb = d["fallback_code"].fillna(-1).astype(float).values >= 0        # -1 = no fallback
        if fb.any():
            yl = ax.get_ylim()
            ax.plot(x[fb], np.full(fb.sum(), yl[0]), ls="none", marker="|", ms=7, color="#000000",
                    label="fallback rule active")
        ax.axhline(0, color="#999999", lw=0.5, zorder=0)
        ax.set_ylabel("EV aggregate power (kW)")
        _panel(ax, "a")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6.8)
        axp.step(x, d["price_ch"], where="post", color="#000000", lw=1.1, label=r"$c^{\mathrm{ch}}(t)$")
        if not np.allclose(d["price_ch"], d["price_dis"]):
            axp.step(x, d["price_dis"], where="post", color="#666666", ls="--", lw=1.1, label=r"$c^{\mathrm{dis}}(t)$")
        axp.set_ylabel("Price (€/kWh)")
        axp.set_xlabel("Time since episode start (h)")
        _panel(axp, "b")
        axp.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6.8)
        data = d.copy()
        data["source_exp_id"] = exp_id
        if other is not None:
            o = other[2][["t", "sum_p"]].rename(columns={"sum_p": f"sum_p_{other[0]}"})
            data = data.merge(o, on="t", how="left")
            data["source_exp_id"] = f"{exp_id};{other[1]}"
        self.fw.save(fig, "fig5_example_day", data)

    # ------------------------------------------------------------------ F6
    def f6_ablation(self):
        variants = [(f"{a} + {c}", m) for (a, c), m in C.ABLATION_2X2.items() if m in set(self.ep["method"])]
        variants += [(C.label(m), m) for m in self.present(C.EXTRA_ABLATIONS)]
        if not variants:
            return
        tr_sizes = [n for n in self.sizes if n in C.TRAIN_SIZES]
        zs_sizes = [n for n in self.sizes if n >= 50]
        regimes = [(r"$N \in \{10, 20\}$ (training sizes)", tr_sizes), (r"$N \geq 50$ (zero-shot)", zs_sizes)]
        metrics = [("gap_pct", "Optimality gap (%)", "linear"), ("unserved_pct", "Unserved EVs (%)", "symlog"),
                   ("tr_violation_kwh", "Transformer violation (kWh/episode)", "symlog")]
        fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 0.3 * len(variants) + 1.3), sharey=True, constrained_layout=True)
        ypos = np.arange(len(variants))[::-1]
        bh = 0.38
        recs = []
        for ax, (metric, xl, scale), let in zip(axes, metrics, "abc"):
            vmax = 0
            for r, (rl, sizes) in enumerate(regimes):
                for y, (lab, m) in zip(ypos, variants):
                    g = self.grid[(self.grid["method"] == m) & (self.grid["N"].isin(sizes))]
                    if g.empty:
                        continue
                    s = summarize(g, metric, self.sc, salt=("F6", m, r))
                    recs.append(dict(variant=lab, method=m, regime=rl, metric=metric, mean=s["mean"], std=s["std"],
                                     ci_lo=s["t_lo"], ci_hi=s["t_hi"], exp_ids=";".join(s["exp_ids"])))
                    st = _style(m)
                    yy = y + (0.5 - r) * bh
                    ax.barh(yy, s["mean"], height=bh * 0.92, color=st["color"] if r == 0 else "white",
                            edgecolor=st["color"] if r else "#222222", hatch="////" if r else None, lw=0.7, zorder=3)
                    if np.isfinite(s["t_lo"]) and s["mean"] != 0:
                        ax.errorbar(s["mean"], yy, xerr=[[max(0, s["mean"] - s["t_lo"])], [max(0, s["t_hi"] - s["mean"])]],
                                    color="#222222", elinewidth=0.6, capsize=1.2, ls="none", zorder=4)
                    if s["mean"] == 0:
                        ax.text(0, yy, " 0", va="center", fontsize=6.5)
                    vmax = max(vmax, s["t_hi"] if np.isfinite(s["t_hi"]) else s["mean"])
            if scale == "symlog":
                ax.set_xscale("symlog", linthresh=0.01, linscale=0.5)
                ax.set_xlim(0, max(1, vmax) * 2)
            ax.set_xlabel(xl)
            ax.grid(axis="y", visible=False)
            _panel(ax, let)
        axes[0].set_yticks(ypos)
        axes[0].set_yticklabels([v[0] for v in variants])
        from matplotlib.patches import Patch
        hs = [Patch(facecolor="#666666", edgecolor="#222222", label=regimes[0][0]),
              Patch(facecolor="white", edgecolor="#666666", hatch="////", label=regimes[1][0])]
        fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2)
        self.fw.save(fig, "figS2_ablation", pd.DataFrame(recs))

    # ------------------------------------------------------------------ F7
    def f7_cost_of_guarantee(self):
        pairs = [(a, s, p) for a, s, p in C.GUARANTEE_PAIRS
                 if s in set(self.ep["method"]) and p in set(self.ep["method"])]
        if not pairs:
            return
        fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.4), constrained_layout=True)
        recs = []
        for k, (arch, ms, mp) in enumerate(pairs):
            st = _style(ms)
            rows = []
            for n in self.sizes:
                sub = self.grid[self.grid["N"] == n]
                r = diff_summary(sub, mp, ms, "profit_per_charger", self.sc, salt=("F7", n))
                rows.append(dict(arch=arch, N=n, metric="delta_profit_per_charger(penalty - safety layer)",
                                 mean=r["diff"], ci_lo=r["b_lo"], ci_hi=r["b_hi"], ci="bootstrap",
                                 exp_ids=";".join(r["exp_ids"])))
            d = pd.DataFrame(rows)
            recs.append(d)
            j = 1 + 0.03 * (k - 0.5)
            axes[0].errorbar(d["N"] * j, d["mean"], yerr=self._err(d), color=st["color"], marker=st["marker"],
                             mfc=st["color"], ls="-", label=f"{arch}: penalty − safety layer", elinewidth=0.7)
            v = self._byN([mp], "tr_violation_kwh")
            u = self._byN([mp], "energy_not_served_kwh")
            recs += [v, u]
            sp = _style(mp)
            axes[1].errorbar(v["N"] * j, v["mean"], yerr=self._err(v), color=sp["color"], marker=sp["marker"],
                             mfc="white", ls="--", label=f"{C.label(mp)}: transformer violation", elinewidth=0.7)
            axes[1].errorbar(u["N"] * j, u["mean"], yerr=self._err(u), color=sp["color"], marker="d",
                             mfc="white", ls=":", label=f"{C.label(mp)}: energy not served", elinewidth=0.7)
        for ax in axes:
            _train_span(ax, self.sizes)
            _logN(ax, self.sizes)
        axes[0].axhline(0, color="#000000", lw=0.6)
        axes[0].set_ylabel("Cost of the guarantee\n(€ per charger and episode)")
        axes[1].set_yscale("symlog", linthresh=0.01, linscale=0.5)
        axes[1].set_ylim(bottom=0)
        axes[1].set_ylabel("Penalty-based policy (kWh/episode)")
        _panel(axes[0], "a")
        _panel(axes[1], "b")
        _legend_below(fig, axes, ncol=2)
        self.fw.save(fig, "fig6_cost_of_guarantee", pd.concat(recs, ignore_index=True))

    # ------------------------------------------------------------------ F8
    def f8_conformal(self):
        conf = sorted([m for m in self.ep["method"].unique() if C.conf_alpha(m) is not None], key=C.conf_alpha)
        if not conf:
            return
        alphas = np.array([C.conf_alpha(m) for m in conf])
        has_cov = "coverage_frac" in self.ep and self.ep["coverage_frac"].notna().any()
        npan = 3 if has_cov else 2
        fig, axes = plt.subplots(1, npan, figsize=(FULL_W, 2.2), constrained_layout=True)
        recs = []
        st = _style("attn_safe")
        dp = [diff_summary(self.grid, m, "attn_safe", "profit_per_charger", self.sc, salt=("F8",)) for m in conf]
        d = pd.DataFrame(dict(alpha=alphas, mean=[r["diff"] for r in dp], ci_lo=[r["b_lo"] for r in dp],
                              ci_hi=[r["b_hi"] for r in dp], metric="delta_profit_per_charger(CP - known d)",
                              exp_ids=[";".join(r["exp_ids"]) for r in dp]))
        recs.append(d)
        axes[0].errorbar(d["alpha"], d["mean"], yerr=self._err(d), color=st["color"], marker="o", elinewidth=0.7)
        axes[0].axhline(0, color="#000000", lw=0.6)
        axes[0].set_ylabel("Δ profit vs known departure\n(€ per charger and episode)")
        rows = []
        for m, a in zip(conf, alphas):
            s = summarize(self.grid[self.grid["method"] == m], "unserved_pct", self.sc, salt=("F8u", m))
            rows.append(dict(alpha=a, mean=s["mean"], ci_lo=s["t_lo"], ci_hi=s["t_hi"], metric="unserved_pct",
                             exp_ids=";".join(s["exp_ids"])))
        u = pd.DataFrame(rows)
        recs.append(u)
        axes[1].errorbar(u["alpha"], u["mean"], yerr=self._err(u), color=st["color"], marker="o",
                         label="SA-Attn-CP(α)", elinewidth=0.7)
        ref = summarize(self.grid[self.grid["method"] == "attn_safe"], "unserved_pct", self.sc, salt=("F8ref",))
        if np.isfinite(ref["mean"]):
            axes[1].axhline(ref["mean"], color="#000000", ls="--", lw=0.9, label="SA-Attn, known departure step")
            recs.append(pd.DataFrame([dict(alpha=np.nan, mean=ref["mean"], ci_lo=ref["t_lo"], ci_hi=ref["t_hi"],
                                           metric="unserved_pct (known d)", exp_ids=";".join(ref["exp_ids"]))]))
        axes[1].set_ylabel("Unserved EVs (%)")

        if has_cov:
            rows = []
            for m, a in zip(conf, alphas):
                s = summarize(self.grid[self.grid["method"] == m], "coverage_frac", self.sc, salt=("F8c", m))
                rows.append(dict(alpha=a, mean=s["mean"], ci_lo=s["t_lo"], ci_hi=s["t_hi"], metric="coverage_frac",
                                 exp_ids=";".join(s["exp_ids"])))
            c = pd.DataFrame(rows)
            recs.append(c)
            aa = np.linspace(alphas.min() * 0.8, alphas.max() * 1.2, 50)
            axes[2].plot(aa, 1 - aa, color="#000000", ls="--", lw=0.9, label="nominal 1 − α")
            axes[2].errorbar(c["alpha"], c["mean"], yerr=self._err(c), color=st["color"], marker="o",
                             label="empirical (episode mean)", elinewidth=0.7)
            axes[2].set_ylabel("Coverage of departure bound")

        for ax, let in zip(axes, "abc"):
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(FixedLocator(alphas))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.set_xlabel("Miscoverage level α")
            _panel(ax, let)
        _legend_below(fig, axes, ncol=4)
        self.fw.save(fig, "figS3_conformal_alpha", pd.concat(recs, ignore_index=True))

    # ------------------------------------------------------------------ F9
    def f9_inference(self):
        methods = self.present(F9_METHODS)
        if not methods:
            return
        fig, ax = plt.subplots(figsize=(HALF_W, 3.1), constrained_layout=True)
        d = self._byN(methods, "inference_ms_per_step_mean")
        p95 = self._byN(methods, "inference_ms_per_step_p95")
        for m in methods:
            dm = d[d["method"] == m].sort_values("N")
            st = _style(m)
            ax.errorbar(dm["N"], dm["mean"], yerr=self._err(dm), color=st["color"], ls=st["linestyle"], marker=st["marker"],
                        mfc=st["markerfacecolor"], mec=st["color"], label=st["label"], elinewidth=0.7)
        _logN(ax, self.sizes)
        ax.set_yscale("log")
        ax.set_ylabel("Inference time per step (ms)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.5)
        self.fw.save(fig, "figS4_inference_time", pd.concat([d, p95], ignore_index=True))

    def all(self):
        for f in (self.f2_training, self.f3_scaling, self.f4_constraints, self.f5_example_day, self.f6_ablation,
                  self.f7_cost_of_guarantee, self.f8_conformal, self.f9_inference):
            f()
        return self.fw.written
