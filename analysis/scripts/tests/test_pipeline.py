"""End-to-end and unit tests of the analysis pipeline (synthetic data only).

Run:  python -m pytest analysis/scripts/tests -q
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import common as C  # noqa: E402
import load  # noqa: E402
import make_all  # noqa: E402
import stats  # noqa: E402
import synth  # noqa: E402


@pytest.fixture(scope="module")
def synth_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("t") / "synthetic_test"
    raw = synth.generate(root / "raw", replay_scale=0.1, small=True)
    sc = stats.StatConfig(B=200, B_iqm=50)
    ds, tabs = make_all.run(raw, root, sc)
    return root, raw, ds, tabs


# ----------------------------------------------------------------------------- end to end

def test_end_to_end_outputs(synth_run):
    root, raw, ds, tabs = synth_run
    for t in ["T1", "T2", "T3", "T3b", "T4", "T5", "T6", "T7", "TS1", "TS2", "TS3"]:
        for ext in (".tex", ".md", ".csv", "_values.csv"):
            assert (root / "tables" / f"{t}{ext}").exists(), f"{t}{ext} missing"
    for f in ["figS1_training_curves", "fig3_zero_shot_scaling", "fig4_constraint_satisfaction", "fig5_example_day",
              "figS2_ablation", "fig6_cost_of_guarantee", "figS3_conformal_alpha", "figS4_inference_time",
              "figS6_constraints_pooled"]:
        for ext in ("pdf", "png", "svg", "eps"):
            p = root / "figures" / f"{f}.{ext}"
            assert p.exists() and p.stat().st_size > 1000, f"{p} missing/empty"
        assert (root / "figures" / "data" / f"{f}.csv").exists()
    for p in ["tables/STAT_METHODS.md", "tables/provenance.json", "findings_numbers.md",
              "stats/tests_T1.csv", "stats/tests_T2.csv", "stats/tests_T3.csv", "stats/tests_T5.csv",
              "stats/summary_by_config.csv", "README_SYNTHETIC.txt"]:
        assert (root / p).exists(), p


def test_synthetic_watermarks(synth_run):
    root, *_ = synth_run
    assert C.SYNTH_MARK in (root / "findings_numbers.md").read_text()
    for t in ["T1", "T5"]:
        assert "SYNTHETIC TEST DATA" in (root / "tables" / f"{t}.tex").read_text()
        assert C.SYNTH_MARK in (root / "tables" / f"{t}.md").read_text()
        assert C.SYNTH_MARK in (root / "tables" / f"{t}.csv").read_text(encoding="utf-8-sig")
    assert "SYNTHETIC TEST DATA" in (root / "figures" / "fig3_zero_shot_scaling.svg").read_text()
    assert json.loads((root / "tables" / "provenance.json").read_text())["_synthetic"] is True


def test_guard_refuses_synthetic_output_outside(synth_run, tmp_path):
    _, raw, _, _ = synth_run
    with pytest.raises(SystemExit):
        make_all.run(raw, tmp_path / "analysis", stats.StatConfig(B=50, B_iqm=10))


def test_smoke_rows_dropped(synth_run):
    _, _, ds, _ = synth_run
    assert not ds.episodes["smoke"].any()
    assert not ds.episodes["exp_id"].str.contains("smoke").any()
    assert any("smoke" in r for r in ds.report)


def test_smoke_rows_inside_file_dropped(tmp_path):
    raw = synth.generate(tmp_path / "synthetic_test" / "raw", replay_scale=0.05, small=True, with_smoke=False)
    f = next(p for p in raw.glob("afap__*/episodes.csv"))
    df = pd.read_csv(f)
    df.loc[:2, "smoke"] = True
    df.to_csv(f, index=False)
    ds = load.load_raw(raw)
    afap = ds.episodes[ds.episodes["method"] == "afap"]
    assert len(afap) == len(df) - 3


# ----------------------------------------------------------------------------- schema

def test_schema_missing_column_fails_loudly(tmp_path):
    raw = synth.generate(tmp_path / "synthetic_test" / "raw", replay_scale=0.05, small=True)
    f = next(p for p in raw.glob("attn_safe__trainmix__s0__*/episodes.csv"))
    pd.read_csv(f).drop(columns=["tr_violation_kwh", "fb1_steps"]).to_csv(f, index=False)
    with pytest.raises(load.SchemaError) as e:
        load.load_raw(raw)
    msg = str(e.value)
    assert "tr_violation_kwh" in msg and "fb1_steps" in msg and str(f) in msg


def test_schema_missing_train_column_fails(tmp_path):
    raw = synth.generate(tmp_path / "synthetic_test" / "raw", replay_scale=0.05, small=True)
    f = next(p for p in raw.glob("attn_safe__trainmix__s0__*/train_log.csv"))
    pd.read_csv(f).drop(columns=["env_steps"]).to_csv(f, index=False)
    with pytest.raises(load.SchemaError, match="env_steps"):
        load.load_raw(raw)


def test_schema_non_numeric_value_fails(tmp_path):
    raw = synth.generate(tmp_path / "synthetic_test" / "raw", replay_scale=0.05, small=True)
    f = next(p for p in raw.glob("llf__*/episodes.csv"))
    df = pd.read_csv(f)
    df["profit_eur"] = df["profit_eur"].astype(object)
    df.loc[0, "profit_eur"] = "abc"
    df.to_csv(f, index=False)
    with pytest.raises(load.SchemaError, match="profit_eur"):
        load.load_raw(raw)


def test_missing_oracle_fails_unless_allowed(tmp_path):
    raw = synth.generate(tmp_path / "synthetic_test" / "raw", replay_scale=0.05, small=True)
    for d in raw.glob("oracle__*"):
        shutil.rmtree(d)
    with pytest.raises(load.SchemaError, match="oracle"):
        load.load_raw(raw)
    ds = load.load_raw(raw, require_oracle=False)
    assert ds.episodes["gap_pct"].isna().all()


# ----------------------------------------------------------------------------- statistics

def test_holm_known_example():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone max 0.06
    np.testing.assert_allclose(stats.holm(p), [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(stats.holm([0.5, 0.9]), [1.0, 1.0])          # capped at 1
    out = stats.holm([0.01, np.nan, 0.02])
    assert np.isnan(out[1]) and np.allclose(out[[0, 2]], [0.02, 0.02])
    assert np.isnan(stats.holm([np.nan])).all()


def test_holm_matches_bruteforce():
    rng = np.random.default_rng(1)
    for _ in range(200):
        p = rng.uniform(0, 0.2, rng.integers(1, 9))
        m = p.size
        o = np.argsort(p)
        ref = np.empty(m)
        for k, i in enumerate(o):
            ref[i] = min(1, max((m - j) * p[o[j]] for j in range(k + 1)))
        np.testing.assert_allclose(stats.holm(p), ref)


def test_gap_computation():
    g = load.gap_pct([90, -60, 5, 1], [100, -50, 0, np.nan])
    np.testing.assert_allclose(g[:2], [10.0, 20.0])
    assert np.isnan(g[2]) and np.isnan(g[3])


def test_gap_join(synth_run):
    _, _, ds, _ = synth_run
    ep, orc = ds.episodes, ds.oracle
    row = ep.sample(1, random_state=3).iloc[0]
    o = orc[(orc["test_cfg_id"] == row["test_cfg_id"]) & (orc["replay_id"] == row["replay_id"])]["profit_eur"].iat[0]
    orow = orc[(orc["test_cfg_id"] == row["test_cfg_id"]) & (orc["replay_id"] == row["replay_id"])]
    oc, op = orow[load.ORACLE_CEIL].iat[0], orow[load.ORACLE_PEN].iat[0]
    assert row["oracle_profit_eur"] == pytest.approx(o)
    assert row["oracle_ceil_eur"] == pytest.approx(oc) and row["oracle_pen_eur"] == pytest.approx(op)

    def g(ref, val):
        return np.nan if abs(ref) < load.GAP_EPS else 100 * (ref - val) / abs(ref)
    # DD32: main gap of J^pen w.r.t. the soft oracle; hard gap w.r.t. J_LP,ceil; supplementary w.r.t. J_LP
    np.testing.assert_allclose(row["gap_pct"], g(op, row[load.PEN_COL]))
    np.testing.assert_allclose(row["gap_hard_pct"], g(oc, row["profit_eur"]))
    np.testing.assert_allclose(row["gap_lp_pct"], g(o, row["profit_eur"]))
    assert ep["oracle_profit_eur"].notna().all() and ep["oracle_pen_eur"].notna().all()


def test_summarize_seed_level():
    rows = []
    for s, mu in enumerate([1.0, 2.0, 4.0]):
        for r in range(4):
            rows.append(dict(test_cfg_id="c", seed=s, replay_id=r, exp_id=f"e{s}", x=mu + (r - 1.5) * 0.1))
    df = pd.DataFrame(rows)
    res = stats.summarize(df, "x", stats.StatConfig(B=500))
    assert res["unit"] == "seed" and res["n_seeds"] == 3
    assert res["mean"] == pytest.approx(7 / 3)
    assert res["std"] == pytest.approx(np.std([1, 2, 4], ddof=1))
    lo, hi = sps.t.interval(0.95, 2, loc=7 / 3, scale=np.std([1, 2, 4], ddof=1) / np.sqrt(3))
    assert res["t_lo"] == pytest.approx(lo) and res["t_hi"] == pytest.approx(hi)
    assert 1.0 <= res["b_lo"] <= res["b_hi"] <= 4.0


def test_iqm():
    x = np.arange(1, 9, dtype=float)       # trims 2 lowest and 2 highest -> mean(3,4,5,6)
    assert stats.iqm(x) == pytest.approx(4.5)


def test_wilcoxon_all_zero_and_effect_size():
    ep = pd.DataFrame(dict(method=["a"] * 5 + ["b"] * 5, test_cfg_id="c", replay_id=list(range(5)) * 2,
                           seed=0, exp_id=["ea"] * 5 + ["eb"] * 5, y=[0.0] * 10))
    r = stats.paired_test(ep, "a", "b", "c", "y")
    assert r["p_wilcoxon"] == 1.0 and "zero" in r["note"]
    assert stats.rank_biserial([1, 2, 3]) == 1.0
    assert stats.rank_biserial([-1, -2]) == -1.0


def test_paired_tests_holm_families(synth_run):
    _, _, ds, _ = synth_run
    t = stats.paired_tests(ds.episodes, "attn_safe", ["afap"], ["profit_eur"])
    g = t[(t["is_grid"]) & (t["profile"] == "public") & (t["rho_tr"] == 4.0)]
    np.testing.assert_allclose(g["p_holm_N"].values, stats.holm(g["p_wilcoxon"].values))
    assert (t["p_holm_all"] >= t["p_wilcoxon"] - 1e-15).all()


# ----------------------------------------------------------------------------- provenance

def test_provenance_completeness(synth_run):
    root, _, ds, tabs = synth_run
    prov = json.loads((root / "tables" / "provenance.json").read_text())
    known = set(ds.episodes["exp_id"]) | {f"oracle__{c}" for c in ds.oracle["test_cfg_id"]} | set(ds.train["exp_id"])
    n_cells = 0
    for name, t in tabs.items():
        for r in t.rows:
            for k, h in t.columns:
                c = r["cells"].get(k)
                if c is None or c.empty:
                    continue
                n_cells += 1
                key = f"{name}|{(r['group'] + ' / ') if r['group'] else ''}{r['label']}|{h.replace(chr(10), ' ')}"
                assert key in prov, f"no provenance for {key}"
                assert prov[key], f"empty provenance for {key}"
                assert set(prov[key]) <= known, f"unknown exp_ids in {key}"
                if "gap" in h.lower() and "Gap" in h:
                    assert any(e.startswith("oracle__") for e in prov[key]), f"gap cell {key} lacks oracle ids"
    assert n_cells == len([k for k in prov if not k.startswith("_")])


def test_t1_cell_matches_direct_computation(synth_run):
    _, _, ds, tabs = synth_run
    ep = ds.episodes
    sub = ep[(ep["method"] == "attn_safe") & ep["is_grid"] & (ep["N"] == 10)]
    direct = sub.groupby(["test_cfg_id", "seed"])["profit_eur"].mean().groupby("test_cfg_id").mean().mean()
    row = next(r for r in tabs["T1"].rows if r["group"] == "N = 10" and r["label"] == C.label("attn_safe"))
    assert row["cells"]["profit_eur"].value == pytest.approx(direct)


def test_optional_inputs_missing(tmp_path):
    """No steps files, no conformal runs, no train logs, no eMPC: pipeline still completes."""
    root = tmp_path / "synthetic_test"
    raw = synth.generate(root / "raw", replay_scale=0.05, small=True)
    for d in raw.iterdir():
        if d.is_dir() and (d.name.startswith("attn_safe_conf_") or d.name.startswith("empc")):
            shutil.rmtree(d)
    for f in list(raw.glob("*/steps.*")) + list(raw.glob("*/train_log.csv")):
        f.unlink()
    ds, tabs = make_all.run(raw, root, stats.StatConfig(B=50, B_iqm=10))
    assert "T7" not in tabs and (root / "tables" / "T1.tex").exists()
    assert not (root / "figures" / "fig5_example_day.pdf").exists()
    assert not (root / "figures" / "figS1_training_curves.pdf").exists()
    assert (root / "figures" / "fig3_zero_shot_scaling.pdf").exists()


# ----------------------------------------------------------------------------- names, smoke

def test_names_registry_covers_a3_strings():
    import names
    a3 = ["attn_safe", "attn_penalty", "mlp_safe", "mlp_penalty", "attn_safe_noadm", "attn_safe_conf_a0.01",
          "attn_safe_conf_a0.05", "attn_safe_conf_a0.1", "attn_safe_conf_a0.2", "attn_safe_conf_point",
          "attn_safe_dt5", "afap", "alap", "rr", "llf", "afap_safe", "alap_safe", "rr_safe", "llf_safe", "empc",
          "empc_safe", "empc_highs", "empc_highs_safe", "evgnn", "evgnn_safe", "oracle", "attn_safe_beta0",
          "attn_safe_curriculum", "attn_safe_small", "attn_penalty_dense", "attn_penalty_w1", "attn_penalty_w100",
          # round 3 (DD33, DD35, DD36)
          "attn_proj_nolb", "attn_lagrangian", "deepsets_safe", "attn_penalty_safedeploy", "attn_penalty_noadm",
          "sllf", "sllf_safe", "price_llf", "price_llf_safe", "empc_highs_24h", "empc_highs_24h_safe",
          "attn_safe_trainN50", "evgnn_id", "evgnn_safe_id"]
    for m in a3:
        assert names.info(m).group != "other", m
    labels = [names.label(m) for m in a3]
    assert len(set(labels)) == len(labels)                       # one display name per method
    for lab in labels:                                           # terminology.md forbidden terms
        assert not any(w in lab.lower() for w in ("shield", "projection layer", "safety filter", "oracle"))
    assert names.conf_alpha("attn_safe_conf_a0.05") == 0.05


def _smoke_fixture(tmp_path):
    """Smoke-like raw tree: tiny partial grid (N in {3, 5}, rho 4 only, _T16 ids), one seed, missing methods
    per configuration, a sensitivity configuration, all rows smoke = true, steps.csv with -1 fallback codes."""
    raw = synth.generate(tmp_path / "raw", replay_scale=0.03, n_seeds=1, small=True, with_smoke=False)
    keep_cfg = {"workplace_N5_rho4", "public_N5_rho4", "public_N20_rho4_soc0.8"}
    for f in list(raw.glob("*/episodes.csv")):
        df = pd.read_csv(f)
        df = df[df["test_cfg_id"].isin(keep_cfg)].copy()
        if df.empty:
            shutil.rmtree(f.parent)
            continue
        df["test_cfg_id"] = df["test_cfg_id"] + "_T16"
        df["smoke"] = True
        if f.parent.name.startswith("oracle__"):
            df["exp_id"] = "oracle__" + df["test_cfg_id"]
        if f.parent.name.startswith("mlp_penalty"):          # partial: one method lacks a configuration
            df = df[~df["test_cfg_id"].str.startswith("public_N5")]
        df.to_csv(f, index=False)
        meta = json.loads((f.parent / "meta.json").read_text())
        meta["smoke"] = True
        (f.parent / "meta.json").write_text(json.dumps(meta))
        if f.parent.name.startswith("oracle__"):
            f.parent.rename(f.parent.with_name(f.parent.name + "_T16"))
    for st in raw.glob("*/steps.*"):
        st.unlink()
    (raw / "SYNTHETIC_DATA").unlink()          # behave like real smoke output (no synthetic sentinel)
    for m in raw.glob("*/meta.json"):
        meta = json.loads(m.read_text())
        meta.pop("synthetic", None)
        m.write_text(json.dumps(meta))
    return raw


def test_smoke_like_partial_grid(tmp_path):
    raw = _smoke_fixture(tmp_path)
    with pytest.raises(load.SchemaError, match="smoke"):          # default: smoke rows are dropped
        load.load_raw(raw)
    out = tmp_path / "smoke_analysis"
    ds, tabs = make_all.run(raw, out, stats.StatConfig(B=50, B_iqm=10), include_smoke=True)
    assert ds.smoke and set(ds.episodes["cfg_family"]) == {"grid", "sensitivity"}
    for t in ["T1", "T2", "T3", "T4", "T5", "T6", "TS1"]:
        assert (out / "tables" / f"{t}.tex").exists()
    assert C.SMOKE_MARK in (out / "tables" / "T2.md").read_text()
    assert C.SMOKE_MARK in (out / "findings_numbers.md").read_text()
    assert (out / "figures" / "fig3_zero_shot_scaling.pdf").exists()
    assert (out / "README_SMOKE.txt").exists()
    C.set_mark(C.SYNTH_MARK)


def test_smoke_never_written_into_analysis(tmp_path):
    raw = _smoke_fixture(tmp_path)
    for bad in (make_all.REPO, make_all.REPO / "figures", tmp_path / "analysis"):
        with pytest.raises(SystemExit):
            make_all.run(raw, bad, stats.StatConfig(B=20, B_iqm=5), include_smoke=True)
    assert not any((make_all.REPO / "figures").glob("*.pdf"))
    assert not any((make_all.REPO / "tables").glob("T*.tex"))


def test_oracle_without_ceil_falls_back(tmp_path):
    raw = synth.generate(tmp_path / "synthetic_test" / "raw", replay_scale=0.05, small=True)
    for f in raw.glob("oracle__*/episodes.csv"):
        pd.read_csv(f).drop(columns=[load.ORACLE_CEIL]).to_csv(f, index=False)
    ds = load.load_raw(raw)
    np.testing.assert_allclose(ds.episodes["gap_hard_pct"], ds.episodes["gap_lp_pct"])
    assert any("falls back" in r for r in ds.report)


def test_conceptual_figures_copied(synth_run):
    root, *_ = synth_run
    for n in ("fig2_concepts", "figS5_pipeline"):
        assert (root / "figures" / f"{n}.pdf").exists(), n


# ----------------------------------------------------------------------------- round 3 (DD32, DD38, M2-M8)

def test_gap_eps_one_euro_and_count(tmp_path):
    raw = synth.generate(tmp_path / "synthetic_test" / "raw", replay_scale=0.05, small=True)
    f = next(p for p in raw.glob("oracle__workplace_N5_rho4/episodes.csv"))
    df = pd.read_csv(f)
    df.loc[0, load.ORACLE_PEN] = 0.5           # |J*| < 1 EUR -> excluded (M2)
    df.to_csv(f, index=False)
    ds = load.load_raw(raw)
    ep = ds.episodes
    sel = (ep["test_cfg_id"] == "workplace_N5_rho4") & (ep["replay_id"] == df.loc[0, "replay_id"]) & \
        ~ep["method"].str.startswith("attn_safe_conf_")
    assert load.GAP_EPS == 1.0
    assert ep.loc[sel, "gap_pct"].isna().all()
    assert ds.gap_excluded["gap_pct"] >= int(sel.sum())


def test_conformal_has_main_gap_dd43(synth_run):
    _, _, ds, tabs = synth_run
    conf = ds.episodes["method"].str.startswith("attn_safe_conf_")
    assert conf.any() and ds.episodes.loc[conf, "gap_pct"].notna().any()      # DD43(1) supersedes DD42(2)
    assert "gap_pct" in [k for k, _ in tabs["T7"].columns]


def test_evgnn_in_distribution_split(synth_run):
    _, _, ds, _ = synth_run
    ep = ds.episodes
    assert {"evgnn_id", "evgnn_safe_id"} <= set(ep["method"])
    assert (ep.loc[ep["method"] == "evgnn_id", "N"] == 20).all()
    assert not ep.loc[ep["method"] == "evgnn", "in_distribution"].any()


def test_seed_comparison_paired_t_and_poi():
    rows = []
    a_means, b_means = [3.0, 4.0, 5.0, 6.5], [1.0, 3.5, 2.0, 2.0]
    for s_, (ma, mb) in enumerate(zip(a_means, b_means)):
        for r in range(6):
            off = (r - 2.5) * 0.01
            rows.append(dict(method="a", seed=s_, test_cfg_id="c", replay_id=r, exp_id=f"a{s_}", N=10, y=ma + off))
            rows.append(dict(method="b", seed=s_, test_cfg_id="c", replay_id=r, exp_id=f"b{s_}", N=10, y=mb + off))
    ep = pd.DataFrame(rows)
    out = stats.seed_comparisons(ep, "a", ["b"], ["y"], ["N"], stats.StatConfig(B=500, B_iqm=200), family="X")
    r = out.iloc[0]
    ref = sps.ttest_rel(a_means, b_means)
    assert r["test"].startswith("paired t") and r["p_t"] == pytest.approx(ref.pvalue)
    assert r["diff"] == pytest.approx(np.mean(a_means) - np.mean(b_means))
    # PoI over all 16 seed pairs: a > b in 15 (4.0 vs ... all bigger except 3.0 vs 3.5)
    assert r["poi"] == pytest.approx(15 / 16)
    assert r["b_lo"] <= r["diff"] <= r["b_hi"]
    assert r["p_t_holm"] == pytest.approx(r["p_t"])            # single comparison in the family


def test_seed_comparison_vs_deterministic():
    rows = []
    for s_, ma in enumerate([2.0, 2.5, 3.0, 2.2, 2.8]):
        for r in range(4):
            rows.append(dict(method="a", seed=s_, test_cfg_id="c", replay_id=r, exp_id=f"a{s_}", N=5, y=ma))
    for r in range(4):
        rows.append(dict(method="h", seed=0, test_cfg_id="c", replay_id=r, exp_id="h0", N=5, y=1.0))
    out = stats.seed_comparisons(pd.DataFrame(rows), "a", ["h"], ["y"], ["N"], stats.StatConfig(B=200, B_iqm=50))
    r = out.iloc[0]
    assert r["test"].startswith("one-sample")
    assert r["p_t"] == pytest.approx(sps.ttest_1samp(np.array([2.0, 2.5, 3.0, 2.2, 2.8]) - 1.0, 0).pvalue)


def test_t3b_fallback_mapping_and_requests(synth_run):
    _, _, _, tabs = synth_run
    heads = [h.replace("\n", " ") for _, h in tabs["T3b"].columns]
    for case, lab in [("(i)", "FB-1*"), ("(ii)", "FB-2"), ("(iii)", "FB-1"), ("(iv)", "FB-0")]:
        assert any(h.startswith(f"{case} {lab}") for h in heads), heads
    h3 = [h.replace("\n", " ") for _, h in tabs["T3"].columns]         # moved to T3 (pooled) in DD43 revision
    assert any("deficit bound" in h for h in h3)
    assert any("Capped" in h for h in h3) and any("Admission" in h for h in h3)


def test_t5_units_per_ev_and_percent(synth_run):
    root, _, _, tabs = synth_run
    heads = " ".join(h for _, h in tabs["T5"].columns)
    assert "per EV" in heads and "|J^LP,ceil|" in heads and "per charger" not in heads
    assert heads.count("J^pen") == 2 and heads.count("profit") >= 2              # DD43(7): primary + secondary
    t5 = pd.read_csv(root / "stats" / "tests_T5.csv")
    assert {"penalised_profit_eur", "profit_eur"} <= set(t5["metric"])
    tex = (root / "tables" / "T5.tex").read_text()
    assert r"$J^{\mathrm{LP,ceil}}$" in tex and r"$J^{\mathrm{pen}}$" in tex


def test_sensitivity_and_literature_tables(synth_run):
    _, _, _, tabs = synth_run
    groups = {r["group"] for r in tabs["TS2"].rows}
    for key in ("desired SoC 0.8", "\u0394t = 5", "throughput cost", "discharge price factor",
                "penalty weight", "stress"):
        assert any(key in g for g in groups), (key, groups)
    assert any(r["label"] == names_label("attn_safe") for r in tabs["TS3"].rows)
    t2 = tabs["T2"]
    assert any(not r["cells"].get("transfer", tables_Cell()).empty for r in t2.rows if r["label"] == names_label("attn_safe"))


def names_label(m):
    import names
    return names.label(m)


def tables_Cell():
    import tables
    return tables.Cell()


# ----------------------------------------------------------------------------- DD43

def test_mlp_only_up_to_n20(synth_run):
    _, _, ds, _ = synth_run
    ep = ds.episodes
    assert (ep.loc[ep["method"].isin(["mlp_safe", "mlp_penalty"]), "N"] <= 20).all()


def test_e4_and_binding_in_t3_t3b_fig4(synth_run):
    root, _, ds, tabs = synth_run
    assert "e4_pct" in [k for k, _ in tabs["T3"].columns]
    keys = [k for k, _ in tabs["T3b"].columns]
    for k in ("e4_pct", "upper_bind_pct", "lower_bind_pct", "lb_active_pct", "fb1star_steps", "fb0_steps"):
        assert k in keys, k
    assert {r["group"] for r in tabs["T3b"].rows} >= {"N = 5", "N = 50"}      # per N
    d = pd.read_csv(root / "figures" / "data" / "fig4_constraint_satisfaction.csv")
    assert "e4_pct" in set(d["metric"])
    t3 = pd.read_csv(root / "stats" / "tests_T3.csv")
    assert {"share_evs_stay_feasible", "tr_violation_kwh"} <= set(t3["metric"])
    ep = ds.episodes
    np.testing.assert_allclose(ep["upper_bind_pct"], 100 * ep["agg_upper_active_steps"] / ep["n_steps"])


def test_ratio_of_sums_difference():
    rows = []
    # one configuration, 2 replays, 2 seeds each; n_evs per replay 4 and 6
    vals_a = {0: [10.0, 20.0], 1: [12.0, 22.0]}      # seed -> replay values
    vals_b = {0: [8.0, 14.0], 1: [6.0, 16.0]}
    for s_ in (0, 1):
        for r, nev in enumerate((4, 6)):
            rows.append(dict(method="a", seed=s_, test_cfg_id="c", replay_id=r, exp_id=f"a{s_}", y=vals_a[s_][r], n_evs=nev))
            rows.append(dict(method="b", seed=s_, test_cfg_id="c", replay_id=r, exp_id=f"b{s_}", y=vals_b[s_][r], n_evs=nev))
    r = stats.ratio_diff(pd.DataFrame(rows), "a", "b", "y", "n_evs", stats.StatConfig(B=300))
    # seed-averaged: a = (11, 21), b = (7, 15) -> sum diff = 10, sum n_evs = 10
    assert r["diff"] == pytest.approx(1.0)
    assert r["b_lo"] <= 1.0 + 1e-12 and r["b_hi"] >= 1.0 - 1e-12


def test_stat_methods_matches_dd43(synth_run):
    root, *_ = synth_run
    txt = (root / "tables" / "STAT_METHODS.md").read_text()
    for phrase in ("Primary test: paired t-test on seed means", "Secondary test: Wilcoxon signed-rank test",
                   "Holm correction per table family", "ratio of", "no adaptive seed rule", "E4 (primary endpoint)",
                   "N ≤ 20"):
        assert phrase in txt, phrase
