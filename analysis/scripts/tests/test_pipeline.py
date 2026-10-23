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
    for t in ["T1", "T2", "T3", "T3b", "T3c", "T3d", "T4", "T5", "T6", "T7", "TS1", "TS2", "TS3"]:
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
    assert ds.episodes["gap_abs"].isna().all()


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
    orow = orc[(orc["test_cfg_id"] == row["test_cfg_id"]) & (orc["replay_id"] == row["replay_id"])]
    o, oc, op = orow["profit_eur"].iat[0], orow[load.ORACLE_CEIL].iat[0], orow[load.ORACLE_PEN].iat[0]
    assert row["oracle_profit_eur"] == pytest.approx(o)
    assert row["oracle_ceil_eur"] == pytest.approx(oc) and row["oracle_pen_eur"] == pytest.approx(op)
    # DD46: J^pen recomputed from the shared shortfall and the overload energy
    jp = row["profit_eur"] - load.W_PEN * (row["energy_not_served_shared_kwh"] + row["tr_violation_kwh"])
    assert row[load.PEN_COL] == pytest.approx(jp)
    # DD47: absolute gap in EUR per charger and episode
    assert row["gap_abs"] == pytest.approx((op - jp) / row["N"])
    assert row["gap_hard_abs"] == pytest.approx((oc - row["profit_eur"]) / row["N"])
    assert ep["oracle_profit_eur"].notna().all() and ep["oracle_pen_eur"].notna().all()
    assert "gap_pct" not in ep.columns


def test_jpen_ignores_file_column_and_uses_shared_shortfall(tmp_path):
    raw = synth.generate(tmp_path / "synthetic_test" / "raw", replay_scale=0.05, small=True)
    f = next(p for p in raw.glob("afap__*/episodes.csv"))
    df = pd.read_csv(f)
    df["penalised_profit_eur"] = 1e9                     # must be ignored
    df["energy_not_served_kwh"] = 0.0                   # own shortfall; the shared one must be used
    df.to_csv(f, index=False)
    ds = load.load_raw(raw)
    a = ds.episodes[ds.episodes["method"] == "afap"]
    exp = a["profit_eur"] - load.W_PEN * (a["energy_not_served_shared_kwh"] + a["tr_violation_kwh"])
    np.testing.assert_allclose(a[load.PEN_COL], exp)
    np.testing.assert_allclose(a["energy_not_served_kwh"], a["energy_not_served_shared_kwh"])


def test_summarize_seed_level():
    rows = []
    for s_, mu in enumerate([1.0, 2.0, 4.0]):
        for r in range(4):
            rows.append(dict(test_cfg_id="c", seed=s_, replay_id=r, exp_id=f"e{s_}", x=mu + (r - 1.5) * 0.1))
    df = pd.DataFrame(rows)
    res = stats.summarize(df, "x", stats.StatConfig(B=500))
    assert res["unit"] == "seeds" and res["n_seeds"] == 3
    assert res["mean"] == pytest.approx(7 / 3)
    assert res["seed_min"] == pytest.approx(1.0) and res["seed_max"] == pytest.approx(4.0)
    assert res["seeds"] == pytest.approx({0: 1.0, 1: 2.0, 2: 4.0})
    assert "t_lo" not in res and np.isnan(res["std"])          # DD48: no seed-level t interval
    # replay bootstrap of the seed-averaged replay values 7/3 + (r - 1.5) * 0.1
    assert 7 / 3 - 0.15 - 1e-12 <= res["b_lo"] <= res["b_hi"] <= 7 / 3 + 0.15 + 1e-12


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
                if "Optimality gap" in h or "Transfer gap" in h:
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
    # no SMOKE-watermarked file may exist anywhere in analysis/ (real figures may exist)
    marks = (C.SMOKE_MARK, "SMOKE TEST")
    outputs = [make_all.REPO / "findings_numbers.md", make_all.REPO / "load_report.txt"]
    for sub in ("figures", "tables", "stats"):
        outputs += list((make_all.REPO / sub).rglob("*"))
    for f in outputs:
        if not f.is_file() or f.suffix.lower() not in (".md", ".tex", ".csv", ".svg", ".txt", ".json", ".eps"):
            continue
        txt = f.read_text(encoding="utf-8", errors="ignore")
        assert not any(m in txt for m in marks), f"smoke-watermarked file in analysis/: {f}"
    assert not (make_all.REPO / "README_SMOKE.txt").exists()


def test_oracle_without_ceil_falls_back(tmp_path):
    raw = synth.generate(tmp_path / "synthetic_test" / "raw", replay_scale=0.05, small=True)
    for f in raw.glob("oracle__*/episodes.csv"):
        pd.read_csv(f).drop(columns=[load.ORACLE_CEIL]).to_csv(f, index=False)
    ds = load.load_raw(raw)
    ep = ds.episodes
    np.testing.assert_allclose(ep["gap_hard_abs"], (ep["oracle_profit_eur"] - ep["profit_eur"]) / ep["N"])
    assert any("falls back" in r for r in ds.report)


def test_conceptual_figures_copied(synth_run):
    root, *_ = synth_run
    for n in ("fig2_concepts", "figS5_pipeline"):
        assert (root / "figures" / f"{n}.pdf").exists(), n


# ----------------------------------------------------------------------------- round 3 (DD32, DD38, M2-M8)

def test_soft_oracle_bound_check(tmp_path):
    root = tmp_path / "synthetic_test"
    raw = synth.generate(root / "raw", replay_scale=0.05, small=True)
    ds = load.load_raw(raw)
    assert ds.bound_violations == 0 and ds.bound_checked == ds.episodes["oracle_pen_eur"].notna().sum()
    f = next(p for p in raw.glob("oracle__workplace_N5_rho4/episodes.csv"))
    df = pd.read_csv(f)
    df.loc[0, load.ORACLE_PEN] = -1e6           # optimum below every method on this replay
    df.to_csv(f, index=False)
    ds = load.load_raw(raw)
    ep = ds.episodes
    n_exp = int(((ep["test_cfg_id"] == "workplace_N5_rho4") & (ep["replay_id"] == df.loc[0, "replay_id"])).sum())
    assert ds.bound_violations == n_exp > 0
    with pytest.raises(SystemExit, match="bound"):
        make_all.run(raw, root, stats.StatConfig(B=20, B_iqm=5))


def test_conformal_has_main_gap_dd43(synth_run):
    _, _, ds, tabs = synth_run
    conf = ds.episodes["method"].str.startswith("attn_safe_conf_")
    assert conf.any() and ds.episodes.loc[conf, "gap_abs"].notna().any()
    keys = [k for k, _ in tabs["T7"].columns]
    for k in ("gap_abs", "coverage_frac", "energy_not_served_kwh", "energy_charged_kwh"):
        assert k in keys, (k, keys)


def test_evgnn_in_distribution_split(synth_run):
    _, _, ds, _ = synth_run
    ep = ds.episodes
    assert {"evgnn_id", "evgnn_safe_id"} <= set(ep["method"])
    assert (ep.loc[ep["method"] == "evgnn_id", "N"] == 20).all()
    assert not ep.loc[ep["method"] == "evgnn", "in_distribution"].any()


def test_seed_comparison_wilcoxon_replay_level():
    rng = np.random.default_rng(5)
    rows = []
    ya = {(s_, c, r): 3.0 + s_ + rng.normal() for s_ in (0, 1) for c in ("c1", "c2") for r in range(12)}
    yb = {(s_, c, r): 2.5 + rng.normal() for s_ in (0, 1) for c in ("c1", "c2") for r in range(12)}
    for (s_, c, r), v in ya.items():
        rows.append(dict(method="a", seed=s_, test_cfg_id=c, replay_id=r, exp_id=f"a{s_}{c}", N=10, y=v))
    for (s_, c, r), v in yb.items():
        rows.append(dict(method="b", seed=s_, test_cfg_id=c, replay_id=r, exp_id=f"b{s_}{c}", N=10, y=v))
    ep = pd.DataFrame(rows)
    out = stats.seed_comparisons(ep, "a", ["b"], ["y"], ["N"], stats.StatConfig(B=500, B_iqm=50), family="X")
    r = out.iloc[0]
    d = np.array([(ya[(0, c, k)] + ya[(1, c, k)]) / 2 - (yb[(0, c, k)] + yb[(1, c, k)]) / 2
                  for c in ("c1", "c2") for k in range(12)])
    assert r["p_wilcoxon"] == pytest.approx(sps.wilcoxon(d, zero_method="wilcox").pvalue)
    assert r["n_pairs"] == 24 and r["diff"] == pytest.approx(d.mean())
    assert r["b_lo"] <= r["diff"] <= r["b_hi"]
    assert r["p_wilcoxon_holm"] == pytest.approx(r["p_wilcoxon"])       # single comparison in the family
    assert "s0=" in r["per_seed_ref"] and "s1=" in r["per_seed_ref"]
    assert not any(c.startswith("p_t") or c == "poi" for c in out.columns)


def test_seed_comparison_holm_family_and_deterministic():
    rows = []
    for s_, ma in enumerate([2.0, 2.5]):
        for n in (5, 10):
            for r in range(8):
                rows.append(dict(method="a", seed=s_, test_cfg_id=f"c{n}", replay_id=r, exp_id=f"a{s_}", N=n,
                                 y=ma + 0.1 * r))
    for n in (5, 10):
        for r in range(8):
            rows.append(dict(method="h", seed=0, test_cfg_id=f"c{n}", replay_id=r, exp_id="h0", N=n, y=1.0 + 0.05 * r * n))
    out = stats.seed_comparisons(pd.DataFrame(rows), "a", ["h"], ["y"], ["N"], stats.StatConfig(B=200, B_iqm=50))
    assert len(out) == 2
    np.testing.assert_allclose(out["p_wilcoxon_holm"], stats.holm(out["p_wilcoxon"].values))
    assert (out["per_seed_other"].str.count("s0=") == 1).all()


def test_t3b_fallback_mapping_and_requests(synth_run):
    _, _, _, tabs = synth_run
    heads = [h.replace("\n", " ") for _, h in tabs["T3b"].columns]
    for case, col in [("(i)", "fb1star_steps"), ("(ii)", "fb2_steps"), ("(iii)", "fb1_steps"), ("(iv)", "fb0_steps")]:
        assert (col, f"Case {case}\n(steps)") in tabs["T3b"].columns, heads      # mapping of the log codes
    note = " ".join(tabs["T3b"].notes)
    for txt in ("(i) contradictory limits", "(ii) excess PV export", "(iii) excess required power", "(iv) negative budget"):
        assert txt in note
    h3 = [h.replace("\n", " ") for _, h in tabs["T3"].columns]         # moved to T3 (pooled) in DD43 revision
    assert any("deficit bound" in h for h in h3)
    assert any("Capped" in h for h in h3) and any("Admission" in h for h in h3)


def test_t5_units_per_ev_and_percent(synth_run):
    root, _, _, tabs = synth_run
    heads = " ".join(h for _, h in tabs["T5"].columns)
    assert "per EV" in heads and "|J^LP,ceil|" in heads and "per charger" not in heads
    assert heads.count("J^pen") == 2 and heads.count("profit") >= 2              # DD43(7): primary + secondary
    t5 = pd.read_csv(root / "stats" / "tests_T5.csv")
    assert {"jpen_eur", "profit_eur"} <= set(t5["metric"])
    tex = (root / "tables" / "T5.tex").read_text()
    assert r"$J^{\mathrm{LP,ceil}}$" in tex and r"$J^{\mathrm{pen}}$" in tex


def test_sensitivity_and_literature_tables(synth_run):
    _, _, _, tabs = synth_run
    groups = {r["group"] for r in tabs["TS2"].rows}
    assert groups and all("3 kW" in g for g in groups), groups              # stress level only (DD45)
    assert any(r["label"] == names_label("attn_safe") for r in tabs["TS3"].rows)
    t2 = tabs["T2"]
    for m in ("attn_safe", "deepsets_safe"):
        assert any(not r["cells"].get("transfer", tables_Cell()).empty for r in t2.rows if r["label"] == names_label(m)), m


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


def test_stat_methods_matches_dd48(synth_run):
    root, *_ = synth_run
    txt = (root / "tables" / "STAT_METHODS.md").read_text()
    for phrase in ("Primary test: Wilcoxon signed-rank test", "Holm", "family = table", "min-max over seeds",
                   "No seed-level t-tests", "No adaptive seed rule", "shared admitted requirement",
                   "penalised_profit_eur` of the result files is not used", "ratios of sums", "N <= 20"):
        assert phrase in txt, phrase
    assert "paired t-test" not in txt


# ----------------------------------------------------------------------------- DD45-DD49

def test_counts_tables_and_corollary_check(synth_run):
    _, _, ds, tabs = synth_run
    ep = ds.episodes
    keys = [k for k, _ in tabs["T3c"].columns]
    assert "unserved_no_infeasible" in keys and "fallback_steps" in keys and "n_evs_stay_infeasible" in keys
    row = next(r for r in tabs["T3c"].rows if r["label"] == names_label("attn_safe"))
    a = ep[ep["method"] == "attn_safe"]
    exp = a.loc[a["infeasible_steps"] == 0, "n_evs_unserved"].sum()
    assert row["cells"]["unserved_no_infeasible"].value == pytest.approx(exp)
    assert row["cells"]["infeasible_steps"].value == pytest.approx(a["infeasible_steps"].sum())
    assert {r["group"] for r in tabs["T3d"].rows} >= {"N = 5", "N = 50"}


def test_t4_rows_only_runs_that_exist(synth_run):
    _, _, ds, tabs = synth_run
    present = set(ds.episodes["method"])
    want = [names_label(m) for m in C.T4_ROWS if m in present]
    assert [r["label"] for r in tabs["T4"].rows] == want
    for gone in ("attn_safe_beta0", "attn_safe_curriculum", "attn_safe_small", "attn_penalty_dense",
                 "attn_penalty_w1", "attn_penalty_w100"):
        assert names_label(gone) not in [r["label"] for r in tabs["T4"].rows]


def test_no_percent_gap_in_main_tables(synth_run):
    _, _, _, tabs = synth_run
    for name in ("T1", "T2", "T4", "T7"):
        for k, h in tabs[name].columns:
            assert not ("gap" in k.lower() and "pct" in k.lower()), (name, k)
            assert not ("Gap" in h and "(%)" in h), (name, h)


def test_publication_text_has_no_internal_labels(synth_run):
    """Captions, headers, notes and group/row labels of the LaTeX tables: no design-decision ids, file names,
    internal table ids or review labels; captions at most 60 words; paper terminology."""
    import re
    root, _, _, tabs = synth_run
    bad = re.compile(r"\bDD\d|_values|provenance|STAT_METHODS|\.csv|\.json|\.py\b|\bT\d[a-d]?\b|\bTS\d\b|"
                     r"\b(?:N-)?M\d+\b|\bR\d+\b|Corollary|FB-|literature_N|test_cfg|stats/")
    for name, t in tabs.items():
        cap = t.caption
        assert len(cap.split()) <= 60, (name, len(cap.split()))
        texts = [cap] + list(t.notes) + [h for _, h in t.columns] + [r["group"] or "" for r in t.rows] + \
            [r["label"] for r in t.rows]
        for x in texts:
            assert not bad.search(x), (name, x)
        tex = (root / "tables" / f"{name}.tex").read_text()
        body = "\n".join(l for l in tex.splitlines() if not l.startswith("%"))
        assert not bad.search(body.replace(r"\label{tab:" + name.lower() + "}", "")), name
    assert "Theorem 1(b)" in " ".join(tabs["T3c"].notes)
    assert all("Optimality gap" in h for k, h in tabs["T1"].columns if k == "gap_abs")


def test_dagger_only_for_single_run_methods(synth_run):
    _, _, ds, tabs = synth_run
    multi = {m for m, g in ds.episodes.groupby("method") if g["seed"].nunique() >= 2}
    import names
    lab2m = {names.label(m): m for m in ds.episodes["method"].unique()}
    for name in ("T1", "T2", "T3", "T4", "T7"):
        for r in tabs[name].rows:
            m = lab2m.get(r["label"])
            if m in multi:
                for k, c in r["cells"].items():
                    assert "\u2020" not in c.text, (name, r["label"], k, c.text)
