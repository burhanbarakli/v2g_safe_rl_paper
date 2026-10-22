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
    for t in ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "TS1"]:
        for ext in (".tex", ".md", ".csv", "_values.csv"):
            assert (root / "tables" / f"{t}{ext}").exists(), f"{t}{ext} missing"
    for f in ["figS1_training_curves", "fig3_zero_shot_scaling", "fig4_constraint_satisfaction", "fig5_example_day",
              "figS2_ablation", "fig6_cost_of_guarantee", "figS3_conformal_alpha", "figS4_inference_time"]:
        for ext in ("pdf", "png", "svg", "eps"):
            p = root / "figures" / f"{f}.{ext}"
            assert p.exists() and p.stat().st_size > 1000, f"{p} missing/empty"
        assert (root / "figures" / "data" / f"{f}.csv").exists()
    for p in ["tables/STAT_METHODS.md", "tables/provenance.json", "findings_numbers.md",
              "stats/paired_tests.csv", "stats/summary_by_config.csv", "README_SYNTHETIC.txt"]:
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
    oc = orc[(orc["test_cfg_id"] == row["test_cfg_id"]) & (orc["replay_id"] == row["replay_id"])][load.ORACLE_CEIL].iat[0]
    assert row["oracle_profit_eur"] == pytest.approx(o)
    assert row["oracle_ceil_eur"] == pytest.approx(oc)
    # DD26: main gap w.r.t. J_LP,ceil; supplementary gap w.r.t. J_LP
    assert row["gap_pct"] == pytest.approx(100 * (oc - row["profit_eur"]) / abs(oc))
    assert row["gap_lp_pct"] == pytest.approx(100 * (o - row["profit_eur"]) / abs(o))
    assert ep["oracle_profit_eur"].notna().all() and ep["oracle_ceil_eur"].notna().all()


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
          "attn_safe_curriculum", "attn_safe_small", "attn_penalty_dense", "attn_penalty_w1", "attn_penalty_w100"]
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
    assert "fallback" in ds.gap_reference
    np.testing.assert_allclose(ds.episodes["gap_pct"], ds.episodes["gap_lp_pct"])


def test_conceptual_figures_copied(synth_run):
    root, *_ = synth_run
    for n in ("fig2_concepts", "figS5_pipeline"):
        assert (root / "figures" / f"{n}.pdf").exists(), n
