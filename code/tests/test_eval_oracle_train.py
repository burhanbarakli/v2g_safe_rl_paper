"""evaluate.py output schema, oracle sanity, MPC port, and the train.py smoke run (code_spec Sec. 3-4)."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from conftest import ROOT, requires_ev2gym

pytestmark = requires_ev2gym

TINY_GRID = """defaults: {dt_min: 15, T: 48, min_time_of_stay: 60, hour: 10, spawn_multiplier: 8}
grid:
  profiles: [public]
  N: [5]
  rho_tr: [3.0]
replays: {default: 2}
"""


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny")
    g = d / "grid.yaml"
    g.write_text(TINY_GRID)
    return d, g


def test_evaluate_cli_writes_all_columns_and_step_logs(tiny):
    import evaluate as ev_cli
    from v2gsafe.eval.metrics import EPISODE_COLUMNS, METHOD_COLUMNS
    d, g = tiny
    out = d / "afap_safe_run"
    ev_cli.main(["--method", "afap_safe", "--test-set", str(g), "--test-sets-dir", str(d / "ts"), "--out", str(out),
                 "--admission-cache", str(d / "adm"),
                 "--smoke", "--log-steps"])
    rows = list(csv.DictReader((out / "episodes.csv").open()))
    assert len(rows) == 2
    header = list(rows[0].keys())
    assert header == METHOD_COLUMNS and header[:len(EPISODE_COLUMNS)] == EPISODE_COLUMNS
    must = [c for c in EPISODE_COLUMNS if c not in ("train_N", "oracle_profit_eur")]
    for r in rows:
        for c in must:
            assert r[c] != "", c
        assert r["method"] == "afap_safe" and r["smoke"] == "true" and r["test_cfg_id"] == "public_N5_rho3_T48"
        assert int(r["tr_violation_steps"]) <= int(r["infeasible_steps"])
        assert not r["replay_fingerprint"].startswith("MISMATCH")
        for c in ("alpha", "conformal_coverage", "n_evs_departed_before_bound"):
            assert r[c] == "", c                                  # conformal/oracle columns empty here
        assert float(r["inference_ms_per_step_p95"]) >= float(r["inference_ms_per_step_mean"]) * 0.5
    import pandas as pd
    st = pd.read_parquet(out / "steps.parquet")
    for c in ["exp_id", "method", "test_cfg_id", "replay_id", "t", "price_ch", "price_dis", "P_tr", "P_inf", "P_pv", "C", "C_rev", "sum_p",
              "sum_l", "sum_u", "n_connected", "fallback_code"]:
        assert c in st.columns, c
    ch = pd.read_parquet(out / "steps_chargers.parquet")
    for c in ["exp_id", "method", "test_cfg_id", "replay_id", "t", "i", "p", "l", "u", "E", "E_req", "tau"]:
        assert c in ch.columns, c
    # resume: a second call finds the complete part file and does not re-run
    ev_cli.main(["--method", "afap_safe", "--test-set", str(g), "--test-sets-dir", str(d / "ts"), "--out", str(out),
                 "--admission-cache", str(d / "adm"),
                 "--smoke"])
    assert len(list(csv.DictReader((out / "episodes.csv").open()))) == 2


def test_oracle_upper_bounds_heuristics_on_tiny_replay(tiny):
    """J_LP,ceil >= profit of every controller with realised shortfall <= z* and no violation (Prop. O1(b))."""
    from v2gsafe.baselines.heuristics import HEURISTICS
    from v2gsafe.baselines.oracle import solve_oracle
    from v2gsafe.envs.scenarios import ScenarioConfig
    from v2gsafe.envs.wrapper import SafeV2GEnv, WrapperOptions
    from v2gsafe.eval.evaluate import run_episode
    for rho in (3.0, 30.0):
        scfg = ScenarioConfig(profile="public", N=5, rho_tr=rho, T=48, min_time_of_stay=60, hour=10, spawn_multiplier=8)
        envs = {s: SafeV2GEnv(scfg, WrapperOptions(safety=s, admission=False)) for s in (True, False)}
        for seed in (3, 4):
            envs[True].reset(seed, scfg)
            orc = solve_oracle(envs[True].env, scfg.dt_h)
            assert orc["solver_status"].startswith("Optimal")
            J, Jc, z = orc["profit_eur"], orc["oracle_profit_ceil_eur"], orc["shortfall_stage_kwh"]
            assert Jc >= J - 1e-9
            checked = 0
            for name in ("afap", "alap", "rr", "llf"):
                for safe in (True, False):
                    m = run_episode(envs[safe], HEURISTICS[name](), seed, scfg)
                    if m["tr_violation_steps"] == 0 and m["energy_not_served_dd2_kwh"] <= z + 1e-6:
                        assert Jc >= m["profit_eur"] - 1e-6, (name, safe, rho, Jc, m["profit_eur"])
                        checked += 1
            assert checked >= 4


def test_empc_highs_runs_and_is_safe_with_layer(tiny):
    from v2gsafe.baselines.mpc import EMPCHighs
    from v2gsafe.envs.scenarios import ScenarioConfig
    from v2gsafe.envs.wrapper import SafeV2GEnv, WrapperOptions
    from v2gsafe.eval.evaluate import run_episode
    scfg = ScenarioConfig(profile="public", N=5, rho_tr=3.0, T=48, min_time_of_stay=60, hour=10, spawn_multiplier=8)
    env = SafeV2GEnv(scfg, WrapperOptions(safety=True, admission=True))
    ctrl = EMPCHighs()
    m = run_episode(env, ctrl, 5, scfg)
    assert ctrl.n_fail == 0 and m["tr_violation_steps"] <= m["infeasible_steps"]


def test_train_smoke_run_and_resume(tmp_path):
    """train.py smoke run (N in {3,5}, 2048 env steps) completes; an interrupted run resumes to the same length."""
    import train as train_cli
    from v2gsafe.algos.ppo import PPOTrainer, TRAIN_LOG_COLUMNS
    from v2gsafe.config import load_config
    out = tmp_path / "run"
    path = train_cli.main(["--config", str(ROOT / "configs/train_attn_safe.yaml"), "--overlay",
                           str(ROOT / "configs/smoke.yaml"), "--seed", "0", "--out", str(out), "--smoke"])
    assert Path(path).exists() and (out / "config.yaml").exists() and (out / "meta.json").exists()
    rows = list(csv.DictReader((out / "train_log.csv").open()))
    assert list(rows[0].keys()) == TRAIN_LOG_COLUMNS and len(rows) == 16 and int(rows[-1]["env_steps"]) == 2048
    import json
    meta = json.loads((out / "meta.json").read_text())
    assert meta["smoke"] is True and meta["ev2gym_commit_pinned"].startswith("55ef368")
    # interrupted run: 4 updates, then resume to 16
    cfg = load_config(ROOT / "configs/train_attn_penalty.yaml", ["smoke=true"], [ROOT / "configs/smoke.yaml"])
    tr = PPOTrainer(cfg, tmp_path / "r2", seed=1)
    tr.num_updates = 4
    tr.train()
    tr2 = PPOTrainer(cfg, tmp_path / "r2", seed=1, resume=True)
    assert tr2.update == 4
    tr2.train()
    rows = list(csv.DictReader((tmp_path / "r2" / "train_log.csv").open()))
    assert len(rows) == 16 and int(rows[-1]["env_steps"]) == 2048
