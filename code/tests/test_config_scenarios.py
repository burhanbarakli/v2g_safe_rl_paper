"""Configuration loading/hashing and scenario definitions (Alg. 6, Alg. 7, DD24)."""
from __future__ import annotations

from pathlib import Path

import pytest

from conftest import ROOT, requires_ev2gym
from v2gsafe.config import config_hash8, exp_id, load_config
from v2gsafe.envs.scenarios import (PROFILES, SEED_RANGES, ScenarioConfig, cal_seed, load_test_grid, replay_seed,
                                    val_seed)


def test_defaults_are_alg6():
    c = load_config(ROOT / "configs/train_attn_safe.yaml")
    assert c["model"]["d_model"] == 128 and c["model"]["n_layers"] == 3 and c["model"]["n_heads"] == 4
    p = c["ppo"]
    assert (p["lr"], p["minibatch"], p["epochs"], p["rollout_len"], p["clip"], p["gamma"], p["gae_lambda"],
            p["vf_coef"], p["max_grad_norm"], p["target_kl"], p["beta"], p["total_env_steps"]) == \
        (3e-4, 512, 10, 128, 0.2, 0.99, 0.95, 0.5, 0.5, 0.03, 0.01, 3_000_000)
    assert sum(g["count"] for g in c["envs"]["groups"]) == 16


@pytest.mark.parametrize("name,method,safety,reward,arch", [
    ("train_attn_safe", "attn_safe", True, "profit", "attention"),
    ("train_attn_penalty", "attn_penalty", False, "penalty", "attention"),
    ("train_mlp_safe", "mlp_safe", True, "profit", "mlp"),
    ("train_mlp_penalty", "mlp_penalty", False, "penalty", "mlp"),
    ("train_attn_safe_conformal", "attn_safe_conformal", True, "profit", "attention")])
def test_training_configs(name, method, safety, reward, arch):
    c = load_config(ROOT / f"configs/{name}.yaml")
    assert c["method"] == method and c["safety"]["enabled"] == safety and c["reward"]["type"] == reward
    assert c["model"]["arch"] == arch
    if reward == "penalty":
        assert c["ppo"]["beta"] == 0.0 and c["reward"]["w_E"] == 10.0 and c["reward"]["w_P"] == 10.0
    if name.endswith("conformal"):
        assert c["conformal"]["enabled"] and c["conformal"]["alpha"] == 0.1


def test_ablation_configs_load():
    for p in sorted((ROOT / "configs/ablations").glob("*.yaml")):
        c = load_config(p)
        assert c["method"].startswith("attn_")


def test_unknown_key_rejected_and_hash_stable(tmp_path):
    with pytest.raises(KeyError):
        load_config(None, ["ppo.lrr=1e-4"])
    bad = tmp_path / "bad.yaml"
    bad.write_text("ppo: {learning_rate: 1}\n")
    with pytest.raises(KeyError):
        load_config(bad)
    a = load_config(ROOT / "configs/train_attn_safe.yaml")
    b = load_config(ROOT / "configs/train_attn_safe.yaml", ["smoke=true"])
    assert config_hash8(a) == config_hash8(b)                       # smoke flag excluded
    c = load_config(ROOT / "configs/train_attn_safe.yaml", ["ppo.lr=1e-4"])
    assert config_hash8(a) != config_hash8(c)
    assert exp_id("attn_safe", "train_attn_safe", 3, "abcd1234") == "attn_safe__train_attn_safe__s3__abcd1234"
    s = load_config(ROOT / "configs/train_attn_safe.yaml", overlays=[ROOT / "configs/smoke.yaml"])
    assert s["smoke"] and s["scenario"]["T"] == 16 and {g["N"] for g in s["envs"]["groups"]} == {3, 5}


def test_test_grid_36_plus_literature_3100_episodes():
    grid = load_test_grid(ROOT / "configs/test_grid.yaml")
    assert len(grid) == 37
    assert sum(n for _, n in grid) == 3100
    ids = {s.cfg_id for s, _ in grid}
    assert "workplace_N50_rho4" in ids and "private_N200_rho7" in ids and "literature_N25" in ids
    lit = next(s for s, _ in grid if s.cfg_id == "literature_N25")
    assert lit.P_tr == 100.0 and not lit.loads and not lit.pv and lit.horizon == 112
    for s, n in grid:
        assert n == (50 if s.N >= 100 else 100)


def test_profiles_and_ev2gym_config():
    assert PROFILES["workplace"] == {"hour": 5, "spawn_multiplier": 5.0, "simulation_days": "weekdays"}
    assert PROFILES["public"]["hour"] == 12 and PROFILES["private"]["spawn_multiplier"] == 2.5
    s = ScenarioConfig(profile="private", N=50, rho_tr=7.0)
    c = s.ev2gym_config()
    assert c["transformer"]["max_power"] == 350.0 and c["number_of_charging_stations"] == 50
    assert c["number_of_ports_per_cs"] == 1 and c["v2g_enabled"] and c["ev"]["transition_soc_multiplier"] == 5
    assert c["inflexible_loads"]["inflexible_loads_capacity_multiplier_mean"] == 0.4
    assert c["solar_power"]["solar_power_capacity_multiplier_mean"] == 0.4
    assert c["hour"] == 12 and c["simulation_days"] == "both" and c["simulation_length"] == 112
    assert c["charging_station"]["min_charge_current"] == 0 and c["charging_station"]["min_discharge_current"] == 0
    assert c["ev_specs_file"].endswith("ev_specs_const_eta.json") and Path(c["ev_specs_file"]).exists()
    assert ScenarioConfig(dt_min=5).horizon == 336


def test_seed_ranges_disjoint():
    t = replay_seed("workplace_N10_rho4", 99)
    assert SEED_RANGES["test"][0] <= t < SEED_RANGES["test"][1]
    assert SEED_RANGES["val"][0] <= val_seed("x", 19) < SEED_RANGES["val"][1]
    assert SEED_RANGES["cal"][0] <= cal_seed("x", 4999) < SEED_RANGES["cal"][1]
    assert SEED_RANGES["cal_train"][0] <= cal_seed("x", 1999, "cal_train") < SEED_RANGES["cal_train"][1]
    assert max(hi for _, hi in SEED_RANGES.values()) <= 2**32


@requires_ev2gym
def test_build_test_set_manifest(tmp_path):
    from v2gsafe.envs.scenarios import build_test_set
    s = ScenarioConfig(profile="workplace", N=4, rho_tr=4.0, T=48)
    m1 = build_test_set(s, 3, tmp_path)
    assert (tmp_path / f"{s.cfg_id}.json").exists() and len(m1["replays"]) == 3
    m2 = build_test_set(s, 3, tmp_path / "other")
    assert [r["fingerprint"] for r in m1["replays"]] == [r["fingerprint"] for r in m2["replays"]]
    assert len({r["fingerprint"] for r in m1["replays"]}) == 3


@requires_ev2gym
def test_light_session_sampler_matches_env_date():
    from v2gsafe.envs.scenarios import sample_sessions
    from v2gsafe.envs.wrapper import make_ev2gym_env
    s = ScenarioConfig(profile="workplace", N=5, rho_tr=4.0)
    env = make_ev2gym_env(s)
    env.tr_seed = 11
    env.reset(seed=11)
    d0 = env.sim_starting_date
    evs = sample_sessions(env, 11)
    assert env.sim_starting_date == d0 and len(evs) > 0
    assert all(ev.time_of_departure <= s.horizon - 2 for ev in evs)
