"""SafeV2GEnv with real EV2Gym: guarantees on A5 steps, determinism, observation, reward (code_spec Sec. 4)."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_ev2gym

pytestmark = requires_ev2gym


@pytest.fixture(scope="module")
def tight_env():
    from v2gsafe.envs.scenarios import ScenarioConfig
    from v2gsafe.envs.wrapper import SafeV2GEnv, WrapperOptions
    scfg = ScenarioConfig(profile="public", N=5, rho_tr=2.5, T=48, min_time_of_stay=60, hour=10, spawn_multiplier=8)
    return SafeV2GEnv(scfg, WrapperOptions(admission=False, log_steps=True)), scfg


def _closed_loop(env, scfg, seeds, rng, scale=1.5):
    stats = dict(feasible_steps=0, viol_on_feasible=0, viol_any=0, infeasible=0, evs=0, clean_evs=0,
                 bound_viol=0, margin=0.0, deliv=np.inf)
    for seed in seeds:
        obs = env.reset(seed, scfg)
        clean = {}
        while True:
            x = rng.normal(0, scale, env.N) * 11.0
            p, fb, _ = env.command(x)
            o = obs
            stats["bound_viol"] += int(np.any((p < o.pmin - 1e-9) | (p > o.u + 1e-9)))
            if o.feasible:
                stats["bound_viol"] += int(np.any((p < o.l - 1e-9)[o.occ]))
            out = env.step(None, p_override=(p, fb, None))
            stats["feasible_steps"] += int(o.feasible)
            stats["infeasible"] += int(not o.feasible)
            v = out.info["viol_kw"] > 1e-6
            stats["viol_any"] += int(v)
            stats["viol_on_feasible"] += int(v and o.feasible)
            if out.done:
                break
            obs = env.observe()
        m = env.episode_metrics()
        stats["evs"] += m["n_evs"]
        stats["clean_evs"] += m["n_evs"] - m["n_evs_touched_infeasible"]
        stats["margin"] = max(stats["margin"], m["actuation_margin_max_kw"])
        stats["deliv"] = min(stats["deliv"], m["energy_delivery_min_kwh"])
    return stats


def test_guarantees_under_random_raw_actions_T16_and_T48(tight_env):
    """Zero transformer violations and zero unserved EVs on A5-feasible stays under random raw actions."""
    env, scfg = tight_env
    from v2gsafe.eval import metrics as M
    # wrap on_departure to count unserved EVs that were never touched by an infeasible step
    acc_cls = M.EpisodeAccumulator
    orig = acc_cls.on_departure
    counter = {"clean_unserved": 0, "clean": 0}

    def on_dep(self, short, short2, sat, touched, *a, **k):
        if not touched:
            counter["clean"] += 1
            counter["clean_unserved"] += int(short > 1e-6)
        return orig(self, short, short2, sat, touched, *a, **k)

    acc_cls.on_departure = on_dep
    try:
        rng = np.random.default_rng(0)
        st = _closed_loop(env, scfg, range(30), rng)
        from v2gsafe.envs.wrapper import SafeV2GEnv, WrapperOptions
        s16 = scfg.replace(T=16, min_time_of_stay=30, hour=14, rho_tr=2.0)
        env16 = SafeV2GEnv(s16, WrapperOptions(admission=False))
        st16 = _closed_loop(env16, s16, range(40), rng, scale=3.0)
    finally:
        acc_cls.on_departure = orig
    assert st["evs"] > 30 and st["feasible_steps"] > 500
    assert st["viol_on_feasible"] == 0 and st16["viol_on_feasible"] == 0
    assert counter["clean"] > 20 and counter["clean_unserved"] == 0
    assert st["bound_viol"] == 0
    assert st["margin"] <= 0.008 and st["deliv"] >= -1e-9
    assert st16["bound_viol"] == 0 and st16["evs"] >= 40


def test_T16_smoke_episode_with_safety_layer():
    from v2gsafe.envs.scenarios import ScenarioConfig
    from v2gsafe.envs.wrapper import SafeV2GEnv, WrapperOptions
    scfg = ScenarioConfig(profile="public", N=5, rho_tr=4.0, T=16, min_time_of_stay=30, hour=14)
    env = SafeV2GEnv(scfg, WrapperOptions())
    rng = np.random.default_rng(1)
    for seed in range(5):
        env.reset(seed, scfg)
        n = 0
        while True:
            out = env.step(rng.normal(0, 2, 5) * 11)
            n += 1
            if out.done:
                break
            env.observe()
        m = env.episode_metrics()
        assert n == 16 and m["n_steps"] == 16 and m["tr_violation_steps"] <= m["infeasible_steps"]


def test_determinism_and_scenario_mutation():
    """Same (scenario, seed) gives the same episode; a reused env equals a fresh env (fingerprints)."""
    from v2gsafe.envs.scenarios import ScenarioConfig, fingerprint
    from v2gsafe.envs.wrapper import SafeV2GEnv, WrapperOptions
    a = ScenarioConfig(profile="workplace", N=6, rho_tr=4.0, T=48)
    b = a.replace(profile="private", rho_tr=7.0)
    env1 = SafeV2GEnv(a, WrapperOptions(admission=False))
    env1.reset(123, a)
    fa = fingerprint(env1.env)
    env1.reset(77, b)
    fb = fingerprint(env1.env)
    env1.reset(123, a)
    assert fingerprint(env1.env) == fa
    env2 = SafeV2GEnv(b, WrapperOptions(admission=False))
    env2.reset(77, b)
    assert fingerprint(env2.env) == fb
    assert fa != fb
    assert env2.env.transformers[0].max_power[0] == 42.0


def test_observation_features_and_reward():
    from v2gsafe.constants import C_REF, F_C, F_G, P_REF
    from v2gsafe.envs.scenarios import ScenarioConfig
    from v2gsafe.envs.wrapper import SafeV2GEnv, WrapperOptions
    scfg = ScenarioConfig(profile="workplace", N=8, rho_tr=4.0)
    env = SafeV2GEnv(scfg, WrapperOptions())
    obs = env.reset(5, scfg)
    rng = np.random.default_rng(0)
    seen_occ = False
    total = 0.0
    while True:
        assert obs.tokens.shape == (8, F_C) and obs.glob.shape == (F_G,)
        assert np.all(obs.tokens[~obs.occ] == 0)
        assert np.all(obs.tokens[obs.occ, 0] == 1)
        np.testing.assert_allclose(obs.tokens[obs.occ, 10] * P_REF, obs.l[obs.occ], rtol=1e-6, atol=1e-5)
        np.testing.assert_allclose(obs.tokens[obs.occ, 11] * P_REF, obs.u[obs.occ], rtol=1e-6, atol=1e-5)
        t = obs.t
        assert abs(obs.glob[0] - (-env.env.charge_prices[0, t]) / C_REF) < 1e-5
        assert abs(obs.glob[13] - (env.T - t) / env.T) < 1e-6
        assert abs(obs.glob[34] - obs.occ.mean()) < 1e-6
        assert abs(obs.glob[35] - 32.0 / (8 * 11.0)) < 1e-6
        seen_occ |= bool(obs.occ.any())
        out = env.step(rng.normal(0, 1, 8) * 11)
        total += out.reward
        if out.done:
            break
        obs = env.observe()
    assert seen_occ
    stats = env.env.stats
    assert abs(total - stats["total_profits"]) < 1e-9            # r(t) = EV2Gym profit (F9)
    assert abs(env.episode_metrics()["profit_eur"] - total) < 1e-9
