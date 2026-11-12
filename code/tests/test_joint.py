"""Capacity-aware (joint) departure lower bound and the bound-relative (box) action (safety/joint.py, wrapper)."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_ev2gym
from v2gsafe.safety.joint import joint_lower_bound, joint_required_energy


def _tables(E_req, tau, pnom, dt, e_min=0.0):
    """Closed-form thresholds without taper: B(k) = max(E_min, E_req - k Pnom dt)."""
    return [np.maximum(e_min, r - np.arange(k + 1) * p * dt) for r, k, p in zip(E_req, tau, pnom)]


def test_counterexample_of_sec_7_1_is_resolved():
    """Two EVs, 10 kWh each in two 1-h steps, C = 10 kW: the per-EV bound is 0 at t = 0 and 10 + 10 > C at t = 1;
    the joint bound asks 5 kW each at t = 0, after which the remaining requirements fit exactly."""
    tau, pnom = np.array([2, 2]), np.array([10.0, 10.0])
    B = _tables([10.0, 10.0], tau, pnom, 1.0)
    M1 = joint_required_energy(tau, B, pnom, np.array([10.0]), 1.0)
    np.testing.assert_allclose(M1, [5.0, 5.0])
    l = joint_lower_bound(np.zeros(2), tau, B, pnom, np.zeros(2), np.full(2, 10.0), np.array([10.0]), 10.0, 1.0)
    np.testing.assert_allclose(l, [5.0, 5.0])


def test_reduces_to_per_ev_bound_when_capacity_is_ample():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(1, 8))
        tau = rng.integers(1, 12, n)
        pnom = rng.uniform(3, 11, n)
        E = rng.uniform(5, 30, n)
        B = _tables(E + rng.uniform(0, 30, n), tau, pnom, 0.25)
        l_dep = np.array([max(-5.0, (b[k - 1] - e) / 0.25) for b, k, e in zip(B, tau, E)])
        u = np.maximum(pnom, l_dep)
        l = joint_lower_bound(E, tau, B, pnom, l_dep, u, np.full(12, 1e6), 1e6, 0.25)
        np.testing.assert_allclose(l, np.minimum(u, l_dep), atol=1e-9)


def test_bounds_and_a5_preservation_random():
    """l_dep <= l <= u always; if A5 holds for l_dep it holds for the joint bound (sum l <= C_tilde)."""
    rng = np.random.default_rng(1)
    n_tight = 0
    for _ in range(300):
        n = int(rng.integers(1, 10))
        tau = rng.integers(1, 20, n)
        pnom = rng.uniform(3, 11, n)
        E = rng.uniform(0, 40, n)
        B = _tables(E + rng.uniform(-10, 40, n), tau, pnom, 0.25)
        Bt = np.array([b[k] for b, k in zip(B, tau)])
        l_dep = np.array([(b[k - 1] - e) / 0.25 for b, k, e in zip(B, tau, E)])
        l_dep = np.where(E >= Bt, np.clip(l_dep, -5.0, pnom), pnom)      # infeasible EVs: l = u (S11)
        u = pnom.copy()
        C_hat = rng.uniform(0, 40, 20)
        Ctil = float(rng.uniform(0, 40))
        l = joint_lower_bound(E, tau, B, pnom, l_dep, u, C_hat, Ctil, 0.25)
        assert np.all(l >= l_dep - 1e-9) and np.all(l <= u + 1e-9)
        if l_dep.sum() <= Ctil:
            assert l.sum() <= Ctil + 1e-9
        n_tight += int(np.any(l > l_dep + 1e-6))
    assert n_tight > 20


@requires_ev2gym
@pytest.mark.parametrize("action", ["raw", "box"])
def test_wrapper_joint_bound_guarantees(action):
    """Closed loop with random raw actions on a tight station: transformer limit and bounds hold on A5 steps, the
    layer's bound lies in [l_dep, u], and the joint bound produces fewer infeasible steps than the per-EV bound."""
    from v2gsafe.envs.scenarios import ScenarioConfig
    from v2gsafe.envs.wrapper import SafeV2GEnv, WrapperOptions
    scfg = ScenarioConfig(profile="workplace", N=5, rho_tr=3.0, T=48, min_time_of_stay=60, hour=7, spawn_multiplier=8)
    counts = {}
    for lb in (True, "joint"):
        env = SafeV2GEnv(scfg, WrapperOptions(admission=False, lower_bound=lb, action=action))
        rng = np.random.default_rng(0)
        infeasible = viol = bound = 0
        for seed in range(12):
            obs = env.reset(seed, scfg)
            while True:
                o = obs
                assert np.all(o.l[o.occ] >= o.l_dep[o.occ] - 1e-9) and np.all(o.l[o.occ] <= o.u[o.occ] + 1e-9)
                p, fb, _ = env.command(rng.normal(0, 1.5, env.N) * 11.0)
                if o.feasible:
                    bound += int(np.any(((p < o.l - 1e-9) | (p > o.u + 1e-9))[o.occ]))
                out = env.step(None, p_override=(p, fb, None))
                infeasible += int(not o.feasible)
                viol += int(o.feasible and out.info["viol_kw"] > 1e-6)
                if out.done:
                    break
                obs = env.observe()
        assert viol == 0 and bound == 0
        counts[lb] = infeasible
    assert counts[True] > 0 and counts["joint"] < counts[True]


@requires_ev2gym
def test_resid_action_zero_reproduces_price_llf_rule():
    """Residual action: a = 0 gives the price-LLF rule clipped into [l, u] (up to the R_MIN clip of r_rule), and the
    charger feature 12 carries r_rule; the token width is F_C + 1."""
    from v2gsafe.baselines.heuristics import PriceLLF
    from v2gsafe.constants import F_C
    from v2gsafe.envs.scenarios import ScenarioConfig
    from v2gsafe.envs.wrapper import SafeV2GEnv, WrapperOptions
    scfg = ScenarioConfig(profile="public", N=5, rho_tr=4.0, T=32, min_time_of_stay=30, hour=10)
    env = SafeV2GEnv(scfg, WrapperOptions(admission=False, lower_bound="joint", action="resid"))
    rule = PriceLLF()
    obs = env.reset(0, scfg)
    rule.reset(env)
    n_checked = 0
    while True:
        assert obs.tokens.shape[1] == F_C + 1
        xr = rule.act(obs, env)
        if obs.occ.any():
            np.testing.assert_allclose(obs.tokens[obs.occ, F_C], obs.r_rule[obs.occ], atol=1e-6)
            w = obs.u - obs.l
            target = np.clip(xr, obs.l, obs.u)
            lo = obs.l + env.R_MIN * w
            hi = obs.u - env.R_MIN * w
            target = np.clip(target, lo, hi)
            x0 = obs.l + w * obs.r_rule
            np.testing.assert_allclose(x0[obs.occ & (w > 1e-6)], target[obs.occ & (w > 1e-6)], atol=1e-6)
            n_checked += 1
        out = env.step(np.zeros(env.N))
        if out.done:
            break
        obs = env.observe()
    assert n_checked > 5


def test_build_networks_token_dim():
    import torch
    from v2gsafe.algos.ppo import build_networks, token_dim
    from v2gsafe.config import load_config
    from v2gsafe.constants import F_C, F_G
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for name, fc in (("train_attn_jres_safe_penrew.yaml", F_C + 1), ("train_attn_joint_safe_penrew.yaml", F_C)):
        cfg = load_config(root / "configs" / name)
        assert token_dim(cfg) == fc
        actor, critic = build_networks(cfg)
        with torch.no_grad():
            mu, _ = actor(torch.zeros(2, 3, fc), torch.zeros(2, F_G), torch.ones(2, 3, dtype=torch.bool))
        assert mu.shape == (2, 3) and float(mu.abs().max()) < 0.1     # near-zero mean head at init


@requires_ev2gym
def test_box_action_maps_into_bounds():
    from v2gsafe.envs.scenarios import ScenarioConfig
    from v2gsafe.envs.wrapper import SafeV2GEnv, WrapperOptions
    scfg = ScenarioConfig(profile="public", N=5, rho_tr=7.0, T=16, min_time_of_stay=30, hour=14)
    env = SafeV2GEnv(scfg, WrapperOptions(admission=False, action="box"))
    obs = env.reset(0, scfg)
    for _ in range(16):
        if obs.feasible and obs.occ.any():
            lo, _, _ = env.command(np.full(env.N, -1e3))     # sigmoid -> 0: lower bound (before the projection)
            hi, _, _ = env.command(np.full(env.N, 1e3))
            mid, _, _ = env.command(np.zeros(env.N))
            o = obs
            assert np.all(lo[o.occ] >= o.l[o.occ] - 1e-9) and np.all(hi[o.occ] <= o.u[o.occ] + 1e-9)
            band = lambda v: -o.Crev <= v[o.occ].sum() <= o.Ctil       # noqa: E731  (aggregate band not binding)
            if band(o.l):
                np.testing.assert_allclose(lo[o.occ], o.l[o.occ], atol=1e-9)
            if band(0.5 * (o.l + o.u)):
                np.testing.assert_allclose(mid[o.occ], 0.5 * (o.l + o.u)[o.occ], atol=1e-9)
        out = env.step(np.zeros(env.N))
        if out.done:
            break
        obs = env.observe()
