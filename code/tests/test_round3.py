"""Round-3 additions: controller-independent admission (DD31), taper/soft oracle (DD32), new methods (DD33),
tuning protocol (DD37)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from conftest import ROOT, requires_ev2gym


def _scfg(rho=3.0, N=5, **kw):
    from v2gsafe.envs.scenarios import ScenarioConfig
    return ScenarioConfig(profile="public", N=N, rho_tr=rho, T=48, min_time_of_stay=60, hour=10, spawn_multiplier=8,
                          **kw)


# ----------------------------------------------------------------------------------------- DD31
def test_forecasts_are_a_function_of_seed_and_step():
    from v2gsafe.envs.forecasts import NoisyForecaster
    s = np.linspace(1, 10, 50)
    a, b = NoisyForecaster(0.1, 8), NoisyForecaster(0.1, 8)
    a.reseed(5)
    b.reseed(5)
    fa = [a.forecast(s, t) for t in range(10)]
    fb = [b.forecast(s, t) for t in reversed(range(10))][::-1]      # call order does not matter
    assert all(np.array_equal(x, y) for x, y in zip(fa, fb))
    assert not np.array_equal(a.forecast(s, 3, 0), a.forecast(s, 3, 1))


@requires_ev2gym
def test_admission_is_controller_independent_and_certified():
    """Same admitted requests for every controller (raw and safe); reduced requests satisfy A4 (re-certified)."""
    from v2gsafe.baselines.heuristics import HEURISTICS
    from v2gsafe.envs.wrapper import SafeV2GEnv, WrapperOptions
    from v2gsafe.eval.evaluate import run_episode
    from v2gsafe.safety.bounds import cap_request
    scfg = _scfg(rho=1.0, N=6)
    found_reduction = False
    for seed in (3, 4, 5):
        reqs = []
        for name, safe in (("afap", False), ("alap", True), ("llf", True), ("random", False)):
            env = SafeV2GEnv(scfg, WrapperOptions(safety=safe, admission=True))
            m = run_episode(env, HEURISTICS[name](), seed, scfg)
            tab = env.adm_table
            reqs.append((m["energy_admission_reduced_kwh"], m["n_evs_admission_reduced"],
                         sorted((k, e["E_req"]) for k, e in tab.items() if k is not None)))
        assert all(r == reqs[0] for r in reqs[1:])
        found_reduction |= reqs[0][0] > 0.01
        for (loc, ta), e in [x for x in tab.items() if x[0] is not None]:
            assert e["E_req"] <= e["E_req_dd2"] + 1e-12
    assert found_reduction, "the tight scenario should trigger at least one admission reduction"
    # A4 after re-certification
    env = SafeV2GEnv(scfg, WrapperOptions(safety=True, admission=True))
    env.reset(3, scfg)
    for k, e in [x for x in env.adm_table.items() if x[0] is not None]:
        from v2gsafe.safety.admission_prepass import arrival_records
        a = next(r for r in arrival_records(env.env, scfg.dt_h) if (r["loc"], r["t_arr"]) == k)
        p = a["params"]
        K = a["d_ctrl"] - a["t_arr"] + 1
        y, B = cap_request(e["E_req"], a["E_arr"], K, p.phi, p.e_min, p.cap)
        assert y == pytest.approx(e["E_req"]) and a["E_arr"] >= B[K] - 1e-12


@requires_ev2gym
def test_admission_table_disk_cache(tmp_path):
    from v2gsafe.envs.wrapper import SafeV2GEnv, WrapperOptions
    from v2gsafe.safety import admission_prepass as ap
    scfg = _scfg(rho=1.0)
    env = SafeV2GEnv(scfg, WrapperOptions(admission=True, admission_cache=str(tmp_path)))
    env.reset(7, scfg)
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    ap._MEM.clear()
    t1 = env.adm_table[None]["entries"]
    env.reset(7, scfg)                                     # loaded from disk
    assert env.adm_table[None]["entries"] == t1


# ----------------------------------------------------------------------------------------- DD32
def test_taper_outer_rho_dominates_ev2gym_taper():
    from v2gsafe.safety.bounds import EVParams, taper_outer_rho
    for s_tr, m, cap, pn, dt in ((0.8, 1.0, 50.0, 11.0, 0.25), (0.6, 2.0, 30.0, 22.0, 0.25), (1.0, 1.0, 60.0, 7.0, 0.25),
                                  (0.9, 0.5, 80.0, 11.0, 1 / 12)):
        p = EVParams(cap=cap, e_min=5.0, pnom=pn, pdis=pn, s_tr=s_tr, m=m, zeta_c=1.0, zeta_d=1.0, dt_h=dt)
        rho = taper_outer_rho(p)
        E = np.random.default_rng(0).uniform(p.e_min, cap - 1e-6, 20000)
        pbar = (p.phi_vec(E) - E) / dt
        assert np.all(pbar <= rho * (cap - E) + 1e-9)
        if s_tr < 1:      # exact upper-region slope (EV2Gym ev.py: 1 - (1-s) exp(-kappa))
            Eu = np.linspace(s_tr * cap + 1e-3, cap - 1e-3, 50)
            pb = (p.phi_vec(Eu) - Eu) / dt
            sel = pb < pn - 1e-9                       # below the constant-power cap
            assert sel.sum() > 10
            assert np.allclose(pb[sel] / (cap - Eu[sel]), (1 - math.exp(-p.kappa)) / dt, rtol=1e-6)
        else:
            assert rho == pytest.approx(1 / dt)


@requires_ev2gym
def test_oracles_bound_every_controller_on_tiny_replays():
    """Soft oracle J^pen_LP,ceil >= penalised_profit(_dd2) of every heuristic (raw and safe); hard taper oracle
    J_LP,ceil >= profit of every controller with no violation and shortfall <= z*; all with admitted requests."""
    from v2gsafe.baselines.heuristics import HEURISTICS
    from v2gsafe.baselines.oracle import solve_oracle
    from v2gsafe.envs.wrapper import SafeV2GEnv, WrapperOptions
    from v2gsafe.eval.evaluate import run_episode
    checked_hard = 0
    for rho, kw in ((1.5, {}), (30.0, {}), (3.0, {"throughput_cost": 0.05, "discharge_price_factor": 0.9})):
        scfg = _scfg(rho=rho, **kw)
        envs = {s: SafeV2GEnv(scfg, WrapperOptions(safety=s, admission=True)) for s in (True, False)}
        for seed in (3, 4):
            envs[True].reset(seed, scfg)
            orc = solve_oracle(envs[True].env, scfg.dt_h, table=envs[True].adm_table, thr=scfg.throughput_cost)
            assert orc["solver_status"].startswith("Optimal")
            Jc, Jpen, z = orc["oracle_profit_ceil_eur"], orc["oracle_pen_profit_eur"], orc["shortfall_stage_kwh"]
            assert Jc >= orc["profit_eur"] - 1e-9
            for name in ("afap", "alap", "rr", "llf", "sllf", "price_llf", "random"):
                for safe in (True, False):
                    m = run_episode(envs[safe], HEURISTICS[name](), seed, scfg)
                    assert Jpen >= m["penalised_profit_eur"] - 1e-6, (name, safe, rho, Jpen, m["penalised_profit_eur"])
                    assert m["penalised_profit_eur"] >= m["penalised_profit_dd2_eur"] - 1e-9
                    if m["tr_violation_steps"] == 0 and m["energy_not_served_kwh"] <= z + 1e-6:
                        assert Jc >= m["profit_eur"] - 1e-6, (name, safe, rho, Jc, m["profit_eur"])
                        checked_hard += 1
    assert checked_hard >= 10


# ----------------------------------------------------------------------------------------- DD33
def test_deepsets_shapes_equivariance_and_padding():
    from v2gsafe.models.deepsets_policy import DeepSetsActor, DeepSetsCritic
    torch.manual_seed(0)
    a, c, c3 = DeepSetsActor(32, 16), DeepSetsCritic(32, 16), DeepSetsCritic(32, 16, n_out=3)
    B, N = 3, 7
    tok, glob = torch.randn(B, N, 12), torch.randn(B, 38)
    occ = torch.rand(B, N) > 0.3
    occ[:, 0] = True
    mu, ls = a(tok, glob, occ)
    assert mu.shape == (B, N) and ls.shape == (B, N) and c(tok, glob, occ).shape == (B,)
    assert c3(tok, glob, occ).shape == (B, 3)
    perm = torch.randperm(N)
    mu2, _ = a(tok[:, perm], glob, occ[:, perm])
    assert torch.allclose(mu2, mu[:, perm], atol=1e-5)
    assert torch.allclose(c(tok[:, perm], glob, occ[:, perm]), c(tok, glob, occ), atol=1e-5)
    pad_tok = torch.cat([tok, torch.randn(B, 4, 12)], 1)
    pad_occ = torch.cat([occ, torch.zeros(B, 4, dtype=torch.bool)], 1)
    mu3, _ = a(pad_tok, glob, pad_occ)
    assert torch.allclose(mu3[:, :N][occ], mu[occ], atol=1e-5)
    assert torch.allclose(c(pad_tok, glob, pad_occ), c(tok, glob, occ), atol=1e-5)


def test_deepsets_width_matches_attention_parameter_counts():
    """DD42(3): the DeepSets actor/critic of configs/train_deepsets_safe.yaml match the attention model within 10 %."""
    from v2gsafe.algos.ppo import build_networks
    from v2gsafe.config import load_config
    from v2gsafe.models.attention_policy import count_params
    a, c = build_networks(load_config(ROOT / "configs/train_deepsets_safe.yaml"))
    assert count_params(a) == 626530 and count_params(c) == 643777
    assert abs(count_params(a) / 610370 - 1) <= 0.10 and abs(count_params(c) / 676481 - 1) <= 0.10


def test_lagrangian_dual_update_sign_and_projection():
    from v2gsafe.algos.ppo import dual_update
    lam = dual_update([1.0, 1.0], [2.0, 0.0], [0.5, 0.5], 0.1, 100.0)
    assert lam[0] > 1.0 and lam[1] < 1.0                    # cost above limit -> increase; below -> decrease
    assert np.all(dual_update([0.01, 0.0], [0.0, 0.0], [1.0, 1.0], 1.0, 100.0) == 0.0)   # projection onto lam >= 0
    assert dual_update([99.0, 0.0], [1e3, 0.0], [0.0, 0.0], 1.0, 100.0)[0] == 100.0


def test_lagrangian_advantage_combination_and_critic_heads():
    from v2gsafe.algos.ppo import build_networks
    from v2gsafe.config import load_config
    cfg = load_config(ROOT / "configs/train_attn_lagrangian.yaml", overlays=[ROOT / "configs/smoke.yaml"])
    _, critic = build_networks(cfg)
    v = critic(torch.zeros(2, 3, 12), torch.zeros(2, 38), torch.ones(2, 3, dtype=torch.bool))
    assert v.shape == (2, 3)
    cfg2 = load_config(ROOT / "configs/train_deepsets_safe.yaml", overlays=[ROOT / "configs/smoke.yaml"])
    actor, critic = build_networks(cfg2)
    assert type(actor).__name__ == "DeepSetsActor"


def test_sllf_toy_feasibility():
    from types import SimpleNamespace
    from v2gsafe.baselines.heuristics import SmoothedLLF
    rng = np.random.default_rng(1)
    for _ in range(50):
        N = 6
        occ = rng.random(N) > 0.2
        occ[0] = True
        u = rng.uniform(0, 11, N)
        E = rng.uniform(10, 40, N)
        slots = [SimpleNamespace(params=SimpleNamespace(pnom=11.0), B=np.sort(rng.uniform(5, 45, 10)), d_ctrl=5)
                 if o else None for o in occ]
        env = SimpleNamespace(slots=slots, dt_h=0.25)
        for Ctil in (rng.uniform(0, u[occ].sum()), u[occ].sum() + 5, -3.0):
            obs = SimpleNamespace(occ=occ, u=u, E=E, Ctil=Ctil, t=0)
            x = SmoothedLLF().act(obs, env)
            assert np.all(x[~occ] == 0) and np.all(x >= -1e-12) and np.all(x <= u + 1e-12)
            assert x.sum() <= max(0.0, Ctil) + 1e-9
            if Ctil >= u[occ].sum():
                assert np.allclose(x[occ], u[occ])
            elif Ctil > 0:
                assert x.sum() >= Ctil - 1e-6 * max(1.0, Ctil)     # the bisection fills the budget


@requires_ev2gym
def test_proj_nolb_layer_uses_pmin_and_safedeploy_name(tmp_path):
    from v2gsafe.envs.wrapper import SafeV2GEnv, WrapperOptions
    scfg = _scfg(rho=3.0)
    env = SafeV2GEnv(scfg, WrapperOptions(safety=True, lower_bound=False, admission=True))
    obs = env.reset(4, scfg)
    for _ in range(20):
        assert np.allclose(obs.l[obs.occ], obs.pmin[obs.occ]) and obs.l_dep is not None
        env.step(np.zeros(scfg.N))
        if env.done:
            break
        obs = env.observe()


# ----------------------------------------------------------------------------------------- DD37
def test_tuning_equal_budget_default_first_and_minibatch_in_grid():
    from v2gsafe import tuning
    from v2gsafe.config import DEFAULTS, load_config
    assert DEFAULTS["ppo"]["minibatch"] in tuning.PPO_GRID["ppo.minibatch"]
    sizes = set()
    for c in ("attn_safe", "attn_penalty", "mlp_safe", "mlp_penalty", "attn_proj_nolb", "attn_lagrangian",
              "deepsets_safe"):
        cfg = load_config(ROOT / f"configs/train_{c}.yaml")
        g = tuning.method_grid(cfg)
        pts = tuning.sampled_points(cfg["method"], g, tuning.K_CONFIGS)
        sizes.add(len(pts))
        assert pts[0]["ppo.lr"] == DEFAULTS["ppo"]["lr"] and pts[0]["ppo.minibatch"] == DEFAULTS["ppo"]["minibatch"]
        assert len({tuning.tag(p) for p in pts}) == len(pts)
        assert ("reward.w" in g) == (cfg["reward"]["type"] == "penalty")
        assert ("lagrangian.lr_lambda" in g) == (c == "attn_lagrangian")
    assert sizes == {tuning.K_CONFIGS}


def test_pipeline_refuses_training_without_tuning(tmp_path, monkeypatch):
    from v2gsafe import pipeline
    ns = type("A", (), dict(smoke=True, device="cpu", seeds=1, extended=False, hparams=None))()
    c = pipeline.make_ctx(ns)
    c.results = tmp_path
    with pytest.raises(SystemExit):
        pipeline.stage_train(c)
    hp = tmp_path / "hp.yaml"
    hp.write_text("default: {}\nattn_penalty: {reward: {w_E: 1.0, w_P: 1.0}, ppo: {lr: 1.0e-4}}\n")
    c.hparams = hp
    jobs = pipeline.stage_train(c)
    assert len(jobs) == len(pipeline.MAIN_CONFIGS) + 1              # + attn_safe_trainN50 (core, DD42)
    assert any("_hparams" in x for j in jobs for x in j.cmd)


# ----------------------------------------------------------------------------------------- DD43
@requires_ev2gym
def test_shared_requirement_written_back_and_new_columns():
    """DD43(1)-(2): EV2Gym's EV objects carry the requirement the method is given; the noadm pass measures shortfall
    also vs the shared admitted requirement; stay-feasibility, binding and lower-bound columns are consistent."""
    from v2gsafe.baselines.heuristics import HEURISTICS
    from v2gsafe.envs.wrapper import SafeV2GEnv, WrapperOptions
    from v2gsafe.eval.evaluate import run_episode
    scfg = _scfg(rho=1.0, N=6)
    env = SafeV2GEnv(scfg, WrapperOptions(safety=True, admission=True))
    obs = env.reset(3, scfg)
    seen = 0
    while not env.done:
        for s in env.slots:
            if s is not None:
                assert s.ev.desired_capacity == pytest.approx(s.E_req) and s.E_req <= s.E_req_shared + 1e-9
                seen += 1
        env.step(np.zeros(scfg.N))
        if not env.done:
            obs = env.observe()
    assert seen > 0
    prof = [ev.desired_capacity for ev in env.env.EVs_profiles]
    from v2gsafe.envs.scenarios import fingerprint
    fp = fingerprint(env.env)
    m_on = run_episode(env, HEURISTICS["llf"](), 3, scfg)
    assert [ev.desired_capacity for ev in env.env.EVs_profiles] == prof and fingerprint(env.env) == fp
    assert m_on["energy_not_served_shared_kwh"] == pytest.approx(m_on["energy_not_served_kwh"])
    env_off = SafeV2GEnv(scfg, WrapperOptions(safety=False, admission=False))
    m_off = run_episode(env_off, HEURISTICS["afap"](), 3, scfg)
    assert m_off["energy_admission_reduced_kwh"] == 0.0
    assert m_off["energy_not_served_shared_kwh"] <= m_off["energy_not_served_kwh"] + 1e-9
    for m in (m_on, m_off):
        assert 0 <= m["n_evs_stay_infeasible"] <= m["n_evs"]
        assert m["share_evs_stay_feasible"] == pytest.approx(1 - m["n_evs_stay_infeasible"] / m["n_evs"])
        assert (m["n_evs_stay_infeasible"] > 0) == (m["infeasible_steps"] > 0)
        assert 0 <= m["agg_upper_active_steps"] <= m["n_steps"] and 0 <= m["agg_lower_active_steps"] <= m["n_steps"]
        assert 0.0 <= m["lb_active_frac"] <= 1.0
        assert m["fb0_steps"] + m["fb1_steps"] + m["fb2_steps"] + m["fb1star_steps"] <= m["infeasible_steps"]
    assert m_on["agg_upper_active_steps"] > 0          # tight transformer: LLF fills C_tilde


@requires_ev2gym
def test_empc_highs_uses_shared_requirement_for_future_arrivals():
    from v2gsafe.baselines.mpc import EMPCHighs
    from v2gsafe.envs.wrapper import SafeV2GEnv, WrapperOptions
    from v2gsafe.eval.evaluate import run_episode
    scfg = _scfg(rho=1.0, N=6)
    env = SafeV2GEnv(scfg, WrapperOptions(safety=True, admission=True))
    ctrl = EMPCHighs()
    m = run_episode(env, ctrl, 3, scfg)
    assert ctrl.n_fail == 0 and m["tr_violation_steps"] <= m["infeasible_steps"]


def test_pipeline_evaluates_mlp_only_up_to_N20(tmp_path, capsys):
    from v2gsafe import pipeline
    ns = type("A", (), dict(smoke=False, device="cpu", seeds=1, extended=False, hparams=None))()
    c = pipeline.make_ctx(ns)
    c.results = tmp_path
    for name in ("mlp_safe__train_mlp_safe__s0__aaaaaaaa", "attn_safe__train_attn_safe__s0__bbbbbbbb"):
        (tmp_path / "raw" / name).mkdir(parents=True)
        (tmp_path / "raw" / name / "checkpoint_final.pt").write_bytes(b"")
    jobs = pipeline.stage_evaluate(c)
    mlp = [j.name for j in jobs if "mlp_safe__" in j.name]
    attn = [j.name for j in jobs if "attn_safe__" in j.name]
    assert mlp and all(int(n.rsplit("_N", 1)[1]) <= 20 for n in mlp)
    assert any(n.endswith("_N200") for n in attn)
    assert "MLP baseline only at N <= 20" in capsys.readouterr().out


@requires_ev2gym
def test_gurobi_empc_gets_shared_requirement_and_profiles_are_restored():
    pytest.importorskip("gurobipy")
    from v2gsafe.baselines.mpc import EMPCGurobi
    from v2gsafe.envs.scenarios import fingerprint
    from v2gsafe.envs.wrapper import SafeV2GEnv, WrapperOptions
    scfg = _scfg(rho=1.0, N=5)
    env = SafeV2GEnv(scfg, WrapperOptions(safety=False, admission=True))
    obs = env.reset(3, scfg)
    fp = fingerprint(env.env)
    c = EMPCGurobi()
    c.reset(env)
    assert fingerprint(env.env) == fp
    tab = env.adm_table
    for k, ev in enumerate(env.env.EVs_profiles):
        e = tab.get((int(ev.location), int(ev.time_of_arrival)))
        if e is not None:
            assert c.mpc.Cxf[k] == pytest.approx(e["E_req"])
    x = c.act(obs, env)
    assert np.all(x[obs.occ] >= obs.pmin[obs.occ] - 1e-9) and np.all(x[obs.occ] <= obs.u[obs.occ] + 1e-9)
