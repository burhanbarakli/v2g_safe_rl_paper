"""PPO with variable N (Alg. 2): mixed-N update, determinism, masking (DD21), GAE, checkpoint save/load."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from v2gsafe.algos.buffers import EnvBuffer, RolloutBuffer
from v2gsafe.algos.ppo import build_networks, gaussian_logp, ppo_update
from v2gsafe.config import load_config
from v2gsafe.constants import F_C, F_G

SMALL = ["model.d_model=32", "model.n_layers=1", "model.n_heads=2", "model.head_hidden=16", "ppo.minibatch=32",
         "ppo.epochs=2", "model.n_max=8", "model.mlp_hidden=32"]


def _fake_obs(rng, N):
    occ = rng.random(N) < 0.7
    l = np.where(occ, rng.uniform(-5, 3, N), 0.0)
    u = np.where(occ, l + rng.uniform(0, 8, N) * (rng.random(N) > 0.2), 0.0)
    tok = (rng.normal(size=(N, F_C)) * occ[:, None]).astype(np.float32)
    return SimpleNamespace(tokens=tok, glob=rng.normal(size=F_G).astype(np.float32), occ=occ,
                           infl=occ & (u - l > 1e-9), l=l, u=u, Ctil=float(u.sum() + 1.0), Crev=float(-l.sum() + 1.0),
                           feasible=bool(rng.random() < 0.85))


def _batch(seed=0, Ns=(3, 5, 3, 5), S=16):
    rng = np.random.default_rng(seed)
    buf = RolloutBuffer(S, list(Ns))
    for e, N in enumerate(Ns):
        for k in range(S):
            o = _fake_obs(rng, N)
            buf.envs[e].store(k, o, rng.normal(size=N).astype(np.float32), rng.normal(), rng.normal(), rng.normal(),
                              k == S - 1 and e % 2 == 0)
        buf.envs[e].compute_gae(0.3, 0.99, 0.95)
    return buf


def _setup(arch="attention", seed=0):
    cfg = load_config(None, SMALL + [f"model.arch={arch}"])
    torch.manual_seed(seed)
    actor, critic = build_networks(cfg)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3, eps=1e-5)
    gen = torch.Generator().manual_seed(seed)
    return cfg, actor, critic, opt, gen


@pytest.mark.parametrize("arch", ["attention", "mlp"])
def test_update_runs_on_mixed_N_and_is_deterministic(arch):
    outs = []
    for _ in range(2):
        cfg, actor, critic, opt, gen = _setup(arch)
        batch = _batch().flatten("cpu", safety=True)
        assert batch["tokens"].shape == (64, 5, F_C)              # padded to N_max = 5
        st = ppo_update(actor, critic, opt, batch, cfg["ppo"], gen)
        assert np.isfinite(st["policy_loss"]) and np.isfinite(st["value_loss"]) and st["epochs_run"] >= 1
        outs.append(torch.cat([p.detach().flatten() for p in list(actor.parameters()) + list(critic.parameters())]))
    assert torch.equal(outs[0], outs[1])


def test_masked_tokens_do_not_affect_actor_loss():
    """Changing the raw action of non-influential tokens or the whole action on fallback steps changes nothing."""
    cfg, actor, critic, opt, gen = _setup()
    batch = _batch(seed=3).flatten("cpu", safety=True)
    b2 = {k: v.clone() for k, v in batch.items()}
    noise = torch.randn_like(b2["a"]) * 5
    mask = (~b2["infl"]) | (~b2["w"].bool())[:, None]
    b2["a"] = torch.where(mask, b2["a"] + noise, b2["a"])

    def loss_and_grad(b):
        a2 = copy.deepcopy(actor)
        mu, ls = a2(b["tokens"], b["glob"], b["occ"])
        lp = (gaussian_logp(b["a"], mu, ls) * b["infl"].float()).sum(-1)
        ratio = torch.exp(lp - b["logp"])
        L = (b["w"].float() * ratio * b["adv"]).sum() / len(ratio)
        L.backward()
        return L.item(), torch.cat([p.grad.flatten() for p in a2.parameters() if p.grad is not None])

    l1, g1 = loss_and_grad(batch)
    l2, g2 = loss_and_grad(b2)
    assert abs(l1 - l2) < 1e-6 and torch.allclose(g1, g2, atol=1e-6)


def test_gae_matches_manual():
    b = EnvBuffer(3, 1)
    b.reward[:] = [1.0, 2.0, 3.0]
    b.value[:] = [0.5, 0.4, 0.3]
    b.done[:] = [False, True, False]
    b.compute_gae(last_value=0.7, gamma=0.9, lam=0.8)
    d2 = 3.0 + 0.9 * 0.7 - 0.3
    d1 = 2.0 - 0.4                                  # terminal after step 1
    d0 = 1.0 + 0.9 * 0.4 - 0.5
    a1 = d1
    a0 = d0 + 0.9 * 0.8 * a1
    np.testing.assert_allclose(b.adv, [a0, a1, d2], rtol=1e-6)
    np.testing.assert_allclose(b.ret, b.adv + b.value, rtol=1e-6)


def test_penalty_methods_use_all_transitions():
    batch = _batch().flatten("cpu", safety=False)
    assert bool(batch["w"].all())
    batch = _batch().flatten("cpu", safety=True)
    assert torch.equal(batch["w"], batch["feasible"])


def test_beta_regulariser_zero_inside_K():
    """With mu inside K(t) the regulariser vanishes; outside it is positive (S 9)."""
    from v2gsafe.safety.projection import project_batched
    x = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float64)
    l = torch.tensor([[-3.0, -3.0, 0.0]], dtype=torch.float64)
    u = torch.tensor([[3.0, 3.0, 1.0]], dtype=torch.float64)
    p, _, _ = project_batched(x, l, u, torch.tensor([10.0]), torch.tensor([10.0]), torch.ones(1, 3, dtype=torch.bool))
    assert torch.equal(p, x)
    p2, _, _ = project_batched(x * 10, l, u, torch.tensor([1.0]), torch.tensor([10.0]), torch.ones(1, 3, dtype=torch.bool))
    assert ((x * 10 - p2) ** 2).sum() > 0 and p2.sum() <= 1.0 + 1e-9


def test_checkpoint_save_load(tmp_path):
    from v2gsafe.algos.ppo import PPOTrainer, load_actor
    cfg = load_config(None, SMALL + ["envs.backend=sync", "smoke=true"])
    tr = PPOTrainer(cfg, tmp_path / "run", seed=3)
    with torch.no_grad():
        for p in tr.actor.parameters():
            p.add_(0.123)
    tr.update, tr.env_steps = 7, 896
    path = tr.save("checkpoint_last.pt")
    tr2 = PPOTrainer(cfg, tmp_path / "run", seed=3, resume=True)
    assert tr2.update == 7 and tr2.env_steps == 896
    for p1, p2 in zip(tr.actor.parameters(), tr2.actor.parameters()):
        assert torch.equal(p1, p2)
    actor, cfg2, ck = load_actor(path)
    assert cfg2["method"] == cfg["method"] and ck["seed"] == 3
    for p1, p2 in zip(tr.actor.parameters(), actor.parameters()):
        assert torch.equal(p1, p2)
