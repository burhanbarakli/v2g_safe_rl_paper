#!/usr/bin/env python
"""Indicative timings: one wrapper env step (EV2Gym step + safety layer + observation) and one PPO update
(2048 transitions, default network) at N = 10, 50, 200. Results are machine-dependent; not experiment results.

    python scripts/benchmark.py [--N 10 50 200] [--epochs 10] [--device cpu]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v2gsafe.algos.buffers import RolloutBuffer  # noqa: E402
from v2gsafe.algos.ppo import build_networks, ppo_update  # noqa: E402
from v2gsafe.config import load_config  # noqa: E402
from v2gsafe.constants import F_C, F_G  # noqa: E402


def bench_env(N: int, steps: int = 112, seed: int = 1) -> dict:
    from v2gsafe.envs.scenarios import ScenarioConfig
    from v2gsafe.envs.wrapper import SafeV2GEnv, WrapperOptions
    scfg = ScenarioConfig(profile="workplace", N=N, rho_tr=4.0)
    out = {}
    for adm in (False, True):
        env = SafeV2GEnv(scfg, WrapperOptions(admission=adm))
        rng = np.random.default_rng(0)
        env.reset(seed, scfg)
        t_obs = t_step = 0.0
        n = 0
        while n < steps:
            t0 = time.perf_counter()
            if n:
                env.observe()
            t1 = time.perf_counter()
            res = env.step(rng.normal(0, 1, N) * 11)
            t2 = time.perf_counter()
            t_obs += t1 - t0
            t_step += t2 - t1
            n += 1
            if res.done:
                env.reset(seed + n, scfg)
        out["adm" if adm else "noadm"] = {"observe_ms": 1000 * t_obs / n, "step_ms": 1000 * t_step / n}
    return out


def bench_ppo(N: int, epochs: int, device: str, minibatch: int = 512) -> dict:
    cfg = load_config(None, [f"ppo.epochs={epochs}", "ppo.target_kl=null", f"ppo.minibatch={minibatch}"])
    torch.manual_seed(0)
    actor, critic = build_networks(cfg)
    actor.to(device)
    critic.to(device)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4, eps=1e-5)
    rng = np.random.default_rng(0)
    S, E = 128, 16
    buf = RolloutBuffer(S, [N] * E)
    for b in buf.envs:
        for k in range(S):
            occ = rng.random(N) < 0.25
            l = np.where(occ, rng.uniform(-5, 3, N), 0.0)
            u = np.where(occ, l + rng.uniform(0, 8, N), 0.0)
            o = SimpleNamespace(tokens=(rng.normal(size=(N, F_C)) * occ[:, None]).astype(np.float32),
                                glob=rng.normal(size=F_G).astype(np.float32), occ=occ, infl=occ, l=l, u=u,
                                Ctil=float(u.sum() * 0.5), Crev=float(-l.sum() + 10), feasible=True)
            b.store(k, o, rng.normal(size=N), 0.0, 0.0, rng.normal(), False)
        b.compute_gae(0.0, 0.99, 0.95)
    batch = buf.flatten(device, True)
    t0 = time.perf_counter()
    ppo_update(actor, critic, opt, batch, cfg["ppo"], torch.Generator().manual_seed(0))
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return {"update_s": time.perf_counter() - t0, "epochs": epochs, "minibatch": minibatch}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", default=[10, 50, 200])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-ppo", action="store_true")
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--skip-env", action="store_true")
    a = ap.parse_args(argv)
    print(f"torch threads: {torch.get_num_threads()}, device {a.device}")
    for N in a.N:
        line = f"N={N:4d}"
        if not a.skip_env:
            e = bench_env(N)
            line += (f": env observe+bounds {e['noadm']['observe_ms']:.2f} ms, step (layer+EV2Gym) "
                     f"{e['noadm']['step_ms']:.2f} ms; with admission control observe {e['adm']['observe_ms']:.2f} ms")
        if not a.skip_ppo:
            p = bench_ppo(N, a.epochs, a.device, a.minibatch)
            line += (f"; PPO update (2048 transitions, {p['epochs']} epochs, minibatch {p['minibatch']}) "
                     f"{p['update_s']:.1f} s")
        print(line, flush=True)


if __name__ == "__main__":
    main()
