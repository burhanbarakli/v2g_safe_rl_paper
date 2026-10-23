"""Vectorised SafeV2GEnv runner for PPO (algorithm.md Alg. 2: 16 envs, 8 with N=10 and 8 with N=20).

Each env has its own seed stream (numpy SeedSequence of (run seed, env index)); per episode it draws the
profile and rho_tr uniformly (Alg. 7.3) and an episode seed from the training range. Episodes auto-reset.
Backends: 'sync' (in-process, for tests and smoke runs) and 'subproc' (one process per env, 'spawn' context).

EV2Gym (unmodified) cannot build some (seed, date) episodes: its residential load data contains negative values,
and when the 10-house sum of a step is negative the load-forecast noise gets a negative scale and reset raises
'ValueError: scale < 0'. A training episode seed for which this happens is redrawn from the same stream (so runs
that never meet such a seed are unchanged); test/validation replays are fixed and never redrawn.
"""
from __future__ import annotations

import multiprocessing as mp
import traceback
from dataclasses import dataclass

import numpy as np

from .scenarios import SEED_RANGES, ScenarioConfig, TrainMixture
from .wrapper import SafeV2GEnv, WrapperOptions

#: redraws of an unbuildable training episode seed before giving up
MAX_RESET_REDRAWS = 20


@dataclass
class EnvSpec:
    N: int
    index: int
    run_seed: int


class EnvWorker:
    """One env with its own RNG stream and scenario mixture."""

    def __init__(self, spec: EnvSpec, mixture: TrainMixture, options: WrapperOptions):
        self.spec = spec
        self.mixture = mixture
        self.rng = np.random.default_rng(np.random.SeedSequence([int(spec.run_seed), int(spec.index), 1234]))
        self.env = SafeV2GEnv(mixture.sample(spec.N, np.random.default_rng(spec.index)), options)
        self.scfg: ScenarioConfig | None = None
        self.ep_seed = None
        self.n_reset_redraws = 0

    def reset(self):
        self.scfg = self.mixture.sample(self.spec.N, self.rng)
        lo, hi = SEED_RANGES["train"]
        for _ in range(MAX_RESET_REDRAWS):
            self.ep_seed = int(self.rng.integers(lo, hi))
            try:
                return self.env.reset(self.ep_seed, self.scfg)
            except ValueError as e:
                if "scale < 0" not in str(e):
                    raise
                self.n_reset_redraws += 1
                print(f"[vec_env] env {self.spec.index}: EV2Gym cannot build episode seed {self.ep_seed} "
                      f"({self.scfg.cfg_id}): {e}; redrawing", flush=True)
        raise RuntimeError(f"env {self.spec.index}: {MAX_RESET_REDRAWS} unbuildable episode seeds in a row")

    def step(self, x):
        out = self.env.step(x)
        ep = None
        if out.done:
            ep = self.env.episode_metrics()
            ep.update(test_cfg_id=self.scfg.cfg_id, profile=self.scfg.profile, N=self.spec.N,
                      rho_tr=self.scfg.rho_tr, episode_seed=self.ep_seed)
            obs = self.reset()
        else:
            obs = self.env.observe()
        return out, obs, ep

    def get_state(self):
        return self.rng.bit_generator.state

    def set_state(self, st):
        self.rng.bit_generator.state = st


def _worker_main(remote, spec, mixture, options):
    try:
        w = EnvWorker(spec, mixture, options)
        remote.send(("ok", None))
    except Exception:  # noqa: BLE001
        remote.send(("error", traceback.format_exc()))
        return
    while True:
        cmd, data = remote.recv()
        try:
            if cmd == "reset":
                remote.send(("ok", w.reset()))
            elif cmd == "step":
                remote.send(("ok", w.step(data)))
            elif cmd == "get_state":
                remote.send(("ok", w.get_state()))
            elif cmd == "set_state":
                w.set_state(data)
                remote.send(("ok", None))
            elif cmd == "close":
                remote.send(("ok", None))
                break
        except Exception:  # noqa: BLE001
            remote.send(("error", traceback.format_exc()))


class VecEnv:
    def __init__(self, specs: list[EnvSpec], mixture: TrainMixture, options: WrapperOptions, backend: str = "sync"):
        self.specs = specs
        self.Ns = [s.N for s in specs]
        self.backend = backend
        if backend == "sync":
            self.workers = [EnvWorker(s, mixture, options) for s in specs]
        elif backend == "subproc":
            ctx = mp.get_context("spawn")
            self.remotes, self.procs = [], []
            for s in specs:
                a, b = ctx.Pipe()
                p = ctx.Process(target=_worker_main, args=(b, s, mixture, options), daemon=True)
                p.start()
                self.remotes.append(a)
                self.procs.append(p)
            for r in self.remotes:
                self._check(r.recv())
        else:
            raise ValueError(backend)

    @staticmethod
    def _check(msg):
        status, data = msg
        if status != "ok":
            raise RuntimeError(f"env worker failed:\n{data}")
        return data

    def _call(self, cmd, datas=None):
        if self.backend == "sync":
            fn = {"reset": lambda w, d: w.reset(), "step": lambda w, d: w.step(d),
                  "get_state": lambda w, d: w.get_state(), "set_state": lambda w, d: w.set_state(d)}[cmd]
            return [fn(w, None if datas is None else datas[i]) for i, w in enumerate(self.workers)]
        for i, r in enumerate(self.remotes):
            r.send((cmd, None if datas is None else datas[i]))
        return [self._check(r.recv()) for r in self.remotes]

    def reset(self):
        return self._call("reset")

    def step(self, xs):
        return self._call("step", xs)

    def get_states(self):
        return self._call("get_state")

    def set_states(self, states):
        self._call("set_state", states)

    def close(self):
        if self.backend == "subproc":
            for r in self.remotes:
                try:
                    r.send(("close", None))
                    r.recv()
                except Exception:  # noqa: BLE001
                    pass
            for p in self.procs:
                p.join(timeout=5)
