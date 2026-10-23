"""EnvWorker: unbuildable EV2Gym training episodes ('scale < 0' at reset) are redrawn from the worker's seed stream."""
from __future__ import annotations

import numpy as np
import pytest

from v2gsafe.envs.scenarios import TrainMixture
from v2gsafe.envs.vec_env import MAX_RESET_REDRAWS, EnvSpec, EnvWorker


class _FakeEnv:
    def __init__(self, bad: set[int], msg: str = "scale < 0"):
        self.bad, self.msg, self.calls = bad, msg, []

    def reset(self, seed, scfg):
        self.calls.append(seed)
        if seed in self.bad or "*" in self.bad:
            raise ValueError(self.msg)
        return ("obs", seed)


def _worker(env) -> EnvWorker:
    w = object.__new__(EnvWorker)
    w.spec, w.mixture, w.env = EnvSpec(N=10, index=3, run_seed=0), TrainMixture(), env
    w.rng = np.random.default_rng(np.random.SeedSequence([0, 3, 1234]))
    w.scfg, w.ep_seed, w.n_reset_redraws = None, None, 0
    return w


def test_redraw_skips_unbuildable_seed_and_keeps_the_stream():
    ref = _worker(_FakeEnv(set()))
    seeds = [ref.reset()[1] for _ in range(4)]
    w = _worker(_FakeEnv({seeds[1]}))
    got = [w.reset()[1] for _ in range(3)]
    assert w.n_reset_redraws == 1
    assert got[0] == seeds[0] and got[1] != seeds[1] and seeds[1] not in got
    assert w.ep_seed == got[-1]


def test_redraw_only_for_scale_error_and_bounded():
    with pytest.raises(ValueError, match="other"):
        _worker(_FakeEnv({"*"}, msg="other")).reset()
    w = _worker(_FakeEnv({"*"}))
    with pytest.raises(RuntimeError, match="unbuildable"):
        w.reset()
    assert w.n_reset_redraws == MAX_RESET_REDRAWS
