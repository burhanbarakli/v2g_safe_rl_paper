"""Heuristic baselines (algorithm.md Alg. 5 rows 5a-5d) and the controller interface used by the evaluator.

Every controller maps an :class:`~v2gsafe.envs.wrapper.Observation` to a raw action x (kW per charger). The
wrapper then applies either the safety layer (variants ``*_safe`` = "+Pi": projection, fallback, admission
control) or only the per-EV physical clip p = clip(x, p_min, u) (raw variants); actuation is identical.

* AFAP: x_i = u_i (charge at maximum).
* ALAP: x_i = max(0, l_i^G2V), l^G2V = (S11) with p_min := 0 (our implementation; EV2Gym's ignores eta and taper).
* Round Robin (G2V): rotating pointer over chargers; occupied chargers in pointer order get u_i until C_tilde(t)
  is used (the last one partially); the pointer advances by one per step.
* LLF (G2V): allocate C_tilde(t) in increasing laxity chi_i, u_i each.
* sLLF (G2V, price-unaware; smoothed least-laxity-first [chen2022smoothed], DD33(5)): maximise the minimum (a
  concave sum) of next-slot laxities chi_i' = chi_i^0 + p_i/P_nom_i, chi_i^0 = (E_i - B_hat_i(tau-1))/(P_nom dt):
  water-filling p_i = clip(P_nom_i (L - chi_i^0), 0, u_i) with a common level L found by bisection such that
  sum p = max(0, C_tilde) (or p = u if the station is not capacity-bound).
* price-LLF (DD33(6)): charge (LLF allocation of C_tilde) when the current price is below the running median of
  the episode's prices seen so far and the known day-ahead window, discharge at p_min when it is above.
"""
from __future__ import annotations

import numpy as np


class Controller:
    """Interface: ``reset(env)`` after every env reset, ``act(obs, env) -> x`` (kW, length N)."""

    name = "controller"
    uses_future_info = False

    def reset(self, env) -> None:  # noqa: D401
        pass

    def act(self, obs, env) -> np.ndarray:
        raise NotImplementedError


def _fill(order: np.ndarray, u: np.ndarray, budget: float) -> np.ndarray:
    """Give u_i to chargers in ``order`` until ``budget`` is used (last one partially)."""
    x = np.zeros_like(u)
    b = max(0.0, float(budget))
    for i in order:
        if b <= 0:
            break
        give = min(u[i], b)
        x[i] = give
        b -= give
    return x


class AFAP(Controller):
    name = "afap"

    def act(self, obs, env):
        return np.where(obs.occ, obs.u, 0.0)


class ALAP(Controller):
    name = "alap"

    def act(self, obs, env):
        return np.where(obs.occ, np.maximum(0.0, obs.lg2v), 0.0)


class RoundRobin(Controller):
    name = "rr"

    def reset(self, env):
        self.ptr = 0

    def act(self, obs, env):
        N = len(obs.u)
        idx = (self.ptr + np.arange(N)) % N
        order = idx[obs.occ[idx]]
        x = _fill(order, obs.u, obs.Ctil)
        self.ptr = (self.ptr + 1) % N
        return x


class LLF(Controller):
    name = "llf"

    def act(self, obs, env):
        occ = np.flatnonzero(obs.occ)
        order = occ[np.lexsort((occ, obs.chi[occ]))]       # laxity ascending, index as tie-break
        return _fill(order, obs.u, obs.Ctil)


class SmoothedLLF(Controller):
    """Smoothed least-laxity-first (sLLF) [chen2022smoothed] (DD33(5); lit/parts/L1_notes.md Sec. C).

    Paper rule: r_i = [rbar_i (L(t) - l_i(t) + 1)] clipped to [0, min(rbar_i, e_i)], with the common level L(t) found
    by bisection so that sum_i r_i = min(P(t), sum of caps). Our instantiation: rbar_i = Pnom_i, laxity measured
    against the certified thresholds, l_i - 1 = chi0_i = (E_i - B_hat_i(tau_i - 1)) / (Pnom_i dt) (next-slot laxity,
    the "+1" is absorbed into L), cap = u_i (one-step reachable power, incl. taper), P(t) = max(0, C_tilde(t)).
    If sum u_i <= P(t) every EV gets u_i. Charging only (sLLF has no V2G); the bisection keeps the feasible end
    (sum r_i <= P(t)). The '_safe' variant passes r through the safety layer, the raw variant through the clip.
    """

    name = "sllf"

    def act(self, obs, env):
        x = np.zeros_like(obs.u)
        occ = np.flatnonzero(obs.occ)
        if occ.size == 0:
            return x
        u = obs.u[occ]
        budget = max(0.0, float(obs.Ctil))
        if u.sum() <= budget:
            x[occ] = u
            return x
        t = obs.t
        pn = np.array([env.slots[i].params.pnom for i in occ])
        bt1 = np.array([env.slots[i].B[max(1, env.slots[i].d_ctrl - t + 1) - 1] for i in occ])
        chi0 = (obs.E[occ] - bt1) / (pn * env.dt_h)
        rate = lambda L: np.clip(pn * (L - chi0), 0.0, u)  # noqa: E731
        lo, hi = float(chi0.min()), float((chi0 + u / pn).max()) + 1.0
        for _ in range(100):                       # keep the feasible end: sum rate(lo) <= budget
            mid = 0.5 * (lo + hi)
            if rate(mid).sum() <= budget:
                lo = mid
            else:
                hi = mid
        x[occ] = rate(lo)
        return x


class PriceLLF(Controller):
    """Price-aware LLF (DD33(5)): LLF filling when the current charge price is below the median of the prices seen
    so far plus the 8-step price forecast, maximum discharge (p_min) when above, idle when equal. With '_safe' the
    safety layer restores the departure guarantee and the transformer band."""

    name = "price_llf"

    def reset(self, env):
        self.history: list[float] = []

    def act(self, obs, env):
        from ..constants import C_REF
        c_now = float(obs.glob[0]) * C_REF
        self.history.append(c_now)
        window = (np.asarray(obs.glob[2:10], float) * C_REF).tolist()
        med = float(np.median(self.history + window))
        if c_now < med:
            return LLF().act(obs, env)
        if c_now > med:
            return np.where(obs.occ, obs.pmin, 0.0)
        return np.zeros_like(obs.u)


class RandomRaw(Controller):
    """Random raw actions (tests / smoke only): x ~ N(0, (scale * P_ref)^2)."""

    name = "random"

    def __init__(self, scale: float = 1.5, seed: int = 0, p_ref: float = 11.0):
        self.scale, self.seed, self.p_ref = scale, seed, p_ref

    def reset(self, env):
        self.rng = np.random.default_rng([self.seed, int(env.seed or 0)])

    def act(self, obs, env):
        return self.rng.normal(0.0, self.scale, len(obs.u)) * self.p_ref


HEURISTICS = {"afap": AFAP, "alap": ALAP, "rr": RoundRobin, "llf": LLF, "sllf": SmoothedLLF, "price_llf": PriceLLF,
              "random": RandomRaw}
