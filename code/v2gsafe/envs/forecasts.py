"""Noisy forecasts of inflexible load and PV for the policy observation (DD6; algorithm.md Alg. 7.2).

The forecast for step t+k made at step t is the true value plus Gaussian noise with standard deviation
``rel_std`` (10 %) of the value, clipped at 0 (both quantities are non-negative). Noise is drawn
independently for every (t, k) from the wrapper's own RNG, seeded per episode (EV2Gym's own forecasts use
the global ``np.random`` and have odd mean semantics, ev2gym_notes Sec. 5 / Q6).
Beyond the episode end (t + k > T - 1) the last true value is held before the noise is applied.
"""
from __future__ import annotations

import numpy as np

from ..constants import H_F


class NoisyForecaster:
    def __init__(self, rel_std: float = 0.1, horizon: int = H_F, seed: int | None = None):
        self.rel_std = float(rel_std)
        self.horizon = int(horizon)
        self.rng = np.random.default_rng(seed)

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def forecast(self, series: np.ndarray, t: int) -> np.ndarray:
        """Forecasts of series[t+1 .. t+horizon] (length ``horizon``)."""
        T = len(series)
        idx = np.minimum(np.arange(t + 1, t + 1 + self.horizon), T - 1)
        true = np.asarray(series, float)[idx]
        noise = self.rng.standard_normal(self.horizon)
        return np.maximum(0.0, true * (1.0 + self.rel_std * noise))
