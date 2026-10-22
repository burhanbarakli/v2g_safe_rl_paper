"""Split-conformal lower bound on the departure step (DD12, DD23; safety_layer.md Sec. 8; algorithm.md Alg. 4).

Per configuration c and level alpha:

1. predictor q_hat: lower-alpha quantile regressor of d on w = [arrival step mod N_day, weekday,
   E_arr/E_cap, (E_des - E_arr)/E_ref, EV spec class one-hot], fitted on all sessions of n_train episodes;
2. calibration: n_cal independent episodes, ONE session per episode chosen uniformly; scores
   S_j = q_hat(w_j) - d_j; k = ceil((n_cal+1)(1-alpha)); Q = k-th smallest score (+inf if k > n_cal);
3. bound: d_lo = max(t_arr, ceil(q_hat(w) - Q)).

Sessions come from EV2Gym's own arrival model (:func:`v2gsafe.envs.scenarios.sample_sessions`) with seed
ranges disjoint from training/validation/test. The default predictor is
``GradientBoostingRegressor(loss='quantile', alpha=alpha)`` ('gbr'); ``HistGradientBoostingRegressor`` ('hgb',
can be slow when several OpenMP runtimes are loaded) and linear quantile regression ('linear') are options.

CLI::

    python -m v2gsafe.safety.conformal --grid configs/test_grid.yaml --alpha 0.01 0.05 0.1 0.2 --out results/conformal
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..constants import E_REF
from ..specs import spec_classes


# ------------------------------------------------------------------------------------- features
def session_features(ev, t_arr: int, arr_dt: datetime.datetime, dt_min: int, classes=None) -> np.ndarray:
    """Predictor features w of one session (Alg. 4)."""
    classes = classes if classes is not None else spec_classes()
    n_day = int(round(1440 / dt_min))
    E_arr = float(ev.battery_capacity_at_arrival)
    cap = float(ev.battery_capacity)
    onehot = np.zeros(len(classes))
    key = (float(ev.battery_capacity), float(ev.max_ac_charge_power))
    if key in classes:
        onehot[classes.index(key)] = 1.0
    return np.concatenate([[t_arr % n_day, arr_dt.weekday(), E_arr / cap,
                            (float(ev.desired_capacity) - E_arr) / E_REF], onehot])


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """k-th smallest score with k = ceil((n+1)(1-alpha)); +inf if k > n (Alg. 4)."""
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    k = math.ceil((n + 1) * (1.0 - alpha))
    return float(s[k - 1]) if k <= n else math.inf


def make_predictor(kind: str, alpha: float, seed: int = 0):
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(loss="quantile", quantile=alpha, max_iter=200, learning_rate=0.1,
                                             max_leaf_nodes=31, random_state=seed)
    if kind == "gbr":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(loss="quantile", alpha=alpha, n_estimators=200, max_depth=3,
                                         learning_rate=0.1, random_state=seed)
    if kind == "linear":
        from sklearn.linear_model import QuantileRegressor
        return QuantileRegressor(quantile=alpha, alpha=0.0, solver="highs")
    raise ValueError(kind)


@dataclass
class ConformalBound:
    """Calibrated lower bound provider for one (configuration, alpha)."""

    cfg_id: str
    alpha: float
    model: object
    Q: float
    dt_min: int
    n_train_sessions: int = 0
    n_cal: int = 0
    uncalibrated: bool = False      # SA-Attn-point ablation: Q_hat = 0
    classes: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def predict(self, W: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(np.atleast_2d(W)), float)

    def bound_from_features(self, w: np.ndarray, t_arr: int) -> int:
        Q = 0.0 if self.uncalibrated else self.Q
        if not np.isfinite(Q):
            return int(t_arr)
        y = float(self.predict(w)[0]) - Q
        return max(int(t_arr), int(math.ceil(y - 1e-9)))

    def lower_bound(self, ev, t_arr: int, arr_dt, scfg=None) -> int:
        """d_lo for an arriving EV (called by the wrapper; d is NOT read)."""
        w = session_features(ev, t_arr, arr_dt, self.dt_min, self.classes or None)
        return self.bound_from_features(w, t_arr)

    def save(self, path) -> None:
        """Pickle a plain dict (not the class) so files written by the CLI load from any module path."""
        import dataclasses
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({f_.name: getattr(self, f_.name) for f_ in dataclasses.fields(self)}, f)
        os.replace(tmp, path)

    @staticmethod
    def load(path) -> "ConformalBound":
        with open(path, "rb") as f:
            d = pickle.load(f)
        return ConformalBound(**d)


# ----------------------------------------------------------------------------------- calibration
def _sessions_dataset(env, scfg, seeds, one_per_episode: bool, classes):
    """Feature matrix and departure steps from sessions of the given episode seeds."""
    from ..envs.scenarios import sample_sessions

    W, d = [], []
    for s in seeds:
        evs = sample_sessions(env, int(s))
        if one_per_episode:
            evs = [evs[int(np.random.default_rng(int(s) ^ 0xC0FFEE).integers(len(evs)))]]
        start = env.sim_starting_date
        for ev in evs:
            t_arr = int(ev.time_of_arrival)
            arr_dt = start + datetime.timedelta(minutes=scfg.dt_min * t_arr)
            W.append(session_features(ev, t_arr, arr_dt, scfg.dt_min, classes))
            d.append(int(ev.time_of_departure))
    return np.asarray(W), np.asarray(d, float)


def _dataset_worker(args):
    scfg_d, seeds, one = args
    from ..envs.scenarios import ScenarioConfig
    from ..envs.wrapper import make_ev2gym_env
    scfg = ScenarioConfig.from_dict(scfg_d)
    return _sessions_dataset(make_ev2gym_env(scfg), scfg, seeds, one, spec_classes())


def _parallel_dataset(env, scfg, seeds, one: bool, classes, workers: int):
    """Session data set, optionally split over ``workers`` processes (each builds its own EV2Gym env, ~4 s).
    The result is identical to the serial version (same seeds, concatenated in seed order)."""
    if workers <= 1 or len(seeds) < 4 * workers:
        return _sessions_dataset(env, scfg, seeds, one, classes)
    import multiprocessing as mp
    chunks = [list(c) for c in np.array_split(np.asarray(seeds, dtype=np.int64), workers) if len(c)]
    with mp.get_context("spawn").Pool(workers) as pool:
        parts = pool.map(_dataset_worker, [(scfg.to_dict(), [int(s) for s in c], one) for c in chunks])
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def sessions_cache_path(cache_dir, cfg_id: str, n_train: int, n_cal: int) -> Path:
    return Path(cache_dir) / f"{cfg_id}_sessions_t{n_train}_c{n_cal}.npz"


def calibrate_multi(scfg, alphas, n_train: int = 2000, n_cal: int = 5000, predictor: str = "gbr",
                    env=None, verbose: bool = False, workers: int = 1,
                    cache_dir=None) -> list[ConformalBound]:
    """Fit q_hat on S_train(c) and calibrate Q on S_cal(c) for each alpha (Alg. 4).

    The session data sets do not depend on alpha and are generated once (optionally in ``workers`` processes and
    cached as ``<cache_dir>/<cfg_id>_sessions_t<n>_c<n>.npz``, so a later calibration at another alpha or with
    another predictor skips the expensive session generation); one predictor per (c, alpha)."""
    from ..envs.scenarios import cal_seed
    from ..envs.wrapper import make_ev2gym_env

    t0 = time.time()
    classes = spec_classes()
    cpath = sessions_cache_path(cache_dir, scfg.cfg_id, n_train, n_cal) if cache_dir is not None else None
    if cpath is not None and cpath.exists():
        z = np.load(cpath)
        W, d, Wc, dc = z["W"], z["d"], z["Wc"], z["dc"]
    else:
        env = env or (make_ev2gym_env(scfg) if workers <= 1 else None)
        W, d = _parallel_dataset(env, scfg, [cal_seed(scfg.cfg_id, i, "cal_train") for i in range(n_train)],
                                 False, classes, workers)
        Wc, dc = _parallel_dataset(env, scfg, [cal_seed(scfg.cfg_id, j, "cal") for j in range(n_cal)], True,
                                   classes, workers)
        if cpath is not None:
            cpath.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cpath, W=W, d=d, Wc=Wc, dc=dc)
    t_data = time.time() - t0
    out = []
    for alpha in alphas:
        t1 = time.time()
        model = make_predictor(predictor, alpha)
        model.fit(W, d)
        scores = np.asarray(model.predict(Wc), float) - dc
        Q = conformal_quantile(scores, alpha)
        cb = ConformalBound(cfg_id=scfg.cfg_id, alpha=float(alpha), model=model, Q=Q, dt_min=scfg.dt_min,
                            n_train_sessions=len(d), n_cal=len(dc), classes=classes,
                            meta={"predictor": predictor, "data_seconds": t_data, "fit_seconds": time.time() - t1,
                                  "cal_coverage_in_sample": float(np.mean(scores <= Q))})
        out.append(cb)
        if verbose:
            print(f"{scfg.cfg_id} alpha={alpha}: {len(d)} training sessions, n_cal={len(dc)}, Q={Q:.3f}, "
                  f"data {t_data:.1f}s, fit {time.time() - t1:.1f}s", flush=True)
    return out


def calibrate(scfg, alpha: float, n_train: int = 2000, n_cal: int = 5000, predictor: str = "gbr",
              env=None, verbose: bool = False) -> ConformalBound:
    """Single-alpha version of :func:`calibrate_multi`."""
    return calibrate_multi(scfg, [alpha], n_train, n_cal, predictor, env, verbose)[0]


def calibration_path(cache_dir, cfg_id: str, alpha: float) -> Path:
    return Path(cache_dir) / f"{cfg_id}_a{alpha:g}.pkl"


class ConformalProvider:
    """Callable ScenarioConfig -> ConformalBound, with an on-disk cache (one calibration per config and alpha)."""

    def __init__(self, alpha: float, cache_dir, n_train: int = 2000, n_cal: int = 5000, predictor: str = "gbr",
                 uncalibrated: bool = False, build_missing: bool = True):
        self.alpha = float(alpha)
        self.cache_dir = Path(cache_dir)
        self.n_train, self.n_cal, self.predictor = n_train, n_cal, predictor
        self.uncalibrated = uncalibrated
        self.build_missing = build_missing
        self._mem: dict[str, ConformalBound] = {}

    def __call__(self, scfg) -> ConformalBound:
        key = scfg.cfg_id
        if key not in self._mem:
            path = calibration_path(self.cache_dir, key, self.alpha)
            if path.exists():
                cb = ConformalBound.load(path)
            elif self.build_missing:
                cb = calibrate(scfg, self.alpha, self.n_train, self.n_cal, self.predictor)
                cb.save(path)
            else:
                raise FileNotFoundError(f"missing calibration {path}; run the 'conformal' stage first")
            if self.uncalibrated:
                cb.uncalibrated = True
            self._mem[key] = cb
        return self._mem[key]


def _main(argv=None):
    from ..envs.scenarios import ScenarioConfig, TrainMixture, load_test_grid

    ap = argparse.ArgumentParser(description="conformal calibration (Alg. 4)")
    ap.add_argument("--grid", help="test grid YAML (calibrate every test configuration)")
    ap.add_argument("--train-N", type=int, nargs="*", default=[], help="also calibrate training configs at these N")
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.2])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-cal", type=int, default=5000)
    ap.add_argument("--predictor", default="gbr", choices=["hgb", "gbr", "linear"])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--scenario-json", action="append", default=[], help="explicit ScenarioConfig as JSON")
    ap.add_argument("--workers", type=int, default=1, help="processes for session generation")
    ap.add_argument("--no-session-cache", action="store_true")
    args = ap.parse_args(argv)
    cfgs = [s for s, _ in load_test_grid(args.grid)] if args.grid else []
    cfgs += TrainMixture().all_configs(args.train_N)
    cfgs += [ScenarioConfig.from_dict(json.loads(js)) for js in args.scenario_json]
    for scfg in cfgs:
        if args.only and scfg.cfg_id not in args.only:
            continue
        todo = [a for a in args.alpha if not calibration_path(args.out, scfg.cfg_id, a).exists()]
        if not todo:
            print(f"exists: {scfg.cfg_id} (all alphas)")
            continue
        for cb in calibrate_multi(scfg, todo, args.n_train, args.n_cal, args.predictor, verbose=True,
                                  workers=args.workers,
                                  cache_dir=None if args.no_session_cache else Path(args.out) / "sessions"):
            cb.save(calibration_path(args.out, scfg.cfg_id, cb.alpha))


if __name__ == "__main__":
    _main()
