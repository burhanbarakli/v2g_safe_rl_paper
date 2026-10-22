"""Experiment scenarios (algorithm.md Alg. 7, DD24): EV2Gym configurations, seed ranges, test grid, test replays.

* :class:`ScenarioConfig` - one configuration (profile, N, rho_tr, dt, desired SoC, ...) and its EV2Gym YAML
  dict (:meth:`ScenarioConfig.ev2gym_config`).
* Profiles (Alg. 7.1): workplace 05:00 / multiplier 5 / weekdays; public 12:00 / 4 / both;
  private 12:00 / 2.5 / both. T = 28 h (112 steps at 15 min, 336 at 5 min).
* Scaling (Alg. 7.2): P_tr = rho_tr * N, inflexible load and PV multipliers 0.4, one port per charger,
  heterogeneous specs from our constant-efficiency JSON (DD4), transition_soc_multiplier 5 (DD17),
  ``empty_ports_at_end_of_simulation=True``.
* Seeds: disjoint ranges for training, test, validation and conformal calibration (Alg. 4, Alg. 6).
* Test replays are defined by (configuration, seed); :func:`build_test_set` resets the env once per replay and
  stores a fingerprint (hash of sessions, prices, loads, PV) in a JSON manifest, so that every method and
  every machine evaluates on identical episodes (common random numbers). EV2Gym replay pickles are not needed.

CLI::

    python -m v2gsafe.envs.scenarios build-test-sets --grid configs/test_grid.yaml --out results/test_sets
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime
import hashlib
import json
import os
import random
import tempfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

from ..specs import SPEC_PATH

#: Per-profile start hour, spawn multiplier and simulated days (Alg. 7.1, DD24).
PROFILES: dict[str, dict] = {
    "workplace": {"hour": 5, "spawn_multiplier": 5.0, "simulation_days": "weekdays"},
    "public": {"hour": 12, "spawn_multiplier": 4.0, "simulation_days": "both"},
    "private": {"hour": 12, "spawn_multiplier": 2.5, "simulation_days": "both"},
}

#: Seed ranges (all < 2**32 as required by ``np.random.seed``).
SEED_RANGES = {
    "train": (0, 1_000_000_000),
    "test": (1_000_000_000, 2_000_000_000),
    "val": (2_000_000_000, 2_500_000_000),
    "cal_train": (2_500_000_000, 3_000_000_000),
    "cal": (3_000_000_000, 3_500_000_000),
}


def _crc(s: str) -> int:
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def replay_seed(test_cfg_id: str, replay_id: int) -> int:
    """Seed of replay ``replay_id`` (< 10,000) of a test configuration (range 'test')."""
    assert 0 <= replay_id < 10_000
    return SEED_RANGES["test"][0] + (_crc(test_cfg_id) % 99_000) * 10_000 + int(replay_id)


def val_seed(cfg_id: str, i: int) -> int:
    assert 0 <= i < 100_000
    return SEED_RANGES["val"][0] + (_crc(cfg_id) % 4_000) * 100_000 + int(i)


def cal_seed(cfg_id: str, i: int, split: str = "cal") -> int:
    """Seeds of the conformal predictor-training ('cal_train') and calibration ('cal') episodes."""
    assert split in ("cal", "cal_train") and 0 <= i < 100_000
    return SEED_RANGES[split][0] + (_crc(cfg_id) % 4_000) * 100_000 + int(i)


# ---------------------------------------------------------------------------------------- config
@dataclass(frozen=True)
class ScenarioConfig:
    """One scenario configuration (Alg. 7). Defaults are the main experiment settings."""

    profile: str = "workplace"
    N: int = 10
    rho_tr: float = 4.0
    dt_min: int = 15
    T: int | None = None                  # default: 28 h
    desired_soc: float = 1.0              # EV2Gym ev.desired_capacity (DD2: 1.0 main, 0.8 sensitivity)
    loads: bool = True
    load_mult: float = 0.4
    pv: bool = True
    pv_mult: float = 0.4
    hour: int | None = None               # default: profile start hour
    spawn_multiplier: float | None = None  # default: profile multiplier
    simulation_days: str | None = None    # default: profile days
    min_time_of_stay: int = 180           # minutes
    transition_soc_multiplier: float = 5.0  # DD17
    p_tr: float | None = None             # absolute rating (kW); overrides rho_tr * N (literature config)
    discharge_price_factor: float = 1.0   # EV2Gym discharge_price_factor (DD39 sensitivity: 0.9)
    throughput_cost: float = 0.0          # EUR per kWh of battery throughput, in reward and evaluation (DD39: 0.05)
    name: str | None = None               # explicit configuration id

    def __post_init__(self):
        if self.profile not in PROFILES:
            raise ValueError(f"unknown profile {self.profile!r}; choose from {sorted(PROFILES)}")

    # -- derived quantities
    @property
    def horizon(self) -> int:
        return int(self.T) if self.T is not None else int(round(28 * 60 / self.dt_min))

    @property
    def start_hour(self) -> int:
        return int(self.hour if self.hour is not None else PROFILES[self.profile]["hour"])

    @property
    def multiplier(self) -> float:
        return float(self.spawn_multiplier if self.spawn_multiplier is not None
                     else PROFILES[self.profile]["spawn_multiplier"])

    @property
    def days(self) -> str:
        return self.simulation_days or PROFILES[self.profile]["simulation_days"]

    @property
    def P_tr(self) -> float:
        return float(self.p_tr) if self.p_tr is not None else float(self.rho_tr) * int(self.N)

    @property
    def rho(self) -> float:
        return self.P_tr / int(self.N)

    @property
    def dt_h(self) -> float:
        return self.dt_min / 60.0

    @property
    def cfg_id(self) -> str:
        """Configuration id, e.g. ``workplace_N50_rho4`` (code_spec Sec. 3.1, test_cfg_id)."""
        if self.name:
            return self.name
        s = f"{self.profile}_N{self.N}_rho{self.rho:g}"
        if self.dt_min != 15:
            s += f"_dt{self.dt_min}"
        if self.desired_soc != 1.0:
            s += f"_soc{self.desired_soc:g}"
        if self.T is not None and self.T != int(round(28 * 60 / self.dt_min)):
            s += f"_T{self.T}"
        if self.discharge_price_factor != 1.0:
            s += f"_dpf{self.discharge_price_factor:g}"
        if self.throughput_cost != 0.0:
            s += f"_thr{self.throughput_cost:g}"
        if not self.loads or not self.pv:
            s += "_noloadpv" if (not self.loads and not self.pv) else ("_noload" if not self.loads else "_nopv")
        return s

    def replace(self, **kw) -> "ScenarioConfig":
        return dataclasses.replace(self, **kw)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ScenarioConfig":
        names = {f.name for f in dataclasses.fields(ScenarioConfig)}
        unknown = set(d) - names - {"replays"}
        if unknown:
            raise ValueError(f"unknown scenario keys {sorted(unknown)}")
        return ScenarioConfig(**{k: v for k, v in d.items() if k in names})

    # -- EV2Gym configuration
    def ev2gym_config(self, spec_path: str | os.PathLike | None = None) -> dict:
        """EV2Gym YAML dictionary for this scenario (derived from EV2Gym's V2GProfitMax.yaml)."""
        c = copy.deepcopy(_EV2GYM_BASE)
        c.update(timescale=int(self.dt_min), simulation_length=self.horizon, hour=self.start_hour, minute=0,
                 random_day=True, random_hour=False, simulation_days=self.days, scenario=self.profile,
                 spawn_multiplier=self.multiplier, number_of_charging_stations=int(self.N),
                 ev_specs_file=str(spec_path or SPEC_PATH))
        c["transformer"]["max_power"] = self.P_tr
        c["discharge_price_factor"] = float(self.discharge_price_factor)
        c["inflexible_loads"].update(include=bool(self.loads),
                                     inflexible_loads_capacity_multiplier_mean=float(self.load_mult))
        c["solar_power"].update(include=bool(self.pv), solar_power_capacity_multiplier_mean=float(self.pv_mult))
        c["ev"].update(desired_capacity=float(self.desired_soc), min_time_of_stay=int(self.min_time_of_stay),
                       transition_soc_multiplier=float(self.transition_soc_multiplier))
        return c

    def write_ev2gym_yaml(self, directory: str | os.PathLike | None = None) -> str:
        fd, path = tempfile.mkstemp(suffix=".yaml", prefix=f"ev2gym_{self.cfg_id}_", dir=directory)
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(self.ev2gym_config(), f)
        return path


#: EV2Gym configuration template (keys of V2GProfitMax.yaml at the pinned commit). EV2Gym's own load/PV
#: forecasts (used only by the eMPC baseline) are set unbiased with 10 % noise (forecast_mean 100, std 10).
_EV2GYM_BASE: dict = {
    "timescale": 15, "simulation_length": 112,
    "year": 2022, "month": 1, "day": 17, "random_day": True, "random_hour": False, "hour": 5, "minute": 0,
    "simulation_days": "weekdays", "scenario": "workplace", "spawn_multiplier": 5,
    "discharge_price_factor": 1,
    "v2g_enabled": True, "number_of_charging_stations": 25, "number_of_transformers": 1,
    "number_of_ports_per_cs": 1, "charging_network_topology": "None", "simulate_grid": False,
    "network_info": {"vm_pu": 1.0, "s_base": 1000, "load_multiplier": 1, "pv_scale": 80,
                     "bus_info_file": "./ev2gym/data/network_data/node_34/Nodes_34.csv",
                     "branch_info_file": "./ev2gym/data/network_data/node_34/Lines_34.csv"},
    "pf_solver": "Laurent",
    "power_setpoint_enabled": False, "power_setpoint_flexiblity": 80,
    "tr_seed": -1,
    "inflexible_loads": {"include": True, "inflexible_loads_capacity_multiplier_mean": 0.4,
                         "forecast_mean": 100, "forecast_std": 10},
    "solar_power": {"include": True, "solar_power_capacity_multiplier_mean": 0.4,
                    "forecast_mean": 100, "forecast_std": 10},
    "demand_response": {"include": False, "events_per_day": 1, "event_capacity_percentage_mean": 35,
                        "event_capacity_percentage_std": 5, "event_length_minutes_min": 60,
                        "event_length_minutes_max": 60, "event_start_hour_mean": 12, "event_start_hour_std": 2,
                        "notification_of_event_minutes": 60},
    "heterogeneous_ev_specs": True, "ev_specs_file": "",
    "transformer": {"max_power": 100},
    "charging_station": {"min_charge_current": 0, "max_charge_current": 32, "min_discharge_current": 0,
                         "max_discharge_current": -32, "voltage": 400, "phases": 3},
    "ev": {"battery_capacity": 50, "max_ac_charge_power": 11, "min_ac_charge_power": 0,
           "max_dc_charge_power": 50, "max_discharge_power": -11, "min_discharge_power": 0, "ev_phases": 3,
           "charge_efficiency": 1, "discharge_efficiency": 1, "transition_soc": 1,
           "min_battery_capacity": 5, "min_time_of_stay": 180, "min_emergency_battery_capacity": 25,
           "desired_capacity": 1, "transition_soc_multiplier": 5},
}


def literature_config() -> ScenarioConfig:
    """EV2Gym's original V2GProfitMax.yaml (25 chargers, 100 kW, workplace, x5, 05:00, loads/PV off), Alg. 7.3."""
    return ScenarioConfig(profile="workplace", N=25, rho_tr=4.0, p_tr=100.0, loads=False, pv=False,
                          name="literature_N25")


# ------------------------------------------------------------------------------------ test grid
def load_test_grid(path: str | os.PathLike) -> list[tuple[ScenarioConfig, int]]:
    """Read ``configs/test_grid.yaml``: list of (ScenarioConfig, n_replays)."""
    g = yaml.safe_load(Path(path).read_text())
    defaults = g.get("defaults", {})
    rep = g.get("replays", {})
    n_default = int(rep.get("default", 100))
    n_large = int(rep.get("large_N", n_default))
    large_thr = int(rep.get("large_N_threshold", 10**9))
    out: list[tuple[ScenarioConfig, int]] = []
    grid = g.get("grid")
    if grid:
        for prof in grid["profiles"]:
            for N in grid["N"]:
                for rho in grid["rho_tr"]:
                    sc = ScenarioConfig.from_dict({**defaults, "profile": prof, "N": int(N), "rho_tr": float(rho)})
                    out.append((sc, n_large if int(N) >= large_thr else n_default))
    for e in g.get("extra", []) or []:
        e = dict(e)
        n = int(e.pop("replays", n_default))
        out.append((ScenarioConfig.from_dict({**defaults, **e}), n))
    ids = [s.cfg_id for s, _ in out]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate configuration ids in the test grid")
    return out


# ---------------------------------------------------------------------------- test-set manifests
def fingerprint(env) -> str:
    """Hash of everything exogenous in an EV2Gym episode: date, sessions, prices, loads, PV, rating."""
    h = hashlib.sha1()
    h.update(str(env.sim_starting_date).encode())
    for ev in env.EVs_profiles:
        eff = ev.charge_efficiency
        eff = sorted(set(eff.values()))[0] if isinstance(eff, dict) else eff
        h.update(np.array([ev.location, ev.time_of_arrival, ev.time_of_departure, ev.battery_capacity_at_arrival,
                           ev.battery_capacity, ev.desired_capacity, ev.transition_soc, ev.max_ac_charge_power,
                           float(eff)], dtype=float).round(9).tobytes())
    h.update(np.asarray(env.charge_prices[0], float).round(9).tobytes())
    for tr in env.transformers:
        h.update(np.asarray(tr.inflexible_load, float).round(6).tobytes())
        h.update(np.asarray(tr.solar_power, float).round(6).tobytes())
        h.update(np.asarray(tr.max_power, float).round(6).tobytes())
    return h.hexdigest()[:16]


def build_test_set(scfg: ScenarioConfig, n_replays: int, out_dir: str | os.PathLike | None = None,
                   env=None) -> dict:
    """Create (or load, if it exists with the same configuration) the manifest of a test configuration."""
    from .wrapper import SafeV2GEnv  # local import (wrapper imports this module)

    path = Path(out_dir) / f"{scfg.cfg_id}.json" if out_dir is not None else None
    if path is not None and path.exists():
        man = json.loads(path.read_text())
        if man.get("scenario") == scfg.to_dict() and len(man.get("replays", [])) >= n_replays:
            return man
    env = env or SafeV2GEnv(scfg)
    reps = []
    for rid in range(n_replays):
        seed = replay_seed(scfg.cfg_id, rid)
        env.reset(seed=seed, scenario=scfg)
        e = env.env
        reps.append({"replay_id": rid, "seed": seed, "sim_date": str(e.sim_starting_date),
                     "n_evs": len(e.EVs_profiles), "fingerprint": fingerprint(e)})
    man = {"test_cfg_id": scfg.cfg_id, "scenario": scfg.to_dict(), "n_replays": n_replays, "replays": reps}
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(man, indent=1) + "\n")
    return man


def load_test_set(path_or_dir: str | os.PathLike, cfg_id: str | None = None) -> dict:
    p = Path(path_or_dir)
    if p.is_dir():
        p = p / f"{cfg_id}.json"
    return json.loads(p.read_text())


# ------------------------------------------------------------------------------ training mixture
@dataclass
class TrainMixture:
    """Per-episode sampling of profile and tightness for training (Alg. 7.3: uniform, no profile flag)."""

    profiles: list = field(default_factory=lambda: ["workplace", "public", "private"])
    rho_tr: list = field(default_factory=lambda: [4.0, 7.0])
    base: dict = field(default_factory=dict)  # further ScenarioConfig fields (dt_min, T, desired_soc, ...)

    def sample(self, N: int, rng: np.random.Generator) -> ScenarioConfig:
        prof = self.profiles[int(rng.integers(len(self.profiles)))]
        rho = float(self.rho_tr[int(rng.integers(len(self.rho_tr)))])
        return ScenarioConfig.from_dict({**self.base, "profile": prof, "N": int(N), "rho_tr": rho})

    def all_configs(self, Ns: Iterable[int]) -> list[ScenarioConfig]:
        return [ScenarioConfig.from_dict({**self.base, "profile": p, "N": int(N), "rho_tr": float(r)})
                for N in sorted(set(Ns)) for p in self.profiles for r in self.rho_tr]


# ------------------------------------------------------------------- light session sampler (Alg. 4)
def reset_date(env, seed: int) -> None:
    """Replicate EV2Gym ``reset``'s seeding and date logic (ev2gym_env.py:246-292) without building loads."""
    np.random.seed(seed)
    random.seed(seed)
    cfg = env.config
    env.sim_date = datetime.datetime(2022, 1, 1, cfg["hour"], cfg["minute"]) + \
        datetime.timedelta(days=random.randint(0, int(1.5 * 365)))
    if env.scenario == "workplace":
        while env.sim_date.weekday() > 4:
            env.sim_date += datetime.timedelta(days=1)
    if cfg["simulation_days"] == "weekdays":
        while env.sim_date.weekday() > 4:
            env.sim_date += datetime.timedelta(days=1)
    elif cfg["simulation_days"] == "weekends" and env.scenario != "workplace":
        while env.sim_date.weekday() < 5:
            env.sim_date += datetime.timedelta(days=1)
    env.sim_starting_date = env.sim_date


def sample_sessions(env, seed: int) -> list:
    """Sessions (EV2Gym EV objects) of one episode drawn from EV2Gym's own arrival model (``EV_spawner``).

    Same law as the sessions of ``env.reset(seed=...)`` (the spawner draws from the global RNG after the
    date is fixed); used for conformal calibration (Alg. 4), which needs only the arrival model."""
    from ev2gym.utilities.utils import EV_spawner

    reset_date(env, seed)
    evs = EV_spawner(env)
    while len(evs) == 0:
        evs = EV_spawner(env)
    return evs


# ------------------------------------------------------------------------------------------- CLI
def _main(argv=None):
    ap = argparse.ArgumentParser(description="scenario utilities")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-test-sets", help="create test replay manifests with fingerprints")
    b.add_argument("--grid", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--only", nargs="*", default=None, help="configuration ids to build")
    b.add_argument("--max-replays", type=int, default=None, help="cap the number of replays (smoke)")
    s = sub.add_parser("show", help="print the EV2Gym YAML of a configuration")
    s.add_argument("--profile", default="workplace")
    s.add_argument("--N", type=int, default=10)
    s.add_argument("--rho", type=float, default=4.0)
    w = sub.add_parser("write-yaml", help="write the EV2Gym YAML of a configuration (e.g. for EV-GNN training)")
    w.add_argument("--profile", default="workplace")
    w.add_argument("--N", type=int, default=10)
    w.add_argument("--rho", type=float, default=4.0)
    w.add_argument("--dt", type=int, default=15)
    w.add_argument("--T", type=int, default=None)
    w.add_argument("--min-stay", type=int, default=180)
    w.add_argument("--hour", type=int, default=None)
    w.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "write-yaml":
        sc = ScenarioConfig(profile=args.profile, N=args.N, rho_tr=args.rho, dt_min=args.dt, T=args.T,
                            min_time_of_stay=args.min_stay, hour=args.hour)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(yaml.safe_dump(sc.ev2gym_config()))
        print(f"wrote {args.out} ({sc.cfg_id})")
        return
    if args.cmd == "build-test-sets":
        for scfg, n in load_test_grid(args.grid):
            if args.only and scfg.cfg_id not in args.only:
                continue
            n = min(n, args.max_replays) if args.max_replays else n
            man = build_test_set(scfg, n, args.out)
            print(f"{scfg.cfg_id:32s} {man['n_replays']:4d} replays, "
                  f"mean EVs {np.mean([r['n_evs'] for r in man['replays']]):.1f}", flush=True)
    elif args.cmd == "show":
        print(yaml.safe_dump(ScenarioConfig(profile=args.profile, N=args.N, rho_tr=args.rho).ev2gym_config()))


if __name__ == "__main__":
    _main()
