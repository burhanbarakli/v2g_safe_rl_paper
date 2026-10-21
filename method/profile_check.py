"""profile_check.py -- arrival profiles of EV2Gym (workplace / public / private) for algorithm.md Alg. 7.

    cd /home/claude/work/ext/EV2Gym && /home/claude/work/ext/venv/bin/python /home/claude/work/method/profile_check.py

For each profile, episode start hour and spawn_multiplier, runs n_seeds resets of a 20-charger station
(T = 112 steps of 15 min = 28 h, empty_ports_at_end_of_simulation=True) and reports:
  EVs per charger per day, mean stay (h), mean arrival hour, occupancy (fraction of charger-steps occupied),
  fraction of spawn attempts dropped by EV2Gym because the departure would fall within 4 steps of the end
  (utils.py: `if time_of_stay + step + 4 >= simulation_length: return None`), mean requested energy (kWh).
"""
import os
import sys
import tempfile

import numpy as np
import yaml

REPO = "/home/claude/work/ext/EV2Gym"
os.chdir(REPO)
sys.path.insert(0, REPO)
import ev2gym.utilities.utils as U  # noqa: E402
from ev2gym.models.ev2gym_env import EV2Gym  # noqa: E402

COUNT = {"attempt": 0, "dropped": 0}
_orig = U.spawn_single_EV


def _counting(*args, **kw):
    COUNT["attempt"] += 1
    ev = _orig(*args, **kw)
    if ev is None:
        COUNT["dropped"] += 1
    return ev


U.spawn_single_EV = _counting

BASE = yaml.safe_load(open("ev2gym/example_config_files/V2GProfitMax.yaml"))


def run(profile, hour, mult, n_seeds=15, N=20, T=112, days="weekdays"):
    cfg = dict(BASE)
    cfg.update(scenario=profile, spawn_multiplier=mult, hour=hour, minute=0, random_day=True, random_hour=False,
               simulation_length=T, number_of_charging_stations=N, simulation_days=days)
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(cfg, f)
    env = EV2Gym(config_file=path, seed=0, empty_ports_at_end_of_simulation=True)
    COUNT.update(attempt=0, dropped=0)
    nev, stays, arr_h, occ, req = 0, [], [], 0.0, []
    for s in range(n_seeds):
        env.reset(seed=1000 + s)
        for ev in env.EVs_profiles:
            nev += 1
            st = ev.time_of_departure - ev.time_of_arrival + 1
            stays.append(st * env.timescale / 60)
            arr_h.append((hour + ev.time_of_arrival * env.timescale / 60) % 24)
            occ += st
            req.append(ev.desired_capacity - ev.battery_capacity_at_arrival)
    os.remove(path)
    days_ep = T * env.timescale / 60 / 24
    return dict(ev_per_ch_day=nev / (N * n_seeds * days_ep), stay=np.mean(stays) if stays else 0,
                arr=np.median(arr_h) if arr_h else 0, occ=occ / (N * T * n_seeds),
                drop=COUNT["dropped"] / max(COUNT["attempt"], 1), req=np.mean(req) if req else 0, n=nev)


def main():
    rows = []
    for profile, hours, mults in [("workplace", [5], [3, 5, 7]),
                                  ("public", [5, 12], [2, 3, 5, 7]),
                                  ("private", [5, 12], [1, 2, 3, 5])]:
        for h in hours:
            for m in mults:
                r = run(profile, h, m)
                rows.append((profile, h, m, r))
                print(f"{profile:9s} start {h:2d}:00 mult {m:2d}: EVs/charger/day {r['ev_per_ch_day']:.2f}  "
                      f"mean stay {r['stay']:5.2f} h  median arrival {r['arr']:5.2f} h  occupancy {r['occ']:.3f}  "
                      f"dropped {r['drop']*100:5.1f} %  mean request {r['req']:5.1f} kWh  (EVs {r['n']})", flush=True)


if __name__ == "__main__":
    main()
