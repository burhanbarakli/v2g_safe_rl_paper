"""Hyperparameter tuning protocol (algorithm.md Alg. 6, equal budget for all learned methods).

Grid (18 configurations per method): lr in {1e-4, 3e-4, 1e-3} x minibatch in {256, 1024} x ent_coef in
{0, 1e-3, 1e-2}; each trained for 0.5e6 env steps with seed 0 and evaluated on 20 validation episodes of the
10-charger station at rho_tr = 4 kW with the three profiles mixed (validation seeds disjoint from training and
test). Selection by the method's own objective: mean validation return in r (safety-layer methods) or in r_pen
(penalty methods). The selected values are written to ``<tuning_dir>/<method>/selected.yaml`` (an overlay used
by the final training runs).

    python -m v2gsafe.tuning jobs --config configs/train_attn_safe.yaml --out results/tuning   # list of runs
    python -m v2gsafe.tuning validate --run results/tuning/attn_safe/lr0.0003_mb256_ent0
    python -m v2gsafe.tuning select --method-dir results/tuning/attn_safe
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import yaml

GRID = {"ppo.lr": [1e-4, 3e-4, 1e-3], "ppo.minibatch": [256, 1024], "ppo.ent_coef": [0.0, 1e-3, 1e-2]}
TUNING_STEPS = 500_000
N_VAL = 20


def grid_points() -> list[dict]:
    keys = list(GRID)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(GRID[k] for k in keys))]


def tag(point: dict) -> str:
    return f"lr{point['ppo.lr']:g}_mb{point['ppo.minibatch']}_ent{point['ppo.ent_coef']:g}"


def jobs(config: str, out: str, smoke: bool = False) -> list[dict]:
    from .config import load_config
    cfg = load_config(config)
    method = cfg["method"]
    out_jobs = []
    for pt in grid_points():
        sets = [f"{k}={v}" for k, v in pt.items()]
        if not smoke:
            sets.append(f"ppo.total_env_steps={TUNING_STEPS}")
        out_jobs.append({"method": method, "tag": tag(pt), "config": config, "sets": sets,
                         "out": str(Path(out) / method / tag(pt))})
    return out_jobs


def validation_episodes(n: int = N_VAL, N: int = 10, rho: float = 4.0, base: dict | None = None):
    from .envs.scenarios import ScenarioConfig, val_seed
    profiles = ["workplace", "public", "private"]
    out = []
    for i in range(n):
        scfg = ScenarioConfig.from_dict({**(base or {}), "profile": profiles[i % 3], "N": N, "rho_tr": rho})
        out.append((scfg, val_seed(f"val_N{N}_rho{rho:g}", i)))
    return out


def validate(run_dir: str | Path, n: int = N_VAL) -> dict:
    """Evaluate checkpoint_final.pt of a tuning run on the validation episodes; writes validation.json."""
    from .eval.evaluate import MethodSpec, resolve, run_episode
    from .envs.wrapper import SafeV2GEnv

    run_dir = Path(run_dir)
    spec = MethodSpec(method="learned", checkpoint=str(run_dir / "checkpoint_final.pt"))
    ctrl, name, opts, info = resolve(spec)
    from .algos.ppo import load_actor
    _, cfg, _ = load_actor(spec.checkpoint)
    sc = {k: v for k, v in dict(cfg["scenario"]).items() if k not in ("profiles", "rho_tr") and v is not None}
    eps = validation_episodes(n, base=sc)
    env = SafeV2GEnv(eps[0][0], opts)
    prof, pen = [], []
    for scfg, seed in eps:
        m = run_episode(env, ctrl, seed, scfg)
        prof.append(m["profit_eur"])
        pen.append(m["penalty_return_eur"])
    objective = "penalty" if cfg["reward"]["type"] == "penalty" else "profit"
    res = {"method": name, "objective": objective, "val_return_profit": float(np.mean(prof)),
           "val_return_penalty": float(np.mean(pen)), "n": n,
           "score": float(np.mean(pen) if objective == "penalty" else np.mean(prof))}
    (run_dir / "validation.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def select(method_dir: str | Path) -> dict:
    method_dir = Path(method_dir)
    best, best_score = None, -np.inf
    for v in sorted(method_dir.glob("*/validation.json")):
        r = json.loads(v.read_text())
        if r["score"] > best_score:
            best, best_score = v.parent.name, r["score"]
    if best is None:
        raise FileNotFoundError(f"no validation.json under {method_dir}")
    pt = next(p for p in grid_points() if tag(p) == best)
    overlay = {"ppo": {"lr": pt["ppo.lr"], "minibatch": pt["ppo.minibatch"], "ent_coef": pt["ppo.ent_coef"]}}
    (method_dir / "selected.yaml").write_text(f"# selected by v2gsafe.tuning ({best}, score {best_score:.4f})\n"
                                              + yaml.safe_dump(overlay))
    return {"selected": best, "score": best_score, **overlay}


def _main(argv=None):
    ap = argparse.ArgumentParser(description="tuning protocol (Alg. 6)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("jobs")
    j.add_argument("--config", required=True)
    j.add_argument("--out", default="results/tuning")
    v = sub.add_parser("validate")
    v.add_argument("--run", required=True)
    v.add_argument("--n", type=int, default=N_VAL)
    s = sub.add_parser("select")
    s.add_argument("--method-dir", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "jobs":
        print(json.dumps(jobs(a.config, a.out), indent=1))
    elif a.cmd == "validate":
        print(json.dumps(validate(a.run, a.n)))
    else:
        print(json.dumps(select(a.method_dir)))


if __name__ == "__main__":
    _main()
