"""Hyperparameter tuning protocol (algorithm.md Alg. 6; DD37: equal budget for every learned method).

Per method a grid is formed from the common PPO grid
    lr in {1e-4, 3e-4, 1e-3} x minibatch in {256, 512, 1024} x ent_coef in {0, 1e-3, 1e-2}
plus, for methods trained with the penalty reward (attn_penalty, mlp_penalty, attn_proj_nolb), the penalty weight
w = w_E = w_P in {1, 10, 100} EUR/kWh, and for PPO-Lagrangian (attn_lagrangian) the dual step lr_lambda in
{0.5, 2, 8}. Every method gets the SAME budget: K = 12 configurations (the Alg. 6 default point first, then 11 points
drawn without replacement from the rest of its grid with a fixed per-method seed), each trained on the method's own
training mixture for 0.5e6 env steps with seed 0. Validation: 24 episodes drawn from the training mixture
(configurations cycled over profiles x rho_tr x training sizes) with validation seeds (``val_seed``, a seed range
disjoint from training and test replays). Selection criterion, identical for all methods: mean validation
penalised profit J^pen = profit - 10 * shortfall (vs the admitted request) - 10 * overload energy (EUR), i.e. the
evaluation metric ``penalised_profit_eur`` with the fixed w = 10 EUR/kWh (not the tuned training weight).

``select`` writes ``<tuning_dir>/<method>/selected.yaml`` (all tuned keys) and ``selected_ppo.yaml`` (PPO keys only;
used by variants whose reward weight is fixed by design, e.g. the w-sensitivity runs). The final training runs use
these overlays; ``reproduce.sh`` refuses to train without them unless ``--hparams FILE`` is given
(``configs/hparams_defaults.yaml`` = the Alg. 6 defaults, explicitly).

    python -m v2gsafe.tuning jobs --config configs/train_attn_safe.yaml --out results/tuning   # list of runs
    python -m v2gsafe.tuning validate --run results/tuning/attn_safe/lr0.0003_mb512_ent0
    python -m v2gsafe.tuning select --method-dir results/tuning/attn_safe
"""
from __future__ import annotations

import argparse
import itertools
import json
import zlib
from pathlib import Path

import numpy as np
import yaml

PPO_GRID = {"ppo.lr": [1e-4, 3e-4, 1e-3], "ppo.minibatch": [256, 512, 1024], "ppo.ent_coef": [0.0, 1e-3, 1e-2]}
PENALTY_GRID = {"reward.w": [1.0, 10.0, 100.0]}            # sets reward.w_E = reward.w_P
LAGRANGIAN_GRID = {"lagrangian.lr_lambda": [0.5, 2.0, 8.0]}
DEFAULT_POINT = {"ppo.lr": 3e-4, "ppo.minibatch": 512, "ppo.ent_coef": 0.0, "reward.w": 10.0,
                 "lagrangian.lr_lambda": 2.0}
K_CONFIGS = 12
K_CONFIGS_SMOKE = 2
TUNING_STEPS = 500_000
N_VAL = 24
SCORE_W = 10.0


def method_grid(cfg) -> dict:
    g = dict(PPO_GRID)
    if cfg["reward"]["type"] == "penalty":
        g.update(PENALTY_GRID)
    if cfg["reward"]["type"] == "lagrangian":
        g.update(LAGRANGIAN_GRID)
    return g


def grid_points(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def sampled_points(method: str, grid: dict, k: int) -> list[dict]:
    """Equal-budget sample: the default point first, then k-1 further points (fixed per-method seed)."""
    pts = grid_points(grid)
    default = {key: DEFAULT_POINT[key] for key in grid}
    rest = [p for p in pts if p != default]
    rng = np.random.default_rng(zlib.crc32(method.encode()))
    idx = rng.choice(len(rest), size=min(k - 1, len(rest)), replace=False)
    return [default] + [rest[i] for i in sorted(idx)]


def tag(point: dict) -> str:
    s = f"lr{point['ppo.lr']:g}_mb{point['ppo.minibatch']}_ent{point['ppo.ent_coef']:g}"
    if "reward.w" in point:
        s += f"_w{point['reward.w']:g}"
    if "lagrangian.lr_lambda" in point:
        s += f"_llr{point['lagrangian.lr_lambda']:g}"
    return s


def point_sets(point: dict) -> list[str]:
    out = []
    for k, v in point.items():
        if k == "reward.w":
            out += [f"reward.w_E={v}", f"reward.w_P={v}"]
        else:
            out.append(f"{k}={v}")
    return out


def point_overlay(point: dict, ppo_only: bool = False) -> dict:
    ov: dict = {}
    for k, v in point.items():
        if ppo_only and not k.startswith("ppo."):
            continue
        if k == "reward.w":
            ov.setdefault("reward", {}).update({"w_E": float(v), "w_P": float(v)})
        else:
            sec, key = k.split(".")
            ov.setdefault(sec, {})[key] = v
    return ov


def jobs(config: str, out: str, smoke: bool = False) -> list[dict]:
    from .config import load_config
    cfg = load_config(config)
    method = cfg["method"]
    out_jobs = []
    for pt in sampled_points(method, method_grid(cfg), K_CONFIGS_SMOKE if smoke else K_CONFIGS):
        sets = point_sets(pt)
        if not smoke:
            sets.append(f"ppo.total_env_steps={TUNING_STEPS}")
        out_jobs.append({"method": method, "tag": tag(pt), "config": config, "sets": sets, "point": pt,
                         "out": str(Path(out) / method / tag(pt))})
    return out_jobs


def validation_episodes(cfg, n: int = N_VAL):
    """n episodes of the method's training mixture (configurations cycled), validation seeds."""
    from .algos.ppo import train_mixture
    from .envs.scenarios import val_seed
    Ns = sorted({int(g["N"]) for g in cfg["envs"]["groups"]})
    confs = train_mixture(cfg).all_configs(Ns)
    return [(confs[i % len(confs)], val_seed("val_mixture", i)) for i in range(n)]


def validate(run_dir: str | Path, n: int = N_VAL) -> dict:
    """Evaluate checkpoint_final.pt of a tuning run on the validation episodes; writes validation.json."""
    from .eval.evaluate import MethodSpec, resolve, run_episode
    from .envs.wrapper import SafeV2GEnv

    run_dir = Path(run_dir)
    spec = MethodSpec(method="learned", checkpoint=str(run_dir / "checkpoint_final.pt"))
    ctrl, name, opts, info = resolve(spec)
    from .algos.ppo import load_actor
    _, cfg, _ = load_actor(spec.checkpoint)
    eps = validation_episodes(cfg, n)
    envs: dict = {}
    prof, pen, jpen = [], [], []
    for scfg, seed in eps:
        key = (scfg.N, scfg.dt_min, scfg.horizon)
        if key not in envs:
            envs[key] = SafeV2GEnv(scfg, opts)
        m = run_episode(envs[key], ctrl, seed, scfg)
        prof.append(m["profit_eur"])
        pen.append(m["penalty_return_eur"])
        jpen.append(m["profit_eur"] - SCORE_W * m["energy_not_served_kwh"] - SCORE_W * m["tr_violation_kwh"])
    res = {"method": name, "objective": f"penalised_profit_w{SCORE_W:g}", "val_return_profit": float(np.mean(prof)),
           "val_return_penalty": float(np.mean(pen)), "val_penalised_profit": float(np.mean(jpen)), "n": n,
           "score": float(np.mean(jpen))}
    (run_dir / "validation.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def select(method_dir: str | Path, config: str | None = None) -> dict:
    """Pick the best validated configuration; write selected.yaml and selected_ppo.yaml."""
    method_dir = Path(method_dir)
    best, best_score = None, -np.inf
    for v in sorted(method_dir.glob("*/validation.json")):
        r = json.loads(v.read_text())
        if r["score"] > best_score:
            best, best_score = v.parent, r["score"]
    if best is None:
        raise FileNotFoundError(f"no validation.json under {method_dir}")
    cfg = yaml.safe_load((best / "config.yaml").read_text())
    point = {}
    for k in list(PPO_GRID) + list(LAGRANGIAN_GRID):
        sec, key = k.split(".")
        point[k] = cfg[sec][key]
    if cfg["reward"]["type"] == "penalty":
        point["reward.w"] = cfg["reward"]["w_E"]
    if cfg["reward"]["type"] != "lagrangian":
        point.pop("lagrangian.lr_lambda")
    head = f"# selected by v2gsafe.tuning ({best.name}, validation J^pen {best_score:.4f} EUR)\n"
    full, ppo = point_overlay(point), point_overlay(point, ppo_only=True)
    (method_dir / "selected.yaml").write_text(head + yaml.safe_dump(full))
    (method_dir / "selected_ppo.yaml").write_text(head + yaml.safe_dump(ppo))
    return {"selected": best.name, "score": best_score, **full}


def _main(argv=None):
    ap = argparse.ArgumentParser(description="tuning protocol (Alg. 6, DD37)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("jobs")
    j.add_argument("--config", required=True)
    j.add_argument("--out", default="results/tuning")
    j.add_argument("--smoke", action="store_true")
    v = sub.add_parser("validate")
    v.add_argument("--run", required=True)
    v.add_argument("--n", type=int, default=N_VAL)
    s = sub.add_parser("select")
    s.add_argument("--method-dir", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "jobs":
        print(json.dumps(jobs(a.config, a.out, a.smoke), indent=1))
    elif a.cmd == "validate":
        print(json.dumps(validate(a.run, a.n)))
    else:
        print(json.dumps(select(a.method_dir)))


if __name__ == "__main__":
    _main()
