"""Evaluation of any method on fixed test replays (algorithm.md Alg. 3; code_spec.md Sec. 3).

For every test configuration of a grid and every replay (seed from the test-set manifest, fingerprint checked),
the method's controller runs one closed-loop episode through :class:`~v2gsafe.envs.wrapper.SafeV2GEnv`; one row
per episode is written to ``episodes.csv`` (columns :data:`~v2gsafe.eval.metrics.METHOD_COLUMNS`). Inference time
per step = arrival processing + bounds (safety layer) + policy + projection/fallback (wall clock).

Methods: learned checkpoints (attn_safe, attn_penalty, mlp_safe, mlp_penalty, attn_safe_conformal -> evaluated as
attn_safe_conf_a{alpha}), heuristics afap/alap/rr/llf and their '_safe' variants, empc / empc_safe (EV2Gym eMPC,
Gurobi), empc_highs / empc_highs_safe (HiGHS port), random (tests). The oracle has its own entry point
(:mod:`v2gsafe.baselines.oracle`).
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from ..constants import P_REF
from ..envs.scenarios import ScenarioConfig, build_test_set, fingerprint, load_test_grid
from ..envs.wrapper import SafeV2GEnv, WrapperOptions
from ..utils import append_csv
from .metrics import METHOD_COLUMNS

RAW_HEURISTICS = ("afap", "alap", "rr", "llf", "sllf", "price_llf", "random")


@dataclass
class MethodSpec:
    """Picklable description of the evaluated method."""

    method: str                       # e.g. 'afap_safe', 'empc_highs', 'learned'
    checkpoint: str | None = None
    device: str = "cpu"
    admission: bool | None = None     # None: method default (on for every method, DD31); False -> '<name>_noadm'
    deploy_safety: bool = False       # learned penalty checkpoint + safety layer at deployment ('_safedeploy', DD33(2))
    conformal_alpha: float | None = None
    conformal_uncalibrated: bool = False
    conformal_cache: str = "results/conformal"
    admission_existing_targets: str | None = None
    torch_threads: int = 1
    root: str | None = None
    extra: dict = field(default_factory=dict)


class LearnedController:
    name = "learned"
    uses_future_info = False

    def __init__(self, actor, device: str = "cpu"):
        import torch
        self.torch = torch
        self.actor = actor
        self.device = device

    def reset(self, env):
        pass

    def act(self, obs, env):
        torch = self.torch
        with torch.no_grad():
            tok = torch.from_numpy(obs.tokens)[None].to(self.device)
            glob = torch.from_numpy(obs.glob)[None].to(self.device)
            occ = torch.from_numpy(obs.occ)[None].to(self.device)
            mu, _ = self.actor(tok, glob, occ)       # deterministic action a = mu at test time (Alg. 6)
        return P_REF * mu[0].cpu().numpy().astype(np.float64)


def resolve(spec: MethodSpec):
    """Return (controller, method_name, WrapperOptions, info dict)."""
    m = spec.method
    info = {"seed": 0, "train_N": None, "train_N_set": None}
    if spec.checkpoint is not None:
        import torch
        from ..algos.ppo import conformal_provider, load_actor, wrapper_options
        if spec.torch_threads:
            torch.set_num_threads(int(spec.torch_threads))
        actor, cfg, ck = load_actor(spec.checkpoint, spec.device)
        ctrl = LearnedController(actor, spec.device)
        name = cfg["method"]
        root = Path(spec.root) if spec.root else None
        opts = wrapper_options(cfg, root, admission=spec.admission)
        if spec.admission_existing_targets:
            opts.admission_existing_targets = spec.admission_existing_targets
        if spec.deploy_safety and not opts.safety:
            opts.safety, opts.lower_bound = True, True
            name += "_safedeploy"
        if spec.admission is False:
            name += "_noadm"
        if spec.conformal_alpha is not None or spec.conformal_uncalibrated:
            from ..safety.conformal import ConformalProvider
            alpha = spec.conformal_alpha if spec.conformal_alpha is not None else cfg["conformal"]["alpha"]
            cache = Path(spec.conformal_cache)
            if root is not None and not cache.is_absolute():
                cache = root / cache
            c = cfg["conformal"]
            opts.conformal_provider = ConformalProvider(alpha, cache, c["n_train"], c["n_cal"], c["predictor"],
                                                        uncalibrated=spec.conformal_uncalibrated)
            base = name[: -len("_conformal")] if name.endswith("_conformal") else name
            name = f"{base}_conf_point" if spec.conformal_uncalibrated else f"{base}_conf_a{alpha:g}"
        Ns = sorted({int(g["N"]) for g in cfg["envs"]["groups"]})
        run_id = ck.get("exp_id") or ""
        # evaluation variants of one training run (noadm, conformal alphas) get their own exp_id with the evaluated
        # method in the first field: {method}__{train_cfg}__s{seed}__{hash8} stays unique per (variant, replay)
        eid = run_id if (not run_id or name == cfg["method"]) else name + run_id[run_id.index("__"):]
        info.update(seed=int(ck.get("seed", 0)), train_N=max(Ns), train_N_set="+".join(map(str, Ns)),
                    exp_id=eid)
        return ctrl, name, opts, info
    safe = m.endswith("_safe")
    base = m[:-5] if safe else m
    if base in RAW_HEURISTICS:
        from ..baselines.heuristics import HEURISTICS
        ctrl = HEURISTICS[base]()
    elif base == "empc":
        from ..baselines.mpc import EMPCGurobi
        ctrl = EMPCGurobi()
    elif base == "empc_highs":
        from ..baselines.mpc import EMPCHighs
        ctrl = EMPCHighs()
    elif base == "empc_highs_24h":
        from ..baselines.mpc import EMPCHighs
        ctrl = EMPCHighs(horizon_hours=24.0)
    elif base == "evgnn":
        from ..baselines.evgnn import EVGNNController
        ctrl = EVGNNController(spec.extra.get("evgnn_repo"), spec.extra.get("evgnn_model_dir"), device=spec.device)
        info["train_N"] = 20                       # DD27: trained at N = 20, rho 4, per profile
    else:
        raise ValueError(f"unknown method {m!r}")
    info["seed"] = int(spec.extra.get("seed", 0) or 0)
    adm = True if spec.admission is None else bool(spec.admission)     # DD31: request pipeline for all methods
    opts = WrapperOptions(safety=safe, admission=adm)
    if spec.admission_existing_targets:
        opts.admission_existing_targets = spec.admission_existing_targets
    name = m + ("_noadm" if spec.admission is False else "")
    return ctrl, name, opts, info


def run_episode(env: SafeV2GEnv, ctrl, seed: int, scfg: ScenarioConfig) -> dict:
    """One closed-loop episode (Alg. 3); returns the wrapper's episode metrics."""
    obs = env.reset(seed, scfg)
    ctrl.reset(env)
    first = True
    while True:
        t0 = time.perf_counter()
        if not first:
            obs = env.observe()
        first = False
        x = ctrl.act(obs, env)
        cmd = env.command(x)
        ms = (time.perf_counter() - t0) * 1000.0
        out = env.step(None, inference_ms=ms, p_override=cmd)
        if out.done:
            break
    return env.episode_metrics()


def evaluate_config(spec: MethodSpec, scfg: ScenarioConfig, n_replays: int, test_sets: str, exp_id: str,
                    smoke: bool, log_steps: bool = False, env_cache: dict | None = None):
    ctrl, name, opts, info = resolve(spec)
    opts.log_steps = log_steps
    man = build_test_set(scfg, n_replays, test_sets)
    key = (scfg.N, scfg.dt_min, scfg.horizon)
    env = None
    if env_cache is not None and key in env_cache and _same_opts(env_cache[key].opt, opts):
        env = env_cache[key]
        env.opt = opts                     # same settings; take this call's conformal provider
    if env is None:
        env = SafeV2GEnv(scfg, opts)
        if env_cache is not None:
            env_cache[key] = env
    rows, steps, chargers = [], [], []
    for rep in man["replays"][:n_replays]:
        m = run_episode(env, ctrl, rep["seed"], scfg)
        fp = fingerprint(env.env)
        row = {"exp_id": info.get("exp_id") or exp_id, "method": name, "seed": info["seed"],
               "train_N": info["train_N"], "train_N_set": info["train_N_set"], "test_cfg_id": scfg.cfg_id,
               "profile": scfg.profile, "N": scfg.N,
               "rho_tr": scfg.rho, "dt_min": scfg.dt_min, "replay_id": rep["replay_id"], "smoke": smoke,
               "replay_fingerprint": fp if fp == rep["fingerprint"] else f"MISMATCH:{fp}", **m}
        rows.append(row)
        if log_steps:
            for r in env.step_log:
                steps.append({"exp_id": row["exp_id"], "method": name, "test_cfg_id": scfg.cfg_id,
                              "replay_id": rep["replay_id"], **r})
            for r in env.charger_log:
                chargers.append({"exp_id": row["exp_id"], "method": name, "test_cfg_id": scfg.cfg_id,
                                 "replay_id": rep["replay_id"], **r})
    return name, rows, steps, chargers


def _same_opts(a: WrapperOptions, b: WrapperOptions) -> bool:
    import dataclasses
    skip = {"conformal_provider", "log_steps"}
    return all(getattr(a, f.name) == getattr(b, f.name) for f in dataclasses.fields(a) if f.name not in skip)


def _job(args):
    spec, scfg_d, n, test_sets, exp_id, smoke, log_steps, part_dir = args
    scfg = ScenarioConfig.from_dict(scfg_d)
    name, rows, steps, chargers = evaluate_config(spec, scfg, n, test_sets, exp_id, smoke, log_steps)
    _write_part(part_dir, name, scfg.cfg_id, rows, steps, chargers)
    return name, scfg.cfg_id, len(rows)


def _write_part(part_dir: Path, name: str, cfg_id: str, rows, steps, chargers) -> None:
    part_dir = Path(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)
    tmp = part_dir / f"{name}__{cfg_id}.csv.tmp"
    if tmp.exists():
        tmp.unlink()
    append_csv(tmp, rows, METHOD_COLUMNS)
    os.replace(tmp, part_dir / f"{name}__{cfg_id}.csv")
    if steps:
        _write_table(part_dir / f"{name}__{cfg_id}.steps", steps)
        _write_table(part_dir / f"{name}__{cfg_id}.chargers", chargers)


def _write_table(stem: Path, rows: list[dict]) -> None:
    import pandas as pd
    df = pd.DataFrame(rows)
    try:
        df.to_parquet(str(stem) + ".parquet", index=False)
    except Exception:  # noqa: BLE001  (no pyarrow/fastparquet): CSV fallback
        df.to_csv(str(stem) + ".csv", index=False)


def merge_parts(out_dir: Path) -> Path:
    """Concatenate all part files into episodes.csv (and steps*.parquet if present)."""
    import pandas as pd
    out_dir = Path(out_dir)
    parts = sorted((out_dir / "parts").glob("*.csv"))
    parts = [p for p in parts if not p.name.endswith((".steps.csv", ".chargers.csv"))]
    if parts:
        df = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False) for p in parts], ignore_index=True)
        df = df.reindex(columns=METHOD_COLUMNS)
        tmp = out_dir / f".episodes.{os.getpid()}.tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, out_dir / "episodes.csv")          # atomic: parallel evaluate jobs may merge concurrently
    for kind, fname in (("steps", "steps"), ("chargers", "steps_chargers")):
        files = sorted((out_dir / "parts").glob(f"*.{kind}.parquet"))
        if files:
            pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).to_parquet(
                out_dir / f"{fname}.parquet", index=False)
        files = sorted((out_dir / "parts").glob(f"*.{kind}.csv"))
        if files:
            pd.concat([pd.read_csv(f) for f in files], ignore_index=True).to_csv(out_dir / f"{fname}.csv", index=False)
    return out_dir / "episodes.csv"


def part_complete(part_dir: Path, name: str, cfg_id: str, n: int) -> bool:
    p = Path(part_dir) / f"{name}__{cfg_id}.csv"
    if not p.exists():
        return False
    with p.open() as f:
        return sum(1 for _ in f) - 1 >= n


def evaluate(spec: MethodSpec, grid: list[tuple[ScenarioConfig, int]], test_sets: str, out_dir: str | Path,
             exp_id: str, smoke: bool = False, log_steps: bool = False, jobs: int = 1, only=None,
             max_replays: int | None = None, verbose: bool = True) -> Path:
    out_dir = Path(out_dir)
    part_dir = out_dir / "parts"
    _, name, _, _ = resolve_name_only(spec)
    todo = []
    for scfg, n in grid:
        if only and scfg.cfg_id not in only:
            continue
        n = min(n, max_replays) if max_replays else n
        if part_complete(part_dir, name, scfg.cfg_id, n):
            if verbose:
                print(f"[eval] {name} {scfg.cfg_id}: done (part exists)")
            continue
        build_test_set(scfg, n, test_sets)          # create manifests up front (cheap if they exist)
        todo.append((spec, scfg.to_dict(), n, test_sets, exp_id, smoke, log_steps, part_dir))
    t0 = time.time()
    if jobs <= 1:
        cache: dict = {}
        for a in todo:
            spec_, scfg_d, n, ts, eid, sm, ls, pdir = a
            scfg = ScenarioConfig.from_dict(scfg_d)
            nm, rows, steps, ch = evaluate_config(spec_, scfg, n, ts, eid, sm, ls, env_cache=cache)
            _write_part(pdir, nm, scfg.cfg_id, rows, steps, ch)
            if verbose:
                print(f"[eval] {nm} {scfg.cfg_id}: {len(rows)} episodes, mean profit "
                      f"{np.mean([r['profit_eur'] for r in rows]):.2f} EUR ({time.time() - t0:.0f}s)", flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(jobs) as pool:
            for nm, cid, k in pool.imap_unordered(_job, todo):
                if verbose:
                    print(f"[eval] {nm} {cid}: {k} episodes ({time.time() - t0:.0f}s)", flush=True)
    return merge_parts(out_dir)


def resolve_name_only(spec: MethodSpec):
    """Method name without building envs (loads the checkpoint config for learned methods)."""
    if spec.checkpoint is None:
        name = spec.method + ("_noadm" if spec.admission is False else "")
        return None, name, None, None
    return resolve(spec)
