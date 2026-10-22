"""Configuration: YAML files with inheritance, defaults from algorithm.md Alg. 6, CLI overrides, hashing.

A config file may contain ``inherit: base.yaml`` (path relative to the file); keys are deep-merged.
Unknown keys (not present in :data:`DEFAULTS`) raise an error, so typos cannot silently change a run.
``config_hash8`` hashes the resolved configuration without bookkeeping keys (seed, device, paths).
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

#: Defaults = algorithm.md Alg. 6 (main method SA-Attn).
DEFAULTS: dict[str, Any] = {
    "method": "attn_safe",
    "smoke": False,
    "model": {"arch": "attention", "d_model": 128, "n_layers": 3, "n_heads": 4, "head_hidden": 64,
              "mlp_hidden": 256, "n_max": 200},
    "ppo": {"total_env_steps": 3_000_000, "rollout_len": 128, "minibatch": 512, "epochs": 10, "lr": 3e-4,
            "lr_decay": True, "adam_eps": 1e-5, "gamma": 0.99, "gae_lambda": 0.95, "clip": 0.2, "ent_coef": 0.0,
            "vf_coef": 0.5, "max_grad_norm": 0.5, "target_kl": 0.03, "beta": 0.01, "norm_adv": True},
    "envs": {"groups": [{"N": 10, "count": 8}, {"N": 20, "count": 8}], "backend": "subproc",
             "curriculum": {"enabled": False, "frac": 0.2, "groups": [{"N": 10, "count": 16}]}},
    "scenario": {"profiles": ["workplace", "public", "private"], "rho_tr": [4.0, 7.0], "dt_min": 15, "T": None,
                 "desired_soc": 1.0, "loads": True, "pv": True, "load_mult": 0.4, "pv_mult": 0.4,
                 "min_time_of_stay": 180, "hour": None, "spawn_multiplier": None, "simulation_days": None},
    "safety": {"enabled": True, "lower_bound": True, "admission": True, "admission_existing_targets": "lp_reachable", "eps_q": 0.008},
    "reward": {"type": "profit", "w_E": 10.0, "w_P": 10.0, "dense": False},
    "conformal": {"enabled": False, "alpha": 0.1, "cache_dir": "results/conformal", "n_train": 2000,
                  "n_cal": 5000, "predictor": "gbr", "uncalibrated": False},
    "features": {"forecast_rel_std": 0.1},
    "train": {"save_every": 50, "torch_threads": 0, "deterministic": False, "keep_checkpoints": False},
    "eval": {"test_grid": "configs/test_grid.yaml", "test_sets": "results/test_sets", "max_replays": None},
}

#: Keys excluded from the configuration hash.
HASH_EXCLUDE = {"smoke"}


class Cfg(dict):
    """dict with attribute access (recursively)."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v

    def __setattr__(self, k, v):
        self[k] = v


def deep_merge(base: dict, over: dict, path: str = "", strict: bool = True) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        p = f"{path}.{k}" if path else k
        if strict and k not in base:
            raise KeyError(f"unknown configuration key '{p}'")
        if isinstance(v, dict) and isinstance(base.get(k), dict) and k not in ("extra",):
            out[k] = deep_merge(base[k], v, p, strict)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml_chain(path: Path) -> dict:
    d = yaml.safe_load(path.read_text()) or {}
    parent = d.pop("inherit", None)
    if parent:
        base = _load_yaml_chain((path.parent / parent).resolve())
        return deep_merge(base, d, strict=False)
    return d


def parse_override(s: str) -> tuple[list[str], Any]:
    """'ppo.lr=1e-4' -> (['ppo','lr'], 1e-4) (value parsed as YAML)."""
    k, v = s.split("=", 1)
    return k.strip().split("."), yaml.safe_load(v)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None,
                overlays: list[str | Path] | None = None) -> Cfg:
    """Defaults <- config file (with inheritance) <- overlay files (e.g. configs/smoke.yaml) <- --set overrides."""
    cfg = copy.deepcopy(DEFAULTS)
    name = "default"
    if path is not None:
        path = Path(path)
        cfg = deep_merge(cfg, _load_yaml_chain(path))
        name = path.stem
    for ov in overlays or []:
        cfg = deep_merge(cfg, _load_yaml_chain(Path(ov)))
    for o in overrides or []:
        keys, v = parse_override(o)
        d = cfg
        for k in keys[:-1]:
            if k not in d:
                raise KeyError(f"unknown configuration key '{o}'")
            d = d[k]
        if keys[-1] not in d:
            raise KeyError(f"unknown configuration key '{o}'")
        d[keys[-1]] = v
    cfg["_name"] = name
    return Cfg(cfg)


def config_hash8(cfg: dict) -> str:
    c = {k: v for k, v in cfg.items() if k not in HASH_EXCLUDE and not k.startswith("_")}
    s = json.dumps(c, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:8]


def exp_id(method: str, cfg_name: str, seed: int, h8: str) -> str:
    """code_spec.md Sec. 2: {method}__{train_or_test_cfg}__s{seed}__{config_hash8}."""
    return f"{method}__{cfg_name}__s{seed}__{h8}"


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(json.loads(json.dumps(dict(cfg), default=str)), sort_keys=False))
