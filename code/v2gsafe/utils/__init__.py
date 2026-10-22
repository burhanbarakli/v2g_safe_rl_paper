"""Utilities: seeding, run metadata (meta.json), CSV appending, timing."""
from __future__ import annotations

import csv
import datetime
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

EV2GYM_COMMIT_PINNED = "55ef368c25a58476a9eeaed5111f112a63a38ca8"
CODE_ROOT = Path(__file__).resolve().parents[2]


def set_global_seed(seed: int, deterministic_torch: bool = False) -> None:
    """Seed python, numpy and torch (deterministic seeding, code_spec Sec. 4)."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch
        torch.manual_seed(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # pragma: no cover
        pass


def _git(args, cwd) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def project_git_hash() -> str:
    h = _git(["rev-parse", "HEAD"], CODE_ROOT)
    if h is None:
        return "unknown"
    dirty = _git(["status", "--porcelain"], CODE_ROOT)
    return h + ("-dirty" if dirty else "")


def ev2gym_commit() -> dict:
    """Commit of the installed EV2Gym (git checkout of an editable install) and whether it matches the pin."""
    try:
        import ev2gym
        root = Path(ev2gym.__file__).resolve().parents[1]
        h = _git(["rev-parse", "HEAD"], root)
    except Exception:  # noqa: BLE001
        h = None
    return {"ev2gym_commit": h or "unknown", "ev2gym_commit_pinned": EV2GYM_COMMIT_PINNED,
            "ev2gym_commit_matches": bool(h == EV2GYM_COMMIT_PINNED)}


def package_versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for name in ("numpy", "torch", "pandas", "scipy", "sklearn", "highspy", "gymnasium", "yaml", "ev2gym"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "installed")
        except Exception:  # noqa: BLE001
            out[name] = None
    try:
        from importlib.metadata import version
        out["highspy"] = version("highspy")
        out["ev2gym"] = version("ev2gym")
    except Exception:  # noqa: BLE001
        pass
    return out


class RunMeta:
    """Collects meta.json (code_spec Sec. 2)."""

    def __init__(self, out_dir: str | Path, smoke: bool, device: str = "cpu", extra: dict | None = None):
        self.path = Path(out_dir) / "meta.json"
        self.t0 = time.time()
        self.meta = {"project_git_hash": project_git_hash(), **ev2gym_commit(), "packages": package_versions(),
                     "hostname": socket.gethostname(), "platform": platform.platform(), "device": device,
                     "start_time": datetime.datetime.now().isoformat(timespec="seconds"), "end_time": None,
                     "wall_clock_s": None, "smoke": bool(smoke), "argv": sys.argv, **(extra or {})}
        if self.path.exists():  # resumed run: keep history of previous segments
            try:
                old = json.loads(self.path.read_text())
                self.meta["previous_segments"] = old.get("previous_segments", []) + [
                    {k: old.get(k) for k in ("start_time", "end_time", "wall_clock_s", "hostname")}]
            except Exception:  # noqa: BLE001
                pass
        self.write()

    def update(self, **kw) -> None:
        self.meta.update(kw)
        self.write()

    def finish(self, **kw) -> None:
        self.meta.update(end_time=datetime.datetime.now().isoformat(timespec="seconds"),
                         wall_clock_s=time.time() - self.t0, **kw)
        self.write()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.meta, indent=1, default=str) + "\n")


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (float, np.floating)):
        return "" if (isinstance(v, float) and math.isnan(v)) or (isinstance(v, np.floating) and np.isnan(v)) \
            else repr(float(v))
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def append_csv(path: str | Path, rows: list[dict], columns: list[str]) -> None:
    """Append rows (missing keys -> empty) with a fixed header."""
    path = Path(path)
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({c: _fmt(r.get(c)) for c in columns})


class Timer:
    def __init__(self):
        self.t = time.perf_counter()

    def ms(self) -> float:
        return (time.perf_counter() - self.t) * 1000.0
