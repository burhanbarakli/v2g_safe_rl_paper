"""Shared pytest fixtures. Tests never write to results/raw; they use temporary directories."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")


def ev2gym_available() -> bool:
    try:
        import ev2gym  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


requires_ev2gym = pytest.mark.skipif(not ev2gym_available(), reason="EV2Gym not installed")


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def random_instance(rng, N):
    """Random feasible projection instance (as verify_projection.py T1)."""
    l = rng.uniform(-11, 6, N)
    u = l + rng.uniform(0, 11, N) * (rng.random(N) > 0.1)
    a = rng.normal(0, 15, N)
    lo_s, hi_s = l.sum(), u.sum()
    C = rng.uniform(lo_s, hi_s + 20)
    Crev = rng.uniform(-min(C, hi_s), -lo_s + 20) if rng.random() < 0.7 else rng.uniform(-hi_s, -lo_s)
    Crev = max(Crev, -hi_s)
    if -Crev > C:
        Crev = -C
    return a, l, u, C, Crev


def adversarial_families(rng, n_rows=2000, N=20):
    """verify_projection.py T1b families: tight A5 (upper/lower), |x| up to 1e8, hyperplane C = -C_rev."""
    from reference import projection_ref as R

    out = {}
    for fam in ["tight_upper", "tight_lower", "tight_upper_huge", "huge_general", "hyperplane", "general"]:
        L = rng.uniform(-11, 6, (n_rows, N))
        U = L + rng.uniform(0, 11, (n_rows, N))
        scale = 10.0 ** rng.uniform(4, 8, (n_rows, 1)) if "huge" in fam else 15.0
        A = rng.normal(0, 1, (n_rows, N)) * scale
        if fam.startswith("tight_upper"):
            Cs = L.sum(1)
            Cr = -L.sum(1) + rng.uniform(0, 50, n_rows)
            A = np.abs(A) + U
        elif fam == "tight_lower":
            Cr = -U.sum(1)
            Cs = U.sum(1) + rng.uniform(0, 50, n_rows)
            A = -np.abs(A) + L
        elif fam == "hyperplane":
            Cs = rng.uniform(L.sum(1), U.sum(1))
            Cr = -Cs
        else:
            Cs = rng.uniform(L.sum(1), U.sum(1) + 20)
            Cr = rng.uniform(-np.minimum(Cs, U.sum(1)), -L.sum(1) + 20)
        f = R.is_feasible(L, U, Cs, Cr)
        out[fam] = (A[f], L[f], U[f], Cs[f], Cr[f])
    return out
