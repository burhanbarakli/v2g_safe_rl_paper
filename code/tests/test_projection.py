"""Projection (safety_layer.md Sec. 4, Alg. 1.2): feasibility, agreement with projection_ref, Jacobian, Moreau."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import adversarial_families, random_instance
from reference import projection_ref as R
from v2gsafe.constants import EPS_NUM
from v2gsafe.safety import projection as P


def _check_feasible(p, l, u, C, Crev, tol=EPS_NUM):
    sp = p.sum(1)
    assert np.all(p >= l - 1e-12) and np.all(p <= u + 1e-12), "box violated"
    assert np.all(sp <= C + tol) and np.all(sp >= -Crev - tol), "aggregate violated"


def test_10000_random_inputs_satisfy_bounds_numpy_and_torch(rng):
    """10,000 random rows incl. tight A5 and |x| up to 1e8: bounds and aggregate limits hold (code_spec Sec. 4)."""
    fams = adversarial_families(rng, n_rows=1700)
    A = np.concatenate([f[0] for f in fams.values()])
    L = np.concatenate([f[1] for f in fams.values()])
    U = np.concatenate([f[2] for f in fams.values()])
    C = np.concatenate([f[3] for f in fams.values()])
    Cr = np.concatenate([f[4] for f in fams.values()])
    assert A.shape[0] >= 10_000
    p, lam, case = P.project(A, L, U, C, Cr)
    _check_feasible(p, L, U, C, Cr)
    pt, _, _ = P.project_batched(*(torch.tensor(v) for v in (A, L, U, C, Cr)), torch.ones(A.shape, dtype=torch.bool))
    _check_feasible(pt.numpy(), L, U, C, Cr)
    # both tight families must actually exercise the active branch
    assert (case != 0).sum() > 5000


def test_numpy_port_identical_to_reference(rng):
    for fam, (A, L, U, C, Cr) in adversarial_families(rng, 1000).items():
        pr, lr, cr = R.project(A, L, U, C, Cr)
        pn, ln, cn = P.project(A, L, U, C, Cr)
        np.testing.assert_array_equal(pn, pr)
        np.testing.assert_array_equal(cn, cr)


@pytest.mark.parametrize("N", [1, 2, 5, 20, 60, 200])
def test_torch_matches_reference_1e9(rng, N):
    """PyTorch port matches projection_ref to 1e-9 in float64 (algorithm.md Alg. 1 preamble), T1 family."""
    rows = [random_instance(rng, N) for _ in range(200)]
    A, L, U = (np.stack([r[k] for r in rows]) for k in range(3))
    C = np.array([r[3] for r in rows])
    Cr = np.array([r[4] for r in rows])
    pr, _, _ = R.project(A, L, U, C, Cr)
    pt, _, _ = P.project_batched(*(torch.tensor(v) for v in (A, L, U, C, Cr)), torch.ones(A.shape, dtype=torch.bool))
    assert np.abs(pt.numpy() - pr).max() <= 1e-9
    for b in range(0, 200, 20):
        pb, _ = R.project_bisect(A[b], L[b], U[b], C[b], Cr[b])
        assert np.abs(pr[b] - pb).max() < 1e-8


def test_torch_matches_reference_adversarial(rng):
    """T1b families (tight A5, |x| up to 1e8, hyperplane): relative agreement 1e-9."""
    for fam, (A, L, U, C, Cr) in adversarial_families(rng, 1500).items():
        pr, _, _ = R.project(A, L, U, C, Cr)
        pt, _, _ = P.project_batched(*(torch.tensor(v) for v in (A, L, U, C, Cr)),
                                      torch.ones(A.shape, dtype=torch.bool))
        rel = np.abs(pt.numpy() - pr) / (1.0 + np.abs(A).max(1, keepdims=True))
        assert rel.max() <= 1e-9, fam


def test_padding_and_masks(rng):
    """Batched rows of mixed N padded to 200 with garbage in padded slots: identical to unpadded rows."""
    B, Nmax = 64, 200
    A = np.zeros((B, Nmax)); L = np.zeros((B, Nmax)); U = np.zeros((B, Nmax)); M = np.zeros((B, Nmax), bool)
    Cs = np.zeros(B); Cr = np.zeros(B); refs = []
    for b in range(B):
        N = int(rng.integers(1, Nmax + 1))
        a, l, u, C, Crev = random_instance(rng, N)
        A[b, :N], L[b, :N], U[b, :N], M[b, :N], Cs[b], Cr[b] = a, l, u, True, C, Crev
        A[b, N:] = rng.normal(0, 1e3, Nmax - N)       # garbage
        L[b, N:] = rng.normal(0, 10, Nmax - N)
        U[b, N:] = L[b, N:] + 1.0
        refs.append(R.project(a[None], l[None], u[None], np.array([C]), np.array([Crev]))[0][0])
    pn, _, _ = P.project(A, L, U, Cs, Cr, mask=M)
    pt, _, _ = P.project_batched(*(torch.tensor(v) for v in (A, L, U, Cs, Cr, M)))
    for b in range(B):
        n = len(refs[b])
        np.testing.assert_allclose(pn[b, :n], refs[b], atol=1e-12)
        np.testing.assert_allclose(pt.numpy()[b, :n], refs[b], atol=1e-9)
    assert np.all(pn[~M] == 0) and np.all(pt.numpy()[~M] == 0)


def test_nonfinite_guard_and_counters():
    x = np.array([[np.nan, np.inf, -np.inf, 3.0]])
    c = P.GuardCounters()
    p, _, _ = P.project(x, np.zeros((1, 4)), np.ones((1, 4)), np.array([10.0]), np.array([10.0]), counters=c)
    assert c.nonfinite == 3 and np.all(np.isfinite(p))
    np.testing.assert_array_equal(p[0], [0, 0, 0, 1])
    ct = P.GuardCounters()
    pt, _, _ = P.project_batched(torch.tensor(x), torch.zeros(1, 4, dtype=torch.float64), torch.ones(1, 4, dtype=torch.float64),
                                 torch.tensor([10.0]), torch.tensor([10.0]), torch.ones(1, 4, dtype=torch.bool), counters=ct)
    assert ct.nonfinite == 3
    np.testing.assert_array_equal(pt.numpy()[0], [0, 0, 0, 1])


def test_is_feasible_matches_reference(rng):
    for _ in range(200):
        N = 7
        l = rng.uniform(-5, 5, (1, N)); u = l + rng.uniform(0, 3, (1, N))
        C = rng.uniform(-20, 40); Cr = rng.uniform(-20, 40)
        assert bool(P.is_feasible(l, u, C, Cr)[0]) == bool(R.is_feasible(l, u, C, Cr)[0])


def test_jacobian_vs_finite_differences(rng):
    """(S16) against central finite differences (T2), NumPy and batched torch."""
    n = 0
    worst = 0.0
    while n < 150:
        N = int(rng.choice([2, 5, 20]))
        a, l, u, C, Crev = random_instance(rng, N)
        p, lam, case = P.project(a[None], l[None], u[None], np.array([C]), np.array([Crev]))
        lam, case = lam[0], int(case[0])
        z = a - lam
        if np.min(np.abs(np.concatenate([z - l, z - u]))) < 1e-3:
            continue
        if case != 0 and min(abs(p[0].sum() - C), abs(p[0].sum() + Crev)) > 1e-9:
            continue
        J = P.jacobian(a, l, u, lam, case)
        h = 1e-6
        Jfd = np.zeros((N, N))
        for j in range(N):
            e = np.zeros(N); e[j] = h
            pp = P.project((a + e)[None], l[None], u[None], np.array([C]), np.array([Crev]))[0][0]
            pm = P.project((a - e)[None], l[None], u[None], np.array([C]), np.array([Crev]))[0][0]
            Jfd[:, j] = (pp - pm) / (2 * h)
        worst = max(worst, np.abs(J - Jfd).max())
        Jt = P.jacobian_batched(torch.tensor(a[None]), torch.tensor(l[None]), torch.tensor(u[None]),
                                torch.tensor([lam]), torch.tensor([case]), torch.ones(1, N, dtype=torch.bool))
        np.testing.assert_allclose(Jt.numpy()[0], J, atol=1e-12)
        n += 1
    assert worst < 1e-6


def test_moreau_gradient_of_regulariser(rng):
    """grad of 0.5*||x - Pi(x)||^2 with Pi under stop-gradient equals x - Pi(x) (S 9, T3)."""
    for _ in range(50):
        a, l, u, C, Crev = random_instance(rng, 10)
        x = torch.tensor(a[None], requires_grad=True)
        with torch.no_grad():
            p, _, _ = P.project_batched(x.detach(), torch.tensor(l[None]), torch.tensor(u[None]), torch.tensor([C]),
                                        torch.tensor([Crev]), torch.ones(1, 10, dtype=torch.bool))
        loss = 0.5 * ((x - p) ** 2).sum()
        loss.backward()
        g = x.grad.numpy()[0]
        f = lambda v: 0.5 * np.sum((v - R.project(v[None], l[None], u[None], np.array([C]), np.array([Crev]))[0][0]) ** 2)
        h = 1e-6
        gfd = np.array([(f(a + h * np.eye(10)[j]) - f(a - h * np.eye(10)[j])) / (2 * h) for j in range(10)])
        assert np.abs(g - gfd).max() < 1e-5
