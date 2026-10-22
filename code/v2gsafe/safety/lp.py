"""Thin HiGHS (highspy) LP helper shared by admission control, the oracle and the eMPC port.

Builds an LP  min/max c'x  s.t.  row_lo <= A x <= row_hi,  col_lo <= x <= col_hi  from a sparse matrix in
coordinate form and solves it with HiGHS [huangfu2018parallelizing]."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

try:
    import highspy
    INF = highspy.kHighsInf
except ImportError:  # pragma: no cover
    highspy = None
    INF = np.inf


@dataclass
class LPResult:
    status: str
    optimal: bool
    x: np.ndarray | None
    obj: float
    time_s: float


class LPBuilder:
    """Incremental builder: add variables (blocks) and constraint rows in COO form."""

    def __init__(self):
        self.n = 0
        self.col_lo: list[np.ndarray] = []
        self.col_hi: list[np.ndarray] = []
        self.rows, self.cols, self.vals = [], [], []
        self.row_lo: list[float] = []
        self.row_hi: list[float] = []

    def add_vars(self, k: int, lo, hi) -> np.ndarray:
        idx = np.arange(self.n, self.n + k)
        self.n += k
        self.col_lo.append(np.broadcast_to(np.asarray(lo, float), (k,)).copy())
        self.col_hi.append(np.broadcast_to(np.asarray(hi, float), (k,)).copy())
        return idx

    def add_row(self, cols, vals, lo=-INF, hi=INF) -> int:
        r = len(self.row_lo)
        cols = np.atleast_1d(np.asarray(cols, int))
        vals = np.broadcast_to(np.asarray(vals, float), cols.shape)
        self.rows.append(np.full(cols.shape, r))
        self.cols.append(cols)
        self.vals.append(vals)
        self.row_lo.append(float(lo))
        self.row_hi.append(float(hi))
        return r

    def add_rows_coo(self, nrows: int, r, c, v, lo, hi) -> np.ndarray:
        """Add ``nrows`` rows at once; r are local row indices (0..nrows-1). Returns the global row indices."""
        base = len(self.row_lo)
        self.rows.append(base + np.asarray(r, int))
        self.cols.append(np.asarray(c, int))
        self.vals.append(np.asarray(v, float))
        self.row_lo.extend(np.broadcast_to(np.asarray(lo, float), (nrows,)).tolist())
        self.row_hi.extend(np.broadcast_to(np.asarray(hi, float), (nrows,)).tolist())
        return np.arange(base, base + nrows)

    def add_rows(self, rows_cols_vals: list, lo, hi) -> None:
        for (c, v), a, b in zip(rows_cols_vals, np.broadcast_to(lo, (len(rows_cols_vals),)),
                                np.broadcast_to(hi, (len(rows_cols_vals),))):
            self.add_row(c, v, a, b)

    def matrix(self):
        m = len(self.row_lo)
        if m == 0:
            return sp.csc_matrix((0, self.n))
        r = np.concatenate(self.rows)
        c = np.concatenate(self.cols)
        v = np.concatenate(self.vals)
        return sp.coo_matrix((v, (r, c)), shape=(m, self.n)).tocsc()

    def solve(self, cost, sense: str = "min", time_limit: float | None = None, extra_rows=None,
              options: dict | None = None) -> LPResult:
        A = self.matrix()
        row_lo = np.asarray(self.row_lo, float)
        row_hi = np.asarray(self.row_hi, float)
        if extra_rows is not None:  # list of (dense coefficient vector, lo, hi)
            add = sp.csc_matrix(np.vstack([e[0] for e in extra_rows]))
            A = sp.vstack([A, add]).tocsc()
            row_lo = np.concatenate([row_lo, [e[1] for e in extra_rows]])
            row_hi = np.concatenate([row_hi, [e[2] for e in extra_rows]])
        return solve_lp(np.asarray(cost, float), A, row_lo, row_hi, np.concatenate(self.col_lo),
                        np.concatenate(self.col_hi), sense, time_limit, options)


def add_ev_dynamics(lp: LPBuilder, pc: np.ndarray, ec: np.ndarray, E0: float, dt_h: float,
                    rho: float | None = None, cap: float | None = None) -> None:
    """Rows E(1) - dt p(0) = E0, E(k+1) - E(k) - dt p(k) = 0 (g = id, F5) and, if ``rho`` is given, the linear
    taper minorant p(k) + rho E(k) <= rho E_cap for k >= 1 (safety_layer.md Sec. 7.4).
    pc[k] = p(t0+k), ec[k] = E(t0+k+1), k = 0..K-1."""
    K = len(pc)
    k = np.arange(1, K)
    r = np.concatenate([[0, 0], np.repeat(k, 3)])
    c = np.concatenate([[ec[0], pc[0]], np.stack([ec[1:], ec[:-1], pc[1:]], 1).ravel()])
    v = np.concatenate([[1.0, -dt_h], np.tile([1.0, -1.0, -dt_h], K - 1)])
    rhs = np.zeros(K)
    rhs[0] = E0
    lp.add_rows_coo(K, r, c, v, rhs, rhs)
    if rho is not None and K > 1:
        r2 = np.repeat(np.arange(K - 1), 2)
        c2 = np.stack([pc[1:], ec[:-1]], 1).ravel()
        v2 = np.tile([1.0, rho], K - 1)
        lp.add_rows_coo(K - 1, r2, c2, v2, -INF, rho * cap)


def add_ev_dynamics_split(lp: LPBuilder, pp: np.ndarray, pm: np.ndarray, ec: np.ndarray, E0: float, dt_h: float,
                          rho: float | None = None, cap: float | None = None, E0_taper: float | None = None) -> None:
    """Rows E(1) - dt (p+(0) - p-(0)) = E0 and E(k+1) - E(k) - dt (p+(k) - p-(k)) = 0 (eta = 1, F5/O2).

    If ``rho`` is given, add the linear taper constraint p+(k) <= rho (E_cap - E(k)) for every step: for k = 0
    with the known start energy (``E0_taper``, default E0) as a column bound is left to the caller, for k >= 1
    as rows p+(k) + rho E(k) <= rho E_cap (E(k) = ec[k-1])."""
    K = len(pp)
    r = np.concatenate([[0, 0, 0], np.repeat(np.arange(1, K), 4)])
    c = np.concatenate([[ec[0], pp[0], pm[0]], np.stack([ec[1:], ec[:-1], pp[1:], pm[1:]], 1).ravel()])
    v = np.concatenate([[1.0, -dt_h, dt_h], np.tile([1.0, -1.0, -dt_h, dt_h], K - 1)])
    rhs = np.zeros(K)
    rhs[0] = E0
    lp.add_rows_coo(K, r, c, v, rhs, rhs)
    if rho is not None and K > 1:
        r2 = np.repeat(np.arange(K - 1), 2)
        c2 = np.stack([pp[1:], ec[:-1]], 1).ravel()
        v2 = np.tile([1.0, rho], K - 1)
        lp.add_rows_coo(K - 1, r2, c2, v2, -INF, rho * cap)


def solve_lp(c, A, row_lo, row_hi, col_lo, col_hi, sense="min", time_limit=None, options=None) -> LPResult:
    if highspy is None:  # pragma: no cover
        raise ImportError("highspy is required (pip install highspy)")
    t0 = time.perf_counter()
    A = sp.csc_matrix(A)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    if time_limit is not None:
        h.setOptionValue("time_limit", float(time_limit))
    for k, v in (options or {}).items():
        h.setOptionValue(k, v)
    lp = highspy.HighsLp()
    lp.num_col_ = int(A.shape[1])
    lp.num_row_ = int(A.shape[0])
    lp.col_cost_ = np.asarray(c, float)
    lp.col_lower_ = np.where(np.isfinite(col_lo), col_lo, -INF).astype(float)
    lp.col_upper_ = np.where(np.isfinite(col_hi), col_hi, INF).astype(float)
    lp.row_lower_ = np.where(np.isfinite(row_lo), row_lo, -INF).astype(float)
    lp.row_upper_ = np.where(np.isfinite(row_hi), row_hi, INF).astype(float)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(float)
    lp.sense_ = highspy.ObjSense.kMaximize if sense == "max" else highspy.ObjSense.kMinimize
    h.passModel(lp)
    h.run()
    st = h.getModelStatus()
    status = h.modelStatusToString(st)
    ok = st == highspy.HighsModelStatus.kOptimal
    x = np.array(h.getSolution().col_value, float) if ok else None
    obj = float(h.getInfo().objective_function_value) if ok else float("nan")
    return LPResult(status=status, optimal=ok, x=x, obj=obj, time_s=time.perf_counter() - t0)
