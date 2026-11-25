"""Certified-risk frontier utilities.

Uses the public nonnegative denominator:
    S(Y) = max( <w, Y^+[pi_hat]> , STAB ) with STAB = 1e-6.
Normalized certificates: C_tilde_m = C_m / S(Y).
Fail-closed: non-finite C or S<=0 after stab -> C_tilde = +inf (not certified).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

STAB = 1e-6
TOL = 1e-12

TAU_GRID = (0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 4.0)
C_GRID = (0.10, 0.25, 0.50, 0.75, 0.90)


def normalize_certificate(C: float, S: float) -> float:
    """Common normalization; fail-closed on non-finite / nonpositive scales."""
    if not np.isfinite(C) or not np.isfinite(S):
        return float("inf")
    denom = max(float(S), STAB)
    if denom <= 0.0:
        return float("inf")
    val = float(C) / denom
    if not np.isfinite(val):
        return float("inf")
    return val


def release_scale_U(y: np.ndarray, pi_hat: np.ndarray, w: np.ndarray) -> float:
    """Release scale U = <w, Y^+[pi_hat]> (before STAB)."""
    yplus = np.maximum(np.asarray(y, dtype=np.float64), 0.0)
    ww = np.asarray(w, dtype=np.float64)
    pi = np.asarray(pi_hat, dtype=np.int64)
    return float(np.dot(ww[: pi.size], yplus[pi]))


def certified_mask(c_tilde: np.ndarray, tau: float) -> np.ndarray:
    return np.asarray(c_tilde, dtype=np.float64) <= float(tau)


def coverage_K(c_tilde: np.ndarray, tau: float) -> float:
    x = np.asarray(c_tilde, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.mean(certified_mask(x, tau)))


def inverse_T(c_tilde: np.ndarray, c: float) -> float:
    """Empirical T(c)=inf{tau: K(tau)>=c} = ceil(c N)-th order statistic.

    Ties: np.sort is stable for equal values; callers should pass values already
    ordered by a deterministic secondary key if needed for set membership, but
    the quantile value itself is unique for equal C_tilde.
    """
    x = np.sort(np.asarray(c_tilde, dtype=np.float64))
    n = int(x.size)
    if n == 0:
        return float("nan")
    c = float(c)
    if c <= 0.0:
        return float(x[0])
    if c > 1.0:
        return float("inf")
    idx = int(np.ceil(c * n)) - 1
    idx = max(0, min(idx, n - 1))
    return float(x[idx])


def truth_macro_K(rows: list[dict], key: str, tau: float) -> float:
    g: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        g[str(r["truth_id"])].append(float(r[key]))
    if not g:
        return float("nan")
    return float(np.mean([coverage_K(np.asarray(v), tau) for v in g.values()]))


def truth_macro_T(rows: list[dict], key: str, c: float) -> float:
    g: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        g[str(r["truth_id"])].append(float(r[key]))
    if not g:
        return float("nan")
    return float(np.mean([inverse_T(np.asarray(v), c) for v in g.values()]))


def dataset_macro_K(rows: list[dict], key: str, tau: float) -> float:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[str(r["dataset_id"])].append(r)
    if not by:
        return float("nan")
    return float(np.mean([truth_macro_K(v, key, tau) for v in by.values()]))


def dataset_macro_T(rows: list[dict], key: str, c: float) -> float:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[str(r["dataset_id"])].append(r)
    if not by:
        return float("nan")
    return float(np.mean([truth_macro_T(v, key, c) for v in by.values()]))


def surface_macro_K(rows: list[dict], key: str, tau: float) -> float:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[str(r.get("surface_id", r["dataset_id"]))].append(r)
    if not by:
        return float("nan")
    return float(np.mean([truth_macro_K(v, key, tau) for v in by.values()]))


def set_inclusion_holds(c_box: np.ndarray, c_sep: np.ndarray, tau: float) -> bool:
    """D_sep(tau)subseteq D_box(tau) under common normalization values."""
    mb = certified_mask(c_box, tau)
    ms = certified_mask(c_sep, tau)
    return bool(np.all(ms <= mb))


def pointwise_dominance(c_box: np.ndarray, c_sep: np.ndarray, atol: float = TOL) -> bool:
    return bool(np.all(np.asarray(c_box) <= np.asarray(c_sep) + atol))


def incremental_indices(c_box: np.ndarray, c_sep: np.ndarray, tau: float) -> np.ndarray:
    mb = certified_mask(c_box, tau)
    ms = certified_mask(c_sep, tau)
    return np.where(mb & ~ms)[0]
