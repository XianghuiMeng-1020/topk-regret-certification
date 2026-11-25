"""Independent order-statistic reference used only by the order-statistic tests.

The support formula and the linear program are written here. They do not call
orderstat_noise_bound or certificate_os_envelope.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
from scipy.optimize import linprog


class OSReferenceError(ValueError):
    """The independent reference rejected an input or a solver status."""


def prefix_min(thresholds: np.ndarray) -> np.ndarray:
    raw = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if raw.size == 0 or not np.all(np.isfinite(raw)) or np.any(raw < 0.0):
        raise OSReferenceError("thresholds must be nonempty, finite, and nonnegative")
    return np.minimum.accumulate(raw)


def support_formula(coefficients: np.ndarray, u_plus: np.ndarray, u_minus: np.ndarray) -> float:
    """Closed form using prefix-minimum thresholds."""
    coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    upper = prefix_min(u_plus)
    lower = prefix_min(u_minus)
    positive = np.sort(coefficients[coefficients > 0.0])[::-1]
    negative = np.sort(-coefficients[coefficients < 0.0])[::-1]
    if positive.size > upper.size or negative.size > lower.size:
        raise OSReferenceError("threshold table shorter than the coefficient support")
    value = 0.0
    if positive.size:
        value += float(np.dot(positive, upper[: positive.size]))
    if negative.size:
        value += float(np.dot(negative, lower[: negative.size]))
    return float(value)


def support_lp(coefficients: np.ndarray, u_plus: np.ndarray, u_minus: np.ndarray) -> float:
    """Maximize <a, Z> over ordered noise obeying the raw order-statistic inequalities.

    Variables are the ascending coordinates zeta_1 <= ... <= zeta_n. The r-th
    largest is zeta_{n-r+1} and the r-th smallest is zeta_r. Permutation
    symmetry removes the assignment of coordinates to items: the optimum pairs
    the sorted coefficients with this ordered noise vector.
    """
    coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    upper = np.asarray(u_plus, dtype=np.float64).reshape(-1)
    lower = np.asarray(u_minus, dtype=np.float64).reshape(-1)
    n = int(coefficients.size)
    if n == 0 or upper.size == 0 or lower.size == 0:
        raise OSReferenceError("empty support problem")
    if np.any(~np.isfinite(upper)) or np.any(~np.isfinite(lower)):
        raise OSReferenceError("non-finite thresholds")
    if np.any(upper < 0.0) or np.any(lower < 0.0):
        raise OSReferenceError("negative thresholds")
    ascending = np.sort(coefficients)
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for index in range(n - 1):
        row = np.zeros(n, dtype=np.float64)
        row[index] = 1.0
        row[index + 1] = -1.0
        rows.append(row)
        rhs.append(0.0)
    n_upper = min(n, upper.size)
    n_lower = min(n, lower.size)
    for rank in range(1, n_upper + 1):
        row = np.zeros(n, dtype=np.float64)
        row[n - rank] = 1.0
        rows.append(row)
        rhs.append(float(upper[rank - 1]))
    for rank in range(1, n_lower + 1):
        row = np.zeros(n, dtype=np.float64)
        row[rank - 1] = -1.0
        rows.append(row)
        rhs.append(float(lower[rank - 1]))
    result = linprog(
        -ascending,
        A_ub=np.asarray(rows, dtype=np.float64),
        b_ub=np.asarray(rhs, dtype=np.float64),
        bounds=[(None, None)] * n,
        method="highs",
    )
    if not result.success:
        raise OSReferenceError(f"order-statistic LP failed: {result.message}")
    return float(-result.fun)


def enumerate_os(
    y: np.ndarray,
    k: int,
    weights: np.ndarray,
    u_plus: np.ndarray,
    u_minus: np.ndarray,
) -> float:
    """Maximize G_Y + independent OS support over every injection."""
    values = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(values.size)
    ww = np.asarray(weights, dtype=np.float64).reshape(-1)[:k]
    clipped = np.maximum(values, 0.0)
    pi_hat = np.lexsort((np.arange(n), -clipped))[:k]
    base = float(np.dot(ww, values[pi_hat]))
    best = 0.0
    for perm in permutations(range(n), k):
        pi = np.asarray(perm, dtype=np.int64)
        margin = float(np.dot(ww, values[pi]) - base)
        coefficient = np.zeros(n, dtype=np.float64)
        for rank, item in enumerate(pi_hat.tolist()):
            coefficient[int(item)] += float(ww[rank])
        for rank, item in enumerate(pi.tolist()):
            coefficient[int(item)] -= float(ww[rank])
        best = max(best, margin + support_formula(coefficient, u_plus, u_minus))
    return float(best)
