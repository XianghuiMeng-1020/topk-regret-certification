"""Independent count-feasible reference. It does not call the assignment certificate."""

from __future__ import annotations

from itertools import permutations

import numpy as np
from scipy.optimize import linprog


class CountReferenceError(ValueError):
    """The independent count-feasible reference rejected an input."""


def deploy(y: np.ndarray, k: int) -> np.ndarray:
    values = np.maximum(np.asarray(y, dtype=np.float64), 0.0)
    return np.lexsort((np.arange(values.size), -values))[:k].astype(np.int64)


def weight_vector(pi: np.ndarray, weights: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float64)
    for rank, item in enumerate(np.asarray(pi, dtype=np.int64).tolist()):
        out[int(item)] += float(weights[rank])
    return out


def endpoints(y: np.ndarray, u_minus: float, u_plus: float) -> tuple[bool, np.ndarray]:
    values = np.asarray(y, dtype=np.float64).reshape(-1)
    if np.any(values < -float(u_minus) - 1e-9 * (1.0 + abs(float(u_minus)) + np.max(np.abs(values)))):
        return False, np.minimum(float(u_plus), values)
    return True, np.minimum(float(u_plus), np.maximum(values, -float(u_minus)))


def endpoint_inner(coefficients: np.ndarray, u_minus: float, upper: np.ndarray) -> float:
    coeff = np.asarray(coefficients, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    return float(np.dot(np.maximum(coeff, 0.0), high) - float(u_minus) * np.minimum(coeff, 0.0).sum())


def support_lp(coefficients: np.ndarray, u_minus: float, upper: np.ndarray) -> float:
    coeff = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    high = np.asarray(upper, dtype=np.float64).reshape(-1)
    result = linprog(
        -coeff,
        bounds=[(-float(u_minus), float(high[i])) for i in range(coeff.size)],
        method="highs",
    )
    if not result.success:
        raise CountReferenceError(result.message)
    return float(-result.fun)


def enumerate_count_feasible(
    y: np.ndarray, k: int, weights: np.ndarray, u_minus: float, u_plus: float
) -> float:
    values = np.asarray(y, dtype=np.float64).reshape(-1)
    n = values.size
    ww = np.asarray(weights, dtype=np.float64).reshape(-1)[:k]
    ok, upper = endpoints(values, u_minus, u_plus)
    if not ok:
        raise CountReferenceError("empty intersection is not enumerated")
    pi_hat = deploy(values, k)
    hat = weight_vector(pi_hat, ww, n)
    base = float(np.dot(ww, values[pi_hat]))
    best = 0.0
    for perm in permutations(range(n), k):
        pi = np.asarray(perm, dtype=np.int64)
        margin = float(np.dot(ww, values[pi]) - base)
        coeff = hat - weight_vector(pi, ww, n)
        best = max(best, margin + endpoint_inner(coeff, u_minus, upper))
    return float(best)
