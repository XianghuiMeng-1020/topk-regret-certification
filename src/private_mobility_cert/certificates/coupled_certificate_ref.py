"""Independent small-n reference for coupled release-conditioned certificates.

This module enumerates every ordered injection and independently implements
the order-statistic support-function formula used by the certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np


class CoupledCertificateRefError(ValueError):
    """Invalid input to the independent coupled-certificate reference."""


def _validate_weights_ref(w: np.ndarray) -> np.ndarray:
    weights = np.asarray(w, dtype=np.float64).reshape(-1)
    if weights.size == 0:
        raise CoupledCertificateRefError("weights must be nonempty")
    if not np.all(np.isfinite(weights)):
        raise CoupledCertificateRefError("weights must be finite")
    if np.any(weights < 0.0):
        raise CoupledCertificateRefError("weights must be nonnegative")
    if np.any(weights[1:] - weights[:-1] > 1e-12):
        raise CoupledCertificateRefError("weights must be nonincreasing")
    return weights


def _stable_ranking_ref(y: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(y, dtype=np.float64).reshape(-1)
    if k < 0:
        raise CoupledCertificateRefError("k must be nonnegative")
    order = np.lexsort((np.arange(values.size), -values))
    return order[: min(int(k), values.size)].astype(np.int64)


def _deployed_ranking_ref(y: np.ndarray, k: int) -> np.ndarray:
    return _stable_ranking_ref(np.maximum(np.asarray(y, dtype=np.float64), 0.0), k)


def weight_map_ref(pi: np.ndarray, w: np.ndarray, n: int) -> np.ndarray:
    """Return ``W_pi``, with its rank weight at each item in ``pi``."""
    if n < 0:
        raise CoupledCertificateRefError("n must be nonnegative")
    weights = _validate_weights_ref(w)
    prefix = np.asarray(pi, dtype=np.int64).reshape(-1)
    if prefix.size > weights.size:
        raise CoupledCertificateRefError("weights shorter than prefix")
    if np.any(prefix < 0) or np.any(prefix >= n):
        raise CoupledCertificateRefError("prefix index out of bounds")
    out = np.zeros(n, dtype=np.float64)
    for rank, item in enumerate(prefix):
        out[int(item)] += float(weights[rank])
    return out


def coefficient_map_ref(pi: np.ndarray, pi_hat: np.ndarray, w: np.ndarray, n: int) -> np.ndarray:
    """Return ``a = W_hat - W_pi``, so the noise term is ``<a, Z>``."""
    return weight_map_ref(pi_hat, w, n) - weight_map_ref(pi, w, n)


def observable_margin_ref(
    y: np.ndarray, pi: np.ndarray, pi_hat: np.ndarray, w: np.ndarray
) -> float:
    """Compute ``G_Y(pi, pi_hat)`` from the raw (unclipped) release."""
    values = np.asarray(y, dtype=np.float64).reshape(-1)
    weights = _validate_weights_ref(w)
    candidate = np.asarray(pi, dtype=np.int64).reshape(-1)
    deployed = np.asarray(pi_hat, dtype=np.int64).reshape(-1)
    if candidate.size != deployed.size:
        raise CoupledCertificateRefError("pi and pi_hat depth mismatch")
    if candidate.size > weights.size:
        raise CoupledCertificateRefError("weights shorter than prefix")
    if (
        np.any(candidate < 0)
        or np.any(candidate >= values.size)
        or np.any(deployed < 0)
        or np.any(deployed >= values.size)
    ):
        raise CoupledCertificateRefError("prefix index out of bounds")
    return float(np.dot(weights[: candidate.size], values[candidate] - values[deployed]))


def noise_contrast_ref(z: np.ndarray, pi: np.ndarray, pi_hat: np.ndarray, w: np.ndarray) -> float:
    """Compute ``N_Z(pi, pi_hat) = sum_r w_r (Z_pi_hat(r) - Z_pi(r))``."""
    return observable_margin_ref(z, pi_hat, pi, w)


def _validate_thresholds_ref(
    u_plus: np.ndarray, u_minus: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    upper = np.asarray(u_plus, dtype=np.float64).reshape(-1)
    lower = np.asarray(u_minus, dtype=np.float64).reshape(-1)
    if upper.size == 0 or lower.size == 0:
        raise CoupledCertificateRefError("order-stat threshold tables must be nonempty")
    if not np.all(np.isfinite(upper)) or not np.all(np.isfinite(lower)):
        raise CoupledCertificateRefError("order-stat threshold tables must be finite")
    if np.any(upper < 0.0) or np.any(lower < 0.0):
        raise CoupledCertificateRefError("order-stat thresholds must be nonnegative")
    if np.any(upper[1:] - upper[:-1] > 1e-12) or np.any(lower[1:] - lower[:-1] > 1e-12):
        raise CoupledCertificateRefError("order-stat thresholds must be nonincreasing")
    return upper, lower


def orderstat_bound_ref(a: np.ndarray, u_plus: np.ndarray, u_minus: np.ndarray) -> float:
    """Closed-form support bound for the two-sided order-statistic event."""
    coefficients = np.asarray(a, dtype=np.float64).reshape(-1)
    upper, lower = _validate_thresholds_ref(u_plus, u_minus)
    positive = np.sort(coefficients[coefficients > 0.0])[::-1]
    negative = np.sort(-coefficients[coefficients < 0.0])[::-1]
    if positive.size > upper.size or negative.size > lower.size:
        raise CoupledCertificateRefError("order-stat table shorter than coefficient support")
    positive_term = np.dot(positive, upper[: positive.size])
    negative_term = np.dot(negative, lower[: negative.size])
    return float(positive_term + negative_term)


def bos_lp_verify(a: np.ndarray, u_plus: np.ndarray, u_minus: np.ndarray, n: int) -> float:
    """Verify the order-statistic support function by a sorted-coordinate LP.

    The rearrangement inequality pairs descending coefficients with descending
    coordinates.  Thus only the ordered vector ``zeta[0] <= ... <= zeta[n-1]``
    is needed; permutation variables are unnecessary.
    """
    try:
        from scipy.optimize import linprog
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise CoupledCertificateRefError("bos_lp_verify requires scipy") from exc

    coefficients = np.asarray(a, dtype=np.float64).reshape(-1)
    if n <= 0 or coefficients.size != n:
        raise CoupledCertificateRefError("n must be positive and equal len(a)")
    upper, lower = _validate_thresholds_ref(u_plus, u_minus)
    if upper.size > n or lower.size > n:
        raise CoupledCertificateRefError("order-stat tables cannot be longer than n")

    descending = np.sort(coefficients)[::-1]
    objective = -descending[::-1]
    rows: list[np.ndarray] = []
    rhs: list[float] = []

    for index in range(n - 1):
        row = np.zeros(n, dtype=np.float64)
        row[index] = 1.0
        row[index + 1] = -1.0
        rows.append(row)
        rhs.append(0.0)
    for rank, threshold in enumerate(upper, start=1):
        row = np.zeros(n, dtype=np.float64)
        row[n - rank] = 1.0
        rows.append(row)
        rhs.append(float(threshold))
    for rank, threshold in enumerate(lower, start=1):
        row = np.zeros(n, dtype=np.float64)
        row[rank - 1] = -1.0
        rows.append(row)
        rhs.append(float(threshold))

    result = linprog(
        objective,
        A_ub=np.asarray(rows, dtype=np.float64),
        b_ub=np.asarray(rhs, dtype=np.float64),
        bounds=[(None, None)] * n,
        method="highs",
    )
    if not result.success:
        raise CoupledCertificateRefError(f"order-stat LP failed: {result.message}")
    return float(-result.fun)


@dataclass(frozen=True)
class CoupledCertificateRefOutcome:
    """Exact coupled-certificate result for a small item universe."""

    value: float
    pi_hat: np.ndarray
    best_pi: np.ndarray
    best_g: float
    best_b: float
    method: str


def certificate_coupled_ref_exact(
    y: np.ndarray, k: int, w: np.ndarray, u_plus: np.ndarray, u_minus: np.ndarray
) -> CoupledCertificateRefOutcome:
    """Enumerate all ordered injections and maximize ``G_Y + B_os`` exactly."""
    values = np.asarray(y, dtype=np.float64).reshape(-1)
    n = values.size
    if n == 0:
        raise CoupledCertificateRefError("y must be nonempty")
    if k < 0 or k > n:
        raise CoupledCertificateRefError("require 0 <= k <= n")
    if n > 9:
        raise CoupledCertificateRefError("exact reference enumeration only for n<=9")
    weights = _validate_weights_ref(w)
    if k > weights.size:
        raise CoupledCertificateRefError("weights shorter than k")
    upper, lower = _validate_thresholds_ref(u_plus, u_minus)
    pi_hat = _deployed_ranking_ref(values, k)

    best_value = -np.inf
    best_pi = pi_hat.copy()
    best_g = 0.0
    best_b = 0.0
    for permutation in permutations(range(n), k):
        pi = np.asarray(permutation, dtype=np.int64)
        margin = observable_margin_ref(values, pi, pi_hat, weights[:k])
        bound = orderstat_bound_ref(coefficient_map_ref(pi, pi_hat, weights[:k], n), upper, lower)
        value = margin + bound
        if value > best_value:
            best_value = value
            best_pi = pi
            best_g = margin
            best_b = bound

    return CoupledCertificateRefOutcome(
        value=float(max(best_value, 0.0)),
        pi_hat=pi_hat,
        best_pi=best_pi,
        best_g=float(best_g),
        best_b=float(best_b),
        method="ref_exact_orderstat",
    )


def pathwise_identity_ref(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, k: int
) -> tuple[bool, float, float]:
    """Check the pathwise identity by independent enumeration for ``n <= 9``."""
    latent = np.asarray(x, dtype=np.float64).reshape(-1)
    release = np.asarray(y, dtype=np.float64).reshape(-1)
    if latent.size != release.size or latent.size == 0:
        raise CoupledCertificateRefError("x and y must be nonempty and have equal length")
    n = latent.size
    if k < 0 or k > n:
        raise CoupledCertificateRefError("require 0 <= k <= n")
    if n > 9:
        raise CoupledCertificateRefError("pathwise reference enumeration only for n<=9")
    weights = _validate_weights_ref(w)
    if k > weights.size:
        raise CoupledCertificateRefError("weights shorter than k")

    pi_hat = _deployed_ranking_ref(release, k)
    noise = release - latent
    deployed_utility = float(np.dot(weights[:k], latent[pi_hat]))
    true_regret = -np.inf
    coupled = -np.inf
    for permutation in permutations(range(n), k):
        pi = np.asarray(permutation, dtype=np.int64)
        true_regret = max(true_regret, float(np.dot(weights[:k], latent[pi])) - deployed_utility)
        coupled = max(
            coupled,
            observable_margin_ref(release, pi, pi_hat, weights[:k])
            + noise_contrast_ref(noise, pi, pi_hat, weights[:k]),
        )
    true_regret = float(max(true_regret, 0.0))
    coupled = float(max(coupled, 0.0))
    return abs(true_regret - coupled) <= 1e-9, true_regret, coupled


__all__ = [
    "CoupledCertificateRefError",
    "CoupledCertificateRefOutcome",
    "weight_map_ref",
    "coefficient_map_ref",
    "observable_margin_ref",
    "noise_contrast_ref",
    "orderstat_bound_ref",
    "bos_lp_verify",
    "certificate_coupled_ref_exact",
    "pathwise_identity_ref",
]
