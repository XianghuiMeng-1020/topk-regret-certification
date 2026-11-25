"""M2 — Flat two-sided geometric / discrete Laplace mechanism (pure ε-DP)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from private_mobility_cert.privacy.rng import spawn_generator


@dataclass(frozen=True)
class GeometricRelease:
    y_mech: np.ndarray
    epsilon: float
    alpha: float
    mechanism_id: str = "M2_two_sided_geometric"

    @property
    def y_raw(self) -> np.ndarray:
        """Deprecated alias for y_mech."""
        return self.y_mech


def geometric_alpha(epsilon: float) -> float:
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be positive and finite")
    return float(np.exp(-epsilon))


def two_sided_geometric_pmf(k: int, *, epsilon: float) -> float:
    """Pr(Z=k) = ((1-α)/(1+α)) α^{|k|} with α=e^{-ε}."""
    alpha = geometric_alpha(epsilon)
    return float(((1.0 - alpha) / (1.0 + alpha)) * (alpha ** abs(int(k))))


def two_sided_geometric_variance(epsilon: float) -> float:
    """Var(Z) = 2α / (1-α)^2."""
    alpha = geometric_alpha(epsilon)
    return float(2.0 * alpha / (1.0 - alpha) ** 2)


def _sample_zero_based_geometric(
    rng: np.random.Generator, *, p_success: float, size: int
) -> np.ndarray:
    """Zero-based geometric: P(X=k)=(1-p)^k p for k=0,1,...

    NumPy Generator.geometric(p) is one-based (trials until first success).
    Convert via X_zb = Geo_onebased(p) - 1.
    """
    if not 0.0 < p_success <= 1.0:
        raise ValueError("p_success must be in (0, 1]")
    one_based = rng.geometric(p_success, size=size)
    return one_based.astype(np.int64) - 1


def sample_two_sided_geometric(
    rng: np.random.Generator,
    *,
    epsilon: float,
    size: int,
) -> np.ndarray:
    """Z = G1 - G2 for i.i.d. zero-based geometrics with success probability 1-α."""
    alpha = geometric_alpha(epsilon)
    p = 1.0 - alpha
    g1 = _sample_zero_based_geometric(rng, p_success=p, size=size)
    g2 = _sample_zero_based_geometric(rng, p_success=p, size=size)
    return np.asarray(g1 - g2, dtype=np.int64)


def release_geometric_histogram(
    counts: np.ndarray,
    *,
    epsilon: float,
    seed: int,
) -> GeometricRelease:
    """Y_raw = x + Z with integer-valued two-sided geometric noise."""
    x = np.asarray(counts)
    if x.ndim != 1:
        raise ValueError("counts must be 1-D")
    if not np.issubdtype(x.dtype, np.integer):
        if not np.all(np.equal(np.mod(x, 1), 0)):
            raise ValueError("counts must be integer-valued for geometric release")
        x = x.astype(np.int64)
    else:
        x = x.astype(np.int64)
    rng = spawn_generator(seed)
    noise = sample_two_sided_geometric(rng, epsilon=epsilon, size=int(x.size))
    y_mech = x + noise
    return GeometricRelease(
        y_mech=y_mech,
        epsilon=float(epsilon),
        alpha=geometric_alpha(epsilon),
    )


def geometric_privacy_log_ratio(
    x: np.ndarray,
    xp: np.ndarray,
    y: np.ndarray,
    *,
    epsilon: float,
) -> float:
    """Log likelihood ratio for neighboring integer histograms under M2."""
    x = np.asarray(x, dtype=np.int64)
    xp = np.asarray(xp, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    if not (x.shape == xp.shape == y.shape):
        raise ValueError("shape mismatch")
    log_r = 0.0
    for i in range(x.size):
        z = int(y[i] - x[i])
        zp = int(y[i] - xp[i])
        p = two_sided_geometric_pmf(z, epsilon=epsilon)
        pp = two_sided_geometric_pmf(zp, epsilon=epsilon)
        if p <= 0 or pp <= 0:
            raise ValueError("nonpositive PMF")
        log_r += np.log(p) - np.log(pp)
    return float(log_r)
