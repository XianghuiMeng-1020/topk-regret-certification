"""M1 — Flat Laplace histogram mechanism (pure ε-DP, sensitivity 1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from private_mobility_cert.privacy.rng import spawn_generator


@dataclass(frozen=True)
class LaplaceRelease:
    y_mech: np.ndarray
    epsilon: float
    scale: float
    mechanism_id: str = "M1_flat_laplace"

    @property
    def y_raw(self) -> np.ndarray:
        """Deprecated alias for y_mech."""
        return self.y_mech


def laplace_scale(epsilon: float, *, sensitivity: float = 1.0) -> float:
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be positive and finite")
    if sensitivity <= 0 or not np.isfinite(sensitivity):
        raise ValueError("sensitivity must be positive and finite")
    return float(sensitivity / epsilon)


def release_laplace_histogram(
    counts: np.ndarray,
    *,
    epsilon: float,
    seed: int,
    sensitivity: float = 1.0,
) -> LaplaceRelease:
    """Y_raw = x + Z, Z_j iid Laplace(0, sensitivity/ε)."""
    x = np.asarray(counts, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("counts must be 1-D")
    if not np.all(np.isfinite(x)):
        raise ValueError("counts must be finite")
    scale = laplace_scale(epsilon, sensitivity=sensitivity)
    rng = spawn_generator(seed)
    noise = rng.laplace(loc=0.0, scale=scale, size=x.shape)
    y_mech = x + noise
    return LaplaceRelease(y_mech=y_mech, epsilon=float(epsilon), scale=scale)


def laplace_log_density(z: float, *, scale: float) -> float:
    """Log density of Laplace(0, b) at z."""
    b = float(scale)
    if b <= 0:
        raise ValueError("scale must be positive")
    return float(-np.log(2.0 * b) - abs(z) / b)


def laplace_log_density_ratio_bound(
    x: np.ndarray,
    xp: np.ndarray,
    y: np.ndarray,
    *,
    epsilon: float,
    sensitivity: float = 1.0,
) -> float:
    """Max log likelihood ratio log(p(y|x)/p(y|x')) for neighboring histograms."""
    scale = laplace_scale(epsilon, sensitivity=sensitivity)
    x = np.asarray(x, dtype=np.float64)
    xp = np.asarray(xp, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not (x.shape == xp.shape == y.shape):
        raise ValueError("shape mismatch")
    # Independent coordinates: ratio factors per cell
    log_num = 0.0
    log_den = 0.0
    for i in range(x.size):
        log_num += laplace_log_density(float(y[i] - x[i]), scale=scale)
        log_den += laplace_log_density(float(y[i] - xp[i]), scale=scale)
    return float(log_num - log_den)
