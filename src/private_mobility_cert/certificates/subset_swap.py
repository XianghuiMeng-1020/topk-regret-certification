"""Subset-swap observed deficits, noise contrasts, and Top-K certificates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from private_mobility_cert.certificates.topk_regret import (
    TopKCertError,
    topk_select_indices,
    validate_vector,
)


@dataclass(frozen=True)
class CertificateResult:
    name: str
    value: float
    alpha: float
    details: dict[str, float]


def observed_multi_swap_deficit(y: np.ndarray, selected: np.ndarray, m: int) -> float:
    """d_m(y,S) = max_{|A|=|B|=m} (sum_B y - sum_A y) using signed y.

    Equals sum of m largest y outside S minus sum of m smallest y inside S.
    """
    yy = validate_vector(y, name="y")
    s_idx = np.asarray(selected, dtype=np.int64)
    n = yy.size
    if m < 0:
        raise TopKCertError("m must be nonnegative")
    if m == 0:
        return 0.0
    if s_idx.size == 0:
        raise TopKCertError("selected set empty for m>0")
    if m > s_idx.size:
        raise TopKCertError("m exceeds |S|")
    s_mask = np.zeros(n, dtype=bool)
    s_mask[s_idx] = True
    outside = yy[~s_mask]
    if m > outside.size:
        raise TopKCertError("m exceeds |S^c|")
    inside = yy[s_mask]
    # m largest outside, m smallest inside
    if m == 1:
        return float(outside.max() - inside.min())
    out_part = np.partition(outside, -m)[-m:]
    in_part = np.partition(inside, m - 1)[:m]
    return float(out_part.sum() - in_part.sum())


def all_observed_deficits(y: np.ndarray, selected: np.ndarray, k: int) -> np.ndarray:
    """Return d_m for m=0..k (d_0=0). Uses signed y."""
    yy = validate_vector(y, name="y")
    s_idx = np.asarray(selected, dtype=np.int64)
    k_eff = min(int(k), s_idx.size, yy.size - s_idx.size)
    out = np.zeros(int(k) + 1, dtype=np.float64)
    for m in range(1, k_eff + 1):
        out[m] = observed_multi_swap_deficit(yy, s_idx, m)
    return out


def q_m_order_statistic(z: np.ndarray, m: int) -> float:
    """Q_m(z) = sum of m largest z - sum of m smallest z."""
    zz = validate_vector(z, name="z")
    if m < 0:
        raise TopKCertError("m must be nonnegative")
    if m == 0:
        return 0.0
    if 2 * m > zz.size:
        raise TopKCertError("need at least 2m coordinates for Q_m")
    if m == 1:
        return float(zz.max() - zz.min())
    smallest = np.partition(zz, m - 1)[:m]
    largest = np.partition(zz, -m)[-m:]
    return float(largest.sum() - smallest.sum())


def all_q_m(z: np.ndarray, k: int) -> np.ndarray:
    zz = validate_vector(z, name="z")
    max_m = min(int(k), zz.size // 2)
    out = np.zeros(int(k) + 1, dtype=np.float64)
    for m in range(1, max_m + 1):
        out[m] = q_m_order_statistic(zz, m)
    return out


def regret_upper_from_noise(
    y: np.ndarray,
    selected: np.ndarray,
    z: np.ndarray,
    k: int,
) -> float:
    """max_m [d_m(y,S) + Q_m(z)]_+  (oracle noise; not deployable)."""
    deficits = all_observed_deficits(y, selected, k)
    qs = all_q_m(z, k)
    return float(max(0.0, float(np.max(deficits + qs))))


def coordinate_box_radius_laplace(n: int, epsilon: float, alpha: float) -> float:
    """r such that Pr(max_i |Z_i| > r) <= alpha for iid Laplace(b=1/ε).

    Exact: Pr(|Z|>r) = exp(-ε r); union bound: n exp(-ε r) = alpha
    => r = (1/ε) log(n/alpha). Conservative.
    """
    if n <= 0 or epsilon <= 0 or not (0 < alpha < 1):
        raise TopKCertError("invalid box parameters")
    return float(np.log(n / alpha) / epsilon)


def coordinate_box_radius_geometric(n: int, epsilon: float, alpha: float) -> float:
    """Integer half-width for the closed coordinate box of discrete Laplace noise.

    With ``rho = exp(-epsilon)`` and PMF
    ``P(Z=j) = ((1-rho)/(1+rho)) rho^{|j|}``, the tails for integer ``t`` are

    - ``P(Z > t) = rho^{t+1}/(1+rho)`` for ``t >= 0``,
    - ``P(|Z| > t) = 2 rho^{t+1}/(1+rho)`` for ``t >= 0``,
    - ``P(|Z| >= t) = 2 rho^t/(1+rho)`` for ``t >= 1``.

    The identity ``P(|Z| >= t) = rho^t`` is false. The implemented integer ``t``
    is unchanged: it is the smallest integer satisfying ``n * rho^t <= alpha``.
    The certificate uses the closed event ``|Z_i| <= t``, whose failure event is
    ``|Z_i| >= t+1``. Then

    ``n P(|Z| >= t+1) = n * 2 rho^{t+1}/(1+rho) <= alpha * 2 rho/(1+rho) < alpha``.

    So this radius remains a valid union bound for the closed box. It is not a
    direct joint calibration of ``P(max_i |Z_i| > t)``.
    """
    if n <= 0 or epsilon <= 0 or not (0 < alpha < 1):
        raise TopKCertError("invalid box parameters")
    alpha_geo = float(np.exp(-epsilon))
    # t such that n * alpha_geo^t <= alpha
    t = np.ceil(np.log(alpha / n) / np.log(alpha_geo))
    return float(max(t, 0.0))


def certificate_c0_box(
    y: np.ndarray,
    selected: np.ndarray,
    *,
    k: int,
    radius: float,
) -> CertificateResult:
    """C0: under max |z_i|<=r, Q_m(z) <= 2 m r, so C = max_m [d_m + 2 m r]_+."""
    deficits = all_observed_deficits(y, selected, k)
    vals = [max(0.0, float(deficits[m] + 2.0 * m * radius)) for m in range(0, int(k) + 1)]
    return CertificateResult(
        name="C0_coordinate_box",
        value=float(max(vals)),
        alpha=float("nan"),
        details={"radius": float(radius), "max_m_term": float(max(vals))},
    )


def certificate_c1_pairwise(
    y: np.ndarray,
    selected: np.ndarray,
    *,
    k: int,
    q1: float,
) -> CertificateResult:
    """C1: use Q_m <= m * Q_1 bound with common pairwise range quantile q1."""
    if q1 < 0:
        raise TopKCertError("q1 must be nonnegative")
    deficits = all_observed_deficits(y, selected, k)
    vals = [max(0.0, float(deficits[m] + m * q1)) for m in range(0, int(k) + 1)]
    return CertificateResult(
        name="C1_pairwise_range",
        value=float(max(vals)),
        alpha=float("nan"),
        details={"q1": float(q1)},
    )


def certificate_c2_subset_swap(
    y: np.ndarray,
    selected: np.ndarray,
    *,
    k: int,
    q_m: np.ndarray,
    alpha: float,
    name: str = "C2_subset_swap",
) -> CertificateResult:
    """C2: C = max_m [d_m(y,S) + q_{m,α}]_+ with q_0=0."""
    deficits = all_observed_deficits(y, selected, k)
    qq = np.asarray(q_m, dtype=np.float64)
    if qq.shape[0] < int(k) + 1:
        raise TopKCertError("q_m must cover m=0..K")
    if np.any(qq < -1e-15):
        raise TopKCertError("q_m must be nonnegative")
    vals = [max(0.0, float(deficits[m] + qq[m])) for m in range(0, int(k) + 1)]
    return CertificateResult(
        name=name,
        value=float(max(vals)),
        alpha=float(alpha),
        details={"active_m": float(int(np.argmax(vals))), "max_term": float(max(vals))},
    )


def deployable_selection_from_y_act(y_mech: np.ndarray, k: int) -> np.ndarray:
    """Action uses y^+ = max(y,0); certificate may still use signed y."""
    y = validate_vector(y_mech, name="y_mech")
    y_act = np.maximum(y, 0.0)
    return topk_select_indices(y_act, k).selected_idx


def oracle_certificate(
    y: np.ndarray,
    selected: np.ndarray,
    z: np.ndarray,
    k: int,
) -> CertificateResult:
    """ORACLE — NOT DEPLOYABLE: uses true noise realization."""
    val = regret_upper_from_noise(y, selected, z, k)
    return CertificateResult(
        name="C3_oracle_NOT_DEPLOYABLE",
        value=float(val),
        alpha=float("nan"),
        details={"deployable": 0.0},
    )
