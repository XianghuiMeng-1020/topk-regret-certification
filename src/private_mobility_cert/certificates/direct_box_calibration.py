"""Direct joint calibration of the symmetric coordinate box.

This module does not replace ``coordinate_box_radius_laplace`` or
``coordinate_box_radius_geometric``. Those functions remain the union-bound
radii. The events here are ``DIRECT_JOINT_BOX``. They are not
``OS_RANK1_BOX`` and not the full order-statistic event.

For the same noise law, ``n``, ``alpha``, ``epsilon``, and sensitivity, the
direct radius is at most the union radius, so the direct box is a subset of
the union box. An order-statistic event is not nested in either box by this
argument.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from private_mobility_cert.certificates.block_coupled_certificates import (
    certificate_block5_box,
    certificate_full_coupled_box,
    certificate_pair_block_box,
    certificate_rankwise_box,
    matched_separable_box_value,
)
from private_mobility_cert.certificates.certified_risk_frontier import (
    normalize_certificate,
)
from private_mobility_cert.certificates.topk_regret import TopKCertError

# Used only to classify a raw certificate gap as numerically zero.
# Never added to a certification threshold.
RAW_GAP_ZERO_ATOL = 1e-8
_DECISION_KEYS = (
    "C_box",
    "C_sep_general",
    "C_rank",
    "C_pair",
    "C_block5",
    "C_strong",
)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TopKCertError(f"{name} must be a positive integer")
    if isinstance(value, (int, np.integer)):
        out = int(value)
    else:
        raise TopKCertError(f"{name} must be a positive integer")
    if out < 1:
        raise TopKCertError(f"{name} must be a positive integer")
    return out


def _positive_finite(value: object, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise TopKCertError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(out) or out <= 0.0:
        raise TopKCertError(f"{name} must be a positive finite number")
    return out


def _alpha(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise TopKCertError("alpha must lie in (0, 1)") from exc
    if not math.isfinite(out) or not (0.0 < out < 1.0):
        raise TopKCertError("alpha must lie in (0, 1)")
    return out


def _nonnegative_finite(value: object, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise TopKCertError(f"{name} must be a nonnegative finite number") from exc
    if not math.isfinite(out) or out < 0.0:
        raise TopKCertError(f"{name} must be a nonnegative finite number")
    return out


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TopKCertError(f"{name} must be a nonnegative integer")
    if isinstance(value, (int, np.integer)):
        out = int(value)
    elif isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        raw = float(value)
        if raw != math.floor(raw):
            raise TopKCertError(f"{name} must be a nonnegative integer")
        out = int(raw)
    else:
        raise TopKCertError(f"{name} must be a nonnegative integer")
    if out < 0:
        raise TopKCertError(f"{name} must be a nonnegative integer")
    return out


def laplace_scale(epsilon: float, delta: float) -> float:
    """Scale ``b = Delta / epsilon`` of an iid Laplace coordinate."""
    return _positive_finite(delta, "delta") / _positive_finite(epsilon, "epsilon")


def union_box_radius_laplace(
    n: int,
    epsilon: float,
    alpha: float,
    *,
    delta: float = 1.0,
) -> float:
    """Union radius ``t = b log(n/alpha)``, ``b = Delta/epsilon``.

    When ``delta == 1`` the arithmetic matches
    ``coordinate_box_radius_laplace`` and does not replace that function.
    """
    n_int = _positive_int(n, "n")
    eps = _positive_finite(epsilon, "epsilon")
    alp = _alpha(alpha)
    dlt = _positive_finite(delta, "delta")
    if dlt == 1.0:
        return float(np.log(n_int / alp) / eps)
    return float((dlt / eps) * np.log(n_int / alp))


def direct_box_radius_laplace(
    n: int,
    epsilon: float,
    alpha: float,
    *,
    delta: float = 1.0,
) -> float:
    """Exact radius of the joint Laplace box at coverage ``1 - alpha``.

    ``P(|Z_i| <= t) = 1 - exp(-t/b)`` with ``b = Delta/epsilon``, so
    ``P(Z in E_t) = (1 - exp(-t/b))^n``. Solving for coverage ``1-alpha``
    gives ``t = -b log(1 - (1-alpha)^(1/n))``. The implementation uses

        q = -expm1(log1p(-alpha) / n)
        t = -b log(q).
    """
    n_int = _positive_int(n, "n")
    eps = _positive_finite(epsilon, "epsilon")
    alp = _alpha(alpha)
    b = laplace_scale(eps, delta)
    q = -math.expm1(math.log1p(-alp) / n_int)
    if not math.isfinite(q) or q <= 0.0 or q > 1.0:
        raise TopKCertError("direct Laplace calibration is not finite")
    radius = -b * math.log(q)
    if not math.isfinite(radius) or radius < 0.0:
        raise TopKCertError("direct Laplace radius is not a nonnegative finite number")
    return float(radius)


def direct_box_coverage_laplace(
    n: int,
    radius: float,
    epsilon: float,
    *,
    delta: float = 1.0,
) -> float:
    """``P(max_i |Z_i| <= radius)`` for iid Laplace(0, Delta/epsilon)."""
    n_int = _positive_int(n, "n")
    rad = _nonnegative_finite(radius, "radius")
    b = laplace_scale(epsilon, delta)
    if rad == 0.0:
        return 0.0
    tail = math.exp(-rad / b)
    if tail >= 1.0:
        return 0.0
    log_cov = n_int * math.log1p(-tail)
    if log_cov < -745.0:
        return 0.0
    coverage = math.exp(log_cov)
    if not math.isfinite(coverage):
        raise TopKCertError("Laplace box coverage is not finite")
    return float(min(max(coverage, 0.0), 1.0))


def geometric_coordinate_failure(t: int, epsilon: float) -> float:
    """Closed-box coordinate failure ``P(|Z| > t) = 2 rho^{t+1}/(1+rho)``.

    ``t`` is an integer and ``rho = exp(-epsilon)``. This is the failure of
    ``|Z| <= t``. It is not ``P(|Z| >= t) = 2 rho^t/(1+rho)`` for ``t >= 1``.
    The formula is the unit-sensitivity discrete Laplace used by the current
    sampler. It is not a ``B``-bounded user-level mechanism.
    """
    tt = _nonnegative_int(t, "t")
    eps = _positive_finite(epsilon, "epsilon")
    log_rho = -eps
    # 2 * exp((t+1) log rho) / (1+rho), with rho = exp(-epsilon).
    log_num = math.log(2.0) + (tt + 1) * log_rho
    rho = math.exp(log_rho)
    if log_num < -745.0:
        return 0.0
    failure = math.exp(log_num) / (1.0 + rho)
    if not math.isfinite(failure):
        raise TopKCertError("discrete-Laplace tail is not finite")
    return float(min(max(failure, 0.0), 1.0))


def direct_box_coverage_geometric(n: int, radius: int, epsilon: float) -> float:
    """``P(|Z_i| <= radius for every i)`` for unit-sensitivity discrete Laplace."""
    n_int = _positive_int(n, "n")
    tt = _nonnegative_int(radius, "radius")
    failure = geometric_coordinate_failure(tt, epsilon)
    if failure >= 1.0:
        return 0.0
    if failure == 0.0:
        return 1.0
    log_cov = n_int * math.log1p(-failure)
    if log_cov < -745.0:
        return 0.0
    coverage = math.exp(log_cov)
    if not math.isfinite(coverage):
        raise TopKCertError("discrete-Laplace box coverage is not finite")
    return float(min(max(coverage, 0.0), 1.0))


def _geometric_log_coverage(n: int, t: int, epsilon: float) -> float:
    failure = geometric_coordinate_failure(t, epsilon)
    if failure >= 1.0:
        return float("-inf")
    if failure == 0.0:
        return 0.0
    return float(n * math.log1p(-failure))


def direct_box_radius_geometric(n: int, epsilon: float, alpha: float) -> int:
    """Smallest integer ``t >= 0`` with closed-box coverage at least ``1-alpha``.

    The noise law is the unit-sensitivity discrete Laplace,
    ``rho = exp(-epsilon)``, matching ``release_geometric_histogram``.
    A sensitivity-``Delta`` mechanism would require ``rho = exp(-epsilon/Delta)``
    and a sampler with that same law. This function does not apply that
    rescaling, because the current sampler is unit sensitivity.
    """
    n_int = _positive_int(n, "n")
    eps = _positive_finite(epsilon, "epsilon")
    alp = _alpha(alpha)
    log_target = math.log1p(-alp)
    # One-coordinate failure budget of the exact joint event.
    q_allow = -math.expm1(math.log1p(-alp) / n_int)
    rho = math.exp(-eps)
    rhs = q_allow * (1.0 + rho) / 2.0
    if rhs >= 1.0:
        candidate = 0
    elif rhs <= 0.0:
        raise TopKCertError("discrete-Laplace calibration budget is not positive")
    else:
        # rho^{t+1} <= rhs, and log(rho) < 0 so the inequality flips.
        raw = math.log(rhs) / (-eps) - 1.0
        candidate = max(0, int(math.ceil(raw - 1e-12)))
    t = candidate
    # The closed form can be one step off. Walk until the log coverage holds,
    # then step down to the smallest such integer.
    for _ in range(64):
        if _geometric_log_coverage(n_int, t, eps) >= log_target:
            break
        t += 1
    else:
        raise TopKCertError("discrete-Laplace direct radius did not converge")
    while t > 0 and _geometric_log_coverage(n_int, t - 1, eps) >= log_target:
        t -= 1
    return int(t)


def direct_box_radius_geometric_sensitivity(
    n: int,
    epsilon: float,
    alpha: float,
    *,
    delta: float,
) -> int:
    """Integer radius for ``rho = exp(-epsilon/Delta)``.

    This helper is not the current geometric sampler unless ``delta == 1``.
    A release that uses it must draw noise with the same ``rho``. Do not pass
    ``delta = B`` while still calling ``release_geometric_histogram``, which
    always uses ``rho = exp(-epsilon)``.
    """
    dlt = _positive_finite(delta, "delta")
    if dlt == 1.0:
        return direct_box_radius_geometric(n, epsilon, alpha)
    # Reuse the unit-sensitivity search at the scaled privacy parameter.
    # rho = exp(-epsilon/delta) = exp(-(epsilon/delta)).
    return direct_box_radius_geometric(n, epsilon / dlt, alpha)


def matched_box_certificates(
    y: np.ndarray,
    *,
    k: int,
    weights: np.ndarray,
    half_width: float,
) -> dict[str, float]:
    """Six same-event certificates at one fixed box half-width.

    ``C_strong`` is the minimum of the four structural relaxations. This function does
    not call an order-statistic or m-swap routine.
    """
    box = certificate_full_coupled_box(
        y, k=k, weights=weights, half_width=half_width
    )
    separable = matched_separable_box_value(
        y, k=k, weights=weights, half_width=half_width
    )
    rank = certificate_rankwise_box(
        y, k=k, weights=weights, half_width=half_width
    ).value
    pair = certificate_pair_block_box(
        y, k=k, weights=weights, half_width=half_width
    ).value
    block5 = certificate_block5_box(
        y, k=k, weights=weights, half_width=half_width
    ).value
    strong = float(min(separable, rank, pair, block5))
    return {
        "C_box": float(box.value),
        "C_sep_general": float(separable),
        "C_rank": float(rank),
        "C_pair": float(pair),
        "C_block5": float(block5),
        "C_strong": strong,
    }


def canonicalize_k5_certificates(
    certificates: Mapping[str, float],
    *,
    k: int,
) -> dict[str, float]:
    """Choose the values used when two definitions are the same optimization.

    ``C_strong`` is still ``min(C_sep, C_rank, C_pair, C_block5)``. For
    ``k <= 5`` the block-5 partition is the single full block, so ``C_block5``
    and ``C_box`` are the same problem. ``C_box`` is a lower bound for every
    matched relaxation, and therefore ``C_strong = C_box`` as well. Independent
    float evaluations of that common value are kept under ``*_independent``.
    Certification compares one canonical coupled value. For ``k > 5`` the
    decision values are the independently computed ones, so a real compatibility
    gap is left intact.
    """
    depth = _positive_int(k, "k")
    missing = [key for key in _DECISION_KEYS if key not in certificates]
    if missing:
        raise TopKCertError(f"missing certificate values: {missing}")
    out = {key: float(certificates[key]) for key in _DECISION_KEYS}
    for key, value in out.items():
        if not math.isfinite(value):
            raise TopKCertError(f"{key} must be finite")
    out["C_block5_independent"] = out["C_block5"]
    out["C_strong_independent"] = out["C_strong"]
    out["k5_identity_applied"] = 0.0
    if depth <= 5:
        canonical = out["C_box"]
        out["C_block5"] = canonical
        out["C_strong"] = canonical
        out["k5_identity_applied"] = 1.0
    return out


def certification_decision(value: float, scale: float, tau: float) -> bool:
    """Return whether ``value / S(Y) <= tau``.

    The comparison has no additive slack. A value that rounds to one ulp above
    ``tau * S(Y)`` is not certified. Fail-closed normalization maps a non-finite
    input to ``+inf``, which is not certified.
    """
    if not math.isfinite(float(tau)):
        raise TopKCertError("tau must be finite")
    return bool(normalize_certificate(value, scale) <= float(tau))


def raw_gap_is_numerically_zero(gap: float, *, atol: float = RAW_GAP_ZERO_ATOL) -> bool:
    """Classify a raw gap. This predicate is not a certification rule."""
    if not math.isfinite(float(gap)) or not math.isfinite(float(atol)) or float(atol) < 0.0:
        raise TopKCertError("gap classification requires a finite gap and nonnegative atol")
    return abs(float(gap)) <= float(atol)
