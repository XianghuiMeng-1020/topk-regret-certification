"""Count-feasible refinement of a coordinate-box coupled certificate.

The original box is the product of [-u_minus, u_plus]. Latent counts are
nonnegative, so a noise vector that is jointly compatible with that box and
with x = Y - z satisfies z_i <= Y_i as well. The resulting public,
release-dependent set is still a product. Its support is an exact rectangular
assignment. An empty intersection is not solved; the original box certificate
is returned and labeled as a fallback.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from private_mobility_cert.certificates.block_coupled_certificates import (
    block5_partition,
    full_partition,
    pair_partition,
    singleton_partition,
)
from private_mobility_cert.certificates.direct_box_calibration import (
    canonicalize_k5_certificates,
    matched_box_certificates,
)
from private_mobility_cert.certificates.release_conditioned import (
    _assignment_maximize,
    _assignment_minimize,
    coefficient_map,
    deploy_ranking,
    observable_margin,
    weight_map,
)
from private_mobility_cert.certificates.topk_regret import TopKCertError
from private_mobility_cert.certificates.weighted_ranking import validate_weights

STATUS_EXACT = "COUNT_FEASIBLE_EXACT"
STATUS_FALLBACK = "EVENT_INCONSISTENT_FALLBACK"
_ROUND_OFF = 1e-9


@dataclass(frozen=True)
class CountFeasibleBox:
    """Coordinate description of E_box intersect {z : Y - z >= 0}."""

    u_minus: float
    u_plus: float
    consistent: bool
    upper: np.ndarray
    width: np.ndarray
    n_below_upper: int
    n_inconsistent: int


@dataclass(frozen=True)
class CountFeasibleOutcome:
    """One count-feasible certificate, or the labeled original-box fallback."""

    value: float
    pi_hat: np.ndarray
    best_pi: np.ndarray
    best_g: float
    best_b: float
    method: str
    status: str
    n_assignment_solves: int


def _scalars(u_minus: float, u_plus: float) -> tuple[float, float]:
    lower = float(u_minus)
    upper = float(u_plus)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower < 0.0 or upper < 0.0:
        raise TopKCertError("box endpoints must be finite and nonnegative")
    return lower, upper


def count_feasible_box(y: np.ndarray, *, u_minus: float, u_plus: float) -> CountFeasibleBox:
    """Return per-coordinate upper endpoints b_i = min(u_plus, Y_i).

    The intersection is empty when any Y_i < -u_minus. A deficit smaller than
    a relative 1e-9 is treated as an endpoint at -u_minus, so a floating-point
    undershoot does not declare an empty event.
    """
    values = np.asarray(y, dtype=np.float64).reshape(-1)
    lower, upper = _scalars(u_minus, u_plus)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise TopKCertError("release must be a nonempty finite vector")
    scale = 1.0 + lower + float(np.max(np.abs(values)))
    deficit = (-lower) - values
    inconsistent_mask = deficit > _ROUND_OFF * scale
    capped = np.minimum(upper, values)
    if np.any(inconsistent_mask):
        return CountFeasibleBox(
            u_minus=lower,
            u_plus=upper,
            consistent=False,
            upper=capped,
            width=np.full(values.size, np.nan, dtype=np.float64),
            n_below_upper=int(np.count_nonzero(values < upper)),
            n_inconsistent=int(np.count_nonzero(inconsistent_mask)),
        )
    capped = np.maximum(capped, -lower)
    width = capped + lower
    if np.any(width < -_ROUND_OFF * scale):
        raise TopKCertError("count-feasible width became negative on a nonempty intersection")
    width = np.maximum(width, 0.0)
    return CountFeasibleBox(
        u_minus=lower,
        u_plus=upper,
        consistent=True,
        upper=capped,
        width=width,
        n_below_upper=int(np.count_nonzero(values < upper - _ROUND_OFF * scale)),
        n_inconsistent=0,
    )


def endpoint_support(
    coefficients: np.ndarray, *, lower: float, upper: np.ndarray
) -> float:
    """Coordinate-endpoint supremum of <a, z> on a nonempty product interval.

    Positive coefficients take the upper endpoint and negative coefficients
    take the lower endpoint. This expression does not require a zero sum.
    """
    coeff = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    high = np.asarray(upper, dtype=np.float64).reshape(-1)
    if coeff.shape != high.shape:
        raise TopKCertError("coefficient and endpoint lengths differ")
    positive = np.maximum(coeff, 0.0)
    negative = np.minimum(coeff, 0.0)
    return float(np.dot(positive, high) + negative.sum() * (-float(lower)))


def width_support(coefficients: np.ndarray, width: np.ndarray) -> float:
    """sum_i d_i max(a_i, 0). Exact for the endpoint support when sum(a) = 0."""
    coeff = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    gaps = np.asarray(width, dtype=np.float64).reshape(-1)
    if coeff.shape != gaps.shape:
        raise TopKCertError("coefficient and width lengths differ")
    if abs(float(coeff.sum())) > 1e-8 * (1.0 + float(np.sum(np.abs(coeff)))):
        raise TopKCertError("width support requires a zero-sum coefficient vector")
    return float(np.dot(gaps, np.maximum(coeff, 0.0)))


def _positive_part_identity(hat: np.ndarray, competitor: np.ndarray) -> np.ndarray:
    """max(W_hat - W_pi, 0) = W_hat - min(W_hat, W_pi) for nonnegative maps."""
    left = np.maximum(hat - competitor, 0.0)
    right = hat - np.minimum(hat, competitor)
    if np.max(np.abs(left - right)) > 1e-8 * (1.0 + float(np.max(np.abs(hat)))):
        raise TopKCertError("positive-part identity failed for a weight map")
    return right


def _finish(value: float, scale: float) -> float:
    if not np.isfinite(value):
        raise TopKCertError("count-feasible certificate is non-finite")
    if value < 0.0:
        if value < -1e-8 * scale:
            raise TopKCertError(f"count-feasible certificate negative beyond tolerance: {value}")
        return 0.0
    return float(value)


def _cost_and_constant(
    y: np.ndarray, *, k: int, weights: np.ndarray, width: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(y, dtype=np.float64).reshape(-1)
    n = values.size
    if not 0 < k <= n:
        raise TopKCertError("require 0 < k <= n")
    ww = validate_weights(weights)[:k]
    gaps = np.asarray(width, dtype=np.float64).reshape(-1)
    if gaps.shape != values.shape or not np.all(np.isfinite(gaps)) or np.any(gaps < 0.0):
        raise TopKCertError("count-feasible widths must be nonnegative and match Y")
    pi_hat = deploy_ranking(values, k)
    hat = weight_map(pi_hat, ww, n)
    cost = np.zeros((k, n), dtype=np.float64)
    for rank in range(k):
        wr = float(ww[rank])
        cost[rank, :] = wr * values - gaps * np.minimum(hat, wr)
    constant = -float(np.dot(ww, values[pi_hat])) + float(np.dot(gaps, hat))
    return cost, constant, pi_hat, ww, hat


def certificate_count_feasible_box(
    y: np.ndarray,
    *,
    k: int,
    weights: np.ndarray,
    u_minus: float,
    u_plus: float,
) -> CountFeasibleOutcome:
    """Exact coupled certificate on the nonempty count-feasible intersection.

    On an empty intersection this returns the original zero-sum box certificate
    at half-width (u_plus + u_minus) / 2 and sets the fallback status. That
    value is not the optimum of an empty set.
    """
    box = count_feasible_box(y, u_minus=u_minus, u_plus=u_plus)
    if not box.consistent:
        from private_mobility_cert.certificates.release_conditioned import (
            certificate_l1_assignment,
        )

        original = certificate_l1_assignment(
            y,
            k=k,
            weights=weights,
            half_width=0.5 * (box.u_plus + box.u_minus),
        )
        return CountFeasibleOutcome(
            value=float(original.value),
            pi_hat=original.pi_hat,
            best_pi=original.best_pi,
            best_g=float(original.best_g),
            best_b=float(original.best_b),
            method="count_feasible_fallback_box",
            status=STATUS_FALLBACK,
            n_assignment_solves=1,
        )
    cost, constant, pi_hat, ww, hat = _cost_and_constant(
        y, k=k, weights=weights, width=box.width
    )
    best_var, best_pi = _assignment_maximize(cost)
    value = float(best_var + constant)
    margin = observable_margin(y, best_pi, pi_hat, ww)
    coeff = coefficient_map(best_pi, pi_hat, ww, int(np.asarray(y).size))
    _positive_part_identity(hat, weight_map(best_pi, ww, hat.size))
    support = width_support(coeff, box.width)
    endpoint = endpoint_support(coeff, lower=box.u_minus, upper=box.upper)
    scale = 1.0 + abs(value) + abs(margin + support)
    if abs((margin + support) - value) > 1e-6 * scale:
        raise TopKCertError(
            "count-feasible assignment identity failed: "
            f"K+sum c={value} versus G+h={margin + support}"
        )
    if abs(endpoint - support) > 1e-6 * scale:
        raise TopKCertError(
            f"width support {support} disagrees with the coordinate endpoint {endpoint}"
        )
    return CountFeasibleOutcome(
        value=_finish(value, scale),
        pi_hat=pi_hat,
        best_pi=best_pi,
        best_g=float(margin),
        best_b=float(support),
        method="count_feasible_box",
        status=STATUS_EXACT,
        n_assignment_solves=1,
    )


def _block_value(
    y: np.ndarray,
    *,
    k: int,
    weights: np.ndarray,
    box: CountFeasibleBox,
    partition,
) -> tuple[float, int]:
    cost, constant, _pi_hat, _ww, _hat = _cost_and_constant(
        y, k=k, weights=weights, width=box.width
    )
    partition.validate(k)
    total = 0.0
    solves = 0
    for block in partition.blocks:
        idx = np.asarray(block, dtype=np.int64)
        val, _sel = _assignment_maximize(cost[idx, :])
        total += float(val)
        solves += 1
    scale = 1.0 + abs(constant) + abs(total)
    return _finish(float(constant + total), scale), solves


def separable_count_feasible_value(
    y: np.ndarray,
    *,
    k: int,
    weights: np.ndarray,
    box: CountFeasibleBox,
) -> tuple[float, int]:
    """max G_Y + max h_Eplus, with the support maximum from a min-cost assignment."""
    if not box.consistent:
        raise TopKCertError("separable count-feasible value requires a nonempty intersection")
    _cost, _constant, pi_hat, ww, hat = _cost_and_constant(
        y, k=k, weights=weights, width=box.width
    )
    utility = np.vstack([float(wr) * np.asarray(y, dtype=np.float64) for wr in ww])
    best_utility, _ = _assignment_maximize(utility)
    max_margin = float(best_utility - float(np.dot(ww, np.asarray(y, dtype=np.float64)[pi_hat])))
    overlap_cost = np.vstack(
        [box.width * np.minimum(hat, float(wr)) for wr in ww]
    )
    min_overlap, _ = _assignment_minimize(overlap_cost)
    max_support = float(np.dot(box.width, hat) - float(min_overlap))
    if max_support < -1e-8 * (1.0 + abs(float(np.dot(box.width, hat)))):
        raise TopKCertError("count-feasible maximum support is negative")
    max_support = max(max_support, 0.0)
    scale = 1.0 + abs(max_margin) + abs(max_support)
    return _finish(float(max_margin + max_support), scale), 2


def matched_count_feasible_certificates(
    y: np.ndarray,
    *,
    k: int,
    weights: np.ndarray,
    u_minus: float,
    u_plus: float,
) -> dict[str, float | str | int]:
    """Original-event names plus the count-feasible family at one box.

    Keys C_box, C_sep_general, C_rank, C_pair, C_block5, C_strong are the
    count-feasible values when the intersection is nonempty, and the original
    box family when it is not. Canonical k<=5 copies are included. Independent
    block-5 and strong values are retained either way.
    """
    box = count_feasible_box(y, u_minus=u_minus, u_plus=u_plus)
    half = 0.5 * (box.u_plus + box.u_minus)
    started = time.perf_counter()
    original = matched_box_certificates(y, k=k, weights=weights, half_width=half)
    original_s = time.perf_counter() - started
    plus_started = time.perf_counter()
    base = {
        "u_minus": box.u_minus,
        "u_plus": box.u_plus,
        "half_width": half,
        "n_below_upper": box.n_below_upper,
        "n_inconsistent": box.n_inconsistent,
        "C_box_original": original["C_box"],
        "C_sep_original": original["C_sep_general"],
        "C_rank_original": original["C_rank"],
        "C_pair_original": original["C_pair"],
        "C_block5_original": original["C_block5"],
        "C_strong_original": original["C_strong"],
    }
    if not box.consistent:
        canonical = canonicalize_k5_certificates(original, k=k)
        base.update(
            {
                "status": STATUS_FALLBACK,
                "C_box": canonical["C_box"],
                "C_sep_general": canonical["C_sep_general"],
                "C_rank": canonical["C_rank"],
                "C_pair": canonical["C_pair"],
                "C_block5": canonical["C_block5"],
                "C_strong": canonical["C_strong"],
                "C_block5_independent": canonical["C_block5_independent"],
                "C_strong_independent": canonical["C_strong_independent"],
                "k5_identity_applied": canonical["k5_identity_applied"],
                "n_assignment_solves": 0,
                "runtime_original_s": original_s,
                "runtime_plus_s": time.perf_counter() - plus_started,
            }
        )
        return base
    full, full_solves = _block_value(
        y, k=k, weights=weights, box=box, partition=full_partition(k)
    )
    # The one-block problem is the exact assignment. Recompute it through the
    # identity-checked entry point and keep that value as C_box.
    checked = certificate_count_feasible_box(
        y, k=k, weights=weights, u_minus=u_minus, u_plus=u_plus
    )
    if abs(checked.value - full) > 1e-6 * (1.0 + abs(checked.value) + abs(full)):
        raise TopKCertError("block-full and identity-checked count-feasible values disagree")
    rank, rank_solves = _block_value(
        y, k=k, weights=weights, box=box, partition=singleton_partition(k)
    )
    pair, pair_solves = _block_value(
        y, k=k, weights=weights, box=box, partition=pair_partition(k)
    )
    block5, block5_solves = _block_value(
        y, k=k, weights=weights, box=box, partition=block5_partition(k)
    )
    separable, sep_solves = separable_count_feasible_value(
        y, k=k, weights=weights, box=box
    )
    raw = {
        "C_box": float(checked.value),
        "C_sep_general": float(separable),
        "C_rank": float(rank),
        "C_pair": float(pair),
        "C_block5": float(block5),
        "C_strong": float(min(separable, rank, pair, block5)),
    }
    canonical = canonicalize_k5_certificates(raw, k=k)
    solves = full_solves + checked.n_assignment_solves + rank_solves + pair_solves
    solves += block5_solves + sep_solves
    base.update(canonical)
    base.update(
        {
            "status": STATUS_EXACT,
            "n_assignment_solves": int(solves),
            "checked_box": float(checked.value),
            "runtime_original_s": original_s,
            "runtime_plus_s": time.perf_counter() - plus_started,
        }
    )
    return base


def deployed_margin_profile(
    y: np.ndarray, *, k: int, weights: np.ndarray, u_plus: float
) -> dict[str, float]:
    """How many positively weighted deployed coordinates sit below u_plus."""
    values = np.asarray(y, dtype=np.float64).reshape(-1)
    ww = validate_weights(weights)[:k]
    pi_hat = deploy_ranking(values, k)
    positive = pi_hat[ww > 0.0]
    released = values[positive] if positive.size else np.zeros(0, dtype=np.float64)
    below = released < float(u_plus) - _ROUND_OFF * (1.0 + abs(float(u_plus)))
    return {
        "n_positive_deployed": int(positive.size),
        "n_positive_deployed_below": int(np.count_nonzero(below)),
        "min_positive_deployed_y": float(np.min(released)) if released.size else float("inf"),
    }
