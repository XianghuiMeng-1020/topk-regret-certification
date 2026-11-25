"""Scalable, event-valid upper bounds for the coupled weighted Top-K certificate.

All functions here retain the raw release in ``G_Y`` and use the clipped release
only to determine the deployed ranking.  The order-statistic solvers optimize
the same ``max_pi [G_Y(pi) + B_os(pi)]`` object as the small-n reference.
"""

from __future__ import annotations

import numpy as np

from private_mobility_cert.certificates.release_conditioned import (
    CertificateOutcome,
    ThresholdTable,
    _assignment_maximize,
    action_set_size,
    coefficient_map,
    deploy_ranking,
    observable_margin,
    weight_map,
)
from private_mobility_cert.certificates.topk_regret import TopKCertError
from private_mobility_cert.certificates.weighted_ranking import validate_weights


def certificate_l1_range_assignment(
    y: np.ndarray, *, k: int, weights: np.ndarray, table: ThresholdTable
) -> CertificateOutcome:
    """Scalable range-event relaxation, imported lazily to preserve one API."""
    from private_mobility_cert.certificates.release_conditioned import (
        certificate_range_l1_assignment,
    )

    return certificate_range_l1_assignment(y, k=k, weights=weights, table=table)


def certificate_coord_asymmetric_box(
    y: np.ndarray, *, k: int, weights: np.ndarray, table: ThresholdTable
) -> CertificateOutcome:
    """Coordinate-box certificate implied by the OS event.

    On the OS event, ``Z_i <= u_1^+`` and ``Z_i >= -u_1^-``.  Every coefficient
    vector has zero sum, so its positive and negative masses coincide.  Thus
    ``<a,Z> <= (u_1^+ + u_1^-)*||a||_1/2``, exactly the range-L1 relaxation.
    """
    out = certificate_l1_range_assignment(y, k=k, weights=weights, table=table)
    return CertificateOutcome(
        value=out.value,
        pi_hat=out.pi_hat,
        best_pi=out.best_pi,
        best_g=out.best_g,
        best_b=out.best_b,
        method="coord_asymmetric_box",
    )


def _l1_prefix_upper(
    cost: np.ndarray, const: float, prefix: tuple[int, ...], used: set[int]
) -> float:
    """Exact range-L1 assignment optimum conditional on an ordered prefix."""
    depth, n = cost.shape
    fixed = sum(float(cost[r, item]) for r, item in enumerate(prefix))
    if len(prefix) == depth:
        return float(const + fixed)
    remaining_items = np.asarray([i for i in range(n) if i not in used], dtype=np.int64)
    rest, _ = _assignment_maximize(cost[len(prefix) :, remaining_items])
    return float(const + fixed + rest)


def _exact_os_search(
    y: np.ndarray,
    *,
    k: int,
    weights: np.ndarray,
    table: ThresholdTable,
    time_limit_s: float | None = None,
    prune_every_level: bool = True,
) -> CertificateOutcome:
    """DFS exact OS search, pruning with a proved range-L1 upper bound.

    The bound is admissible: range-L1 upper-bounds the OS objective on every
    prefix-restricted subproblem, so pruned branches cannot contain a better
    OS optimum. When ``time_limit_s`` is set and exceeded, raises TopKCertError
    with a timeout message; timeouts are never returned as exact optima.
    """
    import time as _time

    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = y.size
    if not 0 <= k <= n:
        raise TopKCertError("require 0 <= k <= n")
    ww = validate_weights(weights)[:k]
    if table.u_plus.size < k or table.u_minus.size < k:
        raise TopKCertError("threshold table must contain at least k ranks")

    t_start = _time.perf_counter()
    pi_hat = deploy_ranking(y, k)
    w_hat = weight_map(pi_hat, ww, n)
    t = 0.5 * table.range_width
    cost = np.vstack(
        [float(wr) * y - 2.0 * t * np.minimum(w_hat, float(wr)) for wr in ww]
    )
    const = -float(np.dot(ww, y[pi_hat])) + 2.0 * t * float(ww.sum())

    best_pi = pi_hat.copy()
    best_g = 0.0
    best_b = 0.0
    best = 0.0
    used: set[int] = set()
    prefix: list[int] = []
    # This order finds high-G candidates early, increasing safe pruning.
    item_order = np.argsort(-y, kind="stable").tolist()
    nodes = 0

    def visit() -> None:
        nonlocal best, best_pi, best_g, best_b, nodes
        nodes += 1
        if time_limit_s is not None and (_time.perf_counter() - t_start) > time_limit_s:
            raise TopKCertError(
                f"exact_os_timeout after {time_limit_s}s at n={n},k={k},nodes={nodes}"
            )
        # Admissible conditional range-L1 bound at every depth when enabled.
        if prune_every_level or len(prefix) <= 1:
            if _l1_prefix_upper(cost, const, tuple(prefix), used) <= best + 1e-12:
                return
        if len(prefix) == k:
            pi = np.asarray(prefix, dtype=np.int64)
            g = observable_margin(y, pi, pi_hat, ww)
            a = coefficient_map(pi, pi_hat, ww, n)
            # Prefix-minimum support. On a nonincreasing table this is the
            # raw charge, so completed searches are unchanged there.
            from private_mobility_cert.certificates.os_envelope import os_support_effective

            b = os_support_effective(a, table.u_plus, table.u_minus)
            value = g + b
            if value > best:
                best, best_pi, best_g, best_b = float(value), pi, float(g), float(b)
            return
        for item in item_order:
            if item not in used:
                prefix.append(item)
                used.add(item)
                visit()
                used.remove(item)
                prefix.pop()

    visit()
    return CertificateOutcome(
        value=float(max(best, 0.0)),
        pi_hat=pi_hat,
        best_pi=best_pi,
        best_g=best_g,
        best_b=best_b,
        method="os_branch_and_bound",
    )


def certificate_os_branch_and_bound(
    y: np.ndarray,
    *,
    k: int,
    weights: np.ndarray,
    table: ThresholdTable,
    max_n: int = 30,
    max_k: int = 5,
    time_limit_s: float | None = 30.0,
) -> CertificateOutcome:
    """Exact OS coupled certificate for the configured small action universe."""
    n = int(np.asarray(y).size)
    if n > max_n or k > max_k:
        raise TopKCertError(f"branch-and-bound limited to n<={max_n}, k<={max_k}")
    return _exact_os_search(
        y, k=k, weights=weights, table=table, time_limit_s=time_limit_s
    )


def certificate_os_topk_universe(
    y: np.ndarray,
    *,
    k: int,
    weights: np.ndarray,
    table: ThresholdTable,
    max_actions: int = 2_000_000,
) -> CertificateOutcome:
    """Exact OS certificate when the action universe is affordable, else range-L1.

    The fallback remains valid because the OS envelope is bounded pointwise by
    the range-L1 envelope on the same OS event.
    """
    if action_set_size(int(np.asarray(y).size), k) <= max_actions:
        return _exact_os_search(y, k=k, weights=weights, table=table)
    return certificate_l1_range_assignment(y, k=k, weights=weights, table=table)


def certificate_hybrid_os_l1(
    y: np.ndarray,
    *,
    k: int,
    weights: np.ndarray,
    table: ThresholdTable,
    max_actions: int = 2_000_000,
) -> CertificateOutcome:
    """The tighter of exact OS and range-L1 when exact OS is affordable."""
    if action_set_size(int(np.asarray(y).size), k) <= max_actions:
        os_out = _exact_os_search(y, k=k, weights=weights, table=table)
        l1_out = certificate_l1_range_assignment(y, k=k, weights=weights, table=table)
        return os_out if os_out.value <= l1_out.value else l1_out
    return certificate_l1_range_assignment(y, k=k, weights=weights, table=table)


def certificate_os_cut_loop(
    y: np.ndarray,
    *,
    k: int,
    weights: np.ndarray,
    table: ThresholdTable,
    max_actions: int = 2_000_000,
) -> CertificateOutcome:
    """Safe practical replacement for cut heuristics: exact OS or range-L1.

    Candidate-only cut loops cannot certify the omitted actions without a
    residual upper bound.  This routine therefore uses the complete exact
    universe when affordable and otherwise the globally valid L1 relaxation.
    """
    return certificate_os_topk_universe(
        y, k=k, weights=weights, table=table, max_actions=max_actions
    )


def certificate_primary(
    y: np.ndarray, *, k: int, weights: np.ndarray, table: ThresholdTable
) -> CertificateOutcome:
    """Exact OS when the injection count is at most 500_000, else the OS envelope.

    The large-universe branch calls certificate_os_envelope and labels the
    result primary_os_envelope. It does not use certificate_mswap_os, whose
    value is not a valid upper bound.
    """
    n = int(np.asarray(y).size)
    if action_set_size(n, k) <= 500_000:
        out = _exact_os_search(y, k=k, weights=weights, table=table)
        return CertificateOutcome(
            value=out.value,
            pi_hat=out.pi_hat,
            best_pi=out.best_pi,
            best_g=out.best_g,
            best_b=out.best_b,
            method="primary_exact_os",
        )
    from private_mobility_cert.certificates.os_envelope import (
        METHOD_PRIMARY_OS_ENVELOPE,
        certificate_os_envelope,
    )

    out = certificate_os_envelope(y, k=k, weights=weights, table=table)
    return CertificateOutcome(
        value=out.value,
        pi_hat=out.pi_hat,
        best_pi=out.best_pi,
        best_g=out.best_g,
        best_b=out.best_b,
        method=METHOD_PRIMARY_OS_ENVELOPE,
    )


__all__ = [
    "certificate_l1_range_assignment",
    "certificate_coord_asymmetric_box",
    "certificate_os_branch_and_bound",
    "certificate_os_topk_universe",
    "certificate_hybrid_os_l1",
    "certificate_os_cut_loop",
    "certificate_primary",
]
