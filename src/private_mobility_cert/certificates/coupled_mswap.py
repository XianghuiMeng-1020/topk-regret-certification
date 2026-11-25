"""Stratified m-swap routine. Not a valid OS certificate.

The routine indexes psi_m by the size of a set swap and skips pure
reorderings when k > 8. Both choices can return a value below a feasible
order-statistic objective. It is kept as a counterexample so that tests can
observe that failure. certificate_os_envelope is the valid bound.
"""

from __future__ import annotations

from itertools import combinations, permutations

import numpy as np

from private_mobility_cert.certificates.release_conditioned import (
    CertificateOutcome,
    ThresholdTable,
    coefficient_map,
    deploy_ranking,
    observable_margin,
    orderstat_noise_bound,
    weight_map,
)
from private_mobility_cert.certificates.topk_regret import TopKCertError
from private_mobility_cert.certificates.weighted_ranking import validate_weights


def _best_reorder_g(y: np.ndarray, items: list[int], pi_hat: np.ndarray, w: np.ndarray) -> float:
    """Max G over injections whose image equals ``items`` (rearrangement on raw Y)."""
    ww = validate_weights(w)
    k = pi_hat.size
    if len(items) != k:
        raise TopKCertError("item set size must equal k")
    # Best ordered prefix on this set for raw-Y weighted utility: sort by Y desc, stable index
    items_arr = np.asarray(items, dtype=np.int64)
    order_local = np.lexsort((items_arr, -y[items_arr]))
    pi = items_arr[order_local]
    return observable_margin(y, pi, pi_hat, ww[:k])


def _psi_m(w: np.ndarray, u_plus: np.ndarray, u_minus: np.ndarray, m: int) -> float:
    if m <= 0:
        return 0.0
    return float(np.dot(w[:m], u_plus[:m] + u_minus[:m]))


def certificate_mswap_os(
    y: np.ndarray,
    *,
    k: int,
    weights: np.ndarray,
    table: ThresholdTable,
    max_combo: int = 2_000_000,
    unsafe: bool = False,
) -> CertificateOutcome:
    """Stratified m-swap value. Not valid for arbitrary k.

    The default call fails closed. ``unsafe=True`` is only for tests that
    exhibit the known defect. It does not make the value valid.
    """
    if not unsafe:
        raise TopKCertError(
            "certificate_mswap_os is not a valid order-statistic certificate. "
            "Use certificate_os_envelope. unsafe=True only reproduces "
            "the known defect."
        )
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = y.size
    ww = validate_weights(weights)[:k]
    if table.u_plus.size < k or table.u_minus.size < k:
        raise TopKCertError("threshold table shorter than k")
    pi_hat = deploy_ranking(y, k)
    selected = pi_hat.tolist()
    selected_set = set(selected)
    outside = [i for i in range(n) if i not in selected_set]

    best_val = 0.0
    best_pi = pi_hat.copy()
    best_g = 0.0
    best_b = 0.0

    # Pure reorderings of the deployed set (set-symmetric difference 0)
    if k <= 8:
        for pi_t in permutations(selected, k):
            pi = np.asarray(pi_t, dtype=np.int64)
            g = observable_margin(y, pi, pi_hat, ww)
            a = coefficient_map(pi, pi_hat, ww, n)
            b = orderstat_noise_bound(a, table.u_plus, table.u_minus)
            val = g + b
            if val > best_val:
                best_val, best_pi, best_g, best_b = val, pi, g, b

    # Set m-swaps with rearrangement + psi_m noise charge (valid upper on B_os)
    for m in range(0, k + 1):
        psi = _psi_m(ww, table.u_plus, table.u_minus, m)
        if m == 0:
            # already handled reorderings with exact B_os; also evaluate identity
            g0 = 0.0
            val0 = g0 + psi
            if val0 > best_val:
                best_val, best_g, best_b = val0, g0, psi
            continue
        if m > len(outside) or m > len(selected):
            continue
        # combo count guard
        from math import comb

        ncombo = comb(len(selected), m) * comb(len(outside), m)
        if ncombo > max_combo:
            # Fall back: charge full psi_m against the best unrestricted rearrangement G (<=0 often)
            # Use a cheap valid surrogate: top-k by raw Y
            top = np.lexsort((np.arange(n), -y))[:k]
            g = observable_margin(y, top, pi_hat, ww)
            val = g + psi
            if val > best_val:
                best_val, best_pi, best_g, best_b = val, top.astype(np.int64), g, psi
            continue
        for out_idx in combinations(selected, m):
            for in_idx in combinations(outside, m):
                new_set = [i for i in selected if i not in out_idx] + list(in_idx)
                g = _best_reorder_g(y, new_set, pi_hat, ww)
                # Valid noise charge psi_m (may dominate exact B_os)
                val = g + psi
                if val > best_val:
                    # witness pi = rearrangement of new_set
                    items_arr = np.asarray(new_set, dtype=np.int64)
                    order_local = np.lexsort((items_arr, -y[items_arr]))
                    best_pi = items_arr[order_local]
                    best_val, best_g, best_b = val, g, psi

    # Also compare exact B_os on the recorded witness (can only tighten the reported pair,
    # but certificate value must remain an upper bound — keep max with psi-based value).
    return CertificateOutcome(
        value=float(max(best_val, 0.0)),
        pi_hat=pi_hat,
        best_pi=np.asarray(best_pi, dtype=np.int64),
        best_g=float(best_g),
        best_b=float(best_b),
        method="mswap_os_known_invalid",
    )


def certificate_min_valid(
    y: np.ndarray,
    *,
    k: int,
    weights: np.ndarray,
    table: ThresholdTable,
) -> CertificateOutcome:
    """Closed. min(range-L1, m-swap) can mix in an invalid m-swap value.

    The rank-1 box alone is a valid upper bound, and certificate_os_envelope
    is the valid full-universe bound. Returning the minimum would let an
    under-charged m-swap value become the certificate.
    """
    del y, k, weights, table
    raise TopKCertError(
        "certificate_min_valid is closed: min(range-L1, m-swap) can return a "
        "known-invalid m-swap value. Use certificate_os_envelope."
    )
