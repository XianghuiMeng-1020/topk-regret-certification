"""Lightweight checks of the existing order-statistic certificates.

These tests do not add a new OS solver. They compare the current routines
with an independent injection loop on small instances, and they record the
k>8 reordering gap instead of treating that routine as exact.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from private_mobility_cert.certificates.coupled_mswap import certificate_mswap_os
from private_mobility_cert.certificates.os_envelope import certificate_os_envelope
from private_mobility_cert.certificates.coupled_scalable import (
    certificate_os_branch_and_bound,
    certificate_primary,
)
from private_mobility_cert.certificates.release_conditioned import (
    build_threshold_table,
    certificate_orderstat_exact,
    coefficient_map,
    deploy_ranking,
    observable_margin,
    orderstat_noise_bound,
)
from private_mobility_cert.certificates.weighted_ranking import validate_weights, weight_family


def _enum_os(y, k, w, table) -> float:
    y = np.asarray(y, dtype=np.float64)
    ww = validate_weights(w)[:k]
    pi_hat = deploy_ranking(y, k)
    best = 0.0
    for pi_t in permutations(range(y.size), k):
        pi = np.asarray(pi_t, dtype=np.int64)
        g = observable_margin(y, pi, pi_hat, ww)
        a = coefficient_map(pi, pi_hat, ww, y.size)
        b = orderstat_noise_bound(a, table.u_plus, table.u_minus)
        best = max(best, g + b)
    return float(best)


@pytest.mark.parametrize("n,k,family", [(4, 2, "dcg"), (5, 3, "uniform"), (6, 2, "reciprocal"), (6, 3, "dcg")])
def test_branch_and_bound_matches_enumeration(n, k, family):
    rng = np.random.default_rng(20261124 + n + k)
    y = rng.normal(size=n)
    y[0] = -2.0
    w = weight_family(family, k)
    table = build_threshold_table(
        mechanism="M1_flat_laplace", n=n, k_max=k, epsilon=1.0, alpha=0.05, weights=w
    )
    ref = _enum_os(y, k, w, table)
    got = certificate_os_branch_and_bound(
        y, k=k, weights=w, table=table, time_limit_s=None
    ).value
    exact = certificate_orderstat_exact(y, k=k, weights=w, table=table).value
    primary = certificate_primary(y, k=k, weights=w, table=table).value
    assert got == pytest.approx(ref, abs=1e-8)
    assert exact == pytest.approx(ref, abs=1e-8)
    assert primary == pytest.approx(ref, abs=1e-8)


@pytest.mark.parametrize("n,k,family", [(5, 3, "dcg"), (6, 4, "uniform"), (7, 3, "reciprocal")])
def test_mswap_is_an_upper_bound_on_small_instances(n, k, family):
    rng = np.random.default_rng(17 + n * 10 + k)
    y = rng.normal(size=n)
    y[-1] = -3.0
    w = weight_family(family, k)
    table = build_threshold_table(
        mechanism="M2_flat_geometric", n=n, k_max=k, epsilon=1.0, alpha=0.05, weights=w
    )
    ref = _enum_os(y, k, w, table)
    ms = certificate_mswap_os(
        y, k=k, weights=w, table=table, unsafe=True
    ).value
    envelope = certificate_os_envelope(y, k=k, weights=w, table=table).value
    assert ms + 1e-8 >= ref
    assert envelope + 1e-8 >= ref


def test_mswap_misses_pure_reorderings_when_k_exceeds_8():
    """Pre-existing gap, not a repaired claim.

    For k>8 the pure-reordering loop is skipped. When every item is already
    deployed, no set swap is available either, so a profitable reordering of
    nonpositive raw counts is omitted.
    """
    k = 9
    y = -np.linspace(1.0, 0.1, k)
    w = np.arange(k, 0, -1, dtype=np.float64)
    table = build_threshold_table(
        mechanism="M1_flat_laplace", n=k, k_max=k, epsilon=1.0, alpha=0.05, weights=w
    )
    # Enumeration at n=k=9 is 362,880 injections; the branch-and-bound search
    # is the existing exact routine and is compared with one explicit witness.
    pi_hat = deploy_ranking(y, k)
    witness = np.argsort(-y, kind="stable")
    g = observable_margin(y, witness, pi_hat, w)
    a = coefficient_map(witness, pi_hat, w, k)
    b = orderstat_noise_bound(a, table.u_plus, table.u_minus)
    witness_value = float(g + b)
    ms = certificate_mswap_os(
        y, k=k, weights=w, table=table, unsafe=True
    ).value
    assert witness_value > 0.1
    assert ms + 1e-8 < witness_value
