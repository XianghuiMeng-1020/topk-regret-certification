"""Small-n agreement checks for scalable coupled certificates."""

from __future__ import annotations

import numpy as np
import pytest

from private_mobility_cert.certificates.release_conditioned import (
    build_threshold_table,
    certificate_orderstat_exact,
    weight_family,
)


def test_branch_and_bound_matches_exact_orderstat_certificate() -> None:
    from private_mobility_cert.certificates.coupled_scalable import (
        certificate_os_branch_and_bound,
    )

    rng = np.random.default_rng(27)
    n, k = 7, 3
    weights = weight_family("dcg", k)
    table = build_threshold_table(
        mechanism="M1", n=n, k_max=k, epsilon=1.0, alpha=0.05, weights=weights
    )
    for _ in range(5):
        y = rng.normal(3.0, 2.0, size=n)
        expected = certificate_orderstat_exact(y, k=k, weights=weights, table=table)
        actual = certificate_os_branch_and_bound(y, k=k, weights=weights, table=table)
        assert actual.value == pytest.approx(expected.value, abs=1e-9)


def test_hybrid_and_primary_are_valid_relaxations() -> None:
    from private_mobility_cert.certificates.coupled_scalable import (
        certificate_hybrid_os_l1,
        certificate_l1_range_assignment,
        certificate_primary,
    )

    n, k = 7, 3
    y = np.array([7.0, 1.0, 6.0, 3.0, 2.0, 5.0, 4.0])
    weights = weight_family("reciprocal", k)
    table = build_threshold_table(
        mechanism="M1", n=n, k_max=k, epsilon=0.5, alpha=0.05, weights=weights
    )
    exact = certificate_orderstat_exact(y, k=k, weights=weights, table=table)
    l1 = certificate_l1_range_assignment(y, k=k, weights=weights, table=table)
    hybrid = certificate_hybrid_os_l1(y, k=k, weights=weights, table=table)
    primary = certificate_primary(y, k=k, weights=weights, table=table)
    assert l1.value + 1e-10 >= exact.value
    assert hybrid.value == pytest.approx(exact.value, abs=1e-9)
    assert primary.value == pytest.approx(exact.value, abs=1e-9)
