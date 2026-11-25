"""Threshold inversion boundary checks for M1/M2 order statistics."""

from __future__ import annotations

import numpy as np
import pytest

from private_mobility_cert.certificates.rank_adaptive import (
    binomial_sf,
    geometric_positive_tail_strict,
    laplace_positive_tail,
    solve_order_stat_threshold_continuous,
    solve_order_stat_threshold_integer,
)


@pytest.mark.parametrize("n,r,eta,eps", [(10, 1, 0.01, 1.0), (20, 3, 0.005, 0.5), (8, 2, 0.02, 2.0)])
def test_laplace_threshold_boundary(n: int, r: int, eta: float, eps: float) -> None:
    u = solve_order_stat_threshold_continuous(
        n=n, r=r, eta=eta, epsilon=eps, mechanism="M1_flat_laplace"
    )
    p = laplace_positive_tail(u, eps)
    tail = binomial_sf(r, n, p)
    assert tail <= eta + 1e-8
    # small reduction should violate (near-minimal)
    u_lo = max(u - 1e-4, 0.0)
    p_lo = laplace_positive_tail(u_lo, eps)
    tail_lo = binomial_sf(r, n, p_lo)
    assert tail_lo >= eta - 1e-6 or u == 0.0


@pytest.mark.parametrize("n,r,eta,eps", [(10, 1, 0.01, 1.0), (16, 2, 0.01, 0.7), (8, 3, 0.05, 1.5)])
def test_geometric_threshold_integer_minimality(n: int, r: int, eta: float, eps: float) -> None:
    u = int(
        solve_order_stat_threshold_integer(
            n=n, r=r, eta=eta, epsilon=eps, convention="strict"
        )
    )
    p = geometric_positive_tail_strict(u, eps)
    assert binomial_sf(r, n, p) <= eta + 1e-12
    if u > 0:
        p_lo = geometric_positive_tail_strict(u - 1, eps)
        assert binomial_sf(r, n, p_lo) > eta - 1e-12


def test_strict_vs_nonstrict_geometric_convention() -> None:
    eps = 1.0
    # P(Z > u) < P(Z >= u) for u >= 0
    for u in range(0, 6):
        assert geometric_positive_tail_strict(u, eps) < geometric_positive_tail_strict(u - 1, eps) or u == 0
        assert geometric_positive_tail_strict(u, eps) <= np.exp(-eps * u)
