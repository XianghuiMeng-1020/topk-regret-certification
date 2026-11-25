"""Independent tail checks for the Laplace and discrete-Laplace events.

The numeric radius helpers are not modified. These tests compare them with
tails summed from the PMF or from the closed-form scale-b identities.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from private_mobility_cert.certificates.rank_adaptive import (
    geometric_positive_tail_geq,
    geometric_positive_tail_strict,
    laplace_positive_tail,
)
from private_mobility_cert.certificates.subset_swap import (
    coordinate_box_radius_geometric,
    coordinate_box_radius_laplace,
)
from private_mobility_cert.privacy.geometric import two_sided_geometric_pmf


def _summed_one_sided_gt(t: int, epsilon: float, limit: int = 400) -> float:
    return float(sum(two_sided_geometric_pmf(j, epsilon=epsilon) for j in range(t + 1, limit + 1)))


def _summed_abs_gt(t: int, epsilon: float, limit: int = 400) -> float:
    return float(
        sum(
            two_sided_geometric_pmf(j, epsilon=epsilon) + two_sided_geometric_pmf(-j, epsilon=epsilon)
            for j in range(t + 1, limit + 1)
        )
    )


def _summed_abs_geq(t: int, epsilon: float, limit: int = 400) -> float:
    return float(
        sum(
            two_sided_geometric_pmf(j, epsilon=epsilon) + two_sided_geometric_pmf(-j, epsilon=epsilon)
            for j in range(t, limit + 1)
        )
    )


@pytest.mark.parametrize("epsilon", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("u", [0.0, 0.25, 1.0, 3.5])
def test_laplace_positive_and_absolute_tails(epsilon, u):
    b = 1.0 / epsilon
    assert laplace_positive_tail(u, epsilon) == pytest.approx(0.5 * math.exp(-u / b), abs=1e-12)
    # P(|Z| > u) = exp(-u/b) for u >= 0.
    absolute = 2.0 * laplace_positive_tail(u, epsilon)
    assert absolute == pytest.approx(math.exp(-u / b), abs=1e-12)


@pytest.mark.parametrize("n,alpha,epsilon", [(10, 0.05, 1.0), (500, 0.05, 1.0), (2000, 0.01, 0.5)])
def test_laplace_union_radius(n, alpha, epsilon):
    b = 1.0 / epsilon
    got = coordinate_box_radius_laplace(n, epsilon, alpha)
    assert got == pytest.approx(b * math.log(n / alpha), abs=1e-12)
    # Union of one-coordinate exceedances, not a direct joint calibration.
    assert n * math.exp(-got / b) == pytest.approx(alpha, rel=1e-12)


@pytest.mark.parametrize("epsilon", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("t", [0, 1, 2, 5, 8])
def test_discrete_laplace_tails_against_pmf(epsilon, t):
    rho = math.exp(-epsilon)
    strict = geometric_positive_tail_strict(t, epsilon)
    assert strict == pytest.approx(rho ** (t + 1) / (1.0 + rho), abs=1e-12)
    assert strict == pytest.approx(_summed_one_sided_gt(t, epsilon), abs=1e-10)
    abs_gt = 2.0 * rho ** (t + 1) / (1.0 + rho)
    abs_geq = 2.0 * rho ** t / (1.0 + rho) if t >= 1 else 1.0
    assert abs_gt == pytest.approx(_summed_abs_gt(t, epsilon), abs=1e-10)
    if t >= 1:
        assert geometric_positive_tail_geq(t, epsilon) == pytest.approx(
            rho ** t / (1.0 + rho), abs=1e-12
        )
        assert abs_geq == pytest.approx(_summed_abs_geq(t, epsilon), abs=1e-10)
        # The one-sided helper is half of the two-sided tail, not the two-sided tail.
        assert geometric_positive_tail_geq(t, epsilon) * 2.0 == pytest.approx(abs_geq, rel=1e-12)
        assert geometric_positive_tail_geq(t, epsilon) < abs_geq


@pytest.mark.parametrize(
    "n,epsilon,alpha",
    [(10, 1.0, 0.05), (50, 0.5, 0.05), (500, 1.0, 0.05), (2000, 2.0, 0.01), (8, 0.25, 0.1)],
)
def test_geometric_radius_is_a_closed_box_union_bound(n, epsilon, alpha):
    rho = math.exp(-epsilon)
    t = coordinate_box_radius_geometric(n, epsilon, alpha)
    assert t == math.ceil(math.log(alpha / n) / math.log(rho)) or t == 0.0
    # False comment identity: P(|Z| >= t) is not rho^t.
    if t >= 1:
        true_nonstrict = 2.0 * rho ** t / (1.0 + rho)
        assert true_nonstrict > rho ** t
    # Closed event |Z| <= t fails on |Z| >= t+1.
    failure = 2.0 * rho ** (t + 1) / (1.0 + rho)
    assert n * failure <= alpha + 1e-12
