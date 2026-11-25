from __future__ import annotations

import numpy as np

from private_mobility_cert.certificates.certified_risk_frontier import (
    coverage_K,
    inverse_T,
)


def test_inverse_matches_coverage_definition():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    for c in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        t = inverse_T(x, c)
        assert coverage_K(x, t) + 1e-12 >= c
        # any smaller than t that is strictly below the order statistic fails if unique
        if t > x.min():
            assert coverage_K(x, np.nextafter(t, -np.inf)) < c or np.any(x == t)


def test_dominance_of_inverse():
    a = np.array([0.1, 0.2, 0.4, 0.8])
    b = np.array([0.2, 0.3, 0.5, 0.9])
    for c in (0.25, 0.5, 0.75, 1.0):
        assert inverse_T(a, c) <= inverse_T(b, c) + 1e-12
