from __future__ import annotations

import numpy as np

from private_mobility_cert.certificates.certified_risk_frontier import (
    TAU_GRID,
    normalize_certificate,
    set_inclusion_holds,
)


def test_inclusion_from_pointwise_dominance():
    C_box = np.array([0.0, 1.0, 2.0, 5.0])
    C_sep = np.array([0.5, 1.0, 3.0, 9.0])
    S = np.array([1.0, 1.0, 1.0, 1.0])
    cb = np.array([normalize_certificate(a, s) for a, s in zip(C_box, S)])
    cs = np.array([normalize_certificate(a, s) for a, s in zip(C_sep, S)])
    for tau in TAU_GRID:
        assert set_inclusion_holds(cb, cs, tau)


def test_inclusion_fails_without_dominance():
    cb = np.array([2.0, 0.1])
    cs = np.array([1.0, 0.2])
    assert not set_inclusion_holds(cb, cs, 1.5)
