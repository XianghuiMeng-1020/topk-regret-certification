from __future__ import annotations

import numpy as np

from private_mobility_cert.certificates.certified_risk_frontier import (
    certified_mask,
    inverse_T,
)


def test_ties_deterministic_membership():
    x = np.array([0.25, 0.25, 0.25, 1.0])
    m = certified_mask(x, 0.25)
    assert m.sum() == 3
    assert inverse_T(x, 0.5) == 0.25
    assert inverse_T(x, 0.75) == 0.25
    assert inverse_T(x, 1.0) == 1.0
