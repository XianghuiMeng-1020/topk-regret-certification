from __future__ import annotations

import numpy as np

from private_mobility_cert.certificates.certified_risk_frontier import (
    certified_mask,
    normalize_certificate,
)


def test_nan_inf_fail_closed():
    assert normalize_certificate(float("nan"), 1.0) == float("inf")
    assert normalize_certificate(1.0, float("nan")) == float("inf")
    assert normalize_certificate(float("inf"), 1.0) == float("inf")
    assert not certified_mask(np.array([float("inf")]), 1e9)[0]
