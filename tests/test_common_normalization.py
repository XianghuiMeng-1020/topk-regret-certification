from __future__ import annotations

import numpy as np

from private_mobility_cert.certificates.certified_risk_frontier import (
    STAB,
    normalize_certificate,
    release_scale_U,
)
from private_mobility_cert.certificates.release_conditioned import deploy_ranking, weight_family


def test_matches_r4_stab_rule():
    y = np.array([10.0, 5.0, 1.0, 0.0])
    k = 2
    w = weight_family("uniform", k)
    pi = deploy_ranking(y, k)
    U = release_scale_U(y, pi, w)
    assert abs(U - (10.0 + 5.0)) < 1e-12
    assert abs(normalize_certificate(3.0, U) - 3.0 / U) < 1e-12
    assert abs(normalize_certificate(1.0, 0.0) - 1.0 / STAB) < 1e-12


def test_same_denominator_both_methods():
    C_box, C_sep, S = 2.0, 5.0, 4.0
    assert normalize_certificate(C_box, S) < normalize_certificate(C_sep, S)
    # changing S for one method is forbidden; same S preserves order
    assert normalize_certificate(C_box, S) / normalize_certificate(C_sep, S) == C_box / C_sep
