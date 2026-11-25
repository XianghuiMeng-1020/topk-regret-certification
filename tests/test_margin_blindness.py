"""N1 margin-blindness after Y+ sorting."""
from __future__ import annotations

import numpy as np

from private_mobility_cert.certificates.novelty_extensions import (
    separable_is_margin_blind_after_yplus_sort,
)
from private_mobility_cert.certificates.release_conditioned import weight_family


def test_blind_when_deploy_from_yplus():
    rng = np.random.default_rng(0)
    for n, k in [(5, 2), (8, 3), (10, 4)]:
        for _ in range(20):
            y = rng.normal(size=n)
            w = weight_family("dcg", k)
            assert separable_is_margin_blind_after_yplus_sort(y, k, w)


def test_uniform_and_reciprocal():
    y = np.array([3.0, 1.0, -2.0, 0.5, 4.0])
    for wf in ["uniform", "reciprocal", "dcg"]:
        w = weight_family(wf, 3)
        assert separable_is_margin_blind_after_yplus_sort(y, 3, w)
