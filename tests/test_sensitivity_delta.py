"""Sensitivity-Delta / B-bounded scale consistency."""
from __future__ import annotations

from private_mobility_cert.certificates.novelty_extensions import (
    laplace_scale_for_sensitivity,
)


def test_laplace_scale_record_and_bounded():
    assert abs(laplace_scale_for_sensitivity(1.0, 1.0) - 1.0) < 1e-12
    assert abs(laplace_scale_for_sensitivity(1.0, 5.0) - 5.0) < 1e-12
    assert abs(laplace_scale_for_sensitivity(2.0, 1.0) - 0.5) < 1e-12


def test_scale_monotone_in_B():
    scales = [laplace_scale_for_sensitivity(1.0, float(B)) for B in (1, 3, 5)]
    assert scales[0] < scales[1] < scales[2]
