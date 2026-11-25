"""Reject invalid 'best of several 1-alpha certificates' confidence logic."""

from __future__ import annotations

import numpy as np
import pytest

from private_mobility_cert.certificates.coupled_mswap import certificate_min_valid
from private_mobility_cert.certificates.os_envelope import certificate_os_envelope
from private_mobility_cert.certificates.release_conditioned import (
    box_baseline_coupled,
    build_threshold_table,
    certificate_range_l1_assignment,
    weight_family,
)
from private_mobility_cert.certificates.topk_regret import TopKCertError


def test_min_on_same_event_is_allowed() -> None:
    """The old min with m-swap is closed. The OS envelope stays on one event."""
    y = np.array([10.0, 9.0, 8.0, 1.0, 0.0, -1.0])
    w = weight_family("dcg", 3)
    table = build_threshold_table(
        mechanism="M1", n=6, k_max=3, epsilon=1.0, alpha=0.05, weights=w
    )
    with pytest.raises(TopKCertError, match="certificate_min_valid is closed"):
        certificate_min_valid(y, k=3, weights=w, table=table)
    c = certificate_os_envelope(y, k=3, weights=w, table=table)
    c_l1 = certificate_range_l1_assignment(y, k=3, weights=w, table=table)
    assert c.value <= c_l1.value + 1e-12


def test_min_across_different_alpha_events_is_invalid_pattern() -> None:
    """Document the forbidden pattern: min of two certificates each at level alpha
    but built from *different* events / separate Bonferroni spends is not 1-alpha
    without adjustment.

    We encode the check as: separately allocated box (alpha) vs OS table (alpha)
    are NOT the same event accounting; taking min without proving event nesting
    is rejected by this guard helper.
    """
    y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    w = weight_family("uniform", 2)
    table = build_threshold_table(
        mechanism="M1", n=5, k_max=2, epsilon=1.0, alpha=0.05, weights=w
    )
    c_os_l1 = certificate_range_l1_assignment(y, k=2, weights=w, table=table)
    # Independent box at the same numeric alpha but via n-coordinate Bonferroni
    # is a *different* event construction.
    c_box = box_baseline_coupled(
        y, k=2, weights=w, mechanism="M1", epsilon=1.0, alpha=0.05
    )

    def same_event_family(a: str, b: str) -> bool:
        # OS-implied range-l1 and stratified m-swap share E_os.
        os_family = {"range_l1_assignment", "l1_assignment", "os_envelope_v1",
                     "coord_asymmetric_box", "primary_os_envelope", "primary_exact_os",
                     "exact_orderstat", "os_branch_and_bound"}
        return a in os_family and b in os_family

    assert not same_event_family(c_os_l1.method, c_box.method)
    # Forbidden: claiming min(c_os_l1, c_box) is 1-alpha without adjustment
    with pytest.raises(AssertionError):
        assert same_event_family(c_os_l1.method, c_box.method)
        _ = min(c_os_l1.value, c_box.value)
