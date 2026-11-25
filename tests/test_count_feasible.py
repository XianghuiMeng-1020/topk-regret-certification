"""Independent checks for the count-feasible coupled certificate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from private_mobility_cert.certificates.count_feasible import (
    STATUS_EXACT,
    STATUS_FALLBACK,
    certificate_count_feasible_box,
    count_feasible_box,
    endpoint_support,
    matched_count_feasible_certificates,
    width_support,
)
from private_mobility_cert.certificates.direct_box_calibration import matched_box_certificates
from private_mobility_cert.certificates.release_conditioned import certificate_l1_assignment
from private_mobility_cert.certificates.weighted_ranking import weight_family

_path = Path(__file__).with_name("count_reference.py")
_spec = importlib.util.spec_from_file_location("count_reference", _path)
_ref = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_ref)


def _span(*values: float) -> float:
    return 1.0 + max(abs(v) for v in values)


@pytest.mark.parametrize(
    ("y", "weights", "u_minus", "u_plus"),
    [
        (np.array([5.0, 4.0, 1.0]), weight_family("uniform", 1), 0.0, 0.0),
        (np.array([2.0, 2.0, 0.0, -0.2]), weight_family("dcg", 2), 0.5, 0.5),
        (np.array([1.0, 0.4, 0.4, 0.1, -0.1]), weight_family("reciprocal", 3), 0.25, 0.25),
        (np.array([8.0, 1.0, 1.0, 1.0, 0.0, 0.0]), np.array([3.0, 1.0, 0.0]), 1.0, 1.0),
        (np.array([0.2, 0.2, 0.2, 0.2]), weight_family("uniform", 4), 2.0, 2.0),
        (np.array([9.0, 3.0, 3.0, 0.5, 0.5, 0.0, 4.0]), weight_family("dcg", 5), 1.5, 0.4),
        (np.arange(6.0, 0.0, -1.0), weight_family("reciprocal", 6), 0.8, 0.8),
        (np.array([0.3, 0.2, 0.1, 4.0, 4.0, 4.0, 0.0, 0.0]), weight_family("uniform", 3), 1.0, 1.0),
    ],
)
def test_assignment_matches_independent_enumeration(y, weights, u_minus, u_plus) -> None:
    y = np.asarray(y, dtype=np.float64)
    k = int(np.asarray(weights).size)
    exact = _ref.enumerate_count_feasible(y, k, weights, u_minus, u_plus)
    got = certificate_count_feasible_box(
        y, k=k, weights=weights, u_minus=u_minus, u_plus=u_plus
    )
    assert got.status == STATUS_EXACT
    assert got.value == pytest.approx(exact, abs=1e-6 * _span(exact, got.value))
    assert got.value >= -1e-10


def test_lp_matches_coordinate_endpoints_without_calling_the_assignment() -> None:
    rng = np.random.default_rng(20261125)
    gaps = []
    for _ in range(24):
        n = int(rng.integers(3, 8))
        coeff = rng.normal(size=n)
        coeff -= coeff.mean()
        lower = float(rng.uniform(0.0, 2.0))
        upper = rng.uniform(-lower, lower + 3.0, size=n)
        upper = np.maximum(upper, -lower)
        formula = endpoint_support(coeff, lower=lower, upper=upper)
        linear = _ref.support_lp(coeff, lower, upper)
        gaps.append(abs(formula - linear))
        assert formula == pytest.approx(linear, abs=1e-7 * _span(formula, linear))
        if abs(coeff.sum()) < 1e-8:
            width = upper + lower
            assert width_support(coeff, width) == pytest.approx(formula, abs=1e-8 * _span(formula))
    assert max(gaps) < 1e-7


def test_width_formula_requires_zero_sum() -> None:
    from private_mobility_cert.certificates.topk_regret import TopKCertError

    with pytest.raises(TopKCertError, match="zero-sum"):
        width_support(np.array([1.0, 0.2]), np.array([1.0, 1.0]))


def test_count_feasible_never_exceeds_the_same_box() -> None:
    y = np.array([0.4, 3.0, 2.0, -0.1, 5.0, 0.2])
    weights = weight_family("dcg", 3)
    got = matched_count_feasible_certificates(
        y, k=3, weights=weights, u_minus=1.0, u_plus=1.0
    )
    original = matched_box_certificates(y, k=3, weights=weights, half_width=1.0)
    assert got["status"] == STATUS_EXACT
    assert got["C_box"] <= original["C_box"] + 1e-8
    for key in ("C_sep_general", "C_rank", "C_pair", "C_block5"):
        assert got["C_box"] <= got[key] + 1e-8


def test_high_released_deployed_counts_do_not_tighten() -> None:
    y = np.array([10.0, 9.0, 8.0, 7.0, -0.2, 0.4])
    weights = weight_family("reciprocal", 3)
    plus = certificate_count_feasible_box(
        y, k=3, weights=weights, u_minus=1.0, u_plus=1.0
    ).value
    original = certificate_l1_assignment(y, k=3, weights=weights, half_width=1.0).value
    assert plus == pytest.approx(original, abs=1e-8)


def test_unselected_low_coordinate_does_not_tighten_when_deployed_counts_are_high() -> None:
    weights = weight_family("uniform", 2)
    high = certificate_count_feasible_box(
        np.array([6.0, 5.0, 4.0, 3.0]), k=2, weights=weights, u_minus=1.0, u_plus=1.0
    ).value
    low_tail = certificate_count_feasible_box(
        np.array([6.0, 5.0, 4.0, -0.5]), k=2, weights=weights, u_minus=1.0, u_plus=1.0
    ).value
    assert low_tail == pytest.approx(high, abs=1e-8)


def test_low_deployed_release_can_be_strictly_tighter() -> None:
    y = np.array([0.3, 0.2, 0.1, 0.0])
    weights = weight_family("uniform", 2)
    plus = certificate_count_feasible_box(
        y, k=2, weights=weights, u_minus=1.0, u_plus=1.0
    ).value
    original = certificate_l1_assignment(y, k=2, weights=weights, half_width=1.0).value
    assert plus < original - 1e-6


def test_empty_intersection_uses_the_original_box_and_says_so() -> None:
    y = np.array([4.0, 3.0, -2.0])
    weights = weight_family("dcg", 2)
    box = count_feasible_box(y, u_minus=1.0, u_plus=1.0)
    assert box.consistent is False
    got = certificate_count_feasible_box(
        y, k=2, weights=weights, u_minus=1.0, u_plus=1.0
    )
    original = certificate_l1_assignment(y, k=2, weights=weights, half_width=1.0)
    assert got.status == STATUS_FALLBACK
    assert got.method == "count_feasible_fallback_box"
    assert got.value == pytest.approx(original.value, abs=1e-12)
    family = matched_count_feasible_certificates(
        y, k=2, weights=weights, u_minus=1.0, u_plus=1.0
    )
    assert family["status"] == STATUS_FALLBACK
    assert family["C_box"] == pytest.approx(original.value, abs=1e-12)


def test_k5_decision_identity_and_k6_gap() -> None:
    y = np.array([0.2, 0.5, 0.1, 0.4, 0.3, 2.0, 1.5, 0.0])
    for k, expect_flag in ((5, 1.0), (6, 0.0)):
        weights = weight_family("dcg", k)
        got = matched_count_feasible_certificates(
            y, k=k, weights=weights, u_minus=1.0, u_plus=1.0
        )
        assert got["k5_identity_applied"] == expect_flag
        if k <= 5:
            assert got["C_strong"] == got["C_box"]
            assert got["C_block5"] == got["C_box"]
        else:
            assert got["C_strong"] >= got["C_box"] - 1e-8


def test_module_does_not_call_mswap() -> None:
    text = Path("src/private_mobility_cert/certificates/count_feasible.py").read_text()
    assert "certificate_mswap_os" not in text
    assert "certificate_min_valid" not in text
