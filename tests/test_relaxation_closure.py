"""Block relaxations are valid and not necessarily monotone; the monotone closure."""

from __future__ import annotations

import importlib.util
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from private_mobility_cert.certificates.count_feasible import (
    STATUS_EXACT,
    STATUS_FALLBACK,
    matched_count_feasible_certificates,
)
from private_mobility_cert.certificates.direct_box_calibration import certification_decision
from private_mobility_cert.certificates.relaxation_closure import (
    close_matched_family,
    monotone_closed_value,
)
from private_mobility_cert.certificates.release_conditioned import certificate_l1_assignment
from private_mobility_cert.certificates.weighted_ranking import weight_family

_path = Path(__file__).with_name("relaxation_reference.py")
_spec = importlib.util.spec_from_file_location("relaxation_reference", _path)
_ref = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_ref)

_NK7_Y = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
_NK7_W = np.array([7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
_RANK_Y = np.array([2.0, 2.0, 1.0, 0.0])
_RANK_W = np.array([5.0, 4.0, 1.0])
_PAIR_W = np.array([5.0, 4.0, 3.0, 2.0])
_N20_Y = [
    "5.523", "0.865", "4.528", "5.571", "3.621",
    "-1.478", "-2.447", "-2.400", "2.377", "0.490",
    "3.482", "3.504", "3.757", "4.652", "-1.017",
    "-0.734", "-2.608", "-0.347", "3.423", "1.073",
]
_N20_W = [
    "3.845", "2.536", "2.070", "1.824", "1.689",
    "0.842", "0.705", "0.564", "0.328", "0.140",
]


def _library(y, weights, u_minus, u_plus):
    return matched_count_feasible_certificates(
        np.asarray(y, dtype=np.float64),
        k=int(np.asarray(weights).size),
        weights=np.asarray(weights, dtype=np.float64),
        u_minus=u_minus,
        u_plus=u_plus,
    )


def _assert_matches(y, weights, u_minus, u_plus) -> dict:
    exact = _ref.evaluate_small(y, weights, u_minus=u_minus, u_plus=u_plus)
    got = _library(y, weights, u_minus, u_plus)
    assert got["status"] == STATUS_EXACT
    pairs = {
        "C_box_original": exact["original"]["box"]["value"],
        "C_rank_original": exact["original"]["rank"]["value"],
        "C_pair_original": exact["original"]["pair"]["value"],
        "C_block5_original": exact["original"]["block5"]["value"],
        "C_sep_original": exact["original"]["sep"]["value"],
        "C_strong_original": exact["original"]["strong"]["value"],
        "C_box": exact["plus"]["box"]["value"],
        "C_rank": exact["plus"]["rank"]["value"],
        "C_pair": exact["plus"]["pair"]["value"],
        "C_block5": exact["plus"]["block5"]["value"],
        "C_sep_general": exact["plus"]["sep"]["value"],
        "C_strong": exact["plus"]["strong"]["value"],
    }
    for key, value in pairs.items():
        assert got[key] == pytest.approx(value, abs=1e-9)
    return exact


def test_nk7_counterexample_matches_independent_enumeration() -> None:
    exact = _assert_matches(_NK7_Y, _NK7_W, 1.0, 1.0)
    assert exact["original"]["box"]["value"] == 18
    assert exact["plus"]["box"]["value"] == 18
    assert exact["original"]["rank"]["value"] == 24
    assert exact["plus"]["rank"]["value"] == 24
    assert exact["original"]["pair"]["value"] == 22
    assert exact["plus"]["pair"]["value"] == 22
    assert exact["original"]["block5"]["value"] == 18
    assert exact["plus"]["block5"]["value"] == 19
    assert exact["original"]["sep"]["value"] == 24
    assert exact["plus"]["sep"]["value"] == 24
    assert exact["original"]["strong"]["value"] == 18
    assert exact["plus"]["strong"]["value"] == 19
    plus_pairs = exact["plus"]["block5"]["pairs"]
    # This colliding collection is not the original maximizer. Its own
    # objective rises by 2, while the two block maxima differ by 1.
    assert _ref.fixed_assignment_gap(_NK7_Y, _NK7_W, plus_pairs, u_minus=1.0, u_plus=1.0) == 2
    assert exact["plus"]["block5"]["value"] - exact["original"]["block5"]["value"] == 1
    assert exact["plus"]["block5"]["collisions"]
    assert len(set(item for _rank, item in exact["plus"]["box"]["pairs"])) == 7


def test_nk7_raw_strong_reverses_tau_074_and_closure_does_not() -> None:
    got = _library(_NK7_Y, _NK7_W, 1.0, 1.0)
    scale = 25.0
    assert certification_decision(got["C_strong_original"], scale, 0.74)
    assert not certification_decision(got["C_strong"], scale, 0.74)
    closed = monotone_closed_value(got["C_strong_original"], got["C_strong"])
    assert closed == 18
    assert certification_decision(closed, scale, 0.74)
    assert got["C_strong"] > got["C_strong_original"]
    assert got["C_box"] <= got["C_box_original"]


def test_rank_and_pair_counterexamples_increase_without_moving_the_exact_certificate() -> None:
    rank = _assert_matches(_RANK_Y, _RANK_W, 2.0, 2.0)
    assert rank["plus"]["rank"]["value"] == 23
    assert rank["original"]["rank"]["value"] == 22
    assert rank["plus"]["box"]["value"] == rank["original"]["box"]["value"] == 20
    assert rank["plus"]["rank"]["collisions"]
    pair = _assert_matches(_RANK_Y, _PAIR_W, 2.0, 2.0)
    assert pair["plus"]["pair"]["value"] == 12
    assert pair["original"]["pair"]["value"] == 11
    assert pair["plus"]["rank"]["value"] == 12
    assert pair["original"]["rank"]["value"] == 11
    assert pair["plus"]["box"]["value"] == 10


def test_n20_decimal_gap_is_stable_and_not_an_nk_artifact() -> None:
    precise = _ref.decimal_assignment_family(_N20_Y, _N20_W, "3")
    got = _library([Decimal(v) for v in _N20_Y], [Decimal(v) for v in _N20_W], 3.0, 3.0)
    assert precise["box_plus"] - precise["box_original"] == 0
    assert precise["strong_plus"] - precise["strong_original"] == Decimal("0.087220")
    assert precise["block5_plus"] - precise["block5_original"] == Decimal("0.087220")
    assert precise["rank_plus"] - precise["rank_original"] == Decimal("0.087220")
    assert precise["pair_plus"] - precise["pair_original"] == Decimal("0.174440")
    assert precise["sep_plus"] < precise["sep_original"]
    assert got["status"] == STATUS_EXACT
    assert got["C_box"] == pytest.approx(float(precise["box_plus"]), abs=1e-9)
    assert got["C_strong"] == pytest.approx(float(precise["strong_plus"]), abs=1e-9)
    assert got["C_strong"] > got["C_strong_original"] + 1e-6
    assert got["k5_identity_applied"] == 0


def test_raw_relaxations_remain_upper_bounds_when_they_increase() -> None:
    for y, weights, radius in (
        (_NK7_Y, _NK7_W, 1.0),
        (_RANK_Y, _RANK_W, 2.0),
        (_RANK_Y, _PAIR_W, 2.0),
    ):
        got = _library(y, weights, radius, radius)
        for key in ("C_sep_general", "C_rank", "C_pair", "C_block5", "C_strong"):
            assert got[key] + 1e-9 >= got["C_box"]


def test_monotone_closure_dominates_the_count_feasible_box_and_both_inputs() -> None:
    got = _library(_NK7_Y, _NK7_W, 1.0, 1.0)
    original = {
        "sep": got["C_sep_original"],
        "rank": got["C_rank_original"],
        "pair": got["C_pair_original"],
        "block5": got["C_block5_original"],
        "strong": got["C_strong_original"],
    }
    plus = {
        "sep": got["C_sep_general"],
        "rank": got["C_rank"],
        "pair": got["C_pair"],
        "block5": got["C_block5"],
        "strong": got["C_strong"],
    }
    closed = close_matched_family(original, plus, tuple(original))
    assert closed["strong"] == 18
    assert closed["block5"] == 18
    for key in original:
        assert closed[key] <= original[key] + 1e-12
        assert closed[key] <= plus[key] + 1e-12
        assert closed[key] + 1e-9 >= got["C_box"]


def test_k5_closure_preserves_equality_with_the_exact_certificate() -> None:
    y = np.array([0.2, 0.5, 0.1, 0.4, 0.3, 2.0, 1.5, 0.0])
    weights = np.ones(5)
    got = _library(y, weights, 1.0, 1.0)
    assert got["k5_identity_applied"] == 1.0
    assert got["C_strong"] == pytest.approx(got["C_box"], abs=0.0)
    assert got["C_block5"] == pytest.approx(got["C_box"], abs=0.0)
    closed = monotone_closed_value(got["C_strong_original"], got["C_strong"])
    assert closed == pytest.approx(got["C_box"], abs=1e-12)
    assert got["C_box"] <= got["C_box_original"] + 1e-12


def test_k6_compatibility_gap_is_not_canonicalized() -> None:
    y = np.arange(6.0, 0.0, -1.0)
    weights = weight_family("reciprocal", 6)
    got = _library(y, weights, 0.8, 0.8)
    assert got["k5_identity_applied"] == 0.0
    assert got["C_strong_original"] >= got["C_box_original"] - 1e-8
    assert got["C_strong"] >= got["C_box"] - 1e-8
    # The full injection is a stricter constraint than at least one relaxation
    # on this reciprocal instance, or the two coincide. Either way the k<=5
    # rewrite must not have fired.
    assert got["C_sep_original"] + 1e-8 >= got["C_box_original"]


def test_weight_families_ties_negative_and_asymmetric_boxes_stay_valid() -> None:
    cases = [
        (np.array([1.0, 1.0, 0.0, 3.0]), weight_family("uniform", 3), 1.0, 1.0),
        (np.array([4.0, 0.2, 0.2, -0.4]), weight_family("dcg", 3), 1.0, 1.0),
        (np.array([0.0, 2.0, 2.0, 0.5]), weight_family("reciprocal", 2), 0.5, 1.5),
        (np.array([3.0, 1.0, 1.0, 0.0]), np.array([4.0, 0.0, 0.0]), 1.0, 1.0),
    ]
    for y, weights, u_minus, u_plus in cases:
        if max(y.shape[0], 0) <= 7 and int(np.asarray(weights).size) <= y.size:
            _assert_matches(y, weights, u_minus, u_plus)
        got = _library(y, weights, u_minus, u_plus)
        assert got["status"] == STATUS_EXACT
        assert got["C_box"] <= got["C_box_original"] + 1e-8
        for key in ("C_sep_general", "C_rank", "C_pair", "C_block5", "C_strong"):
            assert got[key] + 1e-8 >= got["C_box"]


def test_empty_intersection_fallback_is_not_a_count_feasible_optimum() -> None:
    y = np.array([4.0, 3.0, -2.0])
    weights = np.ones(3)
    got = _library(y, weights, 1.0, 1.0)
    original = certificate_l1_assignment(y, k=3, weights=weights, half_width=1.0)
    assert got["status"] == STATUS_FALLBACK
    assert got["C_box"] == pytest.approx(original.value, abs=0.0)
    closed = monotone_closed_value(got["C_strong_original"], got["C_strong"])
    assert closed == pytest.approx(got["C_strong_original"], abs=0.0)


def test_high_count_zero_gain_does_not_move_the_family() -> None:
    y = np.array([10.0, 9.0, 8.0, 7.0, -0.2, 0.4])
    weights = weight_family("dcg", 3)
    got = _library(y, weights, 1.0, 1.0)
    for raw, plus in (
        ("C_box_original", "C_box"),
        ("C_sep_original", "C_sep_general"),
        ("C_rank_original", "C_rank"),
        ("C_pair_original", "C_pair"),
        ("C_block5_original", "C_block5"),
        ("C_strong_original", "C_strong"),
    ):
        assert got[plus] == pytest.approx(got[raw], abs=1e-9)


def test_closure_module_does_not_replace_the_raw_certificate_path() -> None:
    source = Path("src/private_mobility_cert/certificates/count_feasible.py").read_text()
    assert "monotone_closed_value" not in source
    assert "certificate_mswap_os" not in source
    assert "certificate_min_valid" not in source


def test_no_collision_global_assignment_cannot_increase() -> None:
    exact = _ref.evaluate_small(_NK7_Y, _NK7_W, u_minus=1.0, u_plus=1.0)
    pairs = exact["original"]["box"]["pairs"]
    assert len(pairs) == len({item for _rank, item in pairs})
    gap = _ref.fixed_assignment_gap(_NK7_Y, _NK7_W, pairs, u_minus=1.0, u_plus=1.0)
    assert gap <= 1e-12
