"""Independent checks for the order-statistic envelope."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from private_mobility_cert.certificates.coupled_mswap import (
    certificate_min_valid,
    certificate_mswap_os,
)
from private_mobility_cert.certificates.coupled_scalable import certificate_primary
from private_mobility_cert.certificates.os_envelope import (
    METHOD_OS_ENVELOPE,
    METHOD_PRIMARY_OS_ENVELOPE,
    certificate_os_envelope,
    os_support_effective,
)
from private_mobility_cert.certificates.release_conditioned import (
    build_threshold_table,
    certificate_l1_assignment,
    certificate_range_l1_assignment,
    coefficient_map,
    deploy_ranking,
    observable_margin,
)
from private_mobility_cert.certificates.topk_regret import TopKCertError
from private_mobility_cert.certificates.weighted_ranking import weight_family
_reference_path = Path(__file__).with_name("os_reference.py")
_spec = importlib.util.spec_from_file_location("os_reference", _reference_path)
_reference = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_reference)
enumerate_os = _reference.enumerate_os
support_formula = _reference.support_formula
support_lp = _reference.support_lp

TOL = 1e-6


def _scale(*values: float) -> float:
    return 1.0 + max(abs(value) for value in values)


def test_independent_lp_matches_prefix_minimum_formula() -> None:
    rng = np.random.default_rng(20261124)
    gaps = []
    for _ in range(30):
        n = int(rng.integers(3, 8))
        depth = int(rng.integers(2, n + 1))
        coefficients = np.zeros(n, dtype=np.float64)
        coefficients[:depth] = rng.normal(size=depth)
        upper = rng.uniform(0.0, 6.0, size=depth)
        lower = rng.uniform(0.0, 6.0, size=depth)
        if rng.random() < 0.5:
            upper = np.sort(upper)[::-1]
            lower = np.sort(lower)[::-1]
        else:
            upper[0] = 0.2
            upper[-1] = 5.0
        formula = support_formula(coefficients, upper, lower)
        linear = support_lp(coefficients, upper, lower)
        gaps.append(abs(formula - linear))
        assert formula == pytest.approx(linear, abs=1e-6 * _scale(formula, linear))
    assert max(gaps) < 1e-6


def test_production_support_matches_independent_formula_not_the_raw_charge() -> None:
    coefficients = np.array([3.0, -1.0, -2.0, 0.0])
    upper = np.array([1.0, 4.0])
    lower = np.array([2.0, 0.5])
    assert os_support_effective(coefficients, upper, lower) == pytest.approx(
        support_formula(coefficients, upper, lower)
    )
    # Raw thresholds would charge the second positive mass at 4. The feasible
    # second-largest coordinate cannot exceed the first threshold.
    assert support_formula(coefficients, upper, lower) < 3.0 * 1.0 + 0.0 + 2.0 * 2.0 + 1.0 * 4.0


@pytest.mark.parametrize(
    "y,weights,mechanism",
    [
        (np.array([5.0, 1.0, -2.0]), weight_family("uniform", 1), "M1"),
        (np.array([4.0, 4.0, 0.0, -1.0]), weight_family("dcg", 2), "M1"),
        (np.array([3.0, 2.0, 2.0, 1.0, -4.0]), weight_family("reciprocal", 3), "M2"),
        (np.array([-2.0, -1.0, -0.4, 0.2]), weight_family("uniform", 4), "M2"),
        (np.array([9.0, 1.0, 1.0, 1.0, 0.0, 0.0, 8.0]), weight_family("dcg", 5), "M1"),
        (np.arange(8.0, 0.0, -1.0), weight_family("reciprocal", 8), "M2"),
    ],
)
def test_envelope_bounds_exhaustive_os_and_rank1_box(y, weights, mechanism) -> None:
    y = np.asarray(y, dtype=np.float64)
    k = int(np.asarray(weights).size)
    table = build_threshold_table(
        mechanism=mechanism,
        n=y.size,
        k_max=k,
        epsilon=1.0,
        alpha=0.05,
        weights=weights,
    )
    exact = enumerate_os(y, k, weights, table.u_plus, table.u_minus)
    envelope = certificate_os_envelope(y, k=k, weights=weights, table=table)
    rank1 = certificate_range_l1_assignment(y, k=k, weights=weights, table=table).value
    span = _scale(exact, envelope.value, rank1)
    assert exact <= envelope.value + TOL * span
    assert envelope.value <= rank1 + TOL * span
    assert envelope.method == METHOD_OS_ENVELOPE
    assert envelope.n_assignment_solves >= 1


def test_zero_weight_fixture_uses_an_explicit_table() -> None:
    """The threshold solver rejects a zero alpha share. The envelope does not."""
    y = np.array([1.0, 0.0, 0.0, 0.0, 8.0, -3.0])
    weights = np.array([3.0, 1.0, 0.0])
    base = build_threshold_table(
        mechanism="M1", n=y.size, k_max=2, epsilon=1.0, alpha=0.05, weights=weights[:2]
    )
    table = type(base)(
        mechanism=base.mechanism,
        epsilon=base.epsilon,
        n=base.n,
        k_max=3,
        alpha=base.alpha,
        scheme="explicit_zero_weight",
        u_plus=np.array([base.u_plus[0], base.u_plus[1], base.u_plus[1]]),
        u_minus=np.array([base.u_minus[0], base.u_minus[1], base.u_minus[1]]),
    )
    exact = enumerate_os(y, 3, weights, table.u_plus, table.u_minus)
    envelope = certificate_os_envelope(y, k=3, weights=weights, table=table)
    span = _scale(exact, envelope.value, envelope.rank1_value)
    assert exact <= envelope.value + TOL * span
    assert envelope.value <= envelope.rank1_value + TOL * span


def test_nonmonotone_fixture_still_obeys_the_envelope_order() -> None:
    y = np.array([5.0, 4.0, 0.0, 1.0, -2.0])
    weights = np.array([4.0, 2.0, 1.0])
    table = build_threshold_table(
        mechanism="M1", n=y.size, k_max=3, epsilon=1.0, alpha=0.05, weights=weights
    )
    table = type(table)(
        mechanism=table.mechanism,
        epsilon=table.epsilon,
        n=table.n,
        k_max=table.k_max,
        alpha=table.alpha,
        scheme=table.scheme,
        u_plus=np.array([1.0, 5.0, 0.25]),
        u_minus=np.array([2.0, 0.4, 3.0]),
    )
    exact = enumerate_os(y, 3, weights, table.u_plus, table.u_minus)
    envelope = certificate_os_envelope(y, k=3, weights=weights, table=table)
    rank1 = certificate_l1_assignment(y, k=3, weights=weights, half_width=0.5 * (1.0 + 2.0)).value
    span = _scale(exact, envelope.value, rank1)
    assert exact <= envelope.value + TOL * span
    assert envelope.value <= rank1 + TOL * span


def test_zero_and_low_noise_releases_do_not_go_negative() -> None:
    weights = weight_family("uniform", 3)
    for y in (
        np.array([10.0, 9.0, 8.0, 1.0, 0.0]),
        np.zeros(5),
        np.array([-5.0, -4.0, -3.0, -2.0, -1.0]),
    ):
        table = build_threshold_table(
            mechanism="M1", n=y.size, k_max=3, epsilon=2.0, alpha=0.05, weights=weights
        )
        envelope = certificate_os_envelope(y, k=3, weights=weights, table=table)
        assert envelope.value >= -1e-8
        assert envelope.value <= envelope.rank1_value + 1e-8


def test_k9_mswap_witness_stays_below_the_feasible_objective_and_the_envelope_does_not() -> None:
    k = 9
    y = -np.linspace(1.0, 0.1, k)
    weights = np.arange(k, 0, -1, dtype=np.float64)
    table = build_threshold_table(
        mechanism="M1", n=k, k_max=k, epsilon=1.0, alpha=0.05, weights=weights
    )
    pi_hat = deploy_ranking(y, k)
    witness = np.argsort(-y, kind="stable")
    margin = observable_margin(y, witness, pi_hat, weights)
    coefficients = coefficient_map(witness, pi_hat, weights, k)
    objective = margin + support_formula(coefficients, table.u_plus, table.u_minus)
    mswap = certificate_mswap_os(
        y, k=k, weights=weights, table=table, unsafe=True
    )
    envelope = certificate_os_envelope(y, k=k, weights=weights, table=table)
    assert mswap.method == "mswap_os_known_invalid"
    assert objective > 0.1
    assert mswap.value + 1e-8 < objective
    assert envelope.value + 1e-7 >= objective


def test_psi_indexed_by_set_swap_size_is_not_an_upper_charge() -> None:
    weights = np.array([8.0, 5.0, 3.0, 1.0])
    thresholds = np.array([10.0, 9.0, 2.0, 0.5])
    pi_hat = np.arange(4)
    pi = np.array([1, 3, 4, 2])
    coefficients = coefficient_map(pi, pi_hat, weights, 6)
    support = support_formula(coefficients, thresholds, thresholds)
    psi_set = float(weights[0] * (thresholds[0] + thresholds[0]))
    assert int(np.sum(coefficients > 0.0)) == 2
    assert support == pytest.approx(171.0)
    assert psi_set == pytest.approx(160.0)
    assert support > psi_set


def test_safe_pipeline_rejects_the_invalid_minimum_and_does_not_call_mswap() -> None:
    y = np.linspace(20.0, 0.0, 40)
    weights = weight_family("dcg", 5)
    table = build_threshold_table(
        mechanism="M2", n=y.size, k_max=5, epsilon=1.0, alpha=0.05, weights=weights
    )
    with pytest.raises(TopKCertError, match="certificate_min_valid is closed"):
        certificate_min_valid(y, k=5, weights=weights, table=table)
    with pytest.raises(TopKCertError, match="not a valid order-statistic certificate"):
        certificate_mswap_os(y, k=5, weights=weights, table=table)
    primary = certificate_primary(y, k=5, weights=weights, table=table)
    envelope = certificate_os_envelope(y, k=5, weights=weights, table=table)
    assert primary.method == METHOD_PRIMARY_OS_ENVELOPE
    assert primary.value == pytest.approx(envelope.value, abs=1e-9)
    source = Path("src/private_mobility_cert/certificates/os_envelope.py").read_text()
    assert "certificate_mswap_os" not in source
    assert "certificate_min_valid" not in source


def test_high_noise_tie_release_remains_ordered() -> None:
    y = np.array([1.0, 1.0, 1.0, 1.0, -8.0, 0.0])
    weights = weight_family("reciprocal", 2)
    table = build_threshold_table(
        mechanism="M1", n=y.size, k_max=2, epsilon=0.5, alpha=0.05, weights=weights
    )
    exact = enumerate_os(y, 2, weights, table.u_plus, table.u_minus)
    envelope = certificate_os_envelope(y, k=2, weights=weights, table=table)
    assert exact <= envelope.value + 1e-6
    assert envelope.value <= envelope.rank1_value + 1e-8
