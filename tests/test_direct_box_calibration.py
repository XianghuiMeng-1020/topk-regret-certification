"""Analytic checks for direct joint coordinate-box calibration.

References are decimal arithmetic or PMF sums. They do not call the
implementation under test to obtain the expected radius.
"""
from __future__ import annotations

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest

from private_mobility_cert.certificates.block_coupled_certificates import (
    certificate_full_coupled_box,
    strongest_nonexact_same_event,
)
from private_mobility_cert.certificates.direct_box_calibration import (
    direct_box_coverage_geometric,
    direct_box_coverage_laplace,
    direct_box_radius_geometric,
    direct_box_radius_geometric_sensitivity,
    direct_box_radius_laplace,
    geometric_coordinate_failure,
    matched_box_certificates,
    union_box_radius_laplace,
)
from private_mobility_cert.certificates.release_conditioned import (
    build_threshold_table,
    deploy_ranking,
)
from private_mobility_cert.certificates.subset_swap import (
    coordinate_box_radius_geometric,
    coordinate_box_radius_laplace,
)
from private_mobility_cert.certificates.topk_regret import TopKCertError
from private_mobility_cert.certificates.weighted_ranking import weight_family
from private_mobility_cert.privacy.geometric import two_sided_geometric_pmf

NS = [1, 2, 8, 25, 50, 261, 500, 2000]
EPSILONS = [0.5, 1.0, 2.0]
ALPHAS = [0.01, 0.05, 0.10]
DELTAS = [1.0, 3.0, 5.0]


def _decimal_laplace_radius(n: int, epsilon: float, alpha: float, delta: float) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = 60
        b = Decimal(delta) / Decimal(epsilon)
        inner = (Decimal(1) - Decimal(str(alpha))) ** (Decimal(1) / Decimal(n))
        q = Decimal(1) - inner
        return -b * q.ln()


def _pmf_abs_gt(t: int, epsilon: float) -> float:
    total = 0.0
    for j in range(t + 1, t + 400):
        total += two_sided_geometric_pmf(j, epsilon=epsilon)
        total += two_sided_geometric_pmf(-j, epsilon=epsilon)
    return float(total)


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("epsilon", EPSILONS)
@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("delta", DELTAS)
def test_laplace_direct_radius_matches_decimal_and_dominates_union(n, epsilon, alpha, delta):
    got = direct_box_radius_laplace(n, epsilon, alpha, delta=delta)
    ref = _decimal_laplace_radius(n, epsilon, alpha, delta)
    assert math.isfinite(got) and got >= 0.0
    assert abs(got - float(ref)) <= 1e-9 * (1.0 + abs(float(ref)))
    union = union_box_radius_laplace(n, epsilon, alpha, delta=delta)
    assert got <= union + 1e-12
    coverage = direct_box_coverage_laplace(n, got, epsilon, delta=delta)
    assert abs(coverage - (1.0 - alpha)) <= 1e-10
    union_coverage = direct_box_coverage_laplace(n, union, epsilon, delta=delta)
    assert union_coverage + 1e-12 >= 1.0 - alpha
    if delta == 1.0:
        coordinate = coordinate_box_radius_laplace(n, epsilon, alpha)
        assert union == coordinate


def test_laplace_anchor_values():
    direct = direct_box_radius_laplace(500, 1.0, 0.05, delta=1.0)
    union = coordinate_box_radius_laplace(500, 1.0, 0.05)
    assert union == 9.210340371976184
    # Stated approximate anchor. The stable formula agrees with a 60-digit
    # expansion; the last digits of this printed anchor differ by ~2e-13.
    assert direct == pytest.approx(9.184854640320033, abs=1e-12)
    assert direct <= union


def test_laplace_monotonicity_and_scaling():
    for epsilon in EPSILONS:
        for delta in DELTAS:
            for alpha in ALPHAS:
                radii = [
                    direct_box_radius_laplace(n, epsilon, alpha, delta=delta) for n in NS
                ]
                assert radii == sorted(radii)
            for n in NS:
                by_alpha = [
                    direct_box_radius_laplace(n, epsilon, alpha, delta=delta)
                    for alpha in ALPHAS
                ]
                assert by_alpha == sorted(by_alpha, reverse=True)
                by_eps = [
                    direct_box_radius_laplace(n, eps, 0.05, delta=delta) for eps in EPSILONS
                ]
                assert by_eps == sorted(by_eps, reverse=True)
                base = direct_box_radius_laplace(n, epsilon, 0.05, delta=1.0)
                for delta_value in DELTAS:
                    scaled = direct_box_radius_laplace(
                        n, epsilon, 0.05, delta=delta_value
                    )
                    assert scaled == pytest.approx(delta_value * base, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("epsilon", EPSILONS)
@pytest.mark.parametrize("alpha", ALPHAS)
def test_geometric_direct_radius_is_minimal_integer(n, epsilon, alpha):
    radius = direct_box_radius_geometric(n, epsilon, alpha)
    assert isinstance(radius, int)
    assert radius >= 0
    coverage = direct_box_coverage_geometric(n, radius, epsilon)
    assert coverage + 1e-12 >= 1.0 - alpha
    if radius > 0:
        below = direct_box_coverage_geometric(n, radius - 1, epsilon)
        assert below < 1.0 - alpha
    union = coordinate_box_radius_geometric(n, epsilon, alpha)
    assert float(radius) <= union + 1e-12
    # Independent PMF tail, not the closed form inside the radius helper.
    q = _pmf_abs_gt(radius, epsilon)
    assert q == pytest.approx(geometric_coordinate_failure(radius, epsilon), abs=1e-10)
    pmf_coverage = math.exp(n * math.log1p(-q)) if q < 1.0 else 0.0
    assert pmf_coverage + 1e-8 >= 1.0 - alpha
    if radius >= 1:
        # Strict failure of |Z|<=t is the non-strict tail at t+1, and is smaller.
        assert geometric_coordinate_failure(radius, epsilon) < (
            2.0 * math.exp(-epsilon * radius) / (1.0 + math.exp(-epsilon))
        )


def test_geometric_anchor_and_wider_integer_gap():
    assert direct_box_radius_geometric(500, 1.0, 0.05) == 9
    assert coordinate_box_radius_geometric(500, 1.0, 0.05) == 10.0
    # Integer gap of two: unit discrete Laplace, not a B-scaled sampler.
    direct = direct_box_radius_geometric(500, 0.1, 0.2)
    union = int(coordinate_box_radius_geometric(500, 0.1, 0.2))
    assert union - direct >= 2
    assert direct_box_coverage_geometric(500, direct, 0.1) + 1e-12 >= 0.8
    assert direct_box_coverage_geometric(500, direct - 1, 0.1) < 0.8


def test_geometric_monotonicity():
    for epsilon in EPSILONS:
        for alpha in ALPHAS:
            radii = [direct_box_radius_geometric(n, epsilon, alpha) for n in NS]
            assert radii == sorted(radii)
        for n in NS:
            by_alpha = [direct_box_radius_geometric(n, epsilon, alpha) for alpha in ALPHAS]
            assert by_alpha == sorted(by_alpha, reverse=True)
            by_eps = [direct_box_radius_geometric(n, eps, 0.05) for eps in EPSILONS]
            assert by_eps == sorted(by_eps, reverse=True)


def test_scaled_geometric_helper_is_not_the_unit_sampler():
    unit = direct_box_radius_geometric(100, 1.0, 0.05)
    assert direct_box_radius_geometric_sensitivity(100, 1.0, 0.05, delta=1.0) == unit
    scaled = direct_box_radius_geometric_sensitivity(100, 1.0, 0.05, delta=3.0)
    assert scaled == direct_box_radius_geometric(100, 1.0 / 3.0, 0.05)
    assert scaled >= unit


def test_difficult_but_finite_laplace_inputs():
    for alpha in (1e-8, 1e-12, 1e-15):
        radius = direct_box_radius_laplace(2000, 0.5, alpha, delta=5.0)
        assert math.isfinite(radius) and radius > 0.0
        coverage = direct_box_coverage_laplace(2000, radius, 0.5, delta=5.0)
        assert abs(coverage - (1.0 - alpha)) <= 1e-8
        union = union_box_radius_laplace(2000, 0.5, alpha, delta=5.0)
        assert radius <= union


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 0, "epsilon": 1.0, "alpha": 0.05},
        {"n": -3, "epsilon": 1.0, "alpha": 0.05},
        {"n": True, "epsilon": 1.0, "alpha": 0.05},
        {"n": 10, "epsilon": 0.0, "alpha": 0.05},
        {"n": 10, "epsilon": math.nan, "alpha": 0.05},
        {"n": 10, "epsilon": 1.0, "alpha": 0.0},
        {"n": 10, "epsilon": 1.0, "alpha": 1.0},
        {"n": 10, "epsilon": 1.0, "alpha": math.inf},
    ],
)
def test_laplace_rejects_invalid_inputs(kwargs):
    with pytest.raises(TopKCertError):
        direct_box_radius_laplace(**kwargs)
    with pytest.raises(TopKCertError):
        direct_box_radius_geometric(kwargs["n"], kwargs["epsilon"], kwargs["alpha"])


def test_delta_must_be_positive_and_finite():
    with pytest.raises(TopKCertError):
        direct_box_radius_laplace(10, 1.0, 0.05, delta=0.0)
    with pytest.raises(TopKCertError):
        direct_box_radius_laplace(10, 1.0, 0.05, delta=math.inf)


def test_union_and_os_rank1_anchor_values():
    assert coordinate_box_radius_laplace(500, 1.0, 0.05) == 9.210340371976184
    assert coordinate_box_radius_geometric(500, 1.0, 0.05) == 10.0
    weights = weight_family("uniform", 5)
    laplace = build_threshold_table(
        mechanism="M1_flat_laplace",
        n=500,
        k_max=5,
        epsilon=1.0,
        alpha=0.05,
        weights=weights,
    )
    geometric = build_threshold_table(
        mechanism="M2_two_sided_geometric",
        n=500,
        k_max=5,
        epsilon=1.0,
        alpha=0.05,
        weights=weights,
    )
    assert float(laplace.u_plus[0]) == 10.817278073012886
    assert float(laplace.u_minus[0]) == 10.817278073012886
    assert float(geometric.u_plus[0]) == 11.0


def test_matched_certificates_share_ranking_and_respect_inclusion():
    y = np.array([5.0, 4.0, -1.0, 3.0, 3.0, 0.2, 8.0, 1.0, -2.0, 6.0])
    k = 5
    weights = weight_family("dcg", k)
    t_union = union_box_radius_laplace(y.size, 1.0, 0.05, delta=1.0)
    t_direct = direct_box_radius_laplace(y.size, 1.0, 0.05, delta=1.0)
    union_certs = matched_box_certificates(y, k=k, weights=weights, half_width=t_union)
    direct_certs = matched_box_certificates(y, k=k, weights=weights, half_width=t_direct)
    pi_union = certificate_full_coupled_box(
        y, k=k, weights=weights, half_width=t_union
    ).pi_hat
    pi_direct = certificate_full_coupled_box(
        y, k=k, weights=weights, half_width=t_direct
    ).pi_hat
    assert np.array_equal(pi_union, pi_direct)
    assert np.array_equal(pi_union, deploy_ranking(y, k))
    for certs in (union_certs, direct_certs):
        for key in ("C_sep_general", "C_rank", "C_pair", "C_block5", "C_strong"):
            assert certs["C_box"] <= certs[key] + 1e-8
        assert certs["C_block5"] == pytest.approx(certs["C_box"], abs=1e-8)
        assert certs["C_strong"] == pytest.approx(certs["C_box"], abs=1e-8)
        strong, parts = strongest_nonexact_same_event(
            y,
            k=k,
            weights=weights,
            half_width=t_union if certs is union_certs else t_direct,
        )
        assert strong == pytest.approx(certs["C_strong"], abs=1e-12)
        assert parts["separable"] == pytest.approx(certs["C_sep_general"], abs=1e-12)
    assert direct_certs["C_box"] <= union_certs["C_box"] + 1e-8


def test_calibration_modules_do_not_call_mswap():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    texts = [
        (root / "src/private_mobility_cert/certificates/direct_box_calibration.py").read_text(),
        (root / "experiments/synthetic_box_calibration/run_calibration.py").read_text(),
    ]
    for text in texts:
        assert "certificate_mswap_os" not in text
        assert "certificate_primary" not in text
        assert "certificate_min_valid" not in text
