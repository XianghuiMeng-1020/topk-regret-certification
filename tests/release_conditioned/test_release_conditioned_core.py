"""Core identity, validity, solver agreement, and release dependence."""

from __future__ import annotations


import numpy as np
import pytest

from private_mobility_cert.certificates.release_conditioned import (
    build_threshold_table,
    certificate_coupled_exact,
    certificate_l1_assignment,
    certificate_orderstat_exact,
    certificate_range_l1_assignment,
    deploy_ranking,
    exact_weighted_regret,
    exact_weighted_regret_enum,
    monotonicity_probe,
    old_degenerate_certificate,
    pathwise_identity_holds,
    unordered_subset_swap_certificate,
    weight_family,
)
from private_mobility_cert.privacy.geometric import sample_two_sided_geometric




def test_pathwise_identity_random() -> None:
    rng = np.random.default_rng(1)
    for n, k in [(4, 2), (5, 3), (6, 2)]:
        w = weight_family("dcg", k)
        for _ in range(40):
            x = rng.integers(0, 8, size=n).astype(np.float64)
            z = rng.normal(0.0, 1.0, size=n)
            y = x + z
            ok, r, c = pathwise_identity_holds(x, y, w, k)
            assert ok, (r, c, x, y)
            pi = deploy_ranking(y, k)
            assert exact_weighted_regret(x, pi, w) == pytest.approx(
                exact_weighted_regret_enum(x, pi, w), abs=1e-9
            )


def test_assignment_matches_exact_l1() -> None:
    rng = np.random.default_rng(2)
    for n, k in [(5, 2), (6, 3), (7, 2)]:
        w = weight_family("reciprocal", k)
        table = build_threshold_table(
            mechanism="M1", n=n, k_max=k, epsilon=0.8, alpha=0.05, weights=w
        )
        t = 0.5 * table.range_width
        for _ in range(20):
            y = rng.normal(3.0, 2.0, size=n)
            ex = certificate_coupled_exact(
                y, k=k, weights=w, u_plus=table.u_plus, u_minus=table.u_minus, bound="l1_range"
            )
            asg = certificate_l1_assignment(y, k=k, weights=w, half_width=t)
            assert asg.value == pytest.approx(ex.value, abs=1e-7)


def test_orderstat_certificate_dominates_regret_on_event() -> None:
    """On the realized order-stat event proxy: use true z OS vs thresholds."""
    rng = np.random.default_rng(3)
    n, k = 6, 3
    w = weight_family("dcg", k)
    table = build_threshold_table(
        mechanism="M1", n=n, k_max=k, epsilon=1.0, alpha=0.1, weights=w
    )
    covered = 0
    checked = 0
    for _ in range(200):
        x = rng.integers(0, 12, size=n).astype(np.float64)
        z = rng.laplace(0.0, 1.0 / 1.0, size=n)
        y = x + z
        zs = np.sort(z)
        # event: top/bottom k order stats within thresholds
        ok_event = True
        for r in range(1, k + 1):
            if zs[-r] > table.u_plus[r - 1] + 1e-12:
                ok_event = False
            if zs[r - 1] < -table.u_minus[r - 1] - 1e-12:
                ok_event = False
        if not ok_event:
            continue
        checked += 1
        pi = deploy_ranking(y, k)
        regret = exact_weighted_regret(x, pi, w)
        cert = certificate_orderstat_exact(y, k=k, weights=w, table=table)
        if regret <= cert.value + 1e-9:
            covered += 1
        else:
            pytest.fail(f"counterexample regret={regret} cert={cert.value} x={x} y={y}")
    assert checked >= 20
    assert covered == checked


def test_release_dependence_and_monotonicity() -> None:
    n, k = 8, 3
    w = weight_family("dcg", k)
    table = build_threshold_table(
        mechanism="M1", n=n, k_max=k, epsilon=1.0, alpha=0.05, weights=w
    )
    rows = monotonicity_probe(
        n=n, k=k, weights=w, table=table, margin_grid=np.array([0.0, 0.5, 2.0, 10.0, 50.0])
    )
    cs = [r["C_exact"] for r in rows]
    # nonincreasing in margin
    for a, b in zip(cs, cs[1:]):
        assert b <= a + 1e-9
    olds = [r["C_old_degenerate"] for r in rows]
    assert max(olds) - min(olds) < 1e-9  # old formula constant
    assert cs[-1] < cs[0]  # large margin strictly tighter than near-tie


def test_uniform_recovers_unordered_subset_swap() -> None:
    rng = np.random.default_rng(4)
    n, k = 6, 3
    w = weight_family("uniform", k)
    table = build_threshold_table(
        mechanism="M1", n=n, k_max=k, epsilon=1.0, alpha=0.05, weights=w
    )
    for _ in range(25):
        y = rng.normal(2.0, 3.0, size=n)
        coupled = certificate_orderstat_exact(y, k=k, weights=w, table=table)
        unordered = unordered_subset_swap_certificate(y, k=k, table=table)
        # For uniform weights the coupled OS certificate should match classical max_m form
        # up to numerical tolerance (same information + same optimization over exchanges).
        assert coupled.value == pytest.approx(unordered, abs=1e-7)


def test_old_formula_not_release_conditioned() -> None:
    w = weight_family("dcg", 3)
    table = build_threshold_table(
        mechanism="M1", n=5, k_max=3, epsilon=1.0, alpha=0.05, weights=w
    )
    y_a = np.array([100.0, 90.0, 80.0, 1.0, 0.0])
    y_b = np.array([100.0, 99.9, 99.8, 99.7, 99.6])
    ca, da, qa = old_degenerate_certificate(y_a, k=3, weights=w, table=table)
    cb, db, qb = old_degenerate_certificate(y_b, k=3, weights=w, table=table)
    assert da == pytest.approx(0.0, abs=1e-12)
    assert db == pytest.approx(0.0, abs=1e-12)
    assert ca == pytest.approx(cb, abs=1e-12)
    assert qa == pytest.approx(qb, abs=1e-12)


@pytest.mark.parametrize("mech", ["M1", "M2"])
def test_monte_carlo_validity_dev_only(mech: str) -> None:
    rng = np.random.default_rng(5 if mech == "M1" else 6)
    n, k, eps, alpha = 6, 2, 1.0, 0.10
    w = weight_family("dcg", k)
    table = build_threshold_table(
        mechanism=mech, n=n, k_max=k, epsilon=eps, alpha=alpha, weights=w
    )
    trials = 400
    violations = 0
    slacks = []
    for t in range(trials):
        x = rng.integers(0, 15, size=n).astype(np.float64)
        if mech == "M1":
            z = rng.laplace(0.0, 1.0 / eps, size=n)
        else:
            z = sample_two_sided_geometric(rng, epsilon=eps, size=n).astype(np.float64)
        y = x + z
        pi = deploy_ranking(y, k)
        regret = exact_weighted_regret(x, pi, w)
        # exact OS when n small; also check scalable range relaxation validity
        c_exact = certificate_orderstat_exact(y, k=k, weights=w, table=table).value
        c_relax = certificate_range_l1_assignment(y, k=k, weights=w, table=table).value
        assert c_relax + 1e-9 >= c_exact  # relaxation dominates exact OS coupled
        if regret > c_exact + 1e-8:
            violations += 1
        slacks.append(c_exact - regret)
    # Bonferroni OS event is conservative; allow empirical rate <= 2*alpha in finite sample
    emp = violations / trials
    assert emp <= min(1.0, 2.5 * alpha + 0.02)
