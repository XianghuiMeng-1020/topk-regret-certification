"""Exhaustive small-n algebraic search for counterexamples (geometric noise)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from private_mobility_cert.certificates.release_conditioned import (
    build_threshold_table,
    certificate_orderstat_exact,
    certificate_range_l1_assignment,
    deploy_ranking,
    exact_weighted_regret_enum,
    pathwise_identity_holds,
    weight_family,
)




def _noise_event_ok(z: np.ndarray, table, k: int) -> bool:
    zs = np.sort(np.asarray(z, dtype=np.float64))
    for r in range(1, k + 1):
        if zs[-r] > table.u_plus[r - 1] + 1e-12:
            return False
        if zs[r - 1] < -table.u_minus[r - 1] - 1e-12:
            return False
    return True


@pytest.mark.parametrize("wname", ["uniform", "dcg", "reciprocal", "geometric"])
def test_exhaustive_geometric_small(wname: str) -> None:
    n, k = 4, 2
    eps, alpha = 1.0, 0.2
    w = weight_family(wname, k)
    table = build_threshold_table(
        mechanism="M2", n=n, k_max=k, epsilon=eps, alpha=alpha, weights=w
    )
    x_vals = range(0, 3)  # 0..2
    z_vals = range(-2, 3)  # -2..2
    cases = 0
    event_cases = 0
    identity_fail = 0
    cert_fail = 0
    relax_fail = 0
    for x_t in itertools.product(x_vals, repeat=n):
        x = np.asarray(x_t, dtype=np.float64)
        for z_t in itertools.product(z_vals, repeat=n):
            z = np.asarray(z_t, dtype=np.float64)
            y = x + z
            cases += 1
            ok, _, _ = pathwise_identity_holds(x, y, w, k)
            if not ok:
                identity_fail += 1
            if not _noise_event_ok(z, table, k):
                continue
            event_cases += 1
            pi = deploy_ranking(y, k)
            regret = exact_weighted_regret_enum(x, pi, w)
            c = certificate_orderstat_exact(y, k=k, weights=w, table=table).value
            cr = certificate_range_l1_assignment(y, k=k, weights=w, table=table).value
            if regret > c + 1e-9:
                cert_fail += 1
                pytest.fail(f"OS cert counterexample w={wname} x={x} z={z} R={regret} C={c}")
            if regret > cr + 1e-9:
                relax_fail += 1
                pytest.fail(f"range-l1 counterexample w={wname} x={x} z={z} R={regret} C={cr}")
    assert identity_fail == 0
    assert cert_fail == 0
    assert relax_fail == 0
    assert cases > 0 and event_cases > 0


def test_exhaustive_clipping_ties() -> None:
    n, k = 4, 2
    w = weight_family("dcg", k)
    table = build_threshold_table(
        mechanism="M2", n=n, k_max=k, epsilon=1.0, alpha=0.2, weights=w
    )
    # Force clipping-heavy and tie geometries
    truths = [
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([5.0, 0.0, 0.0, 0.0]),
        np.array([2.0, 2.0, 2.0, 2.0]),
    ]
    for x in truths:
        for z_t in itertools.product(range(-3, 2), repeat=n):
            z = np.asarray(z_t, dtype=np.float64)
            y = x + z
            ok, r, c = pathwise_identity_holds(x, y, w, k)
            assert ok
            if _noise_event_ok(z, table, k):
                pi = deploy_ranking(y, k)
                regret = exact_weighted_regret_enum(x, pi, w)
                cert = certificate_orderstat_exact(y, k=k, weights=w, table=table).value
                assert regret <= cert + 1e-9
