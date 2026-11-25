"""Synthetic agreement checks for the independent coupled-certificate reference."""

from __future__ import annotations

import numpy as np
import pytest

from private_mobility_cert.certificates.coupled_certificate_ref import (
    bos_lp_verify,
    certificate_coupled_ref_exact,
    orderstat_bound_ref,
    pathwise_identity_ref,
)
from private_mobility_cert.certificates.release_conditioned import (
    certificate_coupled_exact,
    orderstat_noise_bound,
    pathwise_identity_holds,
)


@pytest.mark.parametrize("seed", range(24))
def test_reference_certificate_agrees_with_release_conditioned(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 8))
    k = int(rng.integers(1, n + 1))
    y = rng.normal(size=n)
    w = np.sort(rng.uniform(0.1, 2.0, size=k))[::-1]
    u_plus = np.sort(rng.uniform(0.2, 3.0, size=k))[::-1]
    u_minus = np.sort(rng.uniform(0.2, 3.0, size=k))[::-1]

    reference = certificate_coupled_ref_exact(y, k, w, u_plus, u_minus)
    incumbent = certificate_coupled_exact(
        y, k=k, weights=w, u_plus=u_plus, u_minus=u_minus
    )

    assert reference.value == pytest.approx(incumbent.value, abs=1e-10)
    assert reference.best_g == pytest.approx(incumbent.best_g, abs=1e-10)
    assert reference.best_b == pytest.approx(incumbent.best_b, abs=1e-10)
    assert np.array_equal(reference.pi_hat, incumbent.pi_hat)
    assert np.array_equal(reference.best_pi, incumbent.best_pi)


@pytest.mark.parametrize("seed", range(32))
def test_orderstat_closed_form_agrees_with_lp(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 9))
    a = rng.normal(size=n)
    support = max(int(np.count_nonzero(a > 0.0)), int(np.count_nonzero(a < 0.0)), 1)
    u_plus = np.sort(rng.uniform(0.1, 4.0, size=support))[::-1]
    u_minus = np.sort(rng.uniform(0.1, 4.0, size=support))[::-1]

    closed_form = orderstat_bound_ref(a, u_plus, u_minus)
    lp_value = bos_lp_verify(a, u_plus, u_minus, n)

    assert closed_form == pytest.approx(lp_value, abs=1e-6)
    assert closed_form == pytest.approx(orderstat_noise_bound(a, u_plus, u_minus), abs=1e-10)


@pytest.mark.parametrize("seed", range(12))
def test_reference_pathwise_identity_matches_existing_checker(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 8))
    k = int(rng.integers(1, n + 1))
    x = rng.normal(size=n)
    y = x + rng.normal(scale=0.5, size=n)
    w = np.sort(rng.uniform(0.1, 2.0, size=k))[::-1]

    assert pathwise_identity_ref(x, y, w, k) == pathwise_identity_holds(x, y, w, k)
