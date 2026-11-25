"""Brute-force agreement: full coupled box equals robust regret on the box event."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from private_mobility_cert.certificates.block_coupled_certificates import (
    certificate_block5_box,
    certificate_full_coupled_box,
    certificate_pair_block_box,
    certificate_rankwise_box,
    matched_separable_box_value,
    strongest_nonexact_same_event,
)
from private_mobility_cert.certificates.release_conditioned import (
    certificate_l1_assignment,
    deploy_ranking,
)
def _weights(kind: str, k: int) -> np.ndarray:
    if kind == "uniform":
        return np.ones(k)
    if kind == "dcg":
        return 1.0 / np.log2(np.arange(2, k + 2))
    if kind == "reciprocal":
        return 1.0 / np.arange(1, k + 1)
    if kind == "monotone":
        return np.linspace(1.0, 0.2, k)
    raise ValueError(kind)


def _brute_box_robust(y: np.ndarray, k: int, w: np.ndarray, t: float) -> float:
    """C_E = max_π max_{|z|_∞<=t} R(π̂; Y-z) via enum of injections for small n."""
    n = y.size
    pi_hat = deploy_ranking(y, k)
    best = 0.0
    for pi in itertools.permutations(range(n), k):
        # For box L∞ ball, worst-case regret shift is t||a||_1 with a = W_π - W_π̂
        # Equivalent: G_Y(π) + t||a||_1.
        from private_mobility_cert.certificates.release_conditioned import (
            coefficient_map,
            l1_noise_bound,
            observable_margin,
        )

        pi_arr = np.asarray(pi, dtype=np.int64)
        g = observable_margin(y, pi_arr, pi_hat, w)
        a = coefficient_map(pi_arr, pi_hat, w, n)
        b = l1_noise_bound(a, t)
        best = max(best, g + b)
    return float(max(best, 0.0))


@pytest.mark.parametrize("n,k", [(5, 2), (6, 3), (7, 3), (8, 4)])
@pytest.mark.parametrize("wkind", ["uniform", "dcg", "reciprocal", "monotone"])
@pytest.mark.parametrize("t", [0.5, 1.25, 3.0])
def test_full_coupled_equals_brute(n, k, wkind, t):
    rng = np.random.default_rng(1000 + 17 * n + k)
    y = rng.normal(size=n)
    y[0] -= 2.0  # negative coordinate
    w = _weights(wkind, k)
    c_full = certificate_full_coupled_box(y, k=k, weights=w, half_width=t).value
    c_ref = certificate_l1_assignment(y, k=k, weights=w, half_width=t).value
    c_brute = _brute_box_robust(y, k=k, w=w, t=t)
    assert abs(c_full - c_ref) <= 1e-9
    assert abs(c_full - c_brute) <= 1e-7


@pytest.mark.parametrize("n,k", [(6, 3), (8, 4)])
def test_relaxations_dominate_full(n, k):
    rng = np.random.default_rng(42)
    y = rng.normal(size=n)
    w = _weights("dcg", k)
    t = 1.0
    full = certificate_full_coupled_box(y, k=k, weights=w, half_width=t).value
    rank = certificate_rankwise_box(y, k=k, weights=w, half_width=t).value
    pair = certificate_pair_block_box(y, k=k, weights=w, half_width=t).value
    b5 = certificate_block5_box(y, k=k, weights=w, half_width=t).value
    sep = matched_separable_box_value(y, k=k, weights=w, half_width=t)
    assert rank + 1e-9 >= full
    assert pair + 1e-9 >= full
    assert b5 + 1e-9 >= full
    assert sep + 1e-9 >= full


def test_strong_min_valid_and_above_full():
    rng = np.random.default_rng(7)
    y = rng.normal(size=7)
    w = _weights("reciprocal", 3)
    t = 2.0
    full = certificate_full_coupled_box(y, k=3, weights=w, half_width=t).value
    strong, parts = strongest_nonexact_same_event(y, k=3, weights=w, half_width=t)
    assert strong <= min(parts.values()) + 1e-12
    assert strong + 1e-9 >= full
