"""Hall SDR iff rankwise relaxation equals full coupled assignment."""
from __future__ import annotations

import numpy as np

from private_mobility_cert.certificates.block_coupled_certificates import (
    box_cost_matrix,
    certificate_full_coupled_box,
    certificate_rankwise_box,
)
from private_mobility_cert.certificates.coupling_diagnostics import (
    hall_sdr_exists,
    rankwise_argmax_sets,
)


def test_hall_iff_equality_random():
    rng = np.random.default_rng(20261001)
    for _ in range(80):
        n = int(rng.integers(4, 9))
        k = int(rng.integers(2, min(5, n)))
        y = rng.normal(size=n)
        w = 1.0 / np.arange(1, k + 1)
        t = float(rng.uniform(0.3, 2.0))
        cost, _, _, _ = box_cost_matrix(y, k=k, weights=w, half_width=t)
        sets = rankwise_argmax_sets(cost)
        ok, _ = hall_sdr_exists(sets, n)
        full = certificate_full_coupled_box(y, k=k, weights=w, half_width=t).value
        rank = certificate_rankwise_box(y, k=k, weights=w, half_width=t).value
        eq = abs(full - rank) <= 1e-8
        assert ok == eq


def test_forced_collision_gap():
    # Construct costs where both ranks prefer the same unique item
    y = np.array([10.0, 0.0, 0.0, 0.0])
    w = np.array([1.0, 0.9])
    t = 0.01  # tiny noise so cost ≈ w_r Y_i
    full = certificate_full_coupled_box(y, k=2, weights=w, half_width=t).value
    rank = certificate_rankwise_box(y, k=2, weights=w, half_width=t).value
    cost, _, _, _ = box_cost_matrix(y, k=2, weights=w, half_width=t)
    ok, m = hall_sdr_exists(rankwise_argmax_sets(cost), 4)
    assert not ok
    assert rank > full + 1e-8
