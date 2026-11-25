"""T3: equality iff compatible block optima exist."""
from __future__ import annotations

import numpy as np

from private_mobility_cert.certificates.block_coupled_certificates import (
    box_cost_matrix,
    certificate_block_coupled_box,
    certificate_full_coupled_box,
    pair_partition,
    singleton_partition,
)
from private_mobility_cert.certificates.coupling_diagnostics import (
    exists_compatible_block_optima,
)


def test_compatible_equals_numeric():
    rng = np.random.default_rng(11)
    for k in [2, 3, 4]:
        for _ in range(40):
            n = k + int(rng.integers(1, 5))
            y = rng.normal(size=n)
            w = np.linspace(1.0, 0.4, k)
            t = 1.0
            cost, _, _, _ = box_cost_matrix(y, k=k, weights=w, half_width=t)
            for part in [singleton_partition(k), pair_partition(k)]:
                compat = exists_compatible_block_optima(cost, part)
                full = certificate_full_coupled_box(
                    y, k=k, weights=w, half_width=t
                ).value
                rel = certificate_block_coupled_box(
                    y, k=k, weights=w, half_width=t, partition=part
                ).value
                eq = abs(full - rel) <= 1e-8
                if compat:
                    assert eq
                if not eq:
                    assert not compat
