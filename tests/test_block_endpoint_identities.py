"""Endpoint identities: singleton = rankwise; full block = exact coupled box."""
from __future__ import annotations

import numpy as np

from private_mobility_cert.certificates.block_coupled_certificates import (
    certificate_block_coupled_box,
    certificate_full_coupled_box,
    certificate_rankwise_box,
    full_partition,
    singleton_partition,
)
from private_mobility_cert.certificates.release_conditioned import certificate_l1_assignment


def test_singleton_is_rankwise_and_full_is_exact():
    rng = np.random.default_rng(99)
    for n, k in [(5, 2), (8, 4), (7, 3)]:
        y = rng.normal(size=n)
        y[-1] = -abs(y[-1])
        w = np.linspace(1.0, 0.3, k)
        t = 1.1
        rw = certificate_rankwise_box(y, k=k, weights=w, half_width=t).value
        sing = certificate_block_coupled_box(
            y, k=k, weights=w, half_width=t, partition=singleton_partition(k)
        ).value
        full = certificate_full_coupled_box(y, k=k, weights=w, half_width=t).value
        one = certificate_block_coupled_box(
            y, k=k, weights=w, half_width=t, partition=full_partition(k)
        ).value
        ref = certificate_l1_assignment(y, k=k, weights=w, half_width=t).value
        assert abs(rw - sing) <= 1e-10
        assert abs(full - ref) <= 1e-10
        assert abs(one - ref) <= 1e-9
