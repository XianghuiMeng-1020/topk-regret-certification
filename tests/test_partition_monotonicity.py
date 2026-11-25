"""Merging blocks never increases the block-coupled certificate."""
from __future__ import annotations

import numpy as np
import pytest

from private_mobility_cert.certificates.block_coupled_certificates import (
    BlockPartition,
    certificate_block_coupled_box,
    consecutive_blocks,
    full_partition,
    singleton_partition,
)


@pytest.mark.parametrize("n,k", [(7, 4), (8, 5), (6, 3)])
def test_merge_monotone(n, k):
    rng = np.random.default_rng(123 + k)
    y = rng.normal(size=n)
    w = 1.0 / np.arange(1, k + 1)
    t = 1.7
    fine = consecutive_blocks(k, 1)
    mid = consecutive_blocks(k, 2)
    coarse = full_partition(k)
    v_fine = certificate_block_coupled_box(
        y, k=k, weights=w, half_width=t, partition=fine
    ).value
    v_mid = certificate_block_coupled_box(
        y, k=k, weights=w, half_width=t, partition=mid
    ).value
    v_coarse = certificate_block_coupled_box(
        y, k=k, weights=w, half_width=t, partition=coarse
    ).value
    assert v_fine + 1e-9 >= v_mid
    assert v_mid + 1e-9 >= v_coarse


def test_custom_merge():
    y = np.array([3.0, 2.5, 2.0, 1.0, -0.5, 0.2], dtype=float)
    w = np.array([1.0, 0.7, 0.4])
    t = 0.8
    p1 = BlockPartition(((0,), (1,), (2,)))
    p2 = BlockPartition(((0, 1), (2,)))
    p3 = BlockPartition(((0, 1, 2),))
    v1 = certificate_block_coupled_box(y, k=3, weights=w, half_width=t, partition=p1).value
    v2 = certificate_block_coupled_box(y, k=3, weights=w, half_width=t, partition=p2).value
    v3 = certificate_block_coupled_box(y, k=3, weights=w, half_width=t, partition=p3).value
    assert v1 + 1e-9 >= v2 >= v3 - 1e-9
    assert singleton_partition(3).blocks == p1.blocks
