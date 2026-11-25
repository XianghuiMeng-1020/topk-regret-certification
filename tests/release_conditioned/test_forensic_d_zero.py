"""D_{w,k} under the signed-release and clipped-release definitions."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from private_mobility_cert.certificates.release_conditioned import (
    clip_nonnegative,
    deploy_ranking,
    observable_margin,
    stable_ranking,
    weight_family,
)
from private_mobility_cert.certificates.weighted_ranking import observed_deficit


def _d_plus(y: np.ndarray, k: int, w: np.ndarray) -> float:
    yp = clip_nonnegative(y)
    pi_hat = stable_ranking(yp, k)
    top = stable_ranking(yp, k)
    return float(np.dot(w[:k], yp[top] - yp[pi_hat]))


def _d_plus_enum(y: np.ndarray, k: int, w: np.ndarray) -> float:
    yp = clip_nonnegative(y)
    pi_hat = stable_ranking(yp, k)
    best = -np.inf
    for pi in itertools.permutations(range(y.size), k):
        best = max(best, observable_margin(yp, np.asarray(pi), pi_hat, w[:k]))
    return float(max(best, 0.0))


@pytest.mark.parametrize("wname", ["uniform", "dcg", "reciprocal", "geometric"])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_d_plus_identically_zero_all_positive(wname: str, k: int) -> None:
    rng = np.random.default_rng(0)
    w = weight_family(wname, k)
    for _ in range(30):
        y = rng.uniform(0.0, 20.0, size=6)
        assert _d_plus(y, k, w) == pytest.approx(0.0, abs=1e-12)
        assert _d_plus_enum(y, k, w) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("wname", ["uniform", "dcg"])
def test_d_plus_zero_with_clipping_and_ties(wname: str) -> None:
    w = weight_family(wname, 3)
    cases = [
        np.array([-3.0, -2.0, -1.0, 0.0, 5.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0, 0.0, 0.0]),
        np.array([-5.0, 2.0, 2.0, 2.0, -1.0]),
        np.array([10.0, -100.0, -100.0, -100.0, -100.0]),
    ]
    for y in cases:
        assert _d_plus(y, 3, w) == pytest.approx(0.0, abs=1e-12)
        assert _d_plus_enum(y, 3, w) == pytest.approx(0.0, abs=1e-12)


def test_implementation_signed_deficit_can_be_nonzero_under_clipping() -> None:
    """Code's observed_deficit(y_signed, pi_from_yplus) can be >0; the clipped Y+ D cannot."""
    y = np.array([-1.0, -0.5, -0.1, 10.0])
    w = weight_family("dcg", 2)
    pi = deploy_ranking(y, 2)
    d_signed = observed_deficit(y, pi, w)
    assert d_signed > 0.0
    assert _d_plus(y, 2, w) == pytest.approx(0.0, abs=1e-12)


def test_stable_tie_break_smaller_index() -> None:
    y = np.array([1.0, 1.0, 1.0])
    assert stable_ranking(y, 2).tolist() == [0, 1]
    assert deploy_ranking(np.array([-1.0, -1.0, -1.0]), 2).tolist() == [0, 1]
