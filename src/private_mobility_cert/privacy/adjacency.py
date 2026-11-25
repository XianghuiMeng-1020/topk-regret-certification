"""Record-level add/remove adjacency and sensitivity helpers."""

from __future__ import annotations

import numpy as np


def one_hot_contribution(dimension: int, index: int) -> np.ndarray:
    """Unit contribution of one eligible trip to a flat histogram cell."""
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if not 0 <= index < dimension:
        raise ValueError("index out of range")
    vec = np.zeros(dimension, dtype=np.int64)
    vec[index] = 1
    return vec


def l1_sensitivity_flat_histogram() -> int:
    """Add/remove adjacency changes exactly one cell by ±1 ⇒ Δ₁ = 1."""
    return 1


def neighboring_histograms(
    base: np.ndarray, index: int, *, direction: str = "add"
) -> tuple[np.ndarray, np.ndarray]:
    """Return (D, D') under add/remove of one record at `index`."""
    x = np.asarray(base, dtype=np.int64).copy()
    if x.ndim != 1:
        raise ValueError("base must be 1-D")
    if not 0 <= index < x.size:
        raise ValueError("index out of range")
    xp = x.copy()
    if direction == "add":
        xp[index] += 1
        return x, xp
    if direction == "remove":
        if x[index] < 1:
            raise ValueError("cannot remove from empty cell")
        xp[index] -= 1
        return x, xp
    raise ValueError("direction must be 'add' or 'remove'")


def assert_add_remove_not_replace_one() -> str:
    """Document adjacency semantics for audits/tests."""
    return (
        "Primary adjacency is add/remove of one eligible trip. "
        "An ε-DP guarantee under add/remove does not automatically equal "
        "the same ε under replace-one adjacency."
    )
