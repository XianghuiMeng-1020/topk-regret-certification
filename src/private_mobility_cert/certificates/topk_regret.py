"""Top-K regret identities (signed mechanism output versus clipped action)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class TopKCertError(ValueError):
    """Invalid Top-K certificate inputs."""


@dataclass(frozen=True)
class TopKSelection:
    """Deterministic Top-K selection from nonnegative action scores."""

    selected_idx: np.ndarray  # shape (k_eff,), ascending station-id tie-break already applied
    k: int
    n: int


def validate_vector(x: np.ndarray, *, name: str, nonnegative: bool = False) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise TopKCertError(f"{name} must be 1-D")
    if arr.size == 0:
        raise TopKCertError(f"{name} must be nonempty")
    if not np.all(np.isfinite(arr)):
        raise TopKCertError(f"{name} must be finite")
    if nonnegative and np.any(arr < 0):
        raise TopKCertError(f"{name} must be nonnegative")
    return arr


def topk_select_indices(scores: np.ndarray, k: int) -> TopKSelection:
    """Select Top-K by score; ties broken by ascending index (frozen station IDs)."""
    s = validate_vector(scores, name="scores")
    if k < 0:
        raise TopKCertError("k must be nonnegative")
    n = int(s.size)
    k_eff = min(int(k), n)
    if k_eff == 0:
        return TopKSelection(selected_idx=np.zeros(0, dtype=np.int64), k=0, n=n)
    # Maximize score; among equals prefer smaller index.
    order = np.lexsort((np.arange(n), -s))
    selected = np.sort(order[:k_eff].astype(np.int64))
    return TopKSelection(selected_idx=selected, k=k_eff, n=n)


def value_of_set(x: np.ndarray, selected: np.ndarray) -> float:
    xx = validate_vector(x, name="x", nonnegative=True)
    idx = np.asarray(selected, dtype=np.int64)
    if idx.size and (np.any(idx < 0) or np.any(idx >= xx.size)):
        raise TopKCertError("selected indices out of range")
    return float(xx[idx].sum()) if idx.size else 0.0


def oracle_topk_value(x: np.ndarray, k: int) -> float:
    xx = validate_vector(x, name="x", nonnegative=True)
    sel = topk_select_indices(xx, k)
    return value_of_set(xx, sel.selected_idx)


def topk_regret(selected: np.ndarray, x: np.ndarray, k: int | None = None) -> float:
    """R_K(S;x) = V_K(S*;x) - V_K(S;x) with V = sum of true counts on the set."""
    xx = validate_vector(x, name="x", nonnegative=True)
    idx = np.asarray(selected, dtype=np.int64)
    kk = int(k) if k is not None else int(idx.size)
    if kk < 0:
        raise TopKCertError("k must be nonnegative")
    if idx.size != min(kk, xx.size) and kk <= xx.size:
        # Allow an undersized selection only when k > n was requested.
        if idx.size != min(kk, xx.size):
            raise TopKCertError("selected size must equal min(k, n)")
    v_hat = value_of_set(xx, idx)
    v_star = oracle_topk_value(xx, kk)
    regret = v_star - v_hat
    if regret < -1e-9:
        raise TopKCertError(f"negative regret due to inconsistent selection: {regret}")
    return float(max(regret, 0.0))


def missed_mass_loss(selected: np.ndarray, x: np.ndarray) -> float:
    """Equivalent missed-mass loss: sum_{i not in S} x_i - sum of (n-k) smallest? No.

    For fixed |S|=k, maximizing V_K(S;x) is equivalent to minimizing the missed mass
    sum_{i not in S} x_i, and R_K equals the excess missed mass over the oracle.
    """
    xx = validate_vector(x, name="x", nonnegative=True)
    idx = np.asarray(selected, dtype=np.int64)
    total = float(xx.sum())
    kept = value_of_set(xx, idx)
    return total - kept


def regret_from_missed_mass(selected: np.ndarray, x: np.ndarray, k: int) -> float:
    """R_K(S;x) = L_miss(S;x) - L_miss(S*;x)."""
    xx = validate_vector(x, name="x", nonnegative=True)
    l_hat = missed_mass_loss(selected, xx)
    oracle = topk_select_indices(xx, k)
    l_star = missed_mass_loss(oracle.selected_idx, xx)
    return float(max(l_hat - l_star, 0.0))


def subset_swap_regret_identity(selected: np.ndarray, x: np.ndarray, k: int) -> float:
    """Exact max multi-swap representation of Top-K regret for fixed S."""
    xx = validate_vector(x, name="x", nonnegative=True)
    s_idx = np.asarray(selected, dtype=np.int64)
    n = xx.size
    k_eff = min(int(k), n)
    if s_idx.size != k_eff:
        raise TopKCertError("selected size must equal min(k,n)")
    if k_eff == 0:
        return 0.0
    s_mask = np.zeros(n, dtype=bool)
    s_mask[s_idx] = True
    inside = xx[s_mask]
    outside = xx[~s_mask]
    # m=0 term is 0; for m>=1 use order statistics
    best = 0.0
    max_m = min(k_eff, int(outside.size))
    if max_m == 0:
        return 0.0
    inside_sorted = np.sort(inside)  # ascending
    outside_sorted = np.sort(outside)[::-1]  # descending
    for m in range(1, max_m + 1):
        gain = float(outside_sorted[:m].sum() - inside_sorted[:m].sum())
        best = max(best, gain)
    return float(max(best, 0.0))


def observed_noise_decomposition(
    y: np.ndarray,
    z: np.ndarray,
    a_idx: np.ndarray,
    b_idx: np.ndarray,
) -> tuple[float, float, float]:
    """Return (observed_contrast, noise_contrast, true_contrast) for swap (A,B).

    true = sum_B x - sum_A x = (sum_B y - sum_A y) + (sum_A z - sum_B z)
    with x = y - z.
    """
    yy = validate_vector(y, name="y")
    zz = validate_vector(z, name="z")
    if yy.shape != zz.shape:
        raise TopKCertError("y and z shape mismatch")
    a = np.asarray(a_idx, dtype=np.int64)
    b = np.asarray(b_idx, dtype=np.int64)
    if a.size != b.size:
        raise TopKCertError("|A| must equal |B|")
    observed = float(yy[b].sum() - yy[a].sum()) if a.size else 0.0
    noise = float(zz[a].sum() - zz[b].sum()) if a.size else 0.0
    true = observed + noise
    return observed, noise, true
