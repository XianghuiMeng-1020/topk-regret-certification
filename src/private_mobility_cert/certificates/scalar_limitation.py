"""Scalar-fidelity information limitation constructions for Top-K (TG2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from private_mobility_cert.certificates.topk_regret import topk_regret, topk_select_indices


@dataclass(frozen=True)
class ScalarFidelityInstance:
    name: str
    x: np.ndarray
    y: np.ndarray
    k: int
    fidelity: float
    regret: float


def normalized_l1_fidelity(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y_act = np.maximum(np.asarray(y, dtype=np.float64), 0.0)
    return float(np.abs(x - y_act).sum() / max(float(x.sum()), 1.0))


def equal_fidelity_discordant_regret_pair(
    k: int = 1,
) -> tuple[ScalarFidelityInstance, ScalarFidelityInstance]:
    """Two releases with equal scalar fidelity but different Top-K regret.

    Construction (n=4, K=1):
      x  = (5, 4, 0, 0)
      y1 = (5, 4, 1, 1)  -> F = 2/9, Top-1 = {0}, R = 0
      y2 = (4, 5, 0, 0)  -> F = 2/9, Top-1 = {1}, R = 1
    """
    x = np.array([5.0, 4.0, 0.0, 0.0])
    y1 = np.array([5.0, 4.0, 1.0, 1.0])
    y2 = np.array([4.0, 5.0, 0.0, 0.0])
    f1 = normalized_l1_fidelity(x, y1)
    f2 = normalized_l1_fidelity(x, y2)
    s1 = topk_select_indices(np.maximum(y1, 0.0), k).selected_idx
    s2 = topk_select_indices(np.maximum(y2, 0.0), k).selected_idx
    r1 = topk_regret(s1, x, k)
    r2 = topk_regret(s2, x, k)
    inst1 = ScalarFidelityInstance("equal_F_zero_regret", x, y1, k, f1, r1)
    inst2 = ScalarFidelityInstance("equal_F_positive_regret", x, y2, k, f2, r2)
    if abs(f1 - f2) > 1e-12:
        raise RuntimeError(f"construction failed equal fidelity: {f1} vs {f2}")
    if r1 != 0.0 or r2 <= 0.0:
        raise RuntimeError(f"construction failed regret contrast: {r1}, {r2}")
    return inst1, inst2


def scalar_only_worst_case_envelope(
    fidelity: float,
    *,
    total_mass: float,
    k: int,
    n: int,
) -> float:
    """Worst-case Top-K regret compatible with normalized L1 fidelity F.

    Information: only F, n, K, and total mass proxy M=max(||x||_1,1).
    Absolute L1 budget E = F * M.

    A safe general envelope is R <= E = F*M (one-sided mass removal from the
    oracle Top-K can cost 1 in L1 per unit regret). The tighter heuristic E/2
    assumes pure boundary transfers and is FALSE in general (see proof audit).
    """
    del k, n  # public parameters retained for API symmetry
    if fidelity < 0 or total_mass < 0:
        raise ValueError("invalid envelope parameters")
    return float(fidelity * max(total_mass, 1.0))
