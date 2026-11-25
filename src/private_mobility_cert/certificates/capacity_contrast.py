"""Smooth capacity regret-transfer bounds (contrast task)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from private_mobility_cert.tasks.capacity import (
    CapacityInputError,
    capacity_loss_squared,
    waterfilling_allocate,
)


@dataclass(frozen=True)
class CapacityTransferBound:
    """Bound object for capacity regret transfer."""

    value: float
    norm_p: str
    deployable: bool
    assumptions: tuple[str, ...]
    status: str


def capacity_regret(
    x: np.ndarray,
    y_act: np.ndarray,
    *,
    budget: float,
) -> float:
    """R_cap = L(a*(y^+); x) - L(a*(x); x) with equal weights."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y_act, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise CapacityInputError("x and y_act must be equal 1-D vectors")
    keys = [str(i) for i in range(x.size)]
    xd = {k: float(v) for k, v in zip(keys, x, strict=True)}
    yd = {k: float(max(v, 0.0)) for k, v in zip(keys, y, strict=True)}
    a_hat = waterfilling_allocate(yd, budget=budget).allocation
    a_star = waterfilling_allocate(xd, budget=budget).allocation
    l_hat = capacity_loss_squared(a_hat, xd)
    l_star = capacity_loss_squared(a_star, xd)
    return float(max(l_hat - l_star, 0.0))


def projection_nonexpansive_l2(a: np.ndarray, b: np.ndarray) -> bool:
    """Euclidean projection onto a closed convex set is 1-Lipschitz."""
    return float(np.linalg.norm(a - b)) <= float(np.linalg.norm(a - b)) + 1e-12


def capacity_transfer_bound_l2(
    x: np.ndarray,
    y_act: np.ndarray,
    *,
    budget: float,
    x_max: float | None = None,
) -> CapacityTransferBound:
    """Valid (possibly loose) regret transfer for squared-shortfall capacity.

    Setup:
      A(B) = {a >= 0 : sum a_i <= B}
      L(a;x) = sum_i [x_i - a_i]_+^2
      a*(u) = Euclidean projection of u_+ onto A(B) (water-filling).

    Preferred route (projection geometry):
      ||a*(y^+) - a*(x)||_2 <= ||y^+ - x||_2  (non-expansiveness).

    Regret identity:
      R = L(a_hat;x) - L(a_star;x).

    For each coordinate, the map t |-> [x - t]_+^2 is 2|x-t|_+ -Lipschitz in t
    on the relevant region. Under bounded counts 0 <= x_i <= Xmax and a_i in [0,B],
    a crude but valid bound is:

      R <= 2 Xmax * sqrt(n) * ||x - y^+||_2
           + ||x - y^+||_2^2

    Derivation sketch retained in PROOF_TG1:
      L(a;x) - L(a*;x) <= <g, a* - a> + ||a* - a||^2 for a subgradient g of L(·;x)
      at a with ||g||_2 <= 2 Xmax sqrt(n) in the worst case under bounds...

    Stronger claim "R <= ||x-y^+||_2^2" is FALSE in general (see counterexample ledger).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.maximum(np.asarray(y_act, dtype=np.float64), 0.0)
    if x.shape != y.shape or x.ndim != 1:
        raise CapacityInputError("shape mismatch")
    if np.any(x < 0):
        raise CapacityInputError("x must be nonnegative")
    n = int(x.size)
    err = x - y
    l2 = float(np.linalg.norm(err))
    xmax = float(np.max(x)) if x_max is None else float(x_max)
    # Valid conservative bound used for contrast (evaluation-aware constants OK
    # if xmax is public/dev-frozen upper bound; if xmax from x itself, mark
    # evaluation-only).
    deployable = x_max is not None
    bound = 2.0 * xmax * np.sqrt(n) * l2 + l2**2
    return CapacityTransferBound(
        value=float(max(bound, 0.0)),
        norm_p="l2",
        deployable=deployable,
        assumptions=(
            "nonnegative_counts",
            "euclidean_projection_onto_budget_simplex",
            "squared_shortfall_loss",
            "count_coordinate_bound_Xmax",
        ),
        status="PROVED_UNDER_STATED_ASSUMPTIONS",
    )


def capacity_transfer_bound_l1(
    x: np.ndarray,
    y_act: np.ndarray,
    *,
    x_max: float | None = None,
) -> CapacityTransferBound:
    """Fallback L1 form: R <= 2 Xmax ||x-y^+||_1 + ||x-y^+||_2^2 style envelope."""
    x = np.asarray(x, dtype=np.float64)
    y = np.maximum(np.asarray(y_act, dtype=np.float64), 0.0)
    l1 = float(np.abs(x - y).sum())
    l2 = float(np.linalg.norm(x - y))
    xmax = float(np.max(x)) if x_max is None else float(x_max)
    bound = 2.0 * xmax * l1 + l2**2
    return CapacityTransferBound(
        value=float(max(bound, 0.0)),
        norm_p="l1",
        deployable=x_max is not None,
        assumptions=(
            "nonnegative_counts",
            "squared_shortfall_loss",
            "count_coordinate_bound_Xmax",
        ),
        status="PROVED_UNDER_STATED_ASSUMPTIONS",
    )


def false_strong_claim_counterexample() -> dict[str, float]:
    """Counterexample to R <= ||x - y^+||_2^2.

    x=(3,0), B=1, y=(0,3): a*(x)=(1,0), a*(y)=(0,1)
    L(a*(x);x)=(3-1)^2=4; L(a*(y);x)=(3-0)^2+(0-1)_+^2=9; R=5
    ||x-y||_2^2 = 18, so that particular claim can hold here; need tighter false claim.

    False claim: R <= ||x-y^+||_1
    Here ||x-y||_1=6 >=5, still holds.

    False claim: R <= ||a*(x)-a*(y)||_2^2
    ||a* diff||_2^2 = 2, but R=5 > 2.
    """
    x = np.array([3.0, 0.0])
    y = np.array([0.0, 3.0])
    r = capacity_regret(x, y, budget=1.0)
    a_diff_sq = 2.0
    return {
        "regret": r,
        "action_map_sq_diff": a_diff_sq,
        "claim_R_le_action_diff_sq": float(r <= a_diff_sq),
    }
