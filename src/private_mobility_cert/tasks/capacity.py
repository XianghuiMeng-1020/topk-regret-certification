"""Convex budgeted service-capacity allocation (squared shortfall).

Feasible set A(B) = {a >= 0 : sum_s a_s <= B} must be independent of current
protected truth x. Optional fixed caps u_s allowed only if public/dev-frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


class CapacityInputError(ValueError):
    """Invalid capacity-allocation inputs."""


@dataclass(frozen=True)
class CapacityResult:
    allocation: dict[str, float]
    loss: float
    budget_used: float
    budget: float
    dual_lambda: float
    residual: float


def _validate_vector(values: dict[str, float], *, name: str, nonnegative: bool = False) -> None:
    if not values:
        raise CapacityInputError(f"{name} must be nonempty")
    for key, value in values.items():
        if not np.isfinite(value):
            raise CapacityInputError(f"{name} non-finite at {key}: {value}")
        if nonnegative and value < 0:
            raise CapacityInputError(f"{name} negative at {key}: {value}")


def assert_truth_independent_caps(
    upper_bounds: dict[str, float] | None,
    demand: dict[str, float],
) -> None:
    """Fail closed if caps equal current protected demand (a truth-dependent construction)."""
    if upper_bounds is None:
        return
    if set(upper_bounds) != set(demand):
        raise CapacityInputError("upper_bounds keys must match demand")
    if all(abs(upper_bounds[s] - demand[s]) <= 1e-12 for s in demand):
        raise CapacityInputError(
            "truth-dependent caps rejected: u_s must not equal current protected x_s"
        )


def assert_frozen_budget_not_from_current_total(
    budget: float,
    demand: dict[str, float],
    *,
    betas: tuple[float, ...] = (0.1, 0.2, 0.3),
) -> None:
    """Fail closed if budget matches beta * sum(current x) for common betas."""
    total = float(sum(demand.values()))
    for beta in betas:
        if abs(budget - beta * total) <= 1e-9 * max(1.0, abs(total)):
            raise CapacityInputError(
                "truth-dependent budget rejected: B must be frozen from "
                "development/public information, not beta * sum_s x_s"
            )


def capacity_loss_squared(
    allocation: dict[str, float],
    demand: dict[str, float],
    *,
    weights: dict[str, float] | None = None,
) -> float:
    """L(a;x) = sum_s omega_s [x_s - a_s]_+^2."""
    _validate_vector(allocation, name="allocation", nonnegative=True)
    _validate_vector(demand, name="demand", nonnegative=True)
    if weights is None:
        weights = {s: 1.0 for s in demand}
    _validate_vector(weights, name="weights", nonnegative=True)
    if set(allocation) != set(demand) or set(weights) != set(demand):
        raise CapacityInputError("allocation/demand/weights keys must match")
    if any(w <= 0 for w in weights.values()):
        raise CapacityInputError("weights must be strictly positive")
    return float(
        sum(weights[s] * max(demand[s] - allocation[s], 0.0) ** 2 for s in demand)
    )


def _clip_caps(
    raw: dict[str, float],
    upper_bounds: dict[str, float] | None,
) -> dict[str, float]:
    if upper_bounds is None:
        return raw
    return {s: min(raw[s], upper_bounds[s]) for s in raw}


def waterfilling_allocate(
    demand: dict[str, float],
    *,
    budget: float,
    weights: dict[str, float] | None = None,
    upper_bounds: dict[str, float] | None = None,
    reject_truth_dependent: bool = True,
    tol: float = 1e-10,
    max_iter: int = 80,
) -> CapacityResult:
    """Specialized solver for squared shortfall under sum a <= B, a >= 0.

    KKT (uncapped active stations): a_s = [x_s - lambda / (2 omega_s)]_+
    where lambda >= 0 is the budget dual. Equal weights: a_s = [x_s - nu]_+
    with nu = lambda / 2.
    """
    _validate_vector(demand, name="demand", nonnegative=True)
    if not np.isfinite(budget) or budget < 0:
        raise CapacityInputError("budget must be finite and nonnegative")
    if weights is None:
        weights = {s: 1.0 for s in demand}
    _validate_vector(weights, name="weights", nonnegative=True)
    if any(w <= 0 for w in weights.values()):
        raise CapacityInputError("weights must be strictly positive")
    if set(weights) != set(demand):
        raise CapacityInputError("weights keys must match demand")
    if upper_bounds is not None:
        _validate_vector(upper_bounds, name="upper_bounds", nonnegative=True)
        if set(upper_bounds) != set(demand):
            raise CapacityInputError("upper_bounds keys must match demand")
    if reject_truth_dependent:
        assert_truth_independent_caps(upper_bounds, demand)
        assert_frozen_budget_not_from_current_total(budget, demand)

    stations = sorted(demand.keys())
    # Effective individual ceiling: no benefit to a_s > x_s for this loss;
    # optional public/dev caps may further reduce.
    eff_cap = {s: demand[s] for s in stations}
    if upper_bounds is not None:
        eff_cap = {s: min(eff_cap[s], upper_bounds[s]) for s in stations}

    total_cap = float(sum(eff_cap.values()))
    if budget >= total_cap - tol:
        allocation = dict(eff_cap)
        loss = capacity_loss_squared(allocation, demand, weights=weights)
        return CapacityResult(
            allocation=allocation,
            loss=loss,
            budget_used=float(sum(allocation.values())),
            budget=float(budget),
            dual_lambda=0.0,
            residual=0.0,
        )

    # Binary search on nu = lambda / 2 so a_s = [x_s - nu / omega_s]_+
    # since lambda/(2 omega) = (lambda/2)/omega = nu/omega.
    def alloc_for_nu(nu: float) -> dict[str, float]:
        raw = {s: max(demand[s] - nu / weights[s], 0.0) for s in stations}
        return _clip_caps(raw, eff_cap)

    lo = 0.0
    hi = max(demand[s] * weights[s] for s in stations) + 1.0
    # Expand hi until sum(alloc) <= budget
    for _ in range(60):
        if sum(alloc_for_nu(hi).values()) <= budget + tol:
            break
        hi *= 2.0
    else:
        raise CapacityInputError("waterfilling dual search failed to bracket")

    nu = hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        used = float(sum(alloc_for_nu(mid).values()))
        if used > budget + tol:
            lo = mid
        else:
            hi = mid
            nu = mid
    allocation = alloc_for_nu(nu)
    used = float(sum(allocation.values()))
    # Scale tiny numerical overshoot
    if used > budget + 1e-8:
        raise CapacityInputError(f"budget infeasible after waterfilling: used={used} B={budget}")
    dual_lambda = 2.0 * nu  # so a_s = [x_s - lambda/(2 omega_s)]_+
    # KKT residual: for stations with a_s in (0, eff_cap), |2 omega (x-a) - lambda|
    residual = 0.0
    for s in stations:
        a = allocation[s]
        if a <= tol or a >= eff_cap[s] - tol:
            continue
        grad = 2.0 * weights[s] * (demand[s] - a)
        residual = max(residual, abs(grad - dual_lambda))
    if residual > 1e-4:
        raise CapacityInputError(f"KKT residual too large: {residual}")

    loss = capacity_loss_squared(allocation, demand, weights=weights)
    return CapacityResult(
        allocation=allocation,
        loss=loss,
        budget_used=used,
        budget=float(budget),
        dual_lambda=float(dual_lambda),
        residual=float(residual),
    )


def allocate_capacity(
    demand: dict[str, float],
    *,
    budget: float,
    weights: dict[str, float] | None = None,
    upper_bounds: dict[str, float] | None = None,
    reject_truth_dependent: bool = True,
) -> CapacityResult:
    """Public entry point: deterministic specialized squared-shortfall solver."""
    return waterfilling_allocate(
        demand,
        budget=budget,
        weights=weights,
        upper_bounds=upper_bounds,
        reject_truth_dependent=reject_truth_dependent,
    )


def brute_force_squared_oracle(
    demand: dict[str, float],
    *,
    budget: float,
    weights: dict[str, float],
    step: float = 1.0,
    upper_bounds: dict[str, float] | None = None,
) -> CapacityResult:
    """Enumerate tiny grids for reference agreement (no truth-dependent rejection)."""
    stations = sorted(demand.keys())
    if len(stations) > 4:
        raise CapacityInputError("brute_force_squared_oracle only for tiny instances")
    caps = {s: demand[s] for s in stations}
    if upper_bounds is not None:
        caps = {s: min(caps[s], upper_bounds[s]) for s in stations}
    grids: list[list[float]] = []
    for station in stations:
        max_a = min(caps[station], budget)
        grids.append(list(np.arange(0.0, max_a + 1e-12, step)))

    best_loss = float("inf")
    best_alloc = {s: 0.0 for s in stations}
    for combo in product(*grids):
        if sum(combo) - budget > 1e-9:
            continue
        alloc = {station: float(val) for station, val in zip(stations, combo, strict=True)}
        loss = capacity_loss_squared(alloc, demand, weights=weights)
        if loss < best_loss - 1e-12 or (
            abs(loss - best_loss) <= 1e-12
            and tuple(alloc[s] for s in stations) < tuple(best_alloc[s] for s in stations)
        ):
            best_loss = loss
            best_alloc = alloc
    return CapacityResult(
        allocation=best_alloc,
        loss=float(best_loss),
        budget_used=float(sum(best_alloc.values())),
        budget=float(budget),
        dual_lambda=float("nan"),
        residual=float("nan"),
    )


def allocation_regret(
    private_action: dict[str, float],
    oracle_action: dict[str, float],
    demand: dict[str, float],
    *,
    weights: dict[str, float] | None = None,
) -> float:
    private_loss = capacity_loss_squared(private_action, demand, weights=weights)
    oracle_loss = capacity_loss_squared(oracle_action, demand, weights=weights)
    regret = private_loss - oracle_loss
    if regret < -1e-8:
        raise CapacityInputError(f"negative regret beyond tolerance: {regret}")
    return float(max(regret, 0.0))


def permute_stations(
    demand: dict[str, float],
    allocation: dict[str, float],
    perm: dict[str, str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Apply station-id permutation for invariance tests."""
    d2 = {perm[s]: demand[s] for s in demand}
    a2 = {perm[s]: allocation[s] for s in allocation}
    return d2, a2
