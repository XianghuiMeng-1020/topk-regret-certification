"""Synthetic fixture tests for Top-K and capacity tasks (not scientific experiments)."""

from __future__ import annotations

import pytest

from private_mobility_cert.tasks.capacity import (
    CapacityInputError,
    allocate_capacity,
    allocation_regret,
    brute_force_squared_oracle,
    capacity_loss_squared,
)
from private_mobility_cert.tasks.topk import (
    PROHIBITED_TERMS,
    TASK_NAME,
    TaskInputError,
    jaccard,
    topk_loss,
    topk_select,
)


def test_topk_tie_break_by_station_id() -> None:
    scores = {"b": 5.0, "a": 5.0, "c": 1.0}
    result = topk_select(scores, k=1)
    assert result.selected == ("a",)


def test_topk_score_loss_consistency() -> None:
    activity = {"a": 10.0, "b": 9.0, "c": 1.0}
    weights = {"a": 1.0, "b": 2.0, "c": 1.0}
    result = topk_select(activity, k=1, weights=weights)
    # ranks by z=omega*q => b has 18, a has 10
    assert result.selected == ("b",)
    assert result.loss == pytest.approx(topk_loss(set(result.selected), activity, weights=weights))


def test_topk_rejects_nan() -> None:
    with pytest.raises(TaskInputError):
        topk_select({"a": float("nan")}, k=1)


def test_topk_no_shortage_terminology() -> None:
    assert "shortage" not in TASK_NAME.lower()
    joined = " | ".join(PROHIBITED_TERMS).lower()
    assert "shortage detection" in joined
    assert "stockout prediction" in joined
    assert "unmet-demand detection" in joined


def test_jaccard() -> None:
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


def test_reject_truth_dependent_budget() -> None:
    demand = {"a": 2.0, "b": 2.0}
    with pytest.raises(CapacityInputError):
        allocate_capacity(demand, budget=0.2 * sum(demand.values()))


def test_squared_capacity_budget_and_nonnegativity() -> None:
    demand = {"a": 5.0, "b": 3.0, "c": 1.0}
    # frozen absolute budget, not beta*sum(x)
    result = allocate_capacity(demand, budget=4.0)
    assert result.budget_used == pytest.approx(4.0, abs=1e-6)
    assert all(v >= -1e-12 for v in result.allocation.values())
    assert sum(result.allocation.values()) <= 4.0 + 1e-8


def test_inactive_budget_returns_demand_and_zero_dual() -> None:
    demand = {"a": 3.0, "b": 2.0, "c": 1.0}
    result = allocate_capacity(demand, budget=100.0)
    assert result.allocation == demand
    assert result.dual_lambda == pytest.approx(0.0, abs=1e-12)
    assert result.loss == pytest.approx(0.0, abs=1e-12)


def test_squared_oracle_matches_brute_force() -> None:
    demand = {"a": 3.0, "b": 2.0, "c": 1.0}
    weights = {"a": 1.0, "b": 1.0, "c": 1.0}
    greedy = allocate_capacity(demand, budget=3.0, weights=weights)
    brute = brute_force_squared_oracle(demand, budget=3.0, weights=weights, step=1.0)
    assert greedy.loss == pytest.approx(brute.loss, abs=1e-6)


def test_squared_regret_nonnegative_and_zero_when_same() -> None:
    demand = {"a": 2.0, "b": 2.0}
    oracle = allocate_capacity(demand, budget=2.0)
    assert allocation_regret(oracle.allocation, oracle.allocation, demand) == 0.0
    # private uses same feasible set / budget on a different release y
    y = {"a": 3.0, "b": 1.0}
    private = allocate_capacity(y, budget=2.0)
    regret = allocation_regret(private.allocation, oracle.allocation, demand)
    assert regret >= -1e-12


def test_permutation_invariance() -> None:
    demand = {"a": 4.0, "b": 1.0, "c": 3.0}
    res = allocate_capacity(demand, budget=4.0)
    perm = {"a": "x", "b": "y", "c": "z"}
    d2 = {perm[s]: demand[s] for s in demand}
    res2 = allocate_capacity(d2, budget=4.0)
    assert res.loss == pytest.approx(res2.loss, abs=1e-8)


def test_capacity_rejects_infinite_and_bad_weights() -> None:
    with pytest.raises(CapacityInputError):
        allocate_capacity({"a": float("inf")}, budget=1.0)
    with pytest.raises(CapacityInputError):
        allocate_capacity({"a": 1.0}, budget=-1.0)
    with pytest.raises(CapacityInputError):
        allocate_capacity({"a": 1.0, "b": 1.0}, budget=1.0, weights={"a": 0.0, "b": 1.0})


def test_squared_loss_deterministic() -> None:
    demand = {"a": 2.0, "b": 1.0}
    alloc = {"a": 1.0, "b": 0.0}
    assert capacity_loss_squared(alloc, demand) == pytest.approx(1.0 + 1.0)
