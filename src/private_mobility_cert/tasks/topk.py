"""Top-K high-demand station prioritization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TASK_NAME = "Top-K high-demand station prioritization"
PROHIBITED_TERMS = (
    "shortage detection",
    "shortage measurement",
    "stockout prediction",
    "unmet-demand detection",
)


class TaskInputError(ValueError):
    """Invalid Top-K task configuration or inputs."""


@dataclass(frozen=True)
class TopKResult:
    selected: tuple[str, ...]
    scores: dict[str, float]
    ranking_scores: dict[str, float]
    loss: float
    k: int


def _validate_finite(values: dict[str, float], *, name: str) -> None:
    for key, value in values.items():
        if not np.isfinite(value):
            raise TaskInputError(f"{name} contains non-finite value at {key}: {value}")


def build_ranking_scores(
    activity: dict[str, float],
    *,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """z_s = omega_s * q_s; same scores define ranking and loss."""
    _validate_finite(activity, name="activity")
    if weights is None:
        weights = {s: 1.0 for s in activity}
    if set(weights) != set(activity):
        raise TaskInputError("weights keys must match activity keys")
    _validate_finite(weights, name="weights")
    if any(w < 0 for w in weights.values()):
        raise TaskInputError("weights must be nonnegative")
    return {s: weights[s] * activity[s] for s in activity}


def topk_select(
    activity: dict[str, float],
    *,
    k: int,
    weights: dict[str, float] | None = None,
) -> TopKResult:
    """Select Top-K by z_s=omega_s q_s; loss = sum_{s not in S} z_s."""
    if k < 0:
        raise TaskInputError("k must be nonnegative")
    if not activity:
        raise TaskInputError("activity must be nonempty")
    ranking = build_ranking_scores(activity, weights=weights)
    ordered = sorted(ranking.keys(), key=lambda s: (-ranking[s], s))
    k_eff = min(k, len(ordered))
    selected = tuple(ordered[:k_eff])
    selected_set = set(selected)
    loss = float(sum(ranking[s] for s in ranking if s not in selected_set))
    return TopKResult(
        selected=selected,
        scores=dict(activity),
        ranking_scores=ranking,
        loss=loss,
        k=k_eff,
    )


def topk_loss(
    selected: set[str],
    activity: dict[str, float],
    *,
    weights: dict[str, float] | None = None,
) -> float:
    ranking = build_ranking_scores(activity, weights=weights)
    return float(sum(ranking[s] for s in ranking if s not in selected))


def normalized_topk_regret(
    selected: set[str],
    activity: dict[str, float],
    *,
    weights: dict[str, float] | None = None,
    eps_num: float = 1e-12,
) -> float:
    ranking = build_ranking_scores(activity, weights=weights)
    oracle = topk_select(activity, k=len(selected), weights=weights)
    loss = float(sum(ranking[s] for s in ranking if s not in selected))
    denom = max(float(sum(ranking.values())), eps_num)
    oracle_loss = oracle.loss
    regret = loss - oracle_loss
    if regret < -1e-8:
        raise TaskInputError(f"negative topk regret: {regret}")
    return float(max(regret, 0.0) / denom)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)
