"""Privacy ledger helpers and composition rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ReleaseLedgerEntry:
    release_id: str
    city: str
    split: str
    data_manifest_hash: str
    station_set_hash: str
    aggregation_config_hash: str
    mechanism: str
    adjacency: str
    sensitivity: float
    epsilon: float
    delta: float
    composition_rule: str
    per_level_epsilon: list[float] | None
    post_processing_chain: list[str]
    rng_algorithm: str
    seed_id: str
    code_commit: str
    output_checksum: str
    hypothetical: bool
    jointly_used_with: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sequential_compose(epsilons: list[float], *, delta: float = 0.0) -> tuple[float, float]:
    if any(e < 0 for e in epsilons):
        raise ValueError("epsilons must be nonnegative")
    if delta != 0.0:
        raise ValueError("primary accounting is pure DP (delta=0)")
    return float(sum(epsilons)), 0.0


def parallel_compose_disjoint(epsilons: list[float], *, delta: float = 0.0) -> tuple[float, float]:
    """Parallel composition for disjoint partitions of records."""
    if not epsilons:
        raise ValueError("epsilons nonempty")
    if delta != 0.0:
        raise ValueError("primary accounting is pure DP (delta=0)")
    return float(max(epsilons)), 0.0


def monte_carlo_interpretation() -> str:
    return (
        "Experimental replicates are alternative hypothetical realizations of a "
        "single release mechanism. They are not jointly published. Joint publication "
        "of R independent releases would require sequential composition of their budgets."
    )
