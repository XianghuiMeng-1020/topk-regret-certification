"""Immutable seed registry derivation (no outcome-based selection)."""

from __future__ import annotations

from dataclasses import dataclass

from private_mobility_cert.privacy.rng import derive_seed_int


@dataclass(frozen=True)
class SeedKey:
    city: str
    split: str
    mechanism: str
    epsilon: float
    replicate_id: int


def derive_release_seed(master_seed: int, key: SeedKey) -> int:
    return derive_seed_int(
        master_seed,
        city=key.city,
        split=key.split,
        mechanism=key.mechanism,
        epsilon=key.epsilon,
        replicate_id=key.replicate_id,
    )


def confirmatory_replicate_ids(n: int = 100) -> list[int]:
    if n <= 0:
        raise ValueError("n must be positive")
    return list(range(n))


def smoke_replicate_ids() -> list[int]:
    return [0, 1, 2]


def development_diagnostic_ids() -> list[int]:
    return [9000, 9001]
