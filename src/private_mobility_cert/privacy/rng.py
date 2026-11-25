"""Project-controlled RNG factories (no implicit global NumPy state)."""

from __future__ import annotations

import hashlib

import numpy as np


def spawn_generator(seed: int | np.random.SeedSequence) -> np.random.Generator:
    """Return a NumPy Generator bound to an explicit seed / SeedSequence."""
    if isinstance(seed, np.random.SeedSequence):
        return np.random.default_rng(seed)
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be int or SeedSequence")
    if int(seed) < 0:
        raise ValueError("seed must be nonnegative")
    return np.random.default_rng(int(seed))


def derive_seed_int(
    master_seed: int,
    *,
    city: str,
    split: str,
    mechanism: str,
    epsilon: float,
    replicate_id: int,
) -> int:
    """Deterministic 63-bit seed derivation; outcome-independent."""
    if replicate_id < 0:
        raise ValueError("replicate_id must be nonnegative")
    payload = (
        f"{master_seed}|{city}|{split}|{mechanism}|{epsilon:.12g}|{replicate_id}"
    ).encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
