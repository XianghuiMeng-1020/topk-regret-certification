"""Shared pytest fixtures and deterministic seed."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _deterministic_seed() -> None:
    import random

    import numpy as np

    seed = 20260801
    random.seed(seed)
    np.random.seed(seed)
