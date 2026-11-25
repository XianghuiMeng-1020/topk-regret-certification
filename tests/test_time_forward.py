"""Independent time-forward fixtures. The product test does not call tie_keep as its reference."""

import importlib.util
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "experiments/time_forward_utility/run.py"
SPEC = importlib.util.spec_from_file_location("time_forward_run", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_fixtures_pass():
    MODULE.fixtures()


def test_product_reference_disagrees_with_neither_side():
    hashes = np.arange(0, 5000, dtype=np.uint64)
    got = MODULE.tie_keep(hashes, 7, 3)
    expected = np.array([MODULE.product_keeps(int(value), 7, 3) for value in hashes])
    assert np.array_equal(got, expected)


def test_integer_half_is_exact():
    assert 4 * 12150 + 5 * 9720 == 97200
    assert MODULE.TARGETS["0.50"] == 97200
