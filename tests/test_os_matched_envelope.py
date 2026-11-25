"""Order-statistic checks call the independent fixture suite."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _suite():
    path = Path(__file__).resolve().parents[1] / "experiments/real_order_statistic_evaluation/independent_checks.py"
    spec = importlib.util.spec_from_file_location("independent_checks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_suite()


def test_os_matched_envelope_independent_suite():
    report = _suite()
    assert report["n_failures"] == 0, report["failures"][:10]
    assert report["n_fixtures"] > 0
    assert report["n_injections"] > 0
    assert report["n_k5_mismatches"] == 0
    assert report["max_support_above_linear_arm"] <= 1e-6
