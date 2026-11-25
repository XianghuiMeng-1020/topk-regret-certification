"""m-swap is KNOWN_INVALID / DO_NOT_USE_FOR_CERTIFICATION.

The default call and min(range-L1, m-swap) must refuse to return a value.
The method registry must carry the label, and no real-data runner may evaluate it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from private_mobility_cert.certificates.coupled_mswap import certificate_min_valid, certificate_mswap_os
from private_mobility_cert.certificates.release_conditioned import ThresholdTable
from private_mobility_cert.certificates.topk_regret import TopKCertError

CAP = Path(__file__).resolve().parents[2]


def _table(k: int) -> ThresholdTable:
    return ThresholdTable(mechanism="laplace", epsilon=1.0, n=5, k_max=k, alpha=0.05, scheme="test",
                          u_plus=np.ones(k), u_minus=np.ones(k))


def test_default_mswap_call_fails_closed():
    y = np.array([9.0, 7.0, 5.0, 3.0, 1.0])
    with pytest.raises(TopKCertError):
        certificate_mswap_os(y, k=2, weights=np.array([1.0, 0.5]), table=_table(2))


def test_min_valid_is_closed():
    y = np.array([9.0, 7.0, 5.0, 3.0, 1.0])
    with pytest.raises(TopKCertError):
        certificate_min_valid(y, k=2, weights=np.array([1.0, 0.5]), table=_table(2))


def test_registry_labels_mswap_invalid():
    reg = json.loads((CAP / "docs/METHOD_STATUS.json").read_text())
    row = reg["methods"]["certificate_mswap_os"]
    assert row["status"] == "KNOWN_INVALID"
    assert row["use"] == "DO_NOT_USE_FOR_CERTIFICATION"


def test_real_data_protocols_exclude_mswap():
    for rel in ("experiments/real_box_evaluation/protocol.json",):
        proto = json.loads((CAP / rel).read_text())
        assert "certificate_mswap_os" in proto["invalid_methods_excluded"]
    for rel in ("experiments/real_box_evaluation/run.py", "experiments/real_order_statistic_evaluation/run.py"):
        text = (CAP / rel).read_text()
        assert "certificate_mswap_os(" not in text
