"""Regression targets: recompute registered values from the ledger rows."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def recomputed():
    claims = _load("scripts/check_claims.py", "check_claims")
    return claims.recompute(ROOT, claims.load_audit_helpers())


@pytest.fixture(scope="module")
def targets():
    return json.loads((ROOT / "manifests/REGRESSION_TARGETS.json").read_text())


def test_every_recompute_target_matches(recomputed, targets):
    tol = targets["numeric_tolerance_abs"]
    for t in targets["targets"]:
        if not t["recompute"]:
            continue
        got = recomputed[t["recompute"]]
        if t["expected"] in ("True", "False"):
            assert str(bool(got)) == t["expected"], t["id"]
        else:
            assert abs(float(got) - float(t["expected"])) <= tol, (t["id"], got)


def test_tlc_gain_is_exactly_zero(recomputed):
    assert recomputed["box.delta_union.tlc"] == 0.0


def test_time_forward_masks_match_stored_decisions(recomputed):
    assert recomputed["time_forward.accept_masks_match_ledger"]


def test_time_forward_coupled_accepts_less(recomputed):
    assert recomputed["time_forward.D_accept"] < 0


def test_verify_results_passes():
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/verify_results.py")], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-3000:]


def test_source_manifest_shape():
    objs = json.loads((ROOT / "manifests/SOURCE_OBJECTS.json").read_text())["objects"]
    by = {}
    for o in objs:
        by.setdefault((o["source"], o["role"]), 0)
        by[(o["source"], o["role"])] += 1
        assert o["url"].startswith("https://")
        assert len(o["sha256"]) == 64
    assert by == {("tlc", "TRAIN"): 2, ("tlc", "EVAL"): 2, ("gharchive", "TRAIN"): 14,
                  ("gharchive", "EVAL"): 36}
    assert len({o["name"] for o in objs}) == len(objs)
