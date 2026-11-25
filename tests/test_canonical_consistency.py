"""Canonical k<=5 identity. Cases are constructed here, not copied from the aggregator."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "synthetic_box_calibration"))

from private_mobility_cert.certificates.certified_risk_frontier import (  # noqa: E402
    STAB,
    normalize_certificate,
    release_scale_U,
)
from private_mobility_cert.certificates.direct_box_calibration import (  # noqa: E402
    canonicalize_k5_certificates,
    certification_decision,
    direct_box_radius_laplace,
    matched_box_certificates,
    union_box_radius_laplace,
)
from private_mobility_cert.certificates.release_conditioned import deploy_ranking  # noqa: E402
from private_mobility_cert.certificates.topk_regret import TopKCertError  # noqa: E402
from private_mobility_cert.certificates.weighted_ranking import (  # noqa: E402
    clip_nonnegative,
    weight_family,
)

TAUS = (0.05, 0.10, 0.25, 0.50, 1.0)
requires_numpy2 = pytest.mark.skipif(
    int(np.__version__.split(".")[0]) < 2,
    reason="raw float values depend on numpy 2.x summation; runs in .venv-synthetic",
)
LEDGER = ROOT / "experiments" / "synthetic_box_calibration" / "synthetic_releases.jsonl"


def _tied_release() -> np.ndarray:
    """Fifteen tied counts, a lower shelf, and negative coordinates."""
    return np.array([40.0] * 15 + [5.0] * 10 + [-2.0, 0.0, 1.0], dtype=np.float64)


def _bundle(y: np.ndarray, k: int, family: str, half_width: float) -> dict[str, float]:
    weights = weight_family(family, k)
    raw = matched_box_certificates(y, k=k, weights=weights, half_width=half_width)
    scale = max(float(release_scale_U(y, deploy_ranking(y, k), weights)), STAB)
    canon = canonicalize_k5_certificates(raw, k=k)
    return {"raw": raw, "canon": canon, "S": scale, "weights": weights, "y": y}


def test_k5_solver_split_is_removed_for_every_tau_and_weight_family():
    y = _tied_release()
    assert np.any(y < 0.0)
    for family in ("uniform", "dcg", "reciprocal"):
        for half_width in (4.0, 5.0):
            got = _bundle(y, 5, family, half_width)
            assert got["canon"]["k5_identity_applied"] == 1.0
            assert got["canon"]["C_block5"] == got["canon"]["C_box"]
            assert got["canon"]["C_strong"] == got["canon"]["C_box"]
            assert got["canon"]["C_strong_independent"] == got["raw"]["C_strong"]
            assert got["canon"]["C_block5_independent"] == got["raw"]["C_block5"]
            # Y+ clipping: the negative coordinate is not a deployed count.
            deployed = clip_nonnegative(y)[deploy_ranking(y, 5)]
            assert np.all(deployed == 40.0)
            for tau in TAUS:
                box_bit = certification_decision(got["canon"]["C_box"], got["S"], tau)
                strong_bit = certification_decision(got["canon"]["C_strong"], got["S"], tau)
                block_bit = certification_decision(got["canon"]["C_block5"], got["S"], tau)
                assert box_bit is strong_bit
                assert box_bit is block_bit


@requires_numpy2
def test_dcg_raw_values_straddle_tau_and_canonical_values_do_not():
    got = _bundle(_tied_release(), 5, "dcg", 5.0)
    raw_box = certification_decision(got["raw"]["C_box"], got["S"], 0.25)
    raw_strong = certification_decision(got["raw"]["C_strong"], got["S"], 0.25)
    assert raw_box is True
    assert raw_strong is False
    assert certification_decision(got["canon"]["C_strong"], got["S"], 0.25) is True
    assert got["canon"]["C_strong"] - got["canon"]["C_box"] == 0.0


def test_exact_threshold_and_one_ulp_above_it():
    scale = 100.0
    tau = 0.25
    on_threshold = 25.0
    above = float(np.nextafter(on_threshold, np.inf))
    assert certification_decision(on_threshold, scale, tau) is True
    assert certification_decision(above, scale, tau) is False
    raw = {
        "C_box": on_threshold,
        "C_sep_general": above,
        "C_rank": above,
        "C_pair": above,
        "C_block5": above,
        "C_strong": above,
    }
    canon = canonicalize_k5_certificates(raw, k=5)
    kept = canonicalize_k5_certificates(raw, k=6)
    assert certification_decision(canon["C_strong"], scale, tau) is True
    assert certification_decision(kept["C_strong"], scale, tau) is False
    assert kept["C_strong"] - kept["C_box"] == above - on_threshold


def test_machine_precision_gap_with_a_small_normalizer():
    scale = STAB
    tau = 0.25
    on_threshold = tau * scale
    above = float(np.nextafter(on_threshold, np.inf))
    assert scale == 1e-6
    assert certification_decision(on_threshold, scale, tau) is True
    assert certification_decision(above, scale, tau) is False
    raw = {key: above for key in ("C_sep_general", "C_rank", "C_pair", "C_block5", "C_strong")}
    raw["C_box"] = on_threshold
    canon = canonicalize_k5_certificates(raw, k=5)
    assert certification_decision(canon["C_box"], scale, tau) is True
    assert certification_decision(canon["C_strong"], scale, tau) is True


def test_nearly_zero_certificate_and_negative_release():
    y = np.array([9.0, 7.0, 5.0, 3.0, 1.0, -4.0, -6.0])
    got = _bundle(y, 5, "reciprocal", 0.5)
    assert got["raw"]["C_box"] == 0.0
    assert got["canon"]["C_strong"] == 0.0
    assert got["canon"]["C_block5"] == 0.0
    for tau in TAUS:
        assert certification_decision(got["canon"]["C_strong"], got["S"], tau) is True


def test_small_release_uses_the_stability_floor_and_keeps_the_identity():
    y = np.array([0.0, -1.0, 1e-12, -5.0, 0.0, 0.0])
    got = _bundle(y, 5, "uniform", 1.0)
    assert got["S"] == STAB
    assert got["canon"]["C_strong"] == got["canon"]["C_box"]
    assert got["canon"]["C_block5"] == got["canon"]["C_box"]
    clipped = clip_nonnegative(y)
    assert clipped[1] == 0.0
    assert normalize_certificate(got["canon"]["C_strong"], got["S"]) == normalize_certificate(
        got["canon"]["C_box"], got["S"]
    )


def test_k_above_5_keeps_a_real_positive_gap_and_event_order():
    y = _tied_release()
    wide = _bundle(y, 10, "uniform", 5.0)
    narrow = _bundle(y, 10, "uniform", 4.0)
    assert wide["canon"]["k5_identity_applied"] == 0.0
    assert wide["canon"]["C_strong"] - wide["canon"]["C_box"] == 50.0
    assert narrow["canon"]["C_box"] == 40.0
    assert wide["canon"]["C_box"] == 50.0
    assert narrow["canon"]["C_strong"] == 80.0
    assert wide["canon"]["C_strong"] == 100.0
    # At tau=0.20 the coupled certificate certifies and the relaxation does not.
    assert certification_decision(wide["canon"]["C_box"], wide["S"], 0.20) is True
    assert certification_decision(wide["canon"]["C_strong"], wide["S"], 0.20) is False
    for family in ("dcg", "reciprocal"):
        other = _bundle(y, 10, family, 5.0)
        assert other["canon"]["C_strong"] > other["canon"]["C_box"] + 1.0


def test_laplace_radii_keep_direct_box_at_most_union_box():
    y = _tied_release()
    n = y.size
    t_union = union_box_radius_laplace(n, 1.0, 0.05, delta=1.0)
    t_direct = direct_box_radius_laplace(n, 1.0, 0.05, delta=1.0)
    assert t_direct < t_union
    for family in ("uniform", "dcg", "reciprocal"):
        for k in (5, 10):
            union = _bundle(y, k, family, t_union)
            direct = _bundle(y, k, family, t_direct)
            assert direct["canon"]["C_box"] <= union["canon"]["C_box"] + 1e-9
            if k <= 5:
                assert direct["canon"]["C_strong"] == direct["canon"]["C_box"]
                assert union["canon"]["C_strong"] == union["canon"]["C_box"]
            else:
                assert union["canon"]["C_strong"] > union["canon"]["C_box"] + 1.0


def test_no_additive_tolerance_in_the_certification_rule():
    assert certification_decision(25.0001, 100.0, 0.25) is False
    with pytest.raises(TopKCertError):
        canonicalize_k5_certificates({"C_box": 1.0}, k=5)
    with pytest.raises(TopKCertError):
        certification_decision(1.0, 1.0, float("nan"))


def test_calibration_sources_do_not_call_mswap():
    texts = [
        (ROOT / "src/private_mobility_cert/certificates/direct_box_calibration.py").read_text(),
        (ROOT / "experiments/synthetic_box_calibration/run_canonical.py").read_text(),
        (ROOT / "experiments/synthetic_box_calibration/run_calibration.py").read_text(),
    ]
    for text in texts:
        assert "certificate_mswap_os" not in text
        assert "certificate_primary" not in text
        assert "certificate_min_valid" not in text


@requires_numpy2
def test_stored_spurious_rows_reproduce_and_canonicalize():
    if not LEDGER.exists():
        pytest.skip("requires experiments/synthetic_box_calibration/synthetic_releases.jsonl")
    from run_calibration import _release, synthetic_truth

    wanted = {
        "synthetic|ties|n261|M2|eps2.0|rep1|k5|dcg",
        "synthetic|ties|n261|M2|eps2.0|rep2|k5|dcg",
        "synthetic|ties|n261|M2|eps2.0|rep3|k5|dcg",
        "synthetic|ties|n500|M2|eps2.0|rep1|k5|dcg",
    }
    found = 0
    with LEDGER.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["release_id"] not in wanted:
                continue
            found += 1
            truth = synthetic_truth(row["regime"], row["n"])
            y = _release(truth, row["mechanism"], row["epsilon"], row["seed"])
            weights = weight_family(row["weight_family"], row["k"])
            raw = matched_box_certificates(
                y, k=row["k"], weights=weights, half_width=row["t_union"]
            )
            for key, value in raw.items():
                assert value == row[f"{key}_union"]
            assert certification_decision(row["C_box_union"], row["S"], 0.25) is True
            assert certification_decision(row["C_strong_union"], row["S"], 0.25) is False
            canon = canonicalize_k5_certificates(raw, k=5)
            assert canon["C_strong"] == raw["C_box"]
            for tau in TAUS:
                assert certification_decision(canon["C_box"], row["S"], tau) is certification_decision(
                    canon["C_strong"], row["S"], tau
                )
    assert found == 4
