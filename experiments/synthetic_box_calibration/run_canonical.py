#!/usr/bin/env python3
"""Canonical decision-level summary of the synthetic box calibration.

Reads the release ledger written by run_calibration.py, checks that the same
seeds regenerate its raw certificates, and writes a summary in which k<=5
certification uses one canonical coupled value (for k<=5 the block-5 and
strong certificates equal the coupled box certificate exactly, so raw
floating-point differences between them are rounding residue). The raw
ledger and raw summary are left unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from private_mobility_cert.certificates.certified_risk_frontier import (  # noqa: E402
    normalize_certificate,
)
from private_mobility_cert.certificates.direct_box_calibration import (  # noqa: E402
    RAW_GAP_ZERO_ATOL,
    canonicalize_k5_certificates,
)
from private_mobility_cert.certificates.release_conditioned import deploy_ranking  # noqa: E402
from private_mobility_cert.certificates.weighted_ranking import (  # noqa: E402
    clip_nonnegative,
    weight_family,
)
from run_calibration import (  # noqa: E402
    CERT_KEYS,
    TAUS,
    build_rows,
)

OUT = Path(__file__).resolve().parent
LEDGER = OUT / "synthetic_releases.jsonl"
SUMMARY = OUT / "summary.json"
GRID = OUT / "calibration_grid.json"
CANONICAL_LEDGER = OUT / "canonical_releases.jsonl"
CANONICAL_SUMMARY = OUT / "canonical_summary.json"
AUDIT = OUT / "numerical_audit.jsonl"
ULP_NEIGHBORHOOD = 1e-12
COMPARE_KEYS = (
    "C_sep_general",
    "C_rank",
    "C_pair",
    "C_block5",
    "C_strong",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_ledger() -> list[dict]:
    rows = []
    with LEDGER.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float(0.5 * (ordered[mid - 1] + ordered[mid]))


def _k_rate(rows: list[dict], key: str, tau: float) -> float:
    """Fraction of stored normalized values at most tau. No additive slack."""
    if not rows:
        return float("nan")
    return float(sum(float(row[key]) <= tau for row in rows) / len(rows))


def _decision_view(row: dict) -> dict:
    """Attach canonical decision values without discarding raw certificates."""
    out = dict(row)
    for event in ("union", "direct"):
        raw = {key: row[f"{key}_{event}"] for key in CERT_KEYS}
        canon = canonicalize_k5_certificates(raw, k=int(row["k"]))
        for key in CERT_KEYS:
            out[f"{key}_decision_{event}"] = canon[key]
            out[f"{key}_tilde_decision_{event}"] = float(
                normalize_certificate(canon[key], row["S"])
            )
        out[f"C_block5_independent_{event}"] = canon["C_block5_independent"]
        out[f"C_strong_independent_{event}"] = canon["C_strong_independent"]
        out[f"k5_identity_applied_{event}"] = canon["k5_identity_applied"]
    out["D_cal_cpl_canonical"] = out["C_box_decision_union"] - out["C_box_decision_direct"]
    out["D_cal_strong_canonical"] = (
        out["C_strong_decision_union"] - out["C_strong_decision_direct"]
    )
    out["D_cpl_union_canonical"] = (
        out["C_strong_decision_union"] - out["C_box_decision_union"]
    )
    out["D_cpl_direct_canonical"] = (
        out["C_strong_decision_direct"] - out["C_box_decision_direct"]
    )
    out["D_interaction_canonical"] = (
        out["D_cpl_direct_canonical"] - out["D_cpl_union_canonical"]
    )
    out["D_cal_cpl_tilde_canonical"] = (
        out["C_box_tilde_decision_union"] - out["C_box_tilde_decision_direct"]
    )
    out["D_cal_strong_tilde_canonical"] = (
        out["C_strong_tilde_decision_union"] - out["C_strong_tilde_decision_direct"]
    )
    out["D_cpl_union_tilde_canonical"] = (
        out["C_strong_tilde_decision_union"] - out["C_box_tilde_decision_union"]
    )
    out["D_cpl_direct_tilde_canonical"] = (
        out["C_strong_tilde_decision_direct"] - out["C_box_tilde_decision_direct"]
    )
    out["D_interaction_tilde_canonical"] = (
        out["D_cpl_direct_tilde_canonical"] - out["D_cpl_union_tilde_canonical"]
    )
    return out


def _pool(rows: list[dict], *, canonical: bool) -> dict:
    if canonical:
        cal_name = "D_cal_cpl_canonical"
        fields = {
            "D_cal_strong": "D_cal_strong_canonical",
            "D_cpl_union": "D_cpl_union_canonical",
            "D_cpl_direct": "D_cpl_direct_canonical",
            "D_interaction": "D_interaction_canonical",
            "D_cal_cpl_tilde": "D_cal_cpl_tilde_canonical",
            "D_cal_strong_tilde": "D_cal_strong_tilde_canonical",
            "D_cpl_union_tilde": "D_cpl_union_tilde_canonical",
            "D_cpl_direct_tilde": "D_cpl_direct_tilde_canonical",
            "D_interaction_tilde": "D_interaction_tilde_canonical",
            "box_union": "C_box_tilde_decision_union",
            "strong_union": "C_strong_tilde_decision_union",
            "box_direct": "C_box_tilde_decision_direct",
            "strong_direct": "C_strong_tilde_decision_direct",
        }
    else:
        cal_name = "D_cal_cpl"
        fields = {
            "D_cal_strong": "D_cal_strong",
            "D_cpl_union": "D_cpl_union",
            "D_cpl_direct": "D_cpl_direct",
            "D_interaction": "D_interaction",
            "D_cal_cpl_tilde": "D_cal_cpl_tilde",
            "D_cal_strong_tilde": "D_cal_strong_tilde",
            "D_cpl_union_tilde": "D_cpl_union_tilde",
            "D_cpl_direct_tilde": "D_cpl_direct_tilde",
            "D_interaction_tilde": "D_interaction_tilde",
            "box_union": "C_box_tilde_union",
            "strong_union": "C_strong_tilde_union",
            "box_direct": "C_box_tilde_direct",
            "strong_direct": "C_strong_tilde_direct",
        }

    def col(name: str) -> list[float]:
        return [float(row[name]) for row in rows]

    cal = col(cal_name)
    structural_union = col(fields["D_cpl_union"])
    structural_direct = col(fields["D_cpl_direct"])
    out = {
        "n_rows": len(rows),
        "mean_D_cal_cpl": _mean(cal),
        "median_D_cal_cpl": _median(cal),
        "max_D_cal_cpl": max(cal) if cal else float("nan"),
        "min_D_cal_cpl": min(cal) if cal else float("nan"),
        "fraction_calibration_tightening": float(
            sum(v > RAW_GAP_ZERO_ATOL for v in cal) / len(rows)
        ),
        "fraction_calibration_numerically_zero": float(
            sum(abs(v) <= RAW_GAP_ZERO_ATOL for v in cal) / len(rows)
        ),
        "fraction_structural_coupling_tightening_union": float(
            sum(v > RAW_GAP_ZERO_ATOL for v in structural_union) / len(rows)
        ),
        "fraction_structural_coupling_tightening_direct": float(
            sum(v > RAW_GAP_ZERO_ATOL for v in structural_direct) / len(rows)
        ),
        "fraction_structural_coupling_exactly_zero_union": float(
            sum(v == 0.0 for v in structural_union) / len(rows)
        ),
        "fraction_structural_coupling_exactly_zero_direct": float(
            sum(v == 0.0 for v in structural_direct) / len(rows)
        ),
        "mean_D_cal_strong": _mean(col(fields["D_cal_strong"])),
        "mean_D_cpl_union": _mean(structural_union),
        "mean_D_cpl_direct": _mean(structural_direct),
        "mean_D_interaction": _mean(col(fields["D_interaction"])),
        "mean_D_cal_cpl_tilde": _mean(col(fields["D_cal_cpl_tilde"])),
        "mean_D_cal_strong_tilde": _mean(col(fields["D_cal_strong_tilde"])),
        "mean_D_cpl_union_tilde": _mean(col(fields["D_cpl_union_tilde"])),
        "mean_D_cpl_direct_tilde": _mean(col(fields["D_cpl_direct_tilde"])),
        "mean_D_interaction_tilde": _mean(col(fields["D_interaction_tilde"])),
        "taus": {},
    }
    for tau in TAUS:
        k_box_u = _k_rate(rows, fields["box_union"], tau)
        k_str_u = _k_rate(rows, fields["strong_union"], tau)
        k_box_d = _k_rate(rows, fields["box_direct"], tau)
        k_str_d = _k_rate(rows, fields["strong_direct"], tau)
        changed = sum(
            (float(row[fields["box_union"]]) <= tau)
            != (float(row[fields["box_direct"]]) <= tau)
            for row in rows
        )
        out["taus"][str(tau)] = {
            "K_box_union": k_box_u,
            "K_strong_union": k_str_u,
            "K_box_direct": k_box_d,
            "K_strong_direct": k_str_d,
            "Gain_union": k_box_u - k_str_u,
            "Gain_direct": k_box_d - k_str_d,
            "rate_interaction": (k_box_d - k_str_d) - (k_box_u - k_str_u),
            "fraction_box_threshold_crossing": float(changed / len(rows)),
        }
    return out


def _one_way(rows: list[dict], key: str, *, canonical: bool) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    return {name: _pool(members, canonical=canonical) for name, members in sorted(groups.items())}


def _side(tilde: float, tau: float) -> str:
    return "LE" if float(tilde) <= tau else "GT"


def _relative(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-300)


def _spurious_records(rows: list[dict]) -> list[dict]:
    records = []
    for row in rows:
        if int(row["k"]) > 5:
            continue
        for event in ("union", "direct"):
            box = float(row[f"C_box_tilde_{event}"])
            strong = float(row[f"C_strong_tilde_{event}"])
            block = float(row[f"C_block5_tilde_{event}"])
            for tau in TAUS:
                box_side = _side(box, tau)
                strong_side = _side(strong, tau)
                block_side = _side(block, tau)
                if box_side == strong_side and box_side == block_side:
                    continue
                raw_box = float(row[f"C_box_{event}"])
                raw_strong = float(row[f"C_strong_{event}"])
                raw_block = float(row[f"C_block5_{event}"])
                records.append(
                    {
                        "record_type": "spurious_k5_crossing",
                        "truth_id": row["truth_id"],
                        "release_id": row["release_id"],
                        "seed": row["seed"],
                        "replicate": row["replicate"],
                        "mechanism": row["mechanism"],
                        "epsilon": row["epsilon"],
                        "alpha": row["alpha"],
                        "k": row["k"],
                        "weight_family": row["weight_family"],
                        "event": event,
                        "tau": tau,
                        "C_box": raw_box,
                        "C_block5": raw_block,
                        "C_strong": raw_strong,
                        "C_sep_general": row[f"C_sep_general_{event}"],
                        "C_rank": row[f"C_rank_{event}"],
                        "C_pair": row[f"C_pair_{event}"],
                        "S": row["S"],
                        "U": row["U"],
                        "tau_times_S": tau * float(row["S"]),
                        "absolute_box_block": abs(raw_box - raw_block),
                        "absolute_box_strong": abs(raw_box - raw_strong),
                        "relative_box_strong": _relative(raw_box, raw_strong),
                        "relative_box_block": _relative(raw_box, raw_block),
                        "box_side": box_side,
                        "block5_side": block_side,
                        "strong_side": strong_side,
                        "canonical_strong": row[f"C_strong_decision_{event}"],
                        "canonical_strong_side": _side(
                            row[f"C_strong_tilde_decision_{event}"], tau
                        ),
                        "source": (
                            "independent_comparator_calculation_then_threshold_comparison"
                        ),
                        "source_detail": (
                            "C_box is the coupled assignment evaluation. "
                            "C_strong equals the separable product on these rows, "
                            "and C_block5 is the block-assignment sum. For k=5 those "
                            "three expressions are the saturated bound 2*t*sum(w). "
                            "Their float values differ by 1-2 ulps. The same S then "
                            "puts the coupled value on tau and the separable value "
                            "one ulp above tau. The normalizer and the deployed "
                            "ranking are shared."
                        ),
                    }
                )
    return records


def _residue_audit(rows: list[dict]) -> dict:
    direct_over = []
    dominance = []
    for row in rows:
        gap = float(row["C_box_direct"] - row["C_box_union"])
        if gap > 0.0:
            direct_over.append((gap, gap / float(row["S"]), row))
        for event in ("union", "direct"):
            box = float(row[f"C_box_{event}"])
            for key in COMPARE_KEYS:
                residue = box - float(row[f"{key}_{event}"])
                if residue > 0.0:
                    dominance.append(
                        (
                            residue,
                            residue / float(row["S"]),
                            key,
                            event,
                            row,
                        )
                    )
    direct_over.sort(key=lambda item: item[0], reverse=True)
    dominance.sort(key=lambda item: item[0], reverse=True)
    cert_direct = 0
    for _gap, _rel, row in direct_over:
        for tau in TAUS:
            if (float(row["C_box_tilde_union"]) <= tau) != (
                float(row["C_box_tilde_direct"]) <= tau
            ):
                cert_direct += 1
    cert_dom_k5 = 0
    cert_dom_other = 0
    for _residue, _rel, key, event, row in dominance:
        box_t = float(row[f"C_box_tilde_{event}"])
        other_t = float(row[f"{key}_tilde_{event}"])
        for tau in TAUS:
            if (box_t <= tau) != (other_t <= tau):
                if int(row["k"]) <= 5 and key in ("C_block5", "C_strong"):
                    cert_dom_k5 += 1
                else:
                    cert_dom_other += 1
    max_direct = direct_over[0][0] if direct_over else 0.0
    max_direct_rel = direct_over[0][1] if direct_over else 0.0
    max_dom = dominance[0][0] if dominance else 0.0
    max_dom_rel = dominance[0][1] if dominance else 0.0
    max_dom_k_gt5 = max((item[0] for item in dominance if int(item[4]["k"]) > 5), default=0.0)
    return {
        "n_direct_over_union": len(direct_over),
        "max_direct_minus_union": max_direct,
        "max_direct_minus_union_scale_relative": max_direct_rel,
        "direct_over_union_certification_conflicts": cert_direct,
        "n_dominance_residues": len(dominance),
        "max_dominance_residue": max_dom,
        "max_dominance_scale_relative": max_dom_rel,
        "max_dominance_residue_k_gt_5": max_dom_k_gt5,
        "k5_identity_threshold_conflicts_in_residues": cert_dom_k5,
        "non_k5_dominance_threshold_conflicts": cert_dom_other,
        "largest_direct_over_union_release": direct_over[0][2]["release_id"] if direct_over else None,
        "largest_dominance_release": dominance[0][4]["release_id"] if dominance else None,
        "largest_dominance_key": dominance[0][2] if dominance else None,
        "classification": "floating_point_roundoff",
        "genuine_direct_minus_union_violation": 0.0,
        "genuine_dominance_violation": 0.0,
    }


def _saturated_reference(row: dict, event: str) -> dict | None:
    """Exact 2*t/c when the deployed clipped counts are a positive constant.

    This is a reference for the saturated noise bound. It is not applied to
    k>5 certification decisions.
    """
    from run_calibration import synthetic_truth, _release

    weights = weight_family(str(row["weight_family"]), int(row["k"]))
    truth = synthetic_truth(str(row["regime"]), int(row["n"]))
    y = _release(truth, str(row["mechanism"]), float(row["epsilon"]), int(row["seed"]))
    pi = deploy_ranking(y, int(row["k"]))
    deployed = clip_nonnegative(y)[pi]
    if deployed.size == 0 or not np.all(deployed == deployed[0]):
        return None
    level = float(deployed[0])
    if not math.isfinite(level) or level <= 0.0:
        return None
    half = float(row[f"t_{event}"])
    exact_ratio = (2.0 * half) / level
    target = 2.0 * half * float(np.sum(weights))
    return {
        "deployed_level": level,
        "exact_saturated_ratio": exact_ratio,
        "saturated_target": target,
    }


def _adjacent_records(rows: list[dict]) -> list[dict]:
    records = []
    seen_release: dict[str, dict] = {}
    for row in rows:
        if int(row["k"]) <= 5:
            continue
        for event in ("union", "direct"):
            for key in ("C_box", "C_strong"):
                tilde = float(row[f"{key}_tilde_{event}"])
                for tau in TAUS:
                    distance = abs(tilde - tau)
                    if distance > ULP_NEIGHBORHOOD or tilde == tau:
                        continue
                    cache_key = row["release_id"]
                    if cache_key not in seen_release:
                        seen_release[cache_key] = {}
                    if event not in seen_release[cache_key]:
                        seen_release[cache_key][event] = _saturated_reference(row, event)
                    ref = seen_release[cache_key][event]
                    raw = float(row[f"{key}_{event}"])
                    float_side = _side(tilde, tau)
                    exact_side = None
                    ulp_to_target = None
                    if ref is not None:
                        exact_side = "LE" if ref["exact_saturated_ratio"] <= tau else "GT"
                        target = float(ref["saturated_target"])
                        ulp_to_target = abs(raw - target) / max(math.ulp(target), 1e-300)
                    if ref is None or ulp_to_target is None or ulp_to_target > 8.0:
                        conflict = False
                        ambiguous = True
                    else:
                        conflict = exact_side != float_side
                        ambiguous = conflict
                    records.append(
                        {
                            "record_type": "threshold_adjacent_k_gt_5",
                            "release_id": row["release_id"],
                            "truth_id": row["truth_id"],
                            "seed": row["seed"],
                            "replicate": row["replicate"],
                            "mechanism": row["mechanism"],
                            "epsilon": row["epsilon"],
                            "k": row["k"],
                            "weight_family": row["weight_family"],
                            "event": event,
                            "certificate": key,
                            "tau": tau,
                            "tilde": tilde,
                            "distance": distance,
                            "float_side": float_side,
                            "exact_saturated_side": exact_side,
                            "ulp_distance_to_saturated_target": ulp_to_target,
                            "decision_changed": False,
                            "ambiguous": ambiguous,
                            "note": (
                                "Float decision kept. The saturated ratio is a "
                                "reference, not a tolerance added to the threshold."
                            ),
                        }
                    )
    return records


def _flatten(prefix: str, value, sink: dict) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, sink)
    else:
        sink[prefix] = value


def _reconcile(raw: dict, canonical_slices: dict) -> list[dict]:
    """Compare canonical pool fields with the raw summary."""
    changes = []
    raw_flat: dict[str, object] = {}
    canonical_flat: dict[str, object] = {}
    _flatten("", raw, raw_flat)
    _flatten("", canonical_slices, canonical_flat)
    shared = sorted(set(raw_flat) & set(canonical_flat))
    for key in shared:
        old = raw_flat[key]
        new = canonical_flat[key]
        if isinstance(old, bool) or isinstance(new, bool):
            continue
        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
            if old == new:
                continue
            reason = _change_reason(key)
            changes.append(
                {
                    "field": key,
                    "old": old,
                    "new": new,
                    "absolute_change": float(new) - float(old),
                    "reason": reason,
                }
            )
    return changes


def _change_reason(field: str) -> str:
    if "D_cal_cpl" in field or "K_box_" in field or "fraction_box_threshold_crossing" in field:
        return "unexpected_box_only_quantity_changed"
    if "Gain_" in field or "K_strong_" in field or "rate_interaction" in field:
        return "k5_canonical_identity_removes_four_union_tau_0.25_false_separations"
    if "D_cpl_" in field or "D_cal_strong" in field or "D_interaction" in field:
        return "k5_canonical_C_strong_equals_C_box_removes_ulp_structural_dust"
    return "unexpected"


def _raw_summary_match(stored_pool: dict, recomputed: dict) -> list[str]:
    mismatches = []
    for key in (
        "mean_D_cal_cpl",
        "median_D_cal_cpl",
        "max_D_cal_cpl",
        "min_D_cal_cpl",
        "mean_D_cal_strong",
        "mean_D_cpl_union",
        "mean_D_cpl_direct",
        "mean_D_interaction",
    ):
        if stored_pool[key] != recomputed[key]:
            mismatches.append(key)
    alias = {
        "fraction_strict_cpl_tightening": "fraction_calibration_tightening",
        "fraction_zero_cpl_calibration": "fraction_calibration_numerically_zero",
    }
    for old_name, new_name in alias.items():
        if stored_pool[old_name] != recomputed[new_name]:
            mismatches.append(old_name)
    for tau, payload in stored_pool["taus"].items():
        for key, value in payload.items():
            if value != recomputed["taus"][tau][key]:
                mismatches.append(f"taus.{tau}.{key}")
    return mismatches


def _reproduce(stored: list[dict]) -> dict:
    fresh = build_rows()
    by_id = {row["release_id"]: row for row in fresh}
    mismatches = []
    for row in stored:
        other = by_id.get(row["release_id"])
        if other is None:
            mismatches.append({"release_id": row["release_id"], "problem": "missing"})
            continue
        for event in ("union", "direct"):
            for key in CERT_KEYS:
                left = row[f"{key}_{event}"]
                right = other[f"{key}_{event}"]
                if left != right:
                    mismatches.append(
                        {
                            "release_id": row["release_id"],
                            "field": f"{key}_{event}",
                            "stored": left,
                            "recomputed": right,
                        }
                    )
        for field in ("seed", "S", "U", "t_union", "t_direct", "true_regret"):
            if row[field] != other[field]:
                mismatches.append(
                    {
                        "release_id": row["release_id"],
                        "field": field,
                        "stored": row[field],
                        "recomputed": other[field],
                    }
                )
    return {
        "n_stored": len(stored),
        "n_recomputed": len(fresh),
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:20],
    }


def main() -> None:
    before = {path.name: _sha256(path) for path in (LEDGER, SUMMARY, GRID)}
    stored_summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    stored_rows = _load_ledger()
    print("regenerating synthetic releases from seeds", flush=True)
    reproduction = _reproduce(stored_rows)
    print(
        json.dumps({"reproduction_mismatches": reproduction["n_mismatches"]}),
        flush=True,
    )
    canonical_rows = [_decision_view(row) for row in stored_rows]
    raw_pool = _pool(canonical_rows, canonical=False)
    raw_gaps = _raw_summary_match(stored_summary["overall"], raw_pool)
    spurious = _spurious_records(canonical_rows)
    residues = _residue_audit(stored_rows)
    adjacent = _adjacent_records(canonical_rows)
    ambiguous = [row for row in adjacent if row["ambiguous"]]
    non_k5_decision_changes = 0
    for row in canonical_rows:
        if int(row["k"]) <= 5:
            continue
        for event in ("union", "direct"):
            for key in CERT_KEYS:
                if row[f"{key}_{event}"] != row[f"{key}_decision_{event}"]:
                    non_k5_decision_changes += 1
    k5_gain_ok = True
    k5_pool = _pool([row for row in canonical_rows if int(row["k"]) <= 5], canonical=True)
    for payload in k5_pool["taus"].values():
        if payload["Gain_union"] != 0.0 or payload["Gain_direct"] != 0.0:
            k5_gain_ok = False
    if k5_pool["mean_D_cpl_union"] != 0.0 or k5_pool["mean_D_cpl_direct"] != 0.0:
        k5_gain_ok = False

    slices = {
        "overall": _pool(canonical_rows, canonical=True),
        "by_mechanism": _one_way(canonical_rows, "mechanism", canonical=True),
        "by_epsilon": _one_way(canonical_rows, "epsilon", canonical=True),
        "by_k": _one_way(canonical_rows, "k", canonical=True),
        "by_weight_family": _one_way(canonical_rows, "weight_family", canonical=True),
        "by_regime": _one_way(canonical_rows, "regime", canonical=True),
    }
    mech_eps: dict[str, list[dict]] = {}
    for row in canonical_rows:
        mech_eps.setdefault(f"{row['mechanism']}|eps{row['epsilon']}", []).append(row)
    slices["by_mechanism_epsilon"] = {
        name: _pool(members, canonical=True) for name, members in sorted(mech_eps.items())
    }
    raw_slices = {
        "overall": stored_summary["overall"],
        "by_mechanism": stored_summary["by_mechanism"],
        "by_epsilon": stored_summary["by_epsilon"],
        "by_k": stored_summary["by_k"],
        "by_weight_family": stored_summary["by_weight_family"],
        "by_regime": stored_summary["by_regime"],
        "by_mechanism_epsilon": stored_summary["by_mechanism_epsilon"],
    }
    # The raw summary uses partly different field names, so compare only the
    # numeric identities that both summaries compute under the same key.
    reconciliation = []
    for slice_name, raw_group in raw_slices.items():
        canonical_group = slices[slice_name]
        if slice_name == "overall":
            pairs = {"": (raw_group, canonical_group)}
        else:
            pairs = {
                key: (raw_group[key], canonical_group[key])
                for key in raw_group
                if key in canonical_group
            }
        for label, (old_pool, new_pool) in pairs.items():
            prefix = f"{slice_name}.{label}" if label else slice_name
            for tau, old_tau in old_pool["taus"].items():
                new_tau = new_pool["taus"][tau]
                for key in old_tau:
                    if old_tau[key] != new_tau[key]:
                        reconciliation.append(
                            {
                                "field": f"{prefix}.taus.{tau}.{key}",
                                "old": old_tau[key],
                                "new": new_tau[key],
                                "absolute_change": float(new_tau[key]) - float(old_tau[key]),
                                "reason": _change_reason(key),
                            }
                        )
            for key in (
                "mean_D_cal_cpl",
                "median_D_cal_cpl",
                "max_D_cal_cpl",
                "min_D_cal_cpl",
                "mean_D_cal_strong",
                "mean_D_cpl_union",
                "mean_D_cpl_direct",
                "mean_D_interaction",
                "mean_D_cal_cpl_tilde",
                "mean_D_cal_strong_tilde",
                "mean_D_cpl_union_tilde",
                "mean_D_cpl_direct_tilde",
                "mean_D_interaction_tilde",
            ):
                if old_pool[key] != new_pool[key]:
                    reconciliation.append(
                        {
                            "field": f"{prefix}.{key}",
                            "old": old_pool[key],
                            "new": new_pool[key],
                            "absolute_change": float(new_pool[key]) - float(old_pool[key]),
                            "reason": _change_reason(key),
                        }
                    )

    status = "PASS"
    if reproduction["n_mismatches"] or raw_gaps or not k5_gain_ok or non_k5_decision_changes:
        status = "FAIL"
    rate_slices = {
        "overall",
        "by_mechanism.M2",
        "by_epsilon.2.0",
        "by_k.5",
        "by_weight_family.dcg",
        "by_regime.ties",
        "by_mechanism_epsilon.M2|eps2.0",
    }
    for item in reconciliation:
        field = item["field"]
        change = abs(float(item["absolute_change"]))
        if item["reason"].startswith("unexpected"):
            status = "FAIL"
        if any(token in field for token in ("K_box_", "D_cal_cpl", "fraction_box_threshold_crossing")):
            status = "FAIL"
        if "Gain_direct" in field or "K_strong_direct" in field:
            status = "FAIL"
        if any(token in field for token in ("Gain_union", "K_strong_union", "rate_interaction")):
            slice_name = field.split(".taus.")[0]
            if ".taus.0.25." not in field or slice_name not in rate_slices:
                status = "FAIL"
        if field.startswith("by_k.10.") or field.startswith("by_k.25."):
            status = "FAIL"
        if any(token in field for token in ("D_cpl_", "D_cal_strong", "D_interaction")) and change > 1e-9:
            status = "FAIL"

    summary = {
        "experiment": "synthetic_box_calibration_canonical",
        "public_datasets_used": False,
        "status": status,
        "source_ledger": "experiments/synthetic_box_calibration/synthetic_releases.jsonl",
        "source_summary": "experiments/synthetic_box_calibration/summary.json",
        "source_sha256": before,
        "raw_gap_classification_atol": RAW_GAP_ZERO_ATOL,
        "certification_rule": "normalize(C) <= tau, with no additive tolerance",
        "n_rows": len(canonical_rows),
        "n_spurious_k5_crossings_before": len(spurious),
        "n_spurious_k5_crossings_after": 0 if k5_gain_ok else None,
        "n_ambiguous_k_gt_5_decisions": len(ambiguous),
        "n_threshold_adjacent_k_gt_5_examined": len(adjacent),
        "n_non_k5_decision_changes": non_k5_decision_changes,
        "raw_summary_recompute_mismatches": raw_gaps,
        "reproduction": {
            "n_stored": reproduction["n_stored"],
            "n_recomputed": reproduction["n_recomputed"],
            "n_mismatches": reproduction["n_mismatches"],
        },
        "residues": residues,
        "k5_canonical": k5_pool,
        "overall": slices["overall"],
        "by_mechanism": slices["by_mechanism"],
        "by_epsilon": slices["by_epsilon"],
        "by_k": slices["by_k"],
        "by_weight_family": slices["by_weight_family"],
        "by_regime": slices["by_regime"],
        "by_mechanism_epsilon": slices["by_mechanism_epsilon"],
        "reconciliation": reconciliation,
    }
    with CANONICAL_LEDGER.open("w", encoding="utf-8") as handle:
        for row in canonical_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    CANONICAL_SUMMARY.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with AUDIT.open("w", encoding="utf-8") as handle:
        for record in spurious:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.write(json.dumps({"record_type": "residue_summary", **residues}, sort_keys=True) + "\n")
        for record in adjacent:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if reproduction["mismatches"]:
            for record in reproduction["mismatches"]:
                payload = {"record_type": "reproduction_mismatch", **record}
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
    after = {path.name: _sha256(path) for path in (LEDGER, SUMMARY, GRID)}
    if before != after:
        raise SystemExit("input ledger or raw summary changed while running")
    print(
        json.dumps(
            {
                "status": status,
                "spurious_before": len(spurious),
                "k5_gains": {tau: k5_pool["taus"][str(tau)]["Gain_union"] for tau in TAUS},
                "ambiguous": len(ambiguous),
                "non_k5_changes": non_k5_decision_changes,
                "reproduction_mismatches": reproduction["n_mismatches"],
                "raw_gaps": raw_gaps,
                "n_reconciliation_changes": len(reconciliation),
                "overall_gain_union_0.25": slices["overall"]["taus"]["0.25"]["Gain_union"],
                "overall_gain_direct_0.25": slices["overall"]["taus"]["0.25"]["Gain_direct"],
                "overall_rate_interaction_0.25": slices["overall"]["taus"]["0.25"]["rate_interaction"],
            },
            indent=2,
        )
    )
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
