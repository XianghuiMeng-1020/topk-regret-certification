"""Monotone closure of the count-feasible relaxations.

Reads the ledgers written by run_feasibility.py and writes closure_summary.json.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from private_mobility_cert.certificates.relaxation_closure import monotone_closed_value  # noqa: E402

OUT = Path(__file__).resolve().parent
INPUTS = [OUT / "principal_releases.jsonl", OUT / "low_count_stress.jsonl"]
TAUS = (0.05, 0.10, 0.25, 0.50, 1.00)
ATOL = 1e-8
METHODS = {
    "box": ("C_box_original", "C_box_plus"),
    "sep": ("C_sep_original", "C_sep_plus"),
    "rank": ("C_rank_original", "C_rank_plus"),
    "pair": ("C_pair_original", "C_pair_plus"),
    "block5": ("C_block5_original_independent", "C_block5_plus_independent"),
    "strong": ("C_strong_original_independent", "C_strong_plus_independent"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def _scale(value: float, normalizer: float) -> float:
    return float(value) / max(float(normalizer), 1e-6)


def _sign_counts(deltas: np.ndarray) -> dict[str, float | int]:
    genuine_worse = deltas > ATOL
    genuine_tighter = deltas < -ATOL
    numerical = np.abs(deltas) <= ATOL
    return {
        "n": int(deltas.size),
        "genuine_tighter": int(np.count_nonzero(genuine_tighter)),
        "numerical_zero": int(np.count_nonzero(numerical)),
        "genuine_worse": int(np.count_nonzero(genuine_worse)),
        "max_increase": float(np.max(deltas)) if deltas.size else float("nan"),
        "min_increase": float(np.min(deltas)) if deltas.size else float("nan"),
        "mean_original_minus_plus": float(np.mean(-deltas)) if deltas.size else float("nan"),
    }


def _decision_pair(row: dict, event: str, method: str) -> tuple[float, float]:
    """Original and count-feasible decision values.

    Zero-gain and fallback rows use the original value on the plus side.
    For k<=5, block-5 and strong are the exact coupled certificate, so their
    decision values are the canonical coupled numbers. k>5 keeps the
    independent raw numbers, including a genuine increase.
    """
    original_key, plus_key = METHODS[method]
    original = float(row[f"{event}_{original_key}"])
    plus = float(row[f"{event}_{plus_key}"])
    if method in {"block5", "strong"} and int(row["k"]) <= 5:
        original = float(row[f"{event}_C_box_original"])
        plus = float(row[f"{event}_C_box_plus"])
    if row[f"{event}_zero_gain_condition"] or row[f"{event}_fallback"]:
        plus = original
    return original, plus


def _event_report(rows: list[dict], event: str) -> dict:
    report: dict = {"methods": {}, "taus": {}, "validity": {}}
    for method, (original_key, plus_key) in METHODS.items():
        deltas = np.array(
            [
                float(row[f"{event}_{plus_key}"]) - float(row[f"{event}_{original_key}"])
                for row in rows
            ],
            dtype=np.float64,
        )
        report["methods"][method] = _sign_counts(deltas)
    # Raw relaxations must dominate the exact count-feasible certificate.
    violations = {method: 0 for method in ("sep", "rank", "pair", "block5", "strong")}
    for row in rows:
        if row[f"{event}_fallback"]:
            continue
        box = float(row[f"{event}_C_box_plus"])
        for method, (_original_key, plus_key) in METHODS.items():
            if method == "box":
                continue
            if float(row[f"{event}_{plus_key}"]) + ATOL < box:
                violations[method] += 1
    report["validity"]["raw_plus_below_box_plus"] = violations
    feasibility = np.array(
        [float(row[f"{event}_C_box_original"]) - float(row[f"{event}_C_box_plus"]) for row in rows]
    )
    report["exact_count_feasibility_gain"] = {
        "mean": float(np.mean(feasibility)),
        "median": float(np.median(feasibility)),
        "min": float(np.min(feasibility)),
        "max": float(np.max(feasibility)),
        "genuine_positive": int(np.count_nonzero(feasibility > ATOL)),
    }
    stored = np.array([float(row[f"{event}_D_feas_cpl"]) for row in rows])
    report["exact_gain_matches_stored_D_feas_cpl"] = bool(np.max(np.abs(stored - feasibility)) == 0.0)
    coupling = {"original": [], "plus_raw": [], "plus_closed": []}
    for row in rows:
        box_o, box_p = _decision_pair(row, event, "box")
        strong_o, strong_p = _decision_pair(row, event, "strong")
        strong_c = monotone_closed_value(strong_o, strong_p)
        coupling["original"].append(strong_o - box_o)
        coupling["plus_raw"].append(strong_p - box_p)
        coupling["plus_closed"].append(strong_c - box_p)
    report["coupling_gaps"] = {
        name: {
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
            "min": float(np.min(values)),
        }
        for name, values in coupling.items()
    }
    tau_rows = {}
    decision_disagreements = 0
    for tau in TAUS:
        entry = {}
        for method in METHODS:
            original_cert = 0
            raw_cert = 0
            closed_cert = 0
            raw_reversal = 0
            closed_reversal = 0
            raw_gain = 0
            disagree = 0
            for row in rows:
                scale = float(row[f"{event}_U"])
                original, plus = _decision_pair(row, event, method)
                closed = monotone_closed_value(original, plus)
                o_ok = _scale(original, scale) <= tau
                p_ok = _scale(plus, scale) <= tau
                c_ok = _scale(closed, scale) <= tau
                original_cert += int(o_ok)
                raw_cert += int(p_ok)
                closed_cert += int(c_ok)
                raw_reversal += int(o_ok and not p_ok)
                closed_reversal += int(o_ok and not c_ok)
                raw_gain += int((not o_ok) and p_ok)
                disagree += int(p_ok != c_ok)
            n = len(rows)
            entry[method] = {
                "K_original": original_cert / n,
                "K_plus_raw": raw_cert / n,
                "K_plus_closed": closed_cert / n,
                "raw_reversals": raw_reversal,
                "closed_reversals": closed_reversal,
                "raw_gains": raw_gain,
                "raw_versus_closed_decision_differences": disagree,
            }
            if method == "strong":
                decision_disagreements += disagree
        tau_rows[str(tau)] = entry
    report["taus"] = tau_rows
    report["strong_raw_versus_closed_differences_over_tau_grid"] = decision_disagreements
    return report


def main() -> None:
    principal = _load(OUT / "principal_releases.jsonl")
    stress = _load(OUT / "low_count_stress.jsonl")
    summary = {
        "atol": ATOL,
        "tau_grid": list(TAUS),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "principal": {
            "n": len(principal),
            "union": _event_report(principal, "union"),
            "direct": _event_report(principal, "direct"),
        },
        "stress": {
            "n": len(stress),
            "union": _event_report(stress, "union"),
            "direct": _event_report(stress, "direct"),
        },
        "input_sha256": {path.name: _sha256(path) for path in INPUTS},
    }
    (OUT / "closure_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"n_principal": len(principal), "n_stress": len(stress)}))


if __name__ == "__main__":
    main()
