#!/usr/bin/env python3
"""Count-feasible diagnostics on the synthetic box-calibration releases, plus one predeclared low-count panel.

The low-count panel is fixed here, before any result is inspected:

    x_i = 1 for i < 8, and x_i = 0 otherwise.
    n in {8, 261}; k in {5} when n=8 and {5,10,25} when n=261;
    both mechanisms; epsilon in {0.5,1,2}; four replicates;
    uniform, DCG, and reciprocal weights; alpha = 0.05;
    master seed 20261125; split g4_low_count_stress.

No public dataset is read.
"""

from __future__ import annotations

import json
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "synthetic_box_calibration"))

from private_mobility_cert.certificates.certified_risk_frontier import (  # noqa: E402
    normalize_certificate,
    release_scale_U,
)
from private_mobility_cert.certificates.count_feasible import (  # noqa: E402
    STATUS_EXACT,
    STATUS_FALLBACK,
    deployed_margin_profile,
    matched_count_feasible_certificates,
)
from private_mobility_cert.certificates.direct_box_calibration import (  # noqa: E402
    canonicalize_k5_certificates,
)
from private_mobility_cert.certificates.release_conditioned import (  # noqa: E402
    deploy_ranking,
    exact_weighted_regret,
)
from private_mobility_cert.certificates.weighted_ranking import weight_family  # noqa: E402
from private_mobility_cert.privacy.seeds import SeedKey, derive_release_seed  # noqa: E402
from run_calibration import (  # noqa: E402
    ALPHA,
    KS,
    MASTER,
    MECHS,
    REGIMES,
    REPS,
    SPLIT,
    WEIGHTS,
    _radii,
    _release,
    synthetic_truth,
)

OUT = Path(__file__).resolve().parent
LEDGER = ROOT / "experiments" / "synthetic_box_calibration" / "synthetic_releases.jsonl"
TAUS = (0.05, 0.10, 0.25, 0.50, 1.0)
GAP_ATOL = 1e-8
LOW_MASTER = 20261125
# Opaque seed salt: its exact bytes enter the SHA-256 seed derivation.
LOW_SPLIT = "g4_low_count_stress"
LOW_NS = (8, 261)
LOW_REPS = 4
KEYS = ("C_box", "C_sep_general", "C_rank", "C_pair", "C_block5", "C_strong")


def low_count_truth(n: int) -> np.ndarray:
    """Predeclared nonnegative truth: eight ones, and zeros after that."""
    counts = np.zeros(int(n), dtype=np.int64)
    counts[: min(8, int(n))] = 1
    return counts


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _decision_original(family: dict, k: int) -> dict[str, float]:
    raw = {
        "C_box": family["C_box_original"],
        "C_sep_general": family["C_sep_original"],
        "C_rank": family["C_rank_original"],
        "C_pair": family["C_pair_original"],
        "C_block5": family["C_block5_original"],
        "C_strong": family["C_strong_original"],
    }
    return canonicalize_k5_certificates(raw, k=k)


def _one_event(y, latent, k, weights, half_width: float) -> dict:
    family = matched_count_feasible_certificates(
        y, k=k, weights=weights, u_minus=half_width, u_plus=half_width
    )
    decision = _decision_original(family, k)
    profile = deployed_margin_profile(y, k=k, weights=weights, u_plus=half_width)
    pi = deploy_ranking(y, k)
    scale_u = release_scale_U(y, pi, weights)
    regret = exact_weighted_regret(np.asarray(latent, dtype=np.float64), pi, weights)
    box_gap = float(family["C_box_original"] - family["C_box"])
    strong_gap = float(decision["C_strong"] - family["C_strong"])
    exact = family["status"] == STATUS_EXACT
    return {
        "status": family["status"],
        "consistent": exact,
        "fallback": family["status"] == STATUS_FALLBACK,
        "n_inconsistent": int(family["n_inconsistent"]),
        "n_below_upper": int(family["n_below_upper"]),
        "fraction_below_upper": float(family["n_below_upper"]) / float(np.asarray(y).size),
        "n_positive_deployed_below": int(profile["n_positive_deployed_below"]),
        "min_positive_deployed_y": float(profile["min_positive_deployed_y"]),
        "zero_gain_condition": bool(exact and profile["n_positive_deployed_below"] == 0),
        "U": float(scale_u),
        "latent_regret": float(regret),
        "C_box_original": float(family["C_box_original"]),
        "C_sep_original": float(family["C_sep_original"]),
        "C_rank_original": float(family["C_rank_original"]),
        "C_pair_original": float(family["C_pair_original"]),
        "C_strong_original_independent": float(family["C_strong_original"]),
        "C_strong_original": float(decision["C_strong"]),
        "C_block5_original_independent": float(family["C_block5_original"]),
        "C_box_plus": float(family["C_box"]),
        "C_sep_plus": float(family["C_sep_general"]),
        "C_rank_plus": float(family["C_rank"]),
        "C_pair_plus": float(family["C_pair"]),
        "C_block5_plus_independent": float(family["C_block5_independent"]),
        "C_strong_plus_independent": float(family["C_strong_independent"]),
        "C_block5_plus": float(family["C_block5"]),
        "C_strong_plus": float(family["C_strong"]),
        "tilde_box_original": normalize_certificate(family["C_box_original"], scale_u),
        "tilde_strong_original": normalize_certificate(decision["C_strong"], scale_u),
        "tilde_box_plus": normalize_certificate(family["C_box"], scale_u),
        "tilde_strong_plus": normalize_certificate(family["C_strong"], scale_u),
        "D_feas_cpl": box_gap,
        "D_feas_strong": strong_gap,
        "D_cpl_original": float(decision["C_strong"] - family["C_box_original"]),
        "D_cpl_plus": float(family["C_strong"] - family["C_box"]),
        "strict_feas": bool(exact and box_gap > GAP_ATOL),
        "n_assignment_solves": int(family["n_assignment_solves"]),
        "runtime_original_s": float(family["runtime_original_s"]),
        "runtime_plus_s": float(family["runtime_plus_s"]),
        "k5_identity_applied": float(family["k5_identity_applied"]),
    }


def _attach_event(row: dict, prefix: str, event: dict) -> None:
    for key, value in event.items():
        row[f"{prefix}_{key}"] = value


def _ledger() -> dict[str, dict]:
    if not LEDGER.exists():
        return {}
    out = {}
    with LEDGER.open() as handle:
        for line in handle:
            item = json.loads(line)
            out[item["release_id"]] = item
    return out


def build_principal(ledger: dict[str, dict]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    mismatches = []
    peak = 0
    for regime in REGIMES:
        for n in (8, 261, 500, 2000):
            truth_id = f"synthetic|{regime}|n{n}"
            latent = synthetic_truth(regime, n)
            for mechanism in MECHS:
                for epsilon in (0.5, 1.0, 2.0):
                    t_union, t_direct, _cov_u, _cov_d = _radii(mechanism, n, epsilon)
                    for rep in range(REPS[n]):
                        seed = derive_release_seed(
                            MASTER,
                            SeedKey(
                                city=truth_id,
                                split=SPLIT,
                                mechanism=mechanism,
                                epsilon=float(epsilon),
                                replicate_id=rep,
                            ),
                        )
                        y = _release(latent, mechanism, float(epsilon), int(seed))
                        for k in KS:
                            if k > n:
                                continue
                            for family_name in WEIGHTS:
                                weights = weight_family(family_name, k)
                                release_id = (
                                    f"{truth_id}|{mechanism}|eps{epsilon}|rep{rep}|k{k}|{family_name}"
                                )
                                trace = n == 2000 and k == 25 and family_name == "dcg" and rep == 0 and mechanism == "M1"
                                if trace:
                                    tracemalloc.start()
                                union = _one_event(y, latent, k, weights, t_union)
                                direct = _one_event(y, latent, k, weights, t_direct)
                                if trace:
                                    _current, traced = tracemalloc.get_traced_memory()
                                    tracemalloc.stop()
                                    peak = max(peak, int(traced))
                                stored = ledger.get(release_id)
                                if stored is not None:
                                    mapping = {
                                        "C_box": "C_box_original",
                                        "C_sep_general": "C_sep_original",
                                        "C_rank": "C_rank_original",
                                        "C_pair": "C_pair_original",
                                        "C_block5": "C_block5_original_independent",
                                        "C_strong": "C_strong_original_independent",
                                    }
                                    for event_name, event in (("union", union), ("direct", direct)):
                                        for key, local in mapping.items():
                                            got = float(event[local])
                                            old = float(stored[f"{key}_{event_name}"])
                                            if got != old:
                                                mismatches.append(abs(got - old))
                                row = {
                                    "panel": "principal",
                                    "release_id": release_id,
                                    "regime": regime,
                                    "n": n,
                                    "k": k,
                                    "mechanism": mechanism,
                                    "epsilon": float(epsilon),
                                    "weight_family": family_name,
                                    "seed": int(seed),
                                    "t_union": float(t_union),
                                    "t_direct": float(t_direct),
                                }
                                _attach_event(row, "union", union)
                                _attach_event(row, "direct", direct)
                                rows.append(row)
    return rows, {"ledger_mismatches": mismatches, "peak_traced_bytes": peak}


def build_stress() -> list[dict]:
    rows = []
    for n in LOW_NS:
        latent = low_count_truth(n)
        truth_id = f"synthetic|low_count_floor|n{n}"
        ks = (5,) if n == 8 else (5, 10, 25)
        for mechanism in MECHS:
            for epsilon in (0.5, 1.0, 2.0):
                t_union, t_direct, _a, _b = _radii(mechanism, n, epsilon)
                for rep in range(LOW_REPS):
                    seed = derive_release_seed(
                        LOW_MASTER,
                        SeedKey(
                            city=truth_id,
                            split=LOW_SPLIT,
                            mechanism=mechanism,
                            epsilon=float(epsilon),
                            replicate_id=rep,
                        ),
                    )
                    y = _release(latent, mechanism, float(epsilon), int(seed))
                    for k in ks:
                        for family_name in WEIGHTS:
                            weights = weight_family(family_name, k)
                            union = _one_event(y, latent, k, weights, t_union)
                            direct = _one_event(y, latent, k, weights, t_direct)
                            row = {
                                "panel": "low_count_stress",
                                "release_id": (
                                    f"{truth_id}|{mechanism}|eps{epsilon}|rep{rep}|k{k}|{family_name}"
                                ),
                                "regime": "low_count_floor",
                                "n": n,
                                "k": k,
                                "mechanism": mechanism,
                                "epsilon": float(epsilon),
                                "weight_family": family_name,
                                "seed": int(seed),
                                "t_union": float(t_union),
                                "t_direct": float(t_direct),
                            }
                            _attach_event(row, "union", union)
                            _attach_event(row, "direct", direct)
                            rows.append(row)
    return rows


def _event_summary(rows: list[dict], event: str) -> dict:
    def col(name: str) -> list[float]:
        return [float(row[f"{event}_{name}"]) for row in rows]

    strict = [bool(row[f"{event}_strict_feas"]) for row in rows]
    fallback = [bool(row[f"{event}_fallback"]) for row in rows]
    consistent = [row for row in rows if row[f"{event}_consistent"]]
    zero_gain = [row for row in consistent if row[f"{event}_zero_gain_condition"]]
    active = [row for row in consistent if not row[f"{event}_zero_gain_condition"]]
    feas = col("D_feas_cpl")
    feas_pos = [float(row[f"{event}_D_feas_cpl"]) for row in rows if row[f"{event}_C_box_original"] > GAP_ATOL]
    rel = [gap / raw for gap, raw in zip(feas_pos, [float(r[f"{event}_C_box_original"]) for r in rows if r[f"{event}_C_box_original"] > GAP_ATOL])]
    tilde_gap = [
        float(row[f"{event}_tilde_box_original"] - row[f"{event}_tilde_box_plus"]) for row in rows
    ]
    dominance = [
        float(row[f"{event}_C_box_plus"] - row[f"{event}_C_box_original"])
        for row in rows
        if row[f"{event}_consistent"]
    ]
    relax = []
    for row in consistent:
        box = float(row[f"{event}_C_box_plus"])
        for key in ("C_sep_plus", "C_rank_plus", "C_pair_plus", "C_block5_plus_independent"):
            relax.append(box - float(row[f"{event}_{key}"]))
    k5 = [row for row in rows if row["k"] <= 5]
    k5_cpl = [abs(float(row[f"{event}_D_cpl_plus"])) for row in k5]
    def box_plus_decision(row: dict) -> float:
        # The zero-gain condition makes every competitor support identical to
        # the original box, so the coupled values are the same optimization.
        if row[f"{event}_zero_gain_condition"] or row[f"{event}_fallback"]:
            return float(row[f"{event}_tilde_box_original"])
        return float(row[f"{event}_tilde_box_plus"])

    def strong_plus_decision(row: dict) -> float:
        if row[f"{event}_zero_gain_condition"] or row[f"{event}_fallback"]:
            return float(row[f"{event}_tilde_strong_original"])
        return float(row[f"{event}_tilde_strong_plus"])

    taus = {}
    for tau in TAUS:
        k_box_o = float(np.mean([float(row[f"{event}_tilde_box_original"]) <= tau for row in rows]))
        k_str_o = float(np.mean([float(row[f"{event}_tilde_strong_original"]) <= tau for row in rows]))
        k_box_p = float(np.mean([box_plus_decision(row) <= tau for row in rows]))
        k_str_p = float(np.mean([strong_plus_decision(row) <= tau for row in rows]))
        crossings = int(
            sum(
                float(row[f"{event}_tilde_box_original"]) > tau
                and box_plus_decision(row) <= tau
                and row[f"{event}_strict_feas"]
                for row in rows
            )
        )
        taus[str(tau)] = {
            "K_box_original": k_box_o,
            "K_strong_original": k_str_o,
            "K_box_plus": k_box_p,
            "K_strong_plus": k_str_p,
            "Gain_original": k_box_o - k_str_o,
            "Gain_plus": k_box_p - k_str_p,
            "feasibility_rate_gain": k_box_p - k_box_o,
            "crossings_box": crossings,
            "still_above": int(sum(box_plus_decision(row) > tau for row in rows)),
        }
    return {
        "n": len(rows),
        "fallback_count": int(sum(fallback)),
        "fallback_fraction": float(np.mean(fallback)) if rows else float("nan"),
        "strict_fraction": float(np.mean(strict)) if rows else float("nan"),
        "strict_fraction_among_consistent": float(np.mean([row[f"{event}_strict_feas"] for row in consistent])) if consistent else float("nan"),
        "strict_fraction_zero_gain_regime": float(np.mean([row[f"{event}_strict_feas"] for row in zero_gain])) if zero_gain else float("nan"),
        "strict_fraction_below_regime": float(np.mean([row[f"{event}_strict_feas"] for row in active])) if active else float("nan"),
        "n_zero_gain_condition": len(zero_gain),
        "n_below_condition": len(active),
        "max_abs_gap_on_zero_gain_condition": float(np.max(np.abs([row[f"{event}_D_feas_cpl"] for row in zero_gain]))) if zero_gain else 0.0,
        "mean_D_feas_cpl": _mean(feas),
        "median_D_feas_cpl": _percentile(feas, 50),
        "max_D_feas_cpl": float(np.max(feas)) if feas else float("nan"),
        "min_D_feas_cpl": float(np.min(feas)) if feas else float("nan"),
        "median_relative_reduction_when_positive": _percentile(rel, 50),
        "mean_normalized_reduction": _mean(tilde_gap),
        "median_normalized_reduction": _percentile(tilde_gap, 50),
        "max_plus_minus_original": float(np.max(dominance)) if dominance else 0.0,
        "max_box_minus_relaxation": float(np.max(relax)) if relax else 0.0,
        "max_abs_k5_coupling_gap": float(np.max(k5_cpl)) if k5_cpl else 0.0,
        "mean_D_cpl_original": _mean(col("D_cpl_original")),
        "mean_D_cpl_plus": _mean(col("D_cpl_plus")),
        "mean_interaction": _mean(
            [float(row[f"{event}_D_cpl_plus"] - row[f"{event}_D_cpl_original"]) for row in rows]
        ),
        "taus": taus,
    }


def _ordering(rows: list[dict]) -> dict:
    both = []
    fallback_pairs = 0
    for row in rows:
        if row["union_consistent"] and row["direct_consistent"]:
            both.append(float(row["direct_C_box_plus"] - row["union_C_box_plus"]))
        else:
            fallback_pairs += 1
    return {
        "both_exact": len(both),
        "fallback_pairs": fallback_pairs,
        "max_direct_minus_union": float(np.max(both)) if both else 0.0,
    }


def summarize(principal: list[dict], stress: list[dict], meta: dict) -> dict:
    def by(rows: list[dict], key: str, event: str) -> dict:
        groups: dict[str, list[dict]] = {}
        for row in rows:
            groups.setdefault(str(row[key]), []).append(row)
        return {
            name: {
                "n": len(members),
                "strict_fraction": float(np.mean([m[f"{event}_strict_feas"] for m in members])),
                "fallback_fraction": float(np.mean([m[f"{event}_fallback"] for m in members])),
                "median_D_feas": _percentile([float(m[f"{event}_D_feas_cpl"]) for m in members], 50),
            }
            for name, members in sorted(groups.items())
        }

    runtime = {}
    for n in (8, 261, 500, 2000):
        for k in (5, 10, 25):
            subset = [row for row in principal if row["n"] == n and row["k"] == k]
            if not subset:
                continue
            runtime[f"n{n}_k{k}"] = {
                "n": len(subset),
                "median_original_s": _percentile([row["union_runtime_original_s"] for row in subset], 50),
                "p90_original_s": _percentile([row["union_runtime_original_s"] for row in subset], 90),
                "p95_original_s": _percentile([row["union_runtime_original_s"] for row in subset], 95),
                "median_plus_s": _percentile([row["union_runtime_plus_s"] for row in subset], 50),
                "p90_plus_s": _percentile([row["union_runtime_plus_s"] for row in subset], 90),
                "p95_plus_s": _percentile([row["union_runtime_plus_s"] for row in subset], 95),
                "median_solves": _percentile([float(row["union_n_assignment_solves"]) for row in subset], 50),
            }
    mismatches = meta["ledger_mismatches"]
    return {
        "principal": {
            "union": _event_summary(principal, "union"),
            "direct": _event_summary(principal, "direct"),
            "ordering": _ordering(principal),
            "by_regime_union": by(principal, "regime", "union"),
            "by_mechanism_union": by(principal, "mechanism", "union"),
            "by_epsilon_union": by(principal, "epsilon", "union"),
            "by_k_union": by(principal, "k", "union"),
            "by_weight_union": by(principal, "weight_family", "union"),
            "by_n_union": by(principal, "n", "union"),
        },
        "stress": {
            "union": _event_summary(stress, "union"),
            "direct": _event_summary(stress, "direct"),
            "ordering": _ordering(stress),
        },
        "runtime_union_principal": runtime,
        "ledger_comparisons": len(principal) * 2 * 6,
        "ledger_nonzero_mismatches": len(mismatches),
        "ledger_max_abs_mismatch": float(max(mismatches)) if mismatches else 0.0,
        "peak_traced_bytes": meta["peak_traced_bytes"],
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    ledger = _ledger()
    principal, meta = build_principal(ledger)
    stress = build_stress()
    summary = summarize(principal, stress, meta)
    summary["elapsed_s"] = time.perf_counter() - started
    summary["n_principal"] = len(principal)
    summary["n_stress"] = len(stress)
    with (OUT / "principal_releases.jsonl").open("w") as handle:
        for row in principal:
            handle.write(json.dumps(row) + "\n")
    with (OUT / "low_count_stress.jsonl").open("w") as handle:
        for row in stress:
            handle.write(json.dumps(row) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "n_principal": summary["n_principal"],
        "n_stress": summary["n_stress"],
        "ledger_max_abs_mismatch": summary["ledger_max_abs_mismatch"],
        "union_strict": summary["principal"]["union"]["strict_fraction"],
        "union_fallback": summary["principal"]["union"]["fallback_count"],
        "direct_strict": summary["principal"]["direct"]["strict_fraction"],
        "direct_fallback": summary["principal"]["direct"]["fallback_count"],
        "max_plus_minus_original_union": summary["principal"]["union"]["max_plus_minus_original"],
        "max_box_minus_relaxation_union": summary["principal"]["union"]["max_box_minus_relaxation"],
        "ordering": summary["principal"]["ordering"],
        "stress_union_strict": summary["stress"]["union"]["strict_fraction"],
        "stress_union_fallback": summary["stress"]["union"]["fallback_count"],
        "elapsed_s": summary["elapsed_s"],
    }, indent=2))


if __name__ == "__main__":
    main()
