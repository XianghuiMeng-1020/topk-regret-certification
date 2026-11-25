#!/usr/bin/env python3
"""Synthetic order-statistic envelope diagnostic. Reuses the synthetic box
calibration release generator.

UNION_BOUND_BOX and DIRECT_JOINT_BOX are
recorded as separate events and are not minimized with the OS envelope.
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
from private_mobility_cert.certificates.direct_box_calibration import (  # noqa: E402
    direct_box_radius_geometric,
    direct_box_radius_laplace,
    union_box_radius_laplace,
)
from private_mobility_cert.certificates.os_envelope import (  # noqa: E402
    certificate_os_envelope,
    prefix_min_thresholds,
)
from private_mobility_cert.certificates.rank_adaptive import (  # noqa: E402
    binomial_sf,
    geometric_positive_tail_strict,
    laplace_positive_tail,
)
from private_mobility_cert.certificates.release_conditioned import (  # noqa: E402
    build_threshold_table,
    certificate_l1_assignment,
    exact_weighted_regret,
)
from private_mobility_cert.certificates.subset_swap import (  # noqa: E402
    coordinate_box_radius_geometric,
)
from private_mobility_cert.certificates.weighted_ranking import (  # noqa: E402
    allocate_eta_weight_proportional,
    weight_family,
)
from private_mobility_cert.privacy.seeds import SeedKey, derive_release_seed  # noqa: E402
from run_calibration import (  # noqa: E402
    ALPHA,
    MASTER,
    SPLIT,
    _release,
    synthetic_truth,
)

OUT = Path(__file__).resolve().parent
NS = (261, 500, 2000)
REPS = {261: 4, 500: 4, 2000: 2}
EPSILONS = (0.5, 1.0, 2.0)
KS = (5, 10, 25)
WEIGHTS = ("uniform", "dcg", "reciprocal")
MECHS = ("M1", "M2")
REGIMES = ("separated", "close", "ties", "sparse", "heavy_tail")
TAUS = (0.05, 0.10, 0.25, 0.50)
GAP_ATOL = 1e-8


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _radii(mechanism: str, n: int, epsilon: float) -> tuple[float, float]:
    if mechanism == "M1":
        return (
            float(union_box_radius_laplace(n, epsilon, ALPHA, delta=1.0)),
            float(direct_box_radius_laplace(n, epsilon, ALPHA, delta=1.0)),
        )
    return (
        float(coordinate_box_radius_geometric(n, epsilon, ALPHA)),
        float(direct_box_radius_geometric(n, epsilon, ALPHA)),
    )


def _threshold_audit(table, weights: np.ndarray) -> dict:
    eta_plus, eta_minus = allocate_eta_weight_proportional(weights, table.alpha)
    slacks = []
    for rank, eta in enumerate(eta_plus, start=1):
        if table.mechanism.startswith("M1") or "laplace" in table.mechanism.lower():
            tail = laplace_positive_tail(float(table.u_plus[rank - 1]), table.epsilon)
        else:
            tail = geometric_positive_tail_strict(int(table.u_plus[rank - 1]), table.epsilon)
        slacks.append(float(binomial_sf(rank, table.n, tail) - float(eta)))
    upper = np.asarray(table.u_plus, dtype=np.float64)
    lower = np.asarray(table.u_minus, dtype=np.float64)
    return {
        "eta_sum": float(eta_plus.sum() + eta_minus.sum()),
        "max_sf_minus_eta": float(max(slacks) if slacks else 0.0),
        "monotone_upper": bool(np.all(np.diff(upper) <= 1e-12)),
        "monotone_lower": bool(np.all(np.diff(lower) <= 1e-12)),
        "prefix_changes_upper": bool(
            not np.array_equal(prefix_min_thresholds(upper), upper)
        ),
    }


def build_rows() -> tuple[list[dict], dict]:
    rows: list[dict] = []
    stage = {
        "threshold_s": [],
        "rank1_s": [],
        "envelope_s": [],
        "union_s": [],
        "direct_s": [],
    }
    peak_bytes = 0
    for regime in REGIMES:
        for n in NS:
            truth_id = f"synthetic|{regime}|n{n}"
            latent = synthetic_truth(regime, n)
            for mechanism in MECHS:
                for epsilon in EPSILONS:
                    t_union, t_direct = _radii(mechanism, n, epsilon)
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
                            for family in WEIGHTS:
                                weights = weight_family(family, k)
                                t0 = time.perf_counter()
                                table = build_threshold_table(
                                    mechanism="M1_flat_laplace" if mechanism == "M1" else "M2_flat_geometric",
                                    n=n,
                                    k_max=k,
                                    epsilon=float(epsilon),
                                    alpha=ALPHA,
                                    weights=weights,
                                )
                                stage["threshold_s"].append(time.perf_counter() - t0)
                                audit = _threshold_audit(table, weights)
                                t0 = time.perf_counter()
                                rank1 = certificate_l1_assignment(
                                    y, k=k, weights=weights, half_width=0.5 * table.range_width
                                )
                                stage["rank1_s"].append(time.perf_counter() - t0)
                                trace = n == 2000 and k == 25 and family == "dcg" and rep == 0
                                if trace:
                                    tracemalloc.start()
                                t0 = time.perf_counter()
                                envelope = certificate_os_envelope(
                                    y, k=k, weights=weights, table=table
                                )
                                stage["envelope_s"].append(time.perf_counter() - t0)
                                if trace:
                                    _current, traced_peak = tracemalloc.get_traced_memory()
                                    tracemalloc.stop()
                                    peak_bytes = max(peak_bytes, int(traced_peak))
                                t0 = time.perf_counter()
                                union = certificate_l1_assignment(
                                    y, k=k, weights=weights, half_width=t_union
                                )
                                stage["union_s"].append(time.perf_counter() - t0)
                                t0 = time.perf_counter()
                                direct = certificate_l1_assignment(
                                    y, k=k, weights=weights, half_width=t_direct
                                )
                                stage["direct_s"].append(time.perf_counter() - t0)
                                scale_u = release_scale_U(y, rank1.pi_hat, weights)
                                regret = exact_weighted_regret(
                                    latent.astype(np.float64), rank1.pi_hat, weights
                                )
                                raw_gap = float(rank1.value - envelope.value)
                                rows.append(
                                    {
                                        "release_id": f"{truth_id}|{mechanism}|eps{epsilon}|rep{rep}|k{k}|{family}",
                                        "regime": regime,
                                        "n": n,
                                        "k": k,
                                        "mechanism": mechanism,
                                        "epsilon": epsilon,
                                        "weight_family": family,
                                        "seed": int(seed),
                                        "event_os": "FULL_ORDER_STATISTIC_EVENT",
                                        "threshold_scheme": table.scheme,
                                        "method_envelope": envelope.method,
                                        "C_os_rank1": rank1.value,
                                        "C_os_envelope": envelope.value,
                                        "C_union_box": union.value,
                                        "C_direct_box": direct.value,
                                        "U": scale_u,
                                        "tilde_rank1": normalize_certificate(rank1.value, scale_u),
                                        "tilde_envelope": normalize_certificate(envelope.value, scale_u),
                                        "tilde_union": normalize_certificate(union.value, scale_u),
                                        "tilde_direct": normalize_certificate(direct.value, scale_u),
                                        "latent_regret": float(regret),
                                        "chosen_s": envelope.chosen_s,
                                        "n_assignment_solves": envelope.n_assignment_solves,
                                        "delta": envelope.delta,
                                        "fallback_s1": envelope.fallback_used,
                                        "raw_gap_rank1_minus_envelope": raw_gap,
                                        "strict_tightening": bool(raw_gap > GAP_ATOL),
                                        "solver_status": "optimal",
                                        "monotone_thresholds": audit["monotone_upper"] and audit["monotone_lower"],
                                        "max_sf_minus_eta": audit["max_sf_minus_eta"],
                                        "eta_sum": audit["eta_sum"],
                                        "rank1_s": stage["rank1_s"][-1],
                                        "envelope_s": stage["envelope_s"][-1],
                                        "threshold_s": stage["threshold_s"][-1],
                                        "union_s": stage["union_s"][-1],
                                        "direct_s": stage["direct_s"][-1],
                                    }
                                )
    meta = {"peak_traced_bytes": int(peak_bytes), "stage_times": stage}
    return rows, meta


def _rate(rows: list[dict], key: str, tau: float) -> float:
    return float(np.mean([row[key] <= tau for row in rows]))


def summarize(rows: list[dict], meta: dict) -> dict:
    gaps = [row["raw_gap_rank1_minus_envelope"] for row in rows]
    genuine = [row for row in rows if row["strict_tightening"]]
    by_k = {}
    for k in KS:
        subset = [row for row in rows if row["k"] == k]
        by_k[str(k)] = {
            "n": len(subset),
            "strict_fraction": float(np.mean([row["strict_tightening"] for row in subset])),
            "median_raw_gap": float(np.median([row["raw_gap_rank1_minus_envelope"] for row in subset])),
            "fallback_fraction": float(np.mean([row["fallback_s1"] for row in subset])),
        }
    taus = {}
    for tau in TAUS:
        taus[str(tau)] = {
            "K_os_rank1": _rate(rows, "tilde_rank1", tau),
            "K_os_envelope": _rate(rows, "tilde_envelope", tau),
            "K_union_box": _rate(rows, "tilde_union", tau),
            "K_direct_box": _rate(rows, "tilde_direct", tau),
            "genuine_crossings_rank1_to_envelope": int(
                sum(
                    row["tilde_rank1"] > tau >= row["tilde_envelope"] and row["strict_tightening"]
                    for row in rows
                )
            ),
        }
    runtime = {}
    for name, values in meta["stage_times"].items():
        runtime[name] = {
            "median": _percentile(values, 50),
            "p90": _percentile(values, 90),
            "p95": _percentile(values, 95),
        }
    runtime_by_nk = {}
    for n in NS:
        for k in KS:
            subset = [row["envelope_s"] for row in rows if row["n"] == n and row["k"] == k]
            runtime_by_nk[f"n{n}_k{k}"] = {
                "median_envelope_s": _percentile(subset, 50),
                "p90_envelope_s": _percentile(subset, 90),
                "median_solves": float(
                    np.median(
                        [row["n_assignment_solves"] for row in rows if row["n"] == n and row["k"] == k]
                    )
                ),
            }
    return {
        "experiment": "synthetic_order_statistic",
        "n_rows": len(rows),
        "strict_tightening_fraction": float(np.mean([row["strict_tightening"] for row in rows])),
        "no_improvement_fraction": float(np.mean([not row["strict_tightening"] for row in rows])),
        "zero_rank1_fraction": float(np.mean([row["C_os_rank1"] <= GAP_ATOL for row in rows])),
        "zero_envelope_fraction": float(np.mean([row["C_os_envelope"] <= GAP_ATOL for row in rows])),
        "max_raw_gap": float(np.max(gaps)),
        "median_raw_gap": float(np.median(gaps)),
        "min_raw_gap": float(np.min(gaps)),
        "fallback_s1_count": int(sum(row["fallback_s1"] for row in rows)),
        "nonmonotone_count": int(sum(not row["monotone_thresholds"] for row in rows)),
        "max_sf_minus_eta": float(max(row["max_sf_minus_eta"] for row in rows)),
        "max_eta_sum_minus_alpha": float(max(row["eta_sum"] - ALPHA for row in rows)),
        "by_k": by_k,
        "taus": taus,
        "runtime": runtime,
        "runtime_by_nk": runtime_by_nk,
        "peak_traced_bytes": meta["peak_traced_bytes"],
        "ordering_violations": int(sum(row["raw_gap_rank1_minus_envelope"] < -GAP_ATOL for row in rows)),
        "genuine_tightening_rows": len(genuine),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows, meta = build_rows()
    summary = summarize(rows, meta)
    summary["elapsed_s"] = time.perf_counter() - started
    summary["python"] = sys.version
    summary["numpy"] = np.__version__
    summary["platform"] = platform.platform()
    with (OUT / "synthetic_releases.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "n_rows", "strict_tightening_fraction", "max_raw_gap", "min_raw_gap",
        "ordering_violations", "fallback_s1_count", "nonmonotone_count",
        "max_sf_minus_eta", "elapsed_s", "taus",
    )}, indent=2))


if __name__ == "__main__":
    main()
