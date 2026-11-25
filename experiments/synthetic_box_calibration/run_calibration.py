#!/usr/bin/env python3
"""Synthetic diagnostic for UNION_BOUND_BOX versus DIRECT_JOINT_BOX.

Every count vector is generated here; no public dataset is read. The
canonical decision-level summary is computed by run_canonical.py.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from private_mobility_cert.certificates.certified_risk_frontier import (  # noqa: E402
    STAB,
    normalize_certificate,
    release_scale_U,
)
from private_mobility_cert.certificates.direct_box_calibration import (  # noqa: E402
    direct_box_coverage_geometric,
    direct_box_coverage_laplace,
    direct_box_radius_geometric,
    direct_box_radius_laplace,
    matched_box_certificates,
    union_box_radius_laplace,
)
from private_mobility_cert.certificates.release_conditioned import (  # noqa: E402
    deploy_ranking,
    exact_weighted_regret,
)
from private_mobility_cert.certificates.subset_swap import (  # noqa: E402
    coordinate_box_radius_geometric,
    coordinate_box_radius_laplace,
)
from private_mobility_cert.certificates.weighted_ranking import weight_family  # noqa: E402
from private_mobility_cert.privacy.geometric import release_geometric_histogram  # noqa: E402
from private_mobility_cert.privacy.laplace import release_laplace_histogram  # noqa: E402
from private_mobility_cert.privacy.seeds import SeedKey, derive_release_seed  # noqa: E402

OUT = Path(__file__).resolve().parent
MASTER = 20261124
# Opaque seed salt: its exact bytes enter the SHA-256 seed derivation.
SPLIT = "g2_synthetic_direct_box"
ALPHA = 0.05
DELTA = 1.0
TOL = 1e-8
TAUS = (0.05, 0.10, 0.25, 0.50, 1.0)
REGIMES = ("separated", "close", "ties", "sparse", "heavy_tail")
NS = (8, 261, 500, 2000)
REPS = {8: 4, 261: 4, 500: 4, 2000: 2}
EPSILONS = (0.5, 1.0, 2.0)
KS = (5, 10, 25)
WEIGHTS = ("uniform", "dcg", "reciprocal")
MECHS = ("M1", "M2")
CERT_KEYS = ("C_box", "C_sep_general", "C_rank", "C_pair", "C_block5", "C_strong")


def synthetic_truth(regime: str, n: int) -> np.ndarray:
    """Deterministic nonnegative integer counts. Not a public dataset."""
    x = np.zeros(int(n), dtype=np.int64)
    if regime == "separated":
        for i in range(min(n, 40)):
            x[i] = 2000 - 40 * i
        if n > 40:
            x[40:] = 5
    elif regime == "close":
        for i in range(min(n, 60)):
            x[i] = 80 - i
        if n > 60:
            x[60:] = 10
    elif regime == "ties":
        x[: min(n, 15)] = 40
        if n > 15:
            x[15 : min(n, 40)] = 25
        if n > 40:
            x[40:] = 5
    elif regime == "sparse":
        step = max(n // 12, 1)
        x[::step] = 3
        x[0] = 4
    elif regime == "heavy_tail":
        idx = np.arange(1, n + 1, dtype=np.float64)
        x = np.floor(5000.0 / np.power(idx, 1.2)).astype(np.int64)
    else:
        raise ValueError(regime)
    if np.any(x < 0):
        raise RuntimeError("synthetic truth must be nonnegative")
    return x


def _release(x: np.ndarray, mechanism: str, epsilon: float, seed: int) -> np.ndarray:
    if mechanism == "M1":
        return np.asarray(
            release_laplace_histogram(x, epsilon=epsilon, seed=seed).y_mech,
            dtype=np.float64,
        )
    if mechanism == "M2":
        return np.asarray(
            release_geometric_histogram(x, epsilon=epsilon, seed=seed).y_mech,
            dtype=np.float64,
        )
    raise ValueError(mechanism)


def _radii(mechanism: str, n: int, epsilon: float) -> tuple[float, float, float, float]:
    if mechanism == "M1":
        t_union = union_box_radius_laplace(n, epsilon, ALPHA, delta=DELTA)
        t_direct = direct_box_radius_laplace(n, epsilon, ALPHA, delta=DELTA)
        # The union radius must equal the unit-sensitivity coordinate radius.
        reference = coordinate_box_radius_laplace(n, epsilon, ALPHA)
        if t_union != reference:
            raise RuntimeError("union Laplace helper diverged from the coordinate box radius")
        cov_u = direct_box_coverage_laplace(n, t_union, epsilon, delta=DELTA)
        cov_d = direct_box_coverage_laplace(n, t_direct, epsilon, delta=DELTA)
        return float(t_union), float(t_direct), float(cov_u), float(cov_d)
    t_union = float(coordinate_box_radius_geometric(n, epsilon, ALPHA))
    t_direct = float(direct_box_radius_geometric(n, epsilon, ALPHA))
    cov_u = direct_box_coverage_geometric(n, int(t_union), epsilon)
    cov_d = direct_box_coverage_geometric(n, int(t_direct), epsilon)
    return t_union, t_direct, float(cov_u), float(cov_d)


def _check_row(row: dict) -> None:
    for event in ("union", "direct"):
        box = row[f"C_box_{event}"]
        for key in ("C_sep_general", "C_rank", "C_pair", "C_block5", "C_strong"):
            if box > row[f"{key}_{event}"] + 1e-7:
                raise RuntimeError(f"same-event dominance failed: {row['release_id']} {key}")
        if row["k"] <= 5 and abs(row[f"C_block5_{event}"] - box) > 1e-7:
            raise RuntimeError(f"block-5 control failed: {row['release_id']}")
    if row["C_box_direct"] > row["C_box_union"] + 1e-7:
        raise RuntimeError(f"direct coupled certificate exceeds union: {row['release_id']}")


def build_rows() -> list[dict]:
    rows: list[dict] = []
    for regime in REGIMES:
        for n in NS:
            truth_id = f"synthetic|{regime}|n{n}"
            x = synthetic_truth(regime, n)
            for mechanism in MECHS:
                for epsilon in EPSILONS:
                    t_union, t_direct, cov_u, cov_d = _radii(mechanism, n, epsilon)
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
                        y = _release(x, mechanism, float(epsilon), int(seed))
                        for k in KS:
                            if k > n:
                                continue
                            for family in WEIGHTS:
                                weights = weight_family(family, k)
                                started = time.perf_counter()
                                union = matched_box_certificates(
                                    y, k=k, weights=weights, half_width=t_union
                                )
                                direct = matched_box_certificates(
                                    y, k=k, weights=weights, half_width=t_direct
                                )
                                runtime = time.perf_counter() - started
                                pi = deploy_ranking(y, k)
                                scale = release_scale_U(y, pi, weights)
                                s_value = max(float(scale), STAB)
                                regret = float(exact_weighted_regret(x, pi, weights))
                                row = {
                                    "truth_id": truth_id,
                                    "regime": regime,
                                    "n": int(n),
                                    "mechanism": mechanism,
                                    "epsilon": float(epsilon),
                                    "alpha": ALPHA,
                                    "delta": DELTA,
                                    "k": int(k),
                                    "weight_family": family,
                                    "replicate": int(rep),
                                    "seed": int(seed),
                                    "master_seed": MASTER,
                                    "split": SPLIT,
                                    "t_union": t_union,
                                    "t_direct": t_direct,
                                    "coverage_union": cov_u,
                                    "coverage_direct": cov_d,
                                    "S": s_value,
                                    "U": float(scale),
                                    "true_regret": regret,
                                    "runtime_s": float(runtime),
                                    "ranking_tie_break": "stable_yplus_smaller_index",
                                }
                                for key in CERT_KEYS:
                                    row[f"{key}_union"] = float(union[key])
                                    row[f"{key}_direct"] = float(direct[key])
                                    row[f"{key}_tilde_union"] = float(
                                        normalize_certificate(union[key], s_value)
                                    )
                                    row[f"{key}_tilde_direct"] = float(
                                        normalize_certificate(direct[key], s_value)
                                    )
                                row["D_cal_cpl"] = row["C_box_union"] - row["C_box_direct"]
                                row["D_cal_strong"] = row["C_strong_union"] - row["C_strong_direct"]
                                row["D_cpl_union"] = row["C_strong_union"] - row["C_box_union"]
                                row["D_cpl_direct"] = row["C_strong_direct"] - row["C_box_direct"]
                                row["D_interaction"] = row["D_cpl_direct"] - row["D_cpl_union"]
                                row["D_cal_cpl_tilde"] = (
                                    row["C_box_tilde_union"] - row["C_box_tilde_direct"]
                                )
                                row["D_cal_strong_tilde"] = (
                                    row["C_strong_tilde_union"] - row["C_strong_tilde_direct"]
                                )
                                row["D_cpl_union_tilde"] = (
                                    row["C_strong_tilde_union"] - row["C_box_tilde_union"]
                                )
                                row["D_cpl_direct_tilde"] = (
                                    row["C_strong_tilde_direct"] - row["C_box_tilde_direct"]
                                )
                                row["D_interaction_tilde"] = (
                                    row["D_cpl_direct_tilde"] - row["D_cpl_union_tilde"]
                                )
                                row["release_id"] = (
                                    f"{truth_id}|{mechanism}|eps{epsilon}|rep{rep}|k{k}|{family}"
                                )
                                _check_row(row)
                                rows.append(row)
                                if len(rows) % 200 == 0:
                                    print(f"rows={len(rows)}", flush=True)
    return rows


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float(0.5 * (ordered[mid - 1] + ordered[mid]))


def _k_rate(rows: list[dict], key: str, tau: float) -> float:
    if not rows:
        return float("nan")
    hits = sum(float(row[key]) <= tau for row in rows)
    return float(hits / len(rows))


def pool_stats(rows: list[dict]) -> dict:
    def col(name: str) -> list[float]:
        return [float(row[name]) for row in rows]

    cal = col("D_cal_cpl")
    out = {
        "n_rows": len(rows),
        "mean_D_cal_cpl": _mean(cal),
        "median_D_cal_cpl": _median(cal),
        "max_D_cal_cpl": max(cal) if cal else float("nan"),
        "min_D_cal_cpl": min(cal) if cal else float("nan"),
        "fraction_strict_cpl_tightening": float(sum(v > TOL for v in cal) / len(rows)),
        "fraction_zero_cpl_calibration": float(sum(abs(v) <= TOL for v in cal) / len(rows)),
        "mean_D_cal_strong": _mean(col("D_cal_strong")),
        "mean_D_cpl_union": _mean(col("D_cpl_union")),
        "mean_D_cpl_direct": _mean(col("D_cpl_direct")),
        "mean_D_interaction": _mean(col("D_interaction")),
        "mean_D_cal_cpl_tilde": _mean(col("D_cal_cpl_tilde")),
        "mean_D_cal_strong_tilde": _mean(col("D_cal_strong_tilde")),
        "mean_D_cpl_union_tilde": _mean(col("D_cpl_union_tilde")),
        "mean_D_cpl_direct_tilde": _mean(col("D_cpl_direct_tilde")),
        "mean_D_interaction_tilde": _mean(col("D_interaction_tilde")),
        "fraction_strict_tilde_tightening": float(
            sum(v > TOL for v in col("D_cal_cpl_tilde")) / len(rows)
        ),
        "taus": {},
    }
    for tau in TAUS:
        k_box_u = _k_rate(rows, "C_box_tilde_union", tau)
        k_str_u = _k_rate(rows, "C_strong_tilde_union", tau)
        k_box_d = _k_rate(rows, "C_box_tilde_direct", tau)
        k_str_d = _k_rate(rows, "C_strong_tilde_direct", tau)
        changed = sum(
            (row["C_box_tilde_union"] <= tau) != (row["C_box_tilde_direct"] <= tau)
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


def one_way(rows: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    return {name: pool_stats(members) for name, members in sorted(groups.items())}


def calibration_grid() -> list[dict]:
    grid = []
    for n in (1, 2, 8, 25, 50, 261, 500, 2000):
        for epsilon in EPSILONS:
            for alpha in (0.01, 0.05, 0.10):
                for delta in (1.0, 3.0, 5.0):
                    direct = direct_box_radius_laplace(n, epsilon, alpha, delta=delta)
                    union = union_box_radius_laplace(n, epsilon, alpha, delta=delta)
                    grid.append(
                        {
                            "mechanism": "M1",
                            "n": n,
                            "epsilon": epsilon,
                            "alpha": alpha,
                            "delta": delta,
                            "t_direct": direct,
                            "t_union": union,
                            "coverage_direct": direct_box_coverage_laplace(
                                n, direct, epsilon, delta=delta
                            ),
                            "coverage_union": direct_box_coverage_laplace(
                                n, union, epsilon, delta=delta
                            ),
                        }
                    )
                g_direct = direct_box_radius_geometric(n, epsilon, alpha)
                g_union = coordinate_box_radius_geometric(n, epsilon, alpha)
                grid.append(
                    {
                        "mechanism": "M2",
                        "n": n,
                        "epsilon": epsilon,
                        "alpha": alpha,
                        "delta": 1.0,
                        "t_direct": g_direct,
                        "t_union": g_union,
                        "coverage_direct": direct_box_coverage_geometric(n, g_direct, epsilon),
                        "coverage_union": direct_box_coverage_geometric(n, int(g_union), epsilon),
                        "note": "unit-sensitivity discrete Laplace; not a B-scaled sampler",
                    }
                )
    return grid


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = build_rows()
    elapsed = time.perf_counter() - started
    k5 = [row for row in rows if row["k"] <= 5]
    block_gap = max(
        max(abs(row["C_block5_union"] - row["C_box_union"]), abs(row["C_block5_direct"] - row["C_box_direct"]))
        for row in k5
    )
    dominance_gap = 0.0
    for row in rows:
        for event in ("union", "direct"):
            box = row[f"C_box_{event}"]
            for key in ("C_sep_general", "C_rank", "C_pair", "C_block5", "C_strong"):
                dominance_gap = max(dominance_gap, box - row[f"{key}_{event}"])
    direct_vs_union = max(row["C_box_direct"] - row["C_box_union"] for row in rows)
    summary = {
        "experiment": "synthetic_box_calibration",
        "public_datasets_used": False,
        "master_seed": MASTER,
        "split": SPLIT,
        "alpha": ALPHA,
        "delta": DELTA,
        "tolerance": TOL,
        "stab": STAB,
        "elapsed_s": elapsed,
        "n_rows": len(rows),
        "n_truth_vectors": len({row["truth_id"] for row in rows}),
        "n_releases": len({(row["truth_id"], row["mechanism"], row["epsilon"], row["replicate"]) for row in rows}),
        "max_runtime_s": max(row["runtime_s"] for row in rows),
        "max_same_event_dominance_violation": dominance_gap,
        "max_direct_minus_union_coupled": direct_vs_union,
        "max_block5_abs_gap_k_le_5": block_gap,
        "overall": pool_stats(rows),
        "by_mechanism": one_way(rows, "mechanism"),
        "by_epsilon": one_way(rows, "epsilon"),
        "by_k": one_way(rows, "k"),
        "by_weight_family": one_way(rows, "weight_family"),
        "by_regime": one_way(rows, "regime"),
        "by_mechanism_epsilon": {},
    }
    mech_eps: dict[tuple[str, float], list[dict]] = {}
    for row in rows:
        mech_eps.setdefault((row["mechanism"], row["epsilon"]), []).append(row)
    for (mechanism, epsilon), members in sorted(mech_eps.items()):
        summary["by_mechanism_epsilon"][f"{mechanism}|eps{epsilon}"] = pool_stats(members)
    ledger = OUT / "synthetic_releases.jsonl"
    with ledger.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (OUT / "calibration_grid.json").write_text(
        json.dumps(calibration_grid(), indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "n_rows": summary["n_rows"],
        "elapsed_s": elapsed,
        "max_dominance": dominance_gap,
        "max_direct_minus_union": direct_vs_union,
        "max_block5_gap": block_gap,
        "mean_D_cal_cpl": summary["overall"]["mean_D_cal_cpl"],
        "mean_D_interaction": summary["overall"]["mean_D_interaction"],
    }, indent=2))


if __name__ == "__main__":
    main()
