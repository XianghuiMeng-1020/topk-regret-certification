#!/usr/bin/env python3
"""Matched-rate regret comparison on the box-event and order-statistic ledgers.

The protocol file is written before any real-data regret comparison is saved.
Selection never uses latent regret to break ties.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BOX = REPO / "experiments/real_box_evaluation/ledger.parquet"
OS_EVAL = REPO / "experiments/real_order_statistic_evaluation/ledger.parquet"
BOX_PROTOCOL = REPO / "experiments/real_box_evaluation/protocol.json"
OS_EVAL_PROTOCOL = REPO / "experiments/real_order_statistic_evaluation/protocol.json"
PANEL = REPO / "experiments/real_data_panel"
EXPECTED_LEDGER = {
    "box": "281ca9b676184d7450bbd2c8da225b94da312bb37d3c97855b1818ecc8c956cf",
    "os_eval": "eeac20c3ddb213fc9a6c302174162ae0b93bd5489b19f9f984b66265b121a72e",
}
EXPECTED_PROTOCOL = {
    "box": "a85d4177a4614036001fb76c5edeb26d78e3b4182fc709cb2d079414794caccc",
    "os_eval": "a437eb725bcc0aefcb3db42f6f5deb76f2f9a707c9020fbc8eb2a95ec873e979",
}
EXPECTED_MANIFEST = {
    "manifests/tlc_truth_manifest.jsonl": "942a6cff6abfd58e63b0d025b4a45da41d9a550ee64e2be4e3572b3dc7a9532c",
    "manifests/gharchive_truth_manifest.jsonl": "745752764c77212af41fdb53d45934a4ad1cbee44caca4aa403ae2b8ae90978f",
}
TIE_MASTER = 20261128
BOOT_SEED = 20261129
BLOCK_SEED = 20261130
TIE_SENSITIVITY_SEEDS = list(range(20261201, 20261251))
BOOT_B = 1999
COVERAGES = (("0.10", 1, 10), ("0.25", 1, 4), ("0.50", 1, 2), ("0.75", 3, 4), ("0.90", 9, 10), ("0.95", 19, 20))
GAMMAS = (0.0, 0.001, 0.01, 0.05, 0.10)
TAUS = (0.05, 0.10, 0.25, 0.50, 1.00)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def m_of(n: int, numer: int, denom: int) -> int:
    return (numer * int(n)) // denom


def tie_priority(keys: list[tuple]) -> np.ndarray:
    out = np.empty(len(keys), dtype=np.uint64)
    for i, (truth_id, mech, eps, k, wf, rep) in enumerate(keys):
        payload = f"{TIE_MASTER}|{truth_id}|{mech}|{eps}|{k}|{wf}|{rep}".encode()
        out[i] = np.uint64(int(hashlib.sha256(payload).hexdigest()[:16], 16))
    return out


def write_protocol() -> str:
    protocol = {
        "experiment": "matched_rate_utility",
        "primary_estimand": "Equal-weight macro of within-stratum mean realized normalized regret at matched cardinality, union coupled versus union strong, c=0.50",
        "sign": "D = mean regret of the structural or baseline selector minus mean regret of the coupled or refined selector. D>0 means the coupled or refined selector has lower realized regret.",
        "primary_c": {"label": "0.50", "numer": 1, "denom": 2},
        "secondary_c": [
            {"label": label, "numer": numer, "denom": denom}
            for label, numer, denom in COVERAGES if label != "0.50"
        ],
        "cardinality": "m = (numer * N) // denom within each selection group",
        "strata": "dataset x mechanism x epsilon x k x weight family",
        "aggregation": "mean within stratum, equal weight across the 54 configurations inside a dataset, equal weight across TLC and GitHub",
        "source_only_sensitivity": "separate estimand: match cardinality within a dataset and do not stratify by configuration",
        "outcome": "r = true_regret / S, S = max(U, 1e-6) from the input ledgers",
        "score": "q = C / S, lower is more favorable",
        "pairs": {
            "A_union": ["union_C_strong", "union_C_box"],
            "B_direct": ["direct_C_strong", "direct_C_box"],
            "C_os_rank1": ["C_OS_rank1_strong", "C_OS_rank1_coupled"],
            "D_os_envelope": ["C_OS_strong_envelope", "C_OS_coupled_envelope"],
            "E_coupled_refinement": ["C_OS_rank1_coupled", "C_OS_coupled_envelope"],
            "F_strong_refinement": ["C_OS_rank1_strong", "C_OS_strong_envelope"],
        },
        "pair_meaning": "Each pair is [baseline, refined_or_coupled]. D uses baseline minus refined_or_coupled.",
        "secondary_count_feasible_pairs": {
            "union_raw": ["union_C_strong_plus_decision", "union_C_box_plus"],
            "union_closed": ["union_C_strong_plus_closed", "union_C_box_plus"],
            "direct_raw": ["direct_C_strong_plus_decision", "direct_C_box_plus"],
            "direct_closed": ["direct_C_strong_plus_closed", "direct_C_box_plus"],
        },
        "gammas": list(GAMMAS),
        "gamma0": "bad outcome is r > 0",
        "tie_rule": {
            "master": TIE_MASTER,
            "payload": "{master}|{truth_id}|{mechanism}|{str(epsilon)}|{k}|{weight_family}|{replicate}",
            "epsilon_format": "Python str of the float, so 0.5, 1.0, 2.0",
            "digest": "SHA-256 first 16 hex digits as an integer; smaller is selected first",
            "order": ["q ascending", "shared tie priority ascending", "truth date order", "replicate"],
            "sensitivity_seeds": TIE_SENSITIVITY_SEEDS,
            "note": "The primary priority uses master 20261128. Sensitivity seeds replace only that master and are not the primary analysis.",
        },
        "bootstrap": {"B": BOOT_B, "truth_cluster_seed": BOOT_SEED, "circular_block_seed": BLOCK_SEED, "block_days": 7},
        "negative_control": "k<=5 canonical scores are identical, so shared-priority selections and D_regret are exactly identical",
        "frontier": "exact weighted step-function inverse; TLC cell weight 0.5/24300 and GitHub cell weight 0.5/19440",
        "margin_descriptive_rule": "nonpositive if boundary_margin<=0; within_rank1 if 0<margin<=t_rank1; above otherwise. Descriptive partition of accepted cells, not a new selection design.",
        "multiplicity": "Only c=0.50 union stratified D_regret is primary. Other c, gamma, pairs, and slices are secondary or exploratory.",
        "not_a_new_dp_guarantee": "Matched-rate cutoffs are retrospective and are not additional 1-alpha certificates.",
        "input_ledger_sha256": EXPECTED_LEDGER,
        "input_protocol_sha256": EXPECTED_PROTOCOL,
        "input_manifest_sha256": EXPECTED_MANIFEST,
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    path = OUT / "protocol.json"
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    digest = sha_file(path)
    (OUT / "protocol.sha256").write_text(digest + "\n")
    return digest


def synthetic_checks() -> None:
    q = np.array([0.0, 0.0, 1.0, 2.0])
    is_tlc = np.array([True, True, False, False])
    got = exact_frontier(q, is_tlc, (0.10, 0.25, 0.50, 0.75))
    if got["0.50"]["T"] != 0.0 or got["0.50"]["K"] != 0.5 or got["0.50"]["K_prev"] != 0.0:
        raise SystemExit(f"frontier fixture failed {got['0.50']}")
    if got["0.10"]["T"] != 0.0 or got["0.75"]["T"] != 1.0 or got["0.75"]["K_prev"] != 0.5:
        raise SystemExit(f"frontier fixture endpoints failed {got}")
    r = np.array([0.0, 10.0, 0.0, 10.0])
    qq = np.array([0.0, 1.0, 1.0, 1.0])
    expected = expected_boundary_mean(qq, r, 2)
    if abs(expected - (10.0 / 3.0)) > 1e-12:
        raise SystemExit(f"tie expectation fixture failed {expected}")
    best, worst = boundary_bounds(qq, r, 2)
    if abs(best - 0.0) > 1e-12 or abs(worst - 5.0) > 1e-12:
        raise SystemExit(f"tie bounds fixture failed {best} {worst}")
    n = 6
    q1 = np.array([0.2, 0.2, 0.4, 0.5, 0.5, 0.9])
    tie = np.array([3, 1, 0, 4, 2, 5], dtype=np.uint64)
    final = np.arange(n)
    a = select_order(q1, tie, final, 3)
    b = select_order(q1.copy(), tie.copy(), final.copy(), 3)
    if not np.array_equal(a, b):
        raise SystemExit("identical scores did not produce identical selections")
    if m_of(450, 19, 20) != 427 or m_of(360, 1, 2) != 180:
        raise SystemExit("cardinality arithmetic failed")


def exact_frontier(q: np.ndarray, is_tlc: np.ndarray, targets: tuple[float, ...]) -> dict:
    n_tlc = int(np.sum(is_tlc))
    n_gh = int(len(q) - n_tlc)
    if n_tlc == 0 or n_gh == 0:
        raise SystemExit("frontier requires both sources")
    weights = np.where(is_tlc, 0.5 / n_tlc, 0.5 / n_gh)
    order = np.argsort(q, kind="mergesort")
    scores = q[order]
    mass = weights[order]
    change = np.empty(len(scores), dtype=bool)
    change[0] = True
    change[1:] = scores[1:] != scores[:-1]
    starts = np.flatnonzero(change)
    unique = scores[starts]
    grouped = np.add.reduceat(mass, starts)
    cumulative = np.cumsum(grouped)
    out = {}
    for target in targets:
        label = f"{target:.2f}"
        index = int(np.searchsorted(cumulative, target, side="left"))
        if index >= len(unique):
            raise SystemExit(f"frontier never reaches {target}")
        previous = float(cumulative[index - 1]) if index else 0.0
        out[label] = {
            "T": float(unique[index]),
            "K": float(cumulative[index]),
            "K_prev": previous,
            "preceding_ok": bool(index == 0 or previous < target - 1e-15),
            "reached": bool(cumulative[index] >= target - 1e-15),
        }
    return out


def k_at(q: np.ndarray, is_tlc: np.ndarray, threshold: float) -> float:
    n_tlc = int(np.sum(is_tlc))
    n_gh = int(len(q) - n_tlc)
    flag = q <= threshold
    return float(0.5 * np.mean(flag[is_tlc]) + 0.5 * np.mean(flag[~is_tlc]))


def select_order(q: np.ndarray, tie: np.ndarray, final: np.ndarray, m: int) -> np.ndarray:
    order = np.lexsort((final, tie, q))
    return order[:m]


def expected_boundary_mean(q: np.ndarray, r: np.ndarray, m: int) -> float:
    if m <= 0:
        return float("nan")
    order = np.argsort(q, kind="mergesort")
    scores = q[order]
    regret = r[order]
    boundary = scores[m - 1]
    strict = scores < boundary
    tied = scores == boundary
    n_need = m - int(np.sum(strict))
    n_tie = int(np.sum(tied))
    total = float(np.sum(regret[strict]))
    if n_need:
        total += (n_need / n_tie) * float(np.sum(regret[tied]))
    return total / m


def boundary_bounds(q: np.ndarray, r: np.ndarray, m: int) -> tuple[float, float]:
    order = np.argsort(q, kind="mergesort")
    scores = q[order]
    regret = r[order]
    boundary = scores[m - 1]
    strict = scores < boundary
    tied = np.sort(regret[scores == boundary])
    n_need = m - int(np.sum(strict))
    base = float(np.sum(regret[strict]))
    if n_need == 0:
        return base / m, base / m
    best = (base + float(np.sum(tied[:n_need]))) / m
    worst = (base + float(np.sum(tied[-n_need:]))) / m
    return best, worst


def boundary_status(q: np.ndarray, m: int) -> dict:
    order = np.argsort(q, kind="mergesort")
    scores = q[order]
    boundary = scores[m - 1]
    n_tie = int(np.sum(scores == boundary))
    n_strict = int(np.sum(scores < boundary))
    n_need = m - n_strict
    return {
        "boundary_tied": bool(n_tie > 1 and n_need < n_tie),
        "tie_block": n_tie,
        "taken_from_tie": n_need,
    }


def validate_inputs() -> None:
    if sha_file(BOX) != EXPECTED_LEDGER["box"] or sha_file(OS_EVAL) != EXPECTED_LEDGER["os_eval"]:
        raise SystemExit("ledger hash mismatch")
    if sha_file(BOX_PROTOCOL) != EXPECTED_PROTOCOL["box"] or sha_file(OS_EVAL_PROTOCOL) != EXPECTED_PROTOCOL["os_eval"]:
        raise SystemExit("protocol hash mismatch")
    for rel, expected in EXPECTED_MANIFEST.items():
        if sha_file(PANEL / rel) != expected:
            raise SystemExit(f"manifest hash mismatch {rel}")


def load_joined() -> pd.DataFrame:
    c_cols = [
        "dataset_id", "truth_id", "evaluation_date", "mechanism", "epsilon", "k", "weight_family",
        "replicate", "seed", "y_sha256", "pi_sha256", "S", "true_regret", "boundary_margin",
        "union_C_box", "union_C_strong", "direct_C_box", "direct_C_strong",
        "union_C_box_plus", "union_C_strong_plus_decision", "union_C_strong_plus_closed",
        "direct_C_box_plus", "direct_C_strong_plus_decision", "direct_C_strong_plus_closed",
    ]
    d_cols = [
        "dataset_id", "truth_id", "mechanism", "epsilon", "k", "weight_family", "replicate",
        "seed", "y_sha256", "pi_sha256", "S", "true_regret", "boundary_margin", "t_rank1",
        "C_OS_rank1_coupled", "C_OS_rank1_strong", "C_OS_coupled_envelope", "C_OS_strong_envelope",
        "chosen_s_coupled", "in_os_event",
    ]
    left = pd.read_parquet(BOX, columns=c_cols)
    right = pd.read_parquet(OS_EVAL, columns=d_cols)
    key = ["truth_id", "mechanism", "epsilon", "k", "weight_family", "replicate"]
    if left.duplicated(key).any() or right.duplicated(key).any():
        raise SystemExit("duplicate evaluation keys")
    merged = left.merge(right, on=key, how="outer", suffixes=("_c", "_d"), indicator=True)
    if len(merged) != 43740 or (merged["_merge"] != "both").any():
        raise SystemExit(f"join failed rows={len(merged)}")
    for column in ("dataset_id", "seed", "y_sha256", "pi_sha256"):
        if not merged[f"{column}_c"].equals(merged[f"{column}_d"]):
            raise SystemExit(f"join mismatch {column}")
    for column in ("S", "true_regret", "boundary_margin"):
        gap = (merged[f"{column}_c"] - merged[f"{column}_d"]).abs()
        if float(gap.max()) > 1e-8:
            raise SystemExit(f"material mismatch {column} {float(gap.max())}")
    merged["dataset_id"] = merged["dataset_id_c"]
    merged["S"] = merged["S_c"]
    merged["true_regret"] = merged["true_regret_c"]
    merged["boundary_margin"] = merged["boundary_margin_c"]
    merged["seed"] = merged["seed_c"]
    merged["y_sha256"] = merged["y_sha256_c"]
    merged["pi_sha256"] = merged["pi_sha256_c"]
    counts = merged.groupby(["dataset_id", "mechanism", "epsilon", "k", "weight_family"]).size()
    if set(counts.unique()) != {450, 360} or len(counts) != 108:
        raise SystemExit(f"stratum counts failed {counts.unique().tolist()} n={len(counts)}")
    if merged["truth_id"].nunique() != 81:
        raise SystemExit("truth count failed")
    return merged


def prepare(df: pd.DataFrame) -> dict:
    work = df.copy()
    work["epsilon_key"] = work["epsilon"].map(lambda value: str(float(value)))
    keys = list(zip(
        work["truth_id"], work["mechanism"], work["epsilon_key"],
        work["k"].astype(int), work["weight_family"], work["replicate"].astype(int),
    ))
    work["tie"] = tie_priority(keys)
    work["r"] = work["true_regret"].to_numpy() / work["S"].to_numpy()
    date_rank = (
        work[["dataset_id", "truth_id", "evaluation_date"]]
        .drop_duplicates()
        .sort_values(["dataset_id", "evaluation_date", "truth_id"])
    )
    date_rank["truth_index"] = date_rank.groupby("dataset_id").cumcount()
    work = work.merge(date_rank[["dataset_id", "truth_id", "truth_index"]], on=["dataset_id", "truth_id"], how="left")
    work["final_key"] = work["truth_index"].to_numpy() * 100 + work["replicate"].to_numpy()
    return {"frame": work, "dates": date_rank}


def pack_datasets(work: pd.DataFrame, methods: dict[str, str]) -> dict:
    packed = {}
    for dataset, sub in work.groupby("dataset_id", sort=True):
        truths = (
            sub[["truth_id", "truth_index"]]
            .drop_duplicates()
            .sort_values("truth_index")
        )
        n_truth = len(truths)
        strata = {}
        for key, piece in sub.groupby(["mechanism", "epsilon_key", "k", "weight_family"], sort=True):
            piece = piece.sort_values(["truth_index", "replicate"])
            if len(piece) != n_truth * 10:
                raise SystemExit(f"stratum length failed {dataset} {key}")
            block = {
                "r": piece["r"].to_numpy().reshape(n_truth, 10),
                "raw": piece["true_regret"].to_numpy().reshape(n_truth, 10),
                "tie": piece["tie"].to_numpy().reshape(n_truth, 10),
                "final": piece["final_key"].to_numpy().reshape(n_truth, 10),
                "k": int(key[2]),
                "q": {},
            }
            for name, column in methods.items():
                block["q"][name] = piece[column].to_numpy() / piece["S"].to_numpy()
                block["q"][name] = block["q"][name].reshape(n_truth, 10)
            strata[key] = block
        packed[dataset] = {"n": n_truth, "strata": strata}
    return packed


def flatten(index: np.ndarray, array: np.ndarray) -> np.ndarray:
    return array[index].ravel()


def metrics_from_orders(r: np.ndarray, raw: np.ndarray, left: np.ndarray, right: np.ndarray, m: int) -> dict:
    sel_l = left[:m]
    sel_r = right[:m]
    only_l = np.setdiff1d(sel_l, sel_r, assume_unique=False)
    only_r = np.setdiff1d(sel_r, sel_l, assume_unique=False)
    inter = np.intersect1d(sel_l, sel_r)
    union = m + m - len(inter)
    out = {
        "mean_l": float(np.mean(r[sel_l])),
        "mean_r": float(np.mean(r[sel_r])),
        "D": float(np.mean(r[sel_l]) - np.mean(r[sel_r])),
        "median_l": float(np.median(r[sel_l])),
        "median_r": float(np.median(r[sel_r])),
        "p90_l": float(np.quantile(r[sel_l], 0.90)),
        "p90_r": float(np.quantile(r[sel_r], 0.90)),
        "raw_l": float(np.mean(raw[sel_l])),
        "raw_r": float(np.mean(raw[sel_r])),
        "zero_l": float(np.mean(r[sel_l] == 0.0)),
        "zero_r": float(np.mean(r[sel_r] == 0.0)),
        "jaccard": float(len(inter) / union) if union else 1.0,
        "n_only_l": int(len(only_l)),
        "n_only_r": int(len(only_r)),
        "ex_l": float(np.mean(r[only_l])) if len(only_l) else None,
        "ex_r": float(np.mean(r[only_r])) if len(only_r) else None,
        "m": int(m),
        "n": int(len(r)),
    }
    for gamma in GAMMAS:
        if gamma == 0.0:
            bad_l = r[sel_l] > 0.0
            bad_r = r[sel_r] > 0.0
        else:
            bad_l = r[sel_l] > gamma
            bad_r = r[sel_r] > gamma
        out[f"bad_l_{gamma}"] = float(np.mean(bad_l))
        out[f"bad_r_{gamma}"] = float(np.mean(bad_r))
    return out


def evaluate_pack(pack: dict, pairs: dict[str, tuple[str, str]], tie_override: np.ndarray | None = None) -> dict:
    results = {name: [] for name in pairs}
    controls = []
    tie_rows = []
    for dataset, content in pack.items():
        for key, block in content["strata"].items():
            n_truth = content["n"]
            index = np.arange(n_truth)
            r = flatten(index, block["r"])
            raw = flatten(index, block["raw"])
            tie = flatten(index, block["tie"] if tie_override is None else tie_override[dataset][key])
            final = flatten(index, block["final"])
            orders = {}
            for method, scores in block["q"].items():
                orders[method] = select_order(flatten(index, scores), tie, final, len(r))
            for pair, (left, right) in pairs.items():
                m = m_of(len(r), 1, 2)
                # Coverage-specific metrics are filled below; store orders via recomputation per c.
                del m
                for label, numer, denom in COVERAGES:
                    m = m_of(len(r), numer, denom)
                    row = metrics_from_orders(r, raw, orders[left], orders[right], m)
                    row.update({
                        "dataset_id": dataset,
                        "mechanism": key[0],
                        "epsilon": key[1],
                        "k": int(key[2]),
                        "weight_family": key[3],
                        "pair": pair,
                        "c": label,
                        "coverage": m / len(r),
                    })
                    q_left = flatten(index, block["q"][left])
                    q_right = flatten(index, block["q"][right])
                    status_l = boundary_status(q_left, m)
                    status_r = boundary_status(q_right, m)
                    row["boundary_tied_l"] = status_l["boundary_tied"]
                    row["boundary_tied_r"] = status_r["boundary_tied"]
                    row["tie_block_l"] = status_l["tie_block"]
                    row["tie_block_r"] = status_r["tie_block"]
                    row["taken_l"] = status_l["taken_from_tie"]
                    row["taken_r"] = status_r["taken_from_tie"]
                    if label == "0.50" and pair == "A_union":
                        row["expected_l"] = expected_boundary_mean(q_left, r, m)
                        row["expected_r"] = expected_boundary_mean(q_right, r, m)
                        best_l, worst_l = boundary_bounds(q_left, r, m)
                        best_r, worst_r = boundary_bounds(q_right, r, m)
                        row["best_l"] = best_l
                        row["worst_l"] = worst_l
                        row["best_r"] = best_r
                        row["worst_r"] = worst_r
                        # Shared-resolution bounds are not these independent bounds.
                        row["D_tie_low"] = best_l - worst_r
                        row["D_tie_high"] = worst_l - best_r
                    results[pair].append(row)
            if int(key[2]) <= 5:
                for pair in ("A_union", "B_direct", "C_os_rank1", "D_os_envelope"):
                    if pair not in pairs:
                        continue
                    left, right = pairs[pair]
                    if not np.array_equal(orders[left], orders[right]):
                        controls.append({"dataset_id": dataset, "stratum": list(map(str, key)), "pair": pair})
            if tie_override is None:
                q = flatten(index, block["q"]["union_C_box"])
                m = m_of(len(q), 1, 2)
                status = boundary_status(q, m)
                tie_rows.append({"dataset_id": dataset, "k": int(key[2]), **status, "n": len(q), "m": m})
    if controls:
        raise SystemExit(f"k<=5 selection identity failed {controls[:3]}")
    return {"pairs": results, "tie_rows": tie_rows}


def macro_from_rows(rows: list[dict], field: str) -> dict:
    frame = pd.DataFrame(rows)
    out = {}
    for label, _, _ in COVERAGES:
        piece = frame[frame["c"] == label]
        by_dataset = piece.groupby("dataset_id")[field].mean()
        out[label] = {
            "by_dataset": {key: float(value) for key, value in by_dataset.items()},
            "equal_weight_macro": float(by_dataset.mean()),
        }
    return out


def source_only(work: pd.DataFrame, pairs: dict[str, tuple[str, str]]) -> dict:
    out = {name: [] for name in pairs}
    for dataset, sub in work.groupby("dataset_id"):
        r = sub["r"].to_numpy()
        raw = sub["true_regret"].to_numpy()
        tie = sub["tie"].to_numpy()
        final = sub["final_key"].to_numpy()
        orders = {}
        for column in {item for pair in pairs.values() for item in pair}:
            orders[column] = select_order(sub[column].to_numpy() / sub["S"].to_numpy(), tie, final, len(sub))
        for pair, (left, right) in pairs.items():
            for label, numer, denom in COVERAGES:
                m = m_of(len(sub), numer, denom)
                row = metrics_from_orders(r, raw, orders[left], orders[right], m)
                row.update({"dataset_id": dataset, "pair": pair, "c": label, "coverage": m / len(sub)})
                out[pair].append(row)
    return out


def heterogeneity(rows: list[dict]) -> dict:
    frame = pd.DataFrame(rows)
    frame = frame[frame["c"] == "0.50"]
    slices = {
        "k": ["k"],
        "weight": ["weight_family"],
        "epsilon": ["epsilon"],
        "mechanism": ["mechanism"],
        "dataset_k": ["dataset_id", "k"],
    }
    result = {}
    for name, cols in slices.items():
        grouped = []
        for key, sub in frame.groupby(cols, sort=True):
            if not isinstance(key, tuple):
                key = (key,)
            by_dataset = sub.groupby("dataset_id")["D"].mean()
            grouped.append({
                **{col: (int(value) if col == "k" else value) for col, value in zip(cols, key)},
                "by_dataset": {ds: float(val) for ds, val in by_dataset.items()},
                "equal_weight_macro": float(by_dataset.mean()),
            })
        result[name] = grouped
    return result


def bootstrap(pack: dict, pairs: dict[str, tuple[str, str]], seed: int, mode: str) -> dict:
    rng = np.random.default_rng(seed)
    datasets = list(pack)
    store = {
        pair: {dataset: np.empty(BOOT_B) for dataset in datasets}
        for pair in pairs
    }
    tails = {
        gamma: {dataset: np.empty(BOOT_B) for dataset in datasets}
        for gamma in GAMMAS
    }
    needed = {name for pair in pairs.values() for name in pair}
    for draw in range(BOOT_B):
        if draw % 500 == 0:
            print(f"[matched] {mode} bootstrap {draw}", flush=True)
        indices = {}
        for dataset, content in pack.items():
            n = content["n"]
            if mode == "cluster":
                indices[dataset] = rng.integers(0, n, n)
            else:
                indices[dataset] = block_indices(n, rng)
        stratum_d = {dataset: {pair: [] for pair in pairs} for dataset in datasets}
        stratum_tail = {dataset: {gamma: [] for gamma in GAMMAS} for dataset in datasets}
        for dataset, content in pack.items():
            index = indices[dataset]
            for block in content["strata"].values():
                r = flatten(index, block["r"])
                tie = flatten(index, block["tie"])
                final = np.arange(len(r))
                orders = {
                    method: select_order(flatten(index, scores), tie, final, len(r))
                    for method, scores in block["q"].items() if method in needed
                }
                m = m_of(len(r), 1, 2)
                for pair, (left, right) in pairs.items():
                    stratum_d[dataset][pair].append(float(np.mean(r[orders[left][:m]]) - np.mean(r[orders[right][:m]])))
                sel_l = orders["union_C_strong"][:m]
                sel_r = orders["union_C_box"][:m]
                for gamma in GAMMAS:
                    if gamma == 0.0:
                        bad_l = np.mean(r[sel_l] > 0.0)
                        bad_r = np.mean(r[sel_r] > 0.0)
                    else:
                        bad_l = np.mean(r[sel_l] > gamma)
                        bad_r = np.mean(r[sel_r] > gamma)
                    stratum_tail[dataset][gamma].append(float(bad_l - bad_r))
        for dataset in datasets:
            for pair in pairs:
                store[pair][dataset][draw] = float(np.mean(stratum_d[dataset][pair]))
            for gamma in GAMMAS:
                tails[gamma][dataset][draw] = float(np.mean(stratum_tail[dataset][gamma]))
    report = {"D": {}, "union_bad_difference": {}}
    for pair in pairs:
        macro = 0.5 * (store[pair][datasets[0]] + store[pair][datasets[1]])
        report["D"][pair] = interval_block(store[pair], macro)
    for gamma in GAMMAS:
        macro = 0.5 * (tails[gamma][datasets[0]] + tails[gamma][datasets[1]])
        report["union_bad_difference"][str(gamma)] = interval_block(tails[gamma], macro)
    return report


def interval_block(by_dataset: dict, macro: np.ndarray) -> dict:
    out = {
        "macro": quantile_pair(macro),
        "by_dataset": {dataset: quantile_pair(values) for dataset, values in by_dataset.items()},
    }
    return out


def quantile_pair(values: np.ndarray) -> dict:
    return {
        "lo": float(np.quantile(values, 0.025)),
        "hi": float(np.quantile(values, 0.975)),
    }


def block_indices(n: int, rng: np.random.Generator, block: int = 7) -> np.ndarray:
    starts = rng.integers(0, n, int(math.ceil(n / block)))
    idx = [(int(start) + offset) % n for start in starts for offset in range(block)]
    return np.asarray(idx[:n], dtype=int)


def tie_sensitivity(work: pd.DataFrame, pairs: dict[str, tuple[str, str]]) -> dict:
    # Rebuild tie priorities under each seed without using regret.
    records = []
    changed_strata = []
    changed_cells = []
    keys = list(zip(
        work["truth_id"], work["mechanism"], work["epsilon_key"],
        work["k"].astype(int), work["weight_family"], work["replicate"].astype(int),
    ))
    focus = {name: pairs[name] for name in ("A_union", "B_direct", "D_os_envelope")}
    primary_sets = _selected_sets(work, focus["A_union"])
    for seed in TIE_SENSITIVITY_SEEDS:
        priorities = np.empty(len(keys), dtype=np.uint64)
        for i, key in enumerate(keys):
            payload = f"{seed}|{key[0]}|{key[1]}|{key[2]}|{key[3]}|{key[4]}|{key[5]}".encode()
            priorities[i] = np.uint64(int(hashlib.sha256(payload).hexdigest()[:16], 16))
        trial = work.copy()
        trial["tie"] = priorities
        row = {"seed": seed}
        for name, pair in focus.items():
            row[name] = _macro_d(trial, pair)
        alt = _selected_sets(trial, focus["A_union"])
        n_changed = 0
        n_cells = 0
        for key, selected in primary_sets.items():
            gap = np.setxor1d(selected, alt[key])
            if len(gap):
                n_changed += 1
                n_cells += int(len(gap))
        changed_strata.append(n_changed)
        changed_cells.append(n_cells)
        records.append(row)
    frame = pd.DataFrame(records)
    return {
        "n_seeds": len(records),
        "A_union_min": float(frame["A_union"].min()),
        "A_union_max": float(frame["A_union"].max()),
        "A_union_n_negative": int(np.sum(frame["A_union"] < 0)),
        "A_union_n_positive": int(np.sum(frame["A_union"] > 0)),
        "A_union_n_zero": int(np.sum(frame["A_union"] == 0)),
        "B_direct_min": float(frame["B_direct"].min()),
        "B_direct_max": float(frame["B_direct"].max()),
        "D_os_envelope_min": float(frame["D_os_envelope"].min()),
        "D_os_envelope_max": float(frame["D_os_envelope"].max()),
        "D_os_envelope_n_negative": int(np.sum(frame["D_os_envelope"] < 0)),
        "D_os_envelope_n_positive": int(np.sum(frame["D_os_envelope"] > 0)),
        "A_union_strata_changed_mean": float(np.mean(changed_strata)),
        "A_union_strata_changed_max": int(np.max(changed_strata)),
        "A_union_symmetric_difference_cells_mean": float(np.mean(changed_cells)),
        "A_union_symmetric_difference_cells_max": int(np.max(changed_cells)),
    }


def _macro_d(frame: pd.DataFrame, pair: tuple[str, str]) -> float:
    values = []
    for dataset, sub in frame.groupby("dataset_id"):
        stratum_values = []
        for _, piece in sub.groupby(["mechanism", "epsilon_key", "k", "weight_family"]):
            r = piece["r"].to_numpy()
            tie = piece["tie"].to_numpy()
            final = piece["final_key"].to_numpy()
            m = m_of(len(piece), 1, 2)
            left = select_order(piece[pair[0]].to_numpy() / piece["S"].to_numpy(), tie, final, m)
            right = select_order(piece[pair[1]].to_numpy() / piece["S"].to_numpy(), tie, final, m)
            stratum_values.append(float(np.mean(r[left]) - np.mean(r[right])))
        values.append(float(np.mean(stratum_values)))
    return float(np.mean(values))


def _selected_sets(frame: pd.DataFrame, pair: tuple[str, str]) -> dict:
    out = {}
    for dataset, sub in frame.groupby("dataset_id"):
        for key, piece in sub.groupby(["mechanism", "epsilon_key", "k", "weight_family"]):
            tie = piece["tie"].to_numpy()
            final = piece["final_key"].to_numpy()
            m = m_of(len(piece), 1, 2)
            # Compare the coupled selector. Its tie block is the one that can move.
            chosen = select_order(piece[pair[1]].to_numpy() / piece["S"].to_numpy(), tie, final, m)
            ident = piece["truth_index"].to_numpy() * 100 + piece["replicate"].to_numpy()
            out[(dataset, key, pair[1])] = np.sort(ident[chosen])
    return out


def s1_audit(work: pd.DataFrame) -> dict:
    s1 = work["chosen_s_coupled"].to_numpy() == 1
    gap = (work.loc[s1, "C_OS_coupled_envelope"] - work.loc[s1, "C_OS_rank1_coupled"]).to_numpy()
    strong_gap = (work.loc[s1, "C_OS_strong_envelope"] - work.loc[s1, "C_OS_coupled_envelope"]).to_numpy()
    scale = work.loc[s1, "S"].to_numpy()
    env = work.loc[s1, "C_OS_coupled_envelope"].to_numpy()
    rank1 = work.loc[s1, "C_OS_rank1_coupled"].to_numpy()
    strong = work.loc[s1, "C_OS_strong_envelope"].to_numpy()
    decisions = {}
    for tau in TAUS:
        refine = (env <= tau * scale) != (rank1 <= tau * scale)
        coupling = (env <= tau * scale) != (strong <= tau * scale)
        decisions[str(tau)] = {
            "n_refinement_disagreement": int(np.sum(refine)),
            "n_os_coupling_disagreement": int(np.sum(coupling)),
        }
    piece = work.loc[s1].copy()
    piece["coupling_gap"] = np.abs(strong_gap) > 1e-8
    breakdown = []
    for key, sub in piece.groupby(["dataset_id", "k", "weight_family", "epsilon", "mechanism"]):
        breakdown.append({
            "dataset_id": key[0],
            "k": int(key[1]),
            "weight_family": key[2],
            "epsilon": float(key[3]),
            "mechanism": key[4],
            "n": int(len(sub)),
            "n_certificate_gap_gt_1e-8": int(np.sum(np.abs(sub["C_OS_strong_envelope"] - sub["C_OS_coupled_envelope"]) > 1e-8)),
        })
    return {
        "n_s1": int(np.sum(s1)),
        "max_abs_envelope_minus_rank1": float(np.max(np.abs(gap))) if len(gap) else None,
        "n_envelope_rank1_gap_gt_1e-8": int(np.sum(np.abs(gap) > 1e-8)),
        "n_strong_gap_gt_1e-8": int(np.sum(np.abs(strong_gap) > 1e-8)),
        "decisions": decisions,
        "breakdown_nonzero_only": [row for row in breakdown if row["n_certificate_gap_gt_1e-8"]],
        "interpretation": "s=1 forces the coupled envelope to equal the rank-1 coupled box. It does not force equality with the strong envelope.",
    }


def frontier_audit(work: pd.DataFrame) -> dict:
    is_tlc = (work["dataset_id"] == "tlc_yellow").to_numpy()
    columns = {name: work[column].to_numpy() / work["S"].to_numpy() for name, column in {
        "union_coupled": "union_C_box",
        "union_strong": "union_C_strong",
        "direct_coupled": "direct_C_box",
        "direct_strong": "direct_C_strong",
        "os_rank1_coupled": "C_OS_rank1_coupled",
        "os_rank1_strong": "C_OS_rank1_strong",
        "os_coupled_envelope": "C_OS_coupled_envelope",
        "os_strong_envelope": "C_OS_strong_envelope",
    }.items()}
    targets = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    report = {}
    for name, scores in columns.items():
        exact = exact_frontier(scores, is_tlc, targets)
        report[name] = exact
    return report


def baselines(work: pd.DataFrame) -> dict:
    rows = []
    for dataset, sub in work.groupby("dataset_id"):
        for key, piece in sub.groupby(["mechanism", "epsilon_key", "k", "weight_family"]):
            r = piece["r"].to_numpy()
            tie = piece["tie"].to_numpy()
            final = piece["final_key"].to_numpy()
            oracle = select_order(r, tie, final, len(r))
            random_order = select_order(np.zeros(len(r)), tie, final, len(r))
            for label, numer, denom in COVERAGES:
                m = m_of(len(r), numer, denom)
                rows.append({
                    "dataset_id": dataset,
                    "c": label,
                    "random_mean": float(np.mean(r)),
                    "oracle_mean": float(np.mean(r[oracle[:m]])),
                    "seeded_priority_mean": float(np.mean(r[random_order[:m]])),
                    "oracle_zero": float(np.mean(r[oracle[:m]] == 0.0)),
                })
    frame = pd.DataFrame(rows)
    out = {}
    for label, _, _ in COVERAGES:
        piece = frame[frame["c"] == label]
        by = piece.groupby("dataset_id")[["random_mean", "oracle_mean", "seeded_priority_mean", "oracle_zero"]].mean()
        out[label] = {
            "by_dataset": {ds: {col: float(by.loc[ds, col]) for col in by.columns} for ds in by.index},
            "equal_weight_macro": {col: float(by[col].mean()) for col in by.columns},
        }
    return out


def descriptive_margins(work: pd.DataFrame) -> dict:
    sub = work.copy()
    margin = sub["boundary_margin"].to_numpy()
    t1 = sub["t_rank1"].to_numpy()
    sub["margin_group"] = np.where(margin <= 0, "nonpositive", np.where(margin <= t1, "within_rank1", "above_rank1"))
    # Describe the primary union selections, not a new matched design.
    rows = []
    for (dataset, mech, eps, k, wf), piece in sub.groupby(["dataset_id", "mechanism", "epsilon_key", "k", "weight_family"]):
        r = piece["r"].to_numpy()
        tie = piece["tie"].to_numpy()
        final = piece["final_key"].to_numpy()
        q_box = piece["union_C_box"].to_numpy() / piece["S"].to_numpy()
        q_strong = piece["union_C_strong"].to_numpy() / piece["S"].to_numpy()
        m = m_of(len(piece), 1, 2)
        sel_box = select_order(q_box, tie, final, m)
        sel_strong = select_order(q_strong, tie, final, m)
        groups = piece["margin_group"].to_numpy()
        zero = r == 0.0
        for name, sel in (("coupled", sel_box), ("strong", sel_strong)):
            chosen_groups = groups[sel]
            chosen_zero = zero[sel]
            rows.append({
                "dataset_id": dataset,
                "selector": name,
                "fraction_nonpositive": float(np.mean(chosen_groups == "nonpositive")),
                "fraction_within": float(np.mean(chosen_groups == "within_rank1")),
                "fraction_above": float(np.mean(chosen_groups == "above_rank1")),
                "fraction_zero_regret": float(np.mean(chosen_zero)),
                "mean_r_among_positive": float(np.mean(r[sel][r[sel] > 0])) if np.any(r[sel] > 0) else None,
            })
    frame = pd.DataFrame(rows)
    out = {}
    for selector, piece in frame.groupby("selector"):
        means = piece.groupby("dataset_id").mean(numeric_only=True)
        out[selector] = {
            "by_dataset": {ds: {col: float(means.loc[ds, col]) for col in means.columns} for ds in means.index},
            "equal_weight_macro": {col: float(means[col].mean()) for col in means.columns},
        }
    return out


def residue_audit(work: pd.DataFrame) -> dict:
    checks = []
    for left, right, name in (
        ("union_C_box", "union_C_strong", "union"),
        ("direct_C_box", "direct_C_strong", "direct"),
        ("C_OS_rank1_coupled", "C_OS_rank1_strong", "os_rank1"),
        ("C_OS_coupled_envelope", "C_OS_strong_envelope", "os_envelope"),
    ):
        k5 = work["k"] <= 5
        gap = (work.loc[k5, left] - work.loc[k5, right]).to_numpy()
        checks.append({"pair": name, "n": int(k5.sum()), "max_abs_k5_gap": float(np.max(np.abs(gap)))})
    near = 0
    for column in ("union_C_box", "union_C_strong", "C_OS_coupled_envelope", "C_OS_strong_envelope"):
        q = work[column].to_numpy() / work["S"].to_numpy()
        # Adjacent numerical near-equality is assessed within strata at the boundary below.
        near += int(np.sum(~np.isfinite(q)))
    return {"k5_score_gaps": checks, "nonfinite_scores": near}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    validate_inputs()
    digest = write_protocol()
    print(f"[matched] protocol {digest}", flush=True)
    synthetic_checks()
    print("[matched] synthetic checks passed", flush=True)
    merged = load_joined()
    prepared = prepare(merged)
    work = prepared["frame"]
    primary_pairs = {
        "A_union": ("union_C_strong", "union_C_box"),
        "B_direct": ("direct_C_strong", "direct_C_box"),
        "C_os_rank1": ("C_OS_rank1_strong", "C_OS_rank1_coupled"),
        "D_os_envelope": ("C_OS_strong_envelope", "C_OS_coupled_envelope"),
        "E_coupled_refinement": ("C_OS_rank1_coupled", "C_OS_coupled_envelope"),
        "F_strong_refinement": ("C_OS_rank1_strong", "C_OS_strong_envelope"),
    }
    secondary_pairs = {
        "union_raw": ("union_C_strong_plus_decision", "union_C_box_plus"),
        "union_closed": ("union_C_strong_plus_closed", "union_C_box_plus"),
        "direct_raw": ("direct_C_strong_plus_decision", "direct_C_box_plus"),
        "direct_closed": ("direct_C_strong_plus_closed", "direct_C_box_plus"),
    }
    methods = {name: name for pair in {**primary_pairs, **secondary_pairs}.values() for name in pair}
    pack = pack_datasets(work, methods)
    evaluated = evaluate_pack(pack, {**primary_pairs, **secondary_pairs})
    print("[matched] point estimates done", flush=True)
    summaries = {}
    fields = ["D", "mean_l", "mean_r", "median_l", "median_r", "p90_l", "p90_r", "raw_l", "raw_r", "zero_l", "zero_r", "jaccard", "n_only_l", "n_only_r"]
    for gamma in GAMMAS:
        fields.extend([f"bad_l_{gamma}", f"bad_r_{gamma}"])
    for pair, rows in evaluated["pairs"].items():
        summaries[pair] = {field: macro_from_rows(rows, field) for field in fields if field in rows[0]}
    (OUT / "matched_rate_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    pd.DataFrame([row for rows in evaluated["pairs"].values() for row in rows]).to_parquet(OUT / "stratum_rows.parquet", index=False)
    source_rows = source_only(work, primary_pairs)
    source_summary = {pair: macro_from_rows(rows, "D") for pair, rows in source_rows.items()}
    (OUT / "source_only_summary.json").write_text(json.dumps(source_summary, indent=2) + "\n")
    (OUT / "heterogeneity.json").write_text(json.dumps({pair: heterogeneity(rows) for pair, rows in evaluated["pairs"].items() if pair in primary_pairs}, indent=2) + "\n")
    (OUT / "baselines.json").write_text(json.dumps(baselines(work), indent=2) + "\n")
    (OUT / "margin_description.json").write_text(json.dumps(descriptive_margins(work), indent=2) + "\n")
    (OUT / "frontier_exact.json").write_text(json.dumps(frontier_audit(work), indent=2) + "\n")
    (OUT / "s1_audit.json").write_text(json.dumps(s1_audit(work), indent=2) + "\n")
    (OUT / "numerical_audit.json").write_text(json.dumps(residue_audit(work), indent=2) + "\n")
    tie_frame = pd.DataFrame(evaluated["tie_rows"])
    tie_audit = {
        "n_strata": int(len(tie_frame)),
        "n_boundary_tied_union_box_c50": int(tie_frame["boundary_tied"].sum()),
        "mean_tie_block": float(tie_frame["tie_block"].mean()),
        "mean_taken_from_tie": float(tie_frame["taken_from_tie"].mean()),
    }
    primary_rows = [row for row in evaluated["pairs"]["A_union"] if row["c"] == "0.50"]
    tie_audit["expected_D_macro"] = macro_from_rows(
        [{**row, "D": row["expected_l"] - row["expected_r"], "c": "0.50"} for row in primary_rows],
        "D",
    )["0.50"]
    tie_audit["independent_resolution_low_macro"] = macro_from_rows(
        [{**row, "D": row["D_tie_low"], "c": "0.50"} for row in primary_rows], "D"
    )["0.50"]
    tie_audit["independent_resolution_high_macro"] = macro_from_rows(
        [{**row, "D": row["D_tie_high"], "c": "0.50"} for row in primary_rows], "D"
    )["0.50"]
    exchanged = {
        "n_only_strong_sum": int(sum(row["n_only_l"] for row in primary_rows)),
        "n_only_coupled_sum": int(sum(row["n_only_r"] for row in primary_rows)),
        "jaccard_macro": summaries["A_union"]["jaccard"]["0.50"],
        "mean_r_only_strong": _mean_defined(primary_rows, "ex_l"),
        "mean_r_only_coupled": _mean_defined(primary_rows, "ex_r"),
    }
    (OUT / "overlap_primary.json").write_text(json.dumps({"union_c50": exchanged, "all_pairs_jaccard": {
        pair: summaries[pair]["jaccard"]["0.50"] for pair in primary_pairs
    }}, indent=2) + "\n")
    print("[matched] tie sensitivity", flush=True)
    sensitivity = tie_sensitivity(work, primary_pairs)
    tie_audit["sensitivity"] = sensitivity
    (OUT / "tie_audit.json").write_text(json.dumps(tie_audit, indent=2) + "\n")
    print("[matched] bootstrap", flush=True)
    boot_pairs = {key: primary_pairs[key] for key in ("A_union", "B_direct", "D_os_envelope", "C_os_rank1", "E_coupled_refinement", "F_strong_refinement")}
    cluster = bootstrap(pack, boot_pairs, BOOT_SEED, "cluster")
    block = bootstrap(pack, boot_pairs, BLOCK_SEED, "block")
    (OUT / "bootstrap_cluster.json").write_text(json.dumps(cluster, indent=2) + "\n")
    (OUT / "bootstrap_block.json").write_text(json.dumps(block, indent=2) + "\n")
    registry_columns = [
        "dataset_id", "truth_id", "evaluation_date", "mechanism", "epsilon", "k", "weight_family",
        "replicate", "seed", "S", "true_regret", "r", "tie", "boundary_margin", "t_rank1", "chosen_s_coupled",
        "union_C_box", "union_C_strong", "direct_C_box", "direct_C_strong",
        "C_OS_rank1_coupled", "C_OS_rank1_strong", "C_OS_coupled_envelope", "C_OS_strong_envelope",
    ]
    work[registry_columns].to_parquet(OUT / "score_registry.parquet", index=False)
    integrity = {
        "rows": int(len(work)),
        "truths": int(work["truth_id"].nunique()),
        "strata": 108,
        "tlc_stratum_n": 450,
        "github_stratum_n": 360,
        "max_abs_S_join_gap": float((merged["S_c"] - merged["S_d"]).abs().max()),
        "max_abs_regret_join_gap": float((merged["true_regret_c"] - merged["true_regret_d"]).abs().max()),
    }
    (OUT / "join_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")
    (OUT / "stage_times.json").write_text(json.dumps({"total_s": time.perf_counter() - started}, indent=2) + "\n")
    print("[matched] finished", flush=True)


def _mean_defined(rows: list[dict], field: str) -> dict:
    usable = [row for row in rows if row[field] is not None]
    if not usable:
        return {"n_strata": 0}
    frame = pd.DataFrame(usable)
    by = frame.groupby("dataset_id")[field].mean()
    return {
        "n_strata": int(len(frame)),
        "by_dataset": {key: float(value) for key, value in by.items()},
        "equal_weight_macro": float(by.mean()),
    }


if __name__ == "__main__":
    main()
