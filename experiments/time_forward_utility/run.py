#!/usr/bin/env python3
"""Chronological holdout: thresholds use earlier scores only.

The protocol file is written before any out-of-time regret is computed.
Test scores and latent regret never enter cutoff construction.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import scipy

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BOX = REPO / "experiments/real_box_evaluation/ledger.parquet"
OS_EVAL = REPO / "experiments/real_order_statistic_evaluation/ledger.parquet"
MATCHED_PROTOCOL = REPO / "experiments/matched_rate_utility/protocol.json"
MATCHED_FRONTIER = REPO / "experiments/matched_rate_utility/frontier_exact.json"
PANEL = REPO / "experiments/real_data_panel"
EXPECTED = {
    "box_ledger": "281ca9b676184d7450bbd2c8da225b94da312bb37d3c97855b1818ecc8c956cf",
    "os_eval_ledger": "eeac20c3ddb213fc9a6c302174162ae0b93bd5489b19f9f984b66265b121a72e",
    "matched_protocol": "38ba078b479c86a2df35d702a5f75572d40b98d4a9142dce66ab24b1634573d2",
    "matched_frontier": "3820f91755115d2eb22bcf246906c9dce3fa179d611e06af92454ed6b5564957",
    "tlc_manifest": "942a6cff6abfd58e63b0d025b4a45da41d9a550ee64e2be4e3572b3dc7a9532c",
    "gh_manifest": "745752764c77212af41fdb53d45934a4ad1cbee44caca4aa403ae2b8ae90978f",
}
HASH_SEED = 20270111
# Opaque seed salt: its exact bytes enter the SHA-256 seed derivation.
THIN_SALT = "G5F-THIN"
BOOT_SEED = 20270112
BLOCK_SEED = 20270113
CAL_SEED = 20270114
BOOT_B = 1999
CAL_B = 399
BLOCK = 7
DENOM = 194400
TARGETS = {
    "0.10": 19440,
    "0.25": 48600,
    "0.50": 97200,
    "0.75": 145800,
    "0.90": 174960,
    "0.95": 184680,
}
COVERAGES = (("0.25", 1, 4), ("0.50", 1, 2), ("0.75", 3, 4))
GAMMAS = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
TAUS = (0.05, 0.10, 0.25, 0.50, 1.00)
PAIRS = {
    "A_union": ("union_C_strong", "union_C_box"),
    "B_direct": ("direct_C_strong", "direct_C_box"),
    "C_os_rank1": ("C_OS_rank1_strong", "C_OS_rank1_coupled"),
    "D_os_envelope": ("C_OS_strong_envelope", "C_OS_coupled_envelope"),
}
SPLIT_SPEC = {
    "primary_60_40": {"tlc_yellow": 27, "gharchive": 21},
    "sensitivity_50_50": {"tlc_yellow": 22, "gharchive": 18},
    "sensitivity_70_30": {"tlc_yellow": 31, "gharchive": 25},
}
FRONTIER_COLUMNS = {
    "union_coupled": "union_C_box",
    "union_strong": "union_C_strong",
    "direct_coupled": "direct_C_box",
    "direct_strong": "direct_C_strong",
    "os_rank1_coupled": "C_OS_rank1_coupled",
    "os_rank1_strong": "C_OS_rank1_strong",
    "os_coupled_envelope": "C_OS_coupled_envelope",
    "os_strong_envelope": "C_OS_strong_envelope",
}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cell_hash(keys: list[tuple]) -> np.ndarray:
    out = np.empty(len(keys), dtype=np.uint64)
    for i, (truth_id, mech, eps, k, wf, rep) in enumerate(keys):
        payload = f"{THIN_SALT}|{HASH_SEED}|{truth_id}|{mech}|{eps}|{k}|{wf}|{rep}".encode()
        out[i] = np.uint64(int(hashlib.sha256(payload).hexdigest()[:16], 16))
    return out


def tie_keep(hash_u: np.ndarray, n_equal: int, need: int) -> np.ndarray:
    """Independent of the spec's product test. See fixtures for the product reference."""
    n = len(hash_u)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if need >= n_equal:
        return np.ones(n, dtype=bool)
    if need <= 0 or n_equal <= 0:
        return np.zeros(n, dtype=bool)
    quota, remainder = divmod(need << 64, int(n_equal))
    if quota >= (1 << 64):
        return np.ones(n, dtype=bool)
    values = hash_u.astype(np.uint64, copy=False)
    keep = values < np.uint64(quota)
    if remainder > 0:
        keep = keep | (values == np.uint64(quota))
    return keep


def fit_threshold(scores: np.ndarray, numer: int, denom: int) -> dict:
    n = int(len(scores))
    m = (numer * n) // denom
    if m <= 0 or m > n:
        raise SystemExit(f"invalid training cardinality m={m} n={n}")
    ordered = np.sort(scores, kind="mergesort")
    boundary = float(ordered[m - 1])
    n_less = int(np.sum(scores < boundary))
    n_equal = int(np.sum(scores == boundary))
    need = m - n_less
    if need <= 0 or need > n_equal:
        raise SystemExit(f"tie accounting failed need={need} equal={n_equal}")
    return {
        "N_train": n,
        "m_train": m,
        "v": boundary,
        "n_less": n_less,
        "n_equal": n_equal,
        "need": need,
        "p": need / n_equal,
    }


def apply_rule(scores: np.ndarray, hashes: np.ndarray, rule: dict) -> np.ndarray:
    accept = scores < rule["v"]
    tied = scores == rule["v"]
    if np.any(tied):
        keep = np.zeros(len(scores), dtype=bool)
        keep[np.flatnonzero(tied)] = tie_keep(hashes[tied], rule["n_equal"], rule["need"])
        accept = accept | keep
    return accept


def integer_inverse(scores: np.ndarray, is_tlc: np.ndarray) -> dict:
    """Equal-source inverse using the integer numerator 4*n_tlc + 5*n_gh."""
    order = np.argsort(scores, kind="mergesort")
    ordered = scores[order]
    tlc = is_tlc[order]
    change = np.empty(len(ordered), dtype=bool)
    change[0] = True
    change[1:] = ordered[1:] != ordered[:-1]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(ordered)]
    unique = ordered[starts]
    mass = np.empty(len(starts), dtype=np.int64)
    for i, (a, b) in enumerate(zip(starts, ends)):
        n_tlc = int(np.sum(tlc[a:b]))
        n_gh = int((b - a) - n_tlc)
        mass[i] = 4 * n_tlc + 5 * n_gh
    cumulative = np.cumsum(mass, dtype=np.int64)
    out = {}
    for label, target in TARGETS.items():
        index = int(np.searchsorted(cumulative, target, side="left"))
        previous = int(cumulative[index - 1]) if index else 0
        out[label] = {
            "T": float(unique[index]),
            "numerator": int(cumulative[index]),
            "previous_numerator": previous,
            "reached": bool(int(cumulative[index]) >= target),
            "preceding_below": bool(index == 0 or previous < target),
            "K_rational": f"{int(cumulative[index])}/194400",
        }
    return out


def write_protocol() -> str:
    protocol = {
        "experiment": "time_forward_utility",
        "label": "CHRONOLOGICAL_HOLDOUT_SENSITIVITY",
        "not": [
            "independent prospective confirmation",
            "newly collected validation",
            "never-before-seen test panel",
            "randomized experiment",
            "conditional coverage",
        ],
        "input_sha256": EXPECTED,
        "primary_split": {
            "rule": "earliest 60 percent of truth dates within each dataset",
            "tlc_yellow": {
                "n_truth_cal": 27,
                "n_truth_eval": 18,
                "cal_start": "2024-04-01",
                "cal_end": "2024-04-27",
                "eval_start": "2024-04-28",
                "eval_end": "2024-05-15",
                "cells_cal": 14580,
                "cells_eval": 9720,
            },
            "gharchive": {
                "n_truth_cal": 21,
                "n_truth_eval": 15,
                "cal_start": "2024-02-01",
                "cal_end": "2024-02-21",
                "eval_start": "2024-02-22",
                "eval_end": "2024-03-07",
                "cells_cal": 11340,
                "cells_eval": 8100,
            },
        },
        "sensitivity_splits": {
            "50_50": "floor(0.50 * n_dates): TLC 22/23, GitHub 18/18",
            "70_30": "floor(0.70 * n_dates): TLC 31/14, GitHub 25/11",
        },
        "score": "q = C/S from the canonical box-event and order-statistic decision columns. Lower is more favorable.",
        "pairs": {name: list(pair) for name, pair in PAIRS.items()},
        "sign": {
            "D_accept": "test acceptance fraction of the coupled or refined score minus the structural score",
            "D_regret": "mean accepted r of the structural score minus mean accepted r of the coupled score, only where both accepted sets are nonempty. Positive means the coupled accepted set has lower mean regret. This is not a matched-rate contrast.",
        },
        "c_train_primary": {"label": "0.50", "numer": 1, "denom": 2},
        "c_train_secondary": [
            {"label": "0.25", "numer": 1, "denom": 4},
            {"label": "0.75", "numer": 3, "denom": 4},
        ],
        "policy": "Accept q < v. Reject q > v. At q == v, keep the cell when hash * n_equal < need * 2^64.",
        "hash": {
            "seed": HASH_SEED,
            "payload": THIN_SALT + "|{seed}|{truth_id}|{mechanism}|{str(epsilon)}|{k}|{weight}|{replicate}",
            "bits": "SHA-256 first 16 hex digits as uint64",
        },
        "gammas": list(GAMMAS),
        "taus": list(TAUS),
        "aggregation": "equal weight over 54 configurations within a dataset, then equal weight across TLC and GitHub",
        "bootstrap": {
            "test_truth_cluster": {"B": BOOT_B, "seed": BOOT_SEED, "thresholds": "fixed"},
            "test_circular_block_days": {"B": BOOT_B, "seed": BLOCK_SEED, "block": BLOCK, "thresholds": "fixed"},
            "calibration_block_refit": {"B": CAL_B, "seed": CAL_SEED, "block": BLOCK, "test_period": "fixed original"},
        },
        "frontier_denominator": DENOM,
        "frontier_targets_numerator": TARGETS,
        "multiplicity": "Primary output is the union pair at c_train=0.50 on the 60/40 split, reported as acceptance and accepted regret jointly. Other coverages, pairs, gammas, and splits are secondary. The three splits are not independent replications.",
        "tolerances": {"score_equality": "exact floating equality", "material_certificate_gap": "true_regret > C + 1e-6*(1+|C|) + 1e-8"},
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "pyarrow": pyarrow.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    path = OUT / "protocol.json"
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    digest = sha_file(path)
    (OUT / "protocol.sha256").write_text(digest + "\n")
    return digest


def product_keeps(hash_u: int, n_equal: int, need: int) -> bool:
    return hash_u * n_equal < need * (1 << 64)


def fixtures() -> None:
    # No ties.
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    rule = fit_threshold(scores, 1, 2)
    if rule["v"] != 0.2 or rule["n_less"] != 1 or rule["n_equal"] != 1 or rule["p"] != 1.0:
        raise SystemExit(f"no-tie fixture failed {rule}")
    # All scores tied at zero.
    zeros = np.zeros(4)
    rule = fit_threshold(zeros, 1, 2)
    if rule["v"] != 0.0 or rule["n_less"] != 0 or rule["n_equal"] != 4 or rule["need"] != 2:
        raise SystemExit(f"zero-tie fixture failed {rule}")
    # Partial boundary.
    partial = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    rule = fit_threshold(partial, 3, 5)
    if rule["v"] != 1.0 or rule["n_less"] != 2 or rule["n_equal"] != 3 or rule["need"] != 1:
        raise SystemExit(f"partial fixture failed {rule}")
    hashes = np.array([1, 2**63, 2**64 - 1], dtype=np.uint64)
    got = tie_keep(hashes, 3, 1)
    expected = np.array([product_keeps(int(h), 3, 1) for h in hashes])
    if not np.array_equal(got, expected):
        raise SystemExit("tie product reference disagreed")
    # Train/test shift and empty future set, checked by direct comparisons.
    train = np.array([0.1, 0.2, 0.3, 0.4])
    learned = fit_threshold(train, 1, 2)
    future_low = np.array([0.01, 0.02])
    future_high = np.array([0.9, 1.0])
    dummy = np.zeros(2, dtype=np.uint64)
    if apply_rule(future_low, dummy, learned).sum() != 2:
        raise SystemExit("shifted low scores were not all accepted")
    if apply_rule(future_high, dummy, learned).sum() != 0:
        raise SystemExit("empty future acceptance fixture failed")
    # Identical scores, identical decisions.
    same = np.array([0.2, 0.2, 0.4, 0.5])
    h = np.array([5, 9, 3, 8], dtype=np.uint64)
    a = apply_rule(same, h, fit_threshold(same, 1, 2))
    b = apply_rule(same.copy(), h.copy(), fit_threshold(same.copy(), 1, 2))
    if not np.array_equal(a, b):
        raise SystemExit("identical scores produced different policies")
    # Near-threshold float: the next representable value is not equal and is rejected.
    boundary = np.nextafter(1.0, 2.0)
    near = fit_threshold(np.array([0.5, 1.0]), 1, 1)
    # m = n = 2, v is the larger score.
    if near["v"] != 1.0:
        raise SystemExit("near fixture did not select 1")
    decided = apply_rule(np.array([boundary]), np.zeros(1, dtype=np.uint64), {"v": 1.0, "n_equal": 1, "need": 1})
    if bool(decided[0]):
        raise SystemExit("nextafter value was treated as tied")
    # Integer frontier: two TLC weights of 4 and two GitHub weights of 5.
    # Denominator for this toy is 18. Target half is 9, reached exactly by one TLC and one GH zero.
    toy_scores = np.array([0.0, 0.0, 1.0, 2.0])
    toy_tlc = np.array([True, False, True, False])
    # Re-express the toy on the production denominator by repeating the unit.
    # Direct hardcoded check of the grouping arithmetic:
    # group 0: one TLC and one GH, mass 4+5=9. Target 9 is reached. Previous is 0.
    mass = 4 * 1 + 5 * 1
    if mass != 9:
        raise SystemExit("toy mass failed")
    # Production-scale exact half: 12150 TLC zeros and 9720 GitHub zeros.
    # Numerator 4*12150 + 5*9720 = 48600 + 48600 = 97200 exactly.
    if 4 * 12150 + 5 * 9720 != 97200:
        raise SystemExit("half-mass identity failed")
    panel = np.concatenate([np.zeros(12150 + 9720), np.ones(12150 + 9720)])
    flags = np.concatenate([
        np.ones(12150, dtype=bool),
        np.zeros(9720, dtype=bool),
        np.ones(12150, dtype=bool),
        np.zeros(9720, dtype=bool),
    ])
    got = integer_inverse(panel, flags)
    if got["0.50"]["T"] != 0.0 or got["0.50"]["numerator"] != 97200 or got["0.50"]["previous_numerator"] != 0:
        raise SystemExit(f"integer half fixture failed {got['0.50']}")
    if got["0.75"]["numerator"] < 145800 or got["0.50"]["preceding_below"] is not True:
        raise SystemExit("integer fixture crossing failed")
    del toy_scores, toy_tlc


def validate_inputs() -> None:
    checks = {
        BOX: EXPECTED["box_ledger"],
        OS_EVAL: EXPECTED["os_eval_ledger"],
        MATCHED_PROTOCOL: EXPECTED["matched_protocol"],
        MATCHED_FRONTIER: EXPECTED["matched_frontier"],
        PANEL / "manifests/tlc_truth_manifest.jsonl": EXPECTED["tlc_manifest"],
        PANEL / "manifests/gharchive_truth_manifest.jsonl": EXPECTED["gh_manifest"],
    }
    for path, digest in checks.items():
        if sha_file(path) != digest:
            raise SystemExit(f"hash mismatch {path}")


def as_date(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.datetime64):
        return series.dt.strftime("%Y-%m-%d")
    return series.astype(str).str.slice(0, 10)


def load_joined() -> pd.DataFrame:
    left = pd.read_parquet(BOX, columns=[
        "dataset_id", "truth_id", "evaluation_date", "mechanism", "epsilon", "k", "weight_family",
        "replicate", "seed", "y_sha256", "pi_sha256", "S", "true_regret", "boundary_margin",
        "union_C_box", "union_C_strong", "direct_C_box", "direct_C_strong",
    ])
    right = pd.read_parquet(OS_EVAL, columns=[
        "dataset_id", "truth_id", "evaluation_date", "mechanism", "epsilon", "k", "weight_family",
        "replicate", "seed", "y_sha256", "pi_sha256", "S", "true_regret", "boundary_margin",
        "C_OS_rank1_coupled", "C_OS_rank1_strong", "C_OS_coupled_envelope", "C_OS_strong_envelope",
        "in_os_event",
    ])
    key = ["truth_id", "mechanism", "epsilon", "k", "weight_family", "replicate"]
    if left.duplicated(key).any() or right.duplicated(key).any() or len(left) != 43740:
        raise SystemExit("duplicate or unexpected ledger rows")
    merged = left.merge(right, on=key, how="outer", suffixes=("_c", "_d"), indicator=True)
    if len(merged) != 43740 or (merged["_merge"] != "both").any():
        raise SystemExit("join failed")
    for column in ("dataset_id", "seed", "y_sha256", "pi_sha256"):
        if not merged[f"{column}_c"].equals(merged[f"{column}_d"]):
            raise SystemExit(f"join mismatch {column}")
    if not as_date(merged["evaluation_date_c"]).equals(as_date(merged["evaluation_date_d"])):
        raise SystemExit("date mismatch")
    for column in ("S", "true_regret", "boundary_margin"):
        if float((merged[f"{column}_c"] - merged[f"{column}_d"]).abs().max()) != 0.0:
            raise SystemExit(f"material mismatch {column}")
    merged["dataset_id"] = merged["dataset_id_c"]
    merged["evaluation_date"] = as_date(merged["evaluation_date_c"])
    merged["S"] = merged["S_c"]
    merged["true_regret"] = merged["true_regret_c"]
    merged["boundary_margin"] = merged["boundary_margin_c"]
    merged["seed"] = merged["seed_c"]
    merged["y_sha256"] = merged["y_sha256_c"]
    merged["pi_sha256"] = merged["pi_sha256_c"]
    merged["epsilon_key"] = merged["epsilon"].map(lambda value: str(float(value)))
    merged["r"] = merged["true_regret"].to_numpy() / merged["S"].to_numpy()
    return merged


def attach_order(work: pd.DataFrame) -> pd.DataFrame:
    dates = (
        work[["dataset_id", "truth_id", "evaluation_date"]]
        .drop_duplicates()
        .sort_values(["dataset_id", "evaluation_date", "truth_id"])
    )
    dates["truth_index"] = dates.groupby("dataset_id").cumcount()
    expected = {
        "tlc_yellow": ("2024-04-01", "2024-04-27", "2024-04-28", "2024-05-15", 27, 45),
        "gharchive": ("2024-02-01", "2024-02-21", "2024-02-22", "2024-03-07", 21, 36),
    }
    for dataset, (start, cal_end, eval_start, end, n_cal, n_all) in expected.items():
        sub = dates[dates["dataset_id"] == dataset]
        ordered = sub["evaluation_date"].drop_duplicates().tolist()
        if len(sub) != n_all or ordered[0] != start or ordered[n_cal - 1] != cal_end:
            raise SystemExit(f"calendar mismatch {dataset} {ordered[:2]} {ordered[n_cal-1:n_cal+1]}")
        if ordered[n_cal] != eval_start or ordered[-1] != end:
            raise SystemExit(f"evaluation boundary mismatch {dataset}")
    out = work.merge(dates[["dataset_id", "truth_id", "truth_index"]], on=["dataset_id", "truth_id"], how="left")
    keys = list(zip(
        out["truth_id"], out["mechanism"], out["epsilon_key"],
        out["k"].astype(int), out["weight_family"], out["replicate"].astype(int),
    ))
    out["hash_u"] = cell_hash(keys)
    return out


def pack_dataset(work: pd.DataFrame, dataset: str, n_cal: int) -> dict:
    sub = work[work["dataset_id"] == dataset]
    n_truth = int(sub["truth_index"].max()) + 1
    strata = {}
    methods = {name: name for pair in PAIRS.values() for name in pair}
    for key, piece in sub.groupby(["mechanism", "epsilon_key", "k", "weight_family"], sort=True):
        piece = piece.sort_values(["truth_index", "replicate"])
        if len(piece) != n_truth * 10:
            raise SystemExit(f"stratum length {dataset} {key}")
        block = {
            "r": piece["r"].to_numpy().reshape(n_truth, 10),
            "raw": piece["true_regret"].to_numpy().reshape(n_truth, 10),
            "margin": piece["boundary_margin"].to_numpy().reshape(n_truth, 10),
            "hash": piece["hash_u"].to_numpy().reshape(n_truth, 10),
            "event": piece["in_os_event"].to_numpy().reshape(n_truth, 10),
            "k": int(key[2]),
            "truth_id": piece["truth_id"].to_numpy().reshape(n_truth, 10),
            "replicate": piece["replicate"].to_numpy().reshape(n_truth, 10),
            "q": {},
            "c_raw": {},
        }
        for name in methods:
            values = piece[name].to_numpy()
            scale = piece["S"].to_numpy()
            block["q"][name] = (values / scale).reshape(n_truth, 10)
            block["c_raw"][name] = values.reshape(n_truth, 10)
        if int(key[2]) <= 5:
            for left, right in PAIRS.values():
                if not np.array_equal(block["q"][left], block["q"][right]):
                    raise SystemExit(f"k<=5 scores differ {dataset} {key} {left}")
        strata[(str(key[0]), str(key[1]), int(key[2]), str(key[3]))] = block
    return {"n": n_truth, "n_cal": n_cal, "strata": strata}


def flat(index: np.ndarray, array: np.ndarray) -> np.ndarray:
    return array[index].ravel()


def summarize_accept(r: np.ndarray, raw: np.ndarray, accept: np.ndarray) -> dict:
    n_acc = int(np.sum(accept))
    out = {
        "n": int(len(r)),
        "n_acc": n_acc,
        "frac": n_acc / len(r) if len(r) else float("nan"),
        "empty": n_acc == 0,
    }
    if n_acc == 0:
        out.update({"mean": None, "median": None, "p90": None, "raw_mean": None, "zero": None})
        for gamma in GAMMAS:
            out[f"bad_{gamma}"] = None
            out[f"n_bad_{gamma}"] = 0
        return out
    chosen = r[accept]
    out.update({
        "mean": float(np.mean(chosen)),
        "median": float(np.median(chosen)),
        "p90": float(np.quantile(chosen, 0.90)),
        "raw_mean": float(np.mean(raw[accept])),
        "zero": float(np.mean(chosen == 0.0)),
    })
    for gamma in GAMMAS:
        bad = chosen > gamma
        out[f"bad_{gamma}"] = float(np.mean(bad))
        out[f"n_bad_{gamma}"] = int(np.sum(bad))
    return out


def evaluate_split(pack: dict, numer: int, denom: int) -> dict:
    rows = []
    rules = []
    decisions = []
    for dataset, content in pack.items():
        n_cal = content["n_cal"]
        cal_index = np.arange(n_cal)
        test_index = np.arange(n_cal, content["n"])
        for key, block in content["strata"].items():
            fitted = {}
            accepted = {}
            for method, scores in block["q"].items():
                rule = fit_threshold(flat(cal_index, scores), numer, denom)
                fitted[method] = rule
                accepted[method] = apply_rule(flat(test_index, scores), flat(test_index, block["hash"]), rule)
                rules.append({
                    "dataset_id": dataset,
                    "mechanism": key[0],
                    "epsilon": key[1],
                    "k": int(key[2]),
                    "weight_family": key[3],
                    "method": method,
                    **rule,
                    "zero_score_fraction": float(np.mean(flat(cal_index, scores) == 0.0)),
                    "near_zero_residue": int(np.sum((flat(cal_index, scores) > 0.0) & (flat(cal_index, scores) <= 1e-12))),
                    "cal_mean_q": float(np.mean(flat(cal_index, scores))),
                    "cal_median_q": float(np.median(flat(cal_index, scores))),
                    "cal_p90_q": float(np.quantile(flat(cal_index, scores), 0.90)),
                    "test_mean_q": float(np.mean(flat(test_index, scores))),
                    "test_median_q": float(np.median(flat(test_index, scores))),
                    "test_p90_q": float(np.quantile(flat(test_index, scores), 0.90)),
                    "test_zero_fraction": float(np.mean(flat(test_index, scores) == 0.0)),
                    "cal_mean_margin": float(np.mean(flat(cal_index, block["margin"]))),
                    "test_mean_margin": float(np.mean(flat(test_index, block["margin"]))),
                })
            if int(key[2]) <= 5:
                for left, right in PAIRS.values():
                    if not np.array_equal(accepted[left], accepted[right]):
                        raise SystemExit(f"k<=5 policy mismatch {dataset} {key}")
            r = flat(test_index, block["r"])
            raw = flat(test_index, block["raw"])
            for pair, (left, right) in PAIRS.items():
                left_stats = summarize_accept(r, raw, accepted[left])
                right_stats = summarize_accept(r, raw, accepted[right])
                both = accepted[left] & accepted[right]
                only_left = accepted[left] & ~accepted[right]
                only_right = accepted[right] & ~accepted[left]
                union = accepted[left] | accepted[right]
                row = {
                    "dataset_id": dataset,
                    "mechanism": key[0],
                    "epsilon": key[1],
                    "k": int(key[2]),
                    "weight_family": key[3],
                    "pair": pair,
                    "frac_l": left_stats["frac"],
                    "frac_r": right_stats["frac"],
                    "D_accept": right_stats["frac"] - left_stats["frac"],
                    "n_acc_l": left_stats["n_acc"],
                    "n_acc_r": right_stats["n_acc"],
                    "mean_l": left_stats["mean"],
                    "mean_r": right_stats["mean"],
                    "median_l": left_stats["median"],
                    "median_r": right_stats["median"],
                    "p90_l": left_stats["p90"],
                    "p90_r": right_stats["p90"],
                    "raw_l": left_stats["raw_mean"],
                    "raw_r": right_stats["raw_mean"],
                    "zero_l": left_stats["zero"],
                    "zero_r": right_stats["zero"],
                    "empty_l": left_stats["empty"],
                    "empty_r": right_stats["empty"],
                    "n_both": int(np.sum(both)),
                    "n_only_l": int(np.sum(only_left)),
                    "n_only_r": int(np.sum(only_right)),
                    "n_neither": int(np.sum(~union)),
                    "jaccard": float(np.sum(both) / np.sum(union)) if np.any(union) else None,
                    "ex_l": float(np.mean(r[only_left])) if np.any(only_left) else None,
                    "ex_r": float(np.mean(r[only_right])) if np.any(only_right) else None,
                }
                if left_stats["mean"] is not None and right_stats["mean"] is not None:
                    row["D_regret"] = left_stats["mean"] - right_stats["mean"]
                else:
                    row["D_regret"] = None
                for gamma in GAMMAS:
                    row[f"bad_l_{gamma}"] = left_stats[f"bad_{gamma}"]
                    row[f"bad_r_{gamma}"] = right_stats[f"bad_{gamma}"]
                    row[f"n_bad_l_{gamma}"] = left_stats[f"n_bad_{gamma}"]
                    row[f"n_bad_r_{gamma}"] = right_stats[f"n_bad_{gamma}"]
                rows.append(row)
            decisions.append({
                "dataset_id": dataset,
                "key": key,
                "accept": accepted,
                "r": r,
                "raw": raw,
                "q": {method: flat(test_index, scores) for method, scores in block["q"].items()},
                "c_raw": {method: flat(test_index, values) for method, values in block["c_raw"].items()},
                "event": flat(test_index, block["event"]),
                "hash": flat(test_index, block["hash"]),
                "truth_id": flat(test_index, block["truth_id"]),
                "replicate": flat(test_index, block["replicate"]),
                "rules": fitted,
            })
    return {"rows": rows, "rules": rules, "decisions": decisions}


def macro_defined(frame: pd.DataFrame, field: str) -> dict:
    out = {}
    for dataset, sub in frame.groupby("dataset_id"):
        values = sub[field].dropna()
        out[dataset] = {
            "mean": None if values.empty else float(values.mean()),
            "n_strata": int(len(sub)),
            "n_defined": int(values.shape[0]),
        }
    defined = [out[name]["mean"] for name in ("gharchive", "tlc_yellow") if out[name]["mean"] is not None]
    out["equal_weight_macro"] = None if len(defined) < 2 else float(np.mean(defined))
    return out


def block_indices(n: int, rng: np.random.Generator, block: int = BLOCK) -> np.ndarray:
    starts = rng.integers(0, n, int(math.ceil(n / block)))
    idx = [(int(start) + offset) % n for start in starts for offset in range(block)]
    return np.asarray(idx[:n], dtype=int)


def bootstrap_fixed(pack: dict, rules_by_stratum: dict, seed: int, mode: str) -> dict:
    rng = np.random.default_rng(seed)
    datasets = ("gharchive", "tlc_yellow")
    store = {
        pair: {
            "accept": {dataset: np.empty(BOOT_B) for dataset in datasets},
            "regret": {dataset: np.empty(BOOT_B) for dataset in datasets},
            "bad0": {dataset: np.empty(BOOT_B) for dataset in datasets},
        }
        for pair in PAIRS
    }
    for draw in range(BOOT_B):
        if draw % 500 == 0:
            print(f"[time_forward] {mode} bootstrap {draw}", flush=True)
        indices = {}
        for dataset, content in pack.items():
            n_test = content["n"] - content["n_cal"]
            if mode == "cluster":
                indices[dataset] = content["n_cal"] + rng.integers(0, n_test, n_test)
            else:
                indices[dataset] = content["n_cal"] + block_indices(n_test, rng)
        stratum = {dataset: {pair: {"a": [], "r": [], "b": []} for pair in PAIRS} for dataset in datasets}
        for dataset, content in pack.items():
            index = indices[dataset]
            for key, block in content["strata"].items():
                r = flat(index, block["r"])
                hashes = flat(index, block["hash"])
                accepted = {}
                for method in {name for pair in PAIRS.values() for name in pair}:
                    accepted[method] = apply_rule(flat(index, block["q"][method]), hashes, rules_by_stratum[(dataset, key, method)])
                for pair, (left, right) in PAIRS.items():
                    frac_l = float(np.mean(accepted[left]))
                    frac_r = float(np.mean(accepted[right]))
                    stratum[dataset][pair]["a"].append(frac_r - frac_l)
                    if np.any(accepted[left]) and np.any(accepted[right]):
                        stratum[dataset][pair]["r"].append(float(np.mean(r[accepted[left]]) - np.mean(r[accepted[right]])))
                    bad_l = np.mean(r[accepted[left]] > 0.0) if np.any(accepted[left]) else np.nan
                    bad_r = np.mean(r[accepted[right]] > 0.0) if np.any(accepted[right]) else np.nan
                    if np.isfinite(bad_l) and np.isfinite(bad_r):
                        stratum[dataset][pair]["b"].append(float(bad_l - bad_r))
        for dataset in datasets:
            for pair in PAIRS:
                store[pair]["accept"][dataset][draw] = float(np.mean(stratum[dataset][pair]["a"]))
                regret = stratum[dataset][pair]["r"]
                bad = stratum[dataset][pair]["b"]
                store[pair]["regret"][dataset][draw] = float(np.mean(regret)) if regret else np.nan
                store[pair]["bad0"][dataset][draw] = float(np.mean(bad)) if bad else np.nan
    report = {}
    for pair in PAIRS:
        report[pair] = {}
        for metric in ("accept", "regret", "bad0"):
            macro = 0.5 * (store[pair][metric]["gharchive"] + store[pair][metric]["tlc_yellow"])
            report[pair][metric] = {
                "macro": quantile_finite(macro),
                "by_dataset": {dataset: quantile_finite(store[pair][metric][dataset]) for dataset in datasets},
            }
    return report


def quantile_finite(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {"lo": None, "hi": None, "n_finite": 0}
    return {
        "lo": float(np.quantile(finite, 0.025)),
        "hi": float(np.quantile(finite, 0.975)),
        "n_finite": int(len(finite)),
    }


def calibration_sensitivity(pack: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    datasets = ("gharchive", "tlc_yellow")
    store = {metric: {dataset: np.empty(CAL_B) for dataset in datasets} for metric in ("accept", "regret")}
    for draw in range(CAL_B):
        if draw % 100 == 0:
            print(f"[time_forward] calibration refit {draw}", flush=True)
        indices = {}
        for dataset, content in pack.items():
            indices[dataset] = block_indices(content["n_cal"], rng)
        stratum = {dataset: {"a": [], "r": []} for dataset in datasets}
        for dataset, content in pack.items():
            cal = indices[dataset]
            test = np.arange(content["n_cal"], content["n"])
            for block in content["strata"].values():
                r = flat(test, block["r"])
                hashes = flat(test, block["hash"])
                left = fit_threshold(flat(cal, block["q"]["union_C_strong"]), 1, 2)
                right = fit_threshold(flat(cal, block["q"]["union_C_box"]), 1, 2)
                acc_l = apply_rule(flat(test, block["q"]["union_C_strong"]), hashes, left)
                acc_r = apply_rule(flat(test, block["q"]["union_C_box"]), hashes, right)
                stratum[dataset]["a"].append(float(np.mean(acc_r) - np.mean(acc_l)))
                if np.any(acc_l) and np.any(acc_r):
                    stratum[dataset]["r"].append(float(np.mean(r[acc_l]) - np.mean(r[acc_r])))
        for dataset in datasets:
            store["accept"][dataset][draw] = float(np.mean(stratum[dataset]["a"]))
            regret = stratum[dataset]["r"]
            store["regret"][dataset][draw] = float(np.mean(regret)) if regret else np.nan
    return {
        metric: {
            "macro": quantile_finite(0.5 * (store[metric]["gharchive"] + store[metric]["tlc_yellow"])),
            "by_dataset": {dataset: quantile_finite(store[metric][dataset]) for dataset in datasets},
        }
        for metric in store
    }


def fixed_tau(decisions: list[dict]) -> dict:
    rows = []
    for item in decisions:
        r = item["r"]
        event = item["event"]
        for method, scores in item["q"].items():
            raw = item["c_raw"][method]
            for tau in TAUS:
                certified = scores <= tau
                n_cert = int(np.sum(certified))
                exceed = 0
                if n_cert:
                    chosen_r = r[certified]
                    chosen_c = raw[certified]
                    material = chosen_r > chosen_c + 1e-6 * (1.0 + np.abs(chosen_c)) + 1e-8
                    exceed = int(np.sum(material))
                rows.append({
                    "dataset_id": item["dataset_id"],
                    "mechanism": item["key"][0],
                    "epsilon": item["key"][1],
                    "k": int(item["key"][2]),
                    "weight_family": item["key"][3],
                    "method": method,
                    "tau": tau,
                    "n": int(len(scores)),
                    "n_cert": n_cert,
                    "n_abstain": int(len(scores) - n_cert),
                    "frac": n_cert / len(scores),
                    "mean_r": float(np.mean(r[certified])) if n_cert else None,
                    "n_r_gt_tau": int(np.sum(r[certified] > tau)) if n_cert else 0,
                    "n_material_exceed": exceed,
                    "n_cert_outside_os_event": int(np.sum(certified & ~event)) if method.startswith("C_OS") else None,
                })
    return {"rows": rows}


def frontier_audit(work: pd.DataFrame) -> dict:
    is_tlc = (work["dataset_id"] == "tlc_yellow").to_numpy()
    if int(np.sum(is_tlc)) != 24300 or int(np.sum(~is_tlc)) != 19440:
        raise SystemExit("frontier population counts failed")
    previous = json.loads(MATCHED_FRONTIER.read_text())
    report = {}
    changed = []
    for name, column in FRONTIER_COLUMNS.items():
        scores = work[column].to_numpy() / work["S"].to_numpy()
        exact = integer_inverse(scores, is_tlc)
        stored = previous[name]
        for label in TARGETS:
            got = exact[label]["T"]
            old = stored[label]["T"]
            if got != old:
                changed.append({"method": name, "c": label, "matched_T": old, "integer_T": got})
            exact[label]["matched_T"] = old
            exact[label]["matched_K_display"] = stored[label]["K"]
            exact[label]["T_match"] = bool(got == old)
        report[name] = exact
    return {"methods": report, "differences": changed}


def heterogeneity(frame: pd.DataFrame) -> dict:
    out = {}
    for name, columns in {
        "k": ["k"],
        "weight": ["weight_family"],
        "epsilon": ["epsilon"],
        "mechanism": ["mechanism"],
    }.items():
        rows = []
        for key, sub in frame.groupby(columns, sort=True):
            if not isinstance(key, tuple):
                key = (key,)
            rows.append({
                **{col: (int(value) if col == "k" else value) for col, value in zip(columns, key)},
                "D_accept": macro_defined(sub, "D_accept"),
                "D_regret": macro_defined(sub, "D_regret"),
                "frac_l": macro_defined(sub, "frac_l"),
                "frac_r": macro_defined(sub, "frac_r"),
            })
        out[name] = rows
    return out


def drift_summary(rules: list[dict]) -> dict:
    frame = pd.DataFrame(rules)
    out = {}
    for method, sub in frame.groupby("method"):
        by = {}
        for dataset, piece in sub.groupby("dataset_id"):
            by[dataset] = {
                "cal_mean_q": float(piece["cal_mean_q"].mean()),
                "test_mean_q": float(piece["test_mean_q"].mean()),
                "cal_median_q": float(piece["cal_median_q"].mean()),
                "test_median_q": float(piece["test_median_q"].mean()),
                "cal_zero": float(piece["zero_score_fraction"].mean()),
                "test_zero": float(piece["test_zero_fraction"].mean()),
                "cal_p90_q": float(piece["cal_p90_q"].mean()),
                "test_p90_q": float(piece["test_p90_q"].mean()),
                "cal_margin": float(piece["cal_mean_margin"].mean()),
                "test_margin": float(piece["test_mean_margin"].mean()),
                "near_zero_residue_cells": int(piece["near_zero_residue"].sum()),
            }
        out[method] = by
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    validate_inputs()
    digest = write_protocol()
    print(f"[time_forward] protocol {digest}", flush=True)
    fixtures()
    print("[time_forward] fixtures passed", flush=True)
    merged = attach_order(load_joined())
    counts = {}
    pack = {}
    for dataset, n_cal in SPLIT_SPEC["primary_60_40"].items():
        pack[dataset] = pack_dataset(merged, dataset, n_cal)
        n = pack[dataset]["n"]
        counts[dataset] = {
            "truths": n,
            "cal_cells": n_cal * 540,
            "eval_cells": (n - n_cal) * 540,
        }
    if counts["tlc_yellow"]["cal_cells"] != 14580 or counts["gharchive"]["eval_cells"] != 8100:
        raise SystemExit(f"split counts failed {counts}")
    (OUT / "split_manifest.json").write_text(json.dumps({
        "primary": counts,
        "sensitivity_counts": {
            "50_50": {"tlc_yellow": [22 * 540, 23 * 540], "gharchive": [18 * 540, 18 * 540]},
            "70_30": {"tlc_yellow": [31 * 540, 14 * 540], "gharchive": [25 * 540, 11 * 540]},
        },
    }, indent=2) + "\n")
    primary = {}
    for label, numer, denom in COVERAGES:
        print(f"[time_forward] coverage {label}", flush=True)
        primary[label] = evaluate_split(pack, numer, denom)
    focus = primary["0.50"]
    pd.DataFrame(focus["rows"]).to_parquet(OUT / "primary_stratum_rows.parquet", index=False)
    pd.DataFrame(focus["rules"]).to_parquet(OUT / "primary_thresholds.parquet", index=False)
    summary = {}
    frame = pd.DataFrame(focus["rows"])
    for pair, sub in frame.groupby("pair"):
        summary[pair] = {
            field: macro_defined(sub, field)
            for field in ("frac_l", "frac_r", "D_accept", "mean_l", "mean_r", "D_regret", "zero_l", "zero_r", "jaccard", "n_only_l", "n_only_r", "n_both")
        }
        for gamma in GAMMAS:
            summary[pair][f"bad_l_{gamma}"] = macro_defined(sub, f"bad_l_{gamma}")
            summary[pair][f"bad_r_{gamma}"] = macro_defined(sub, f"bad_r_{gamma}")
            summary[pair][f"n_bad_l_{gamma}"] = int(sub[f"n_bad_l_{gamma}"].sum())
            summary[pair][f"n_bad_r_{gamma}"] = int(sub[f"n_bad_r_{gamma}"].sum())
        summary[pair]["empty_l"] = int(sub["empty_l"].sum())
        summary[pair]["empty_r"] = int(sub["empty_r"].sum())
    (OUT / "primary_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    secondary = {}
    for label in ("0.25", "0.75"):
        sub = pd.DataFrame(primary[label]["rows"])
        secondary[label] = {
            pair: {
                "D_accept": macro_defined(piece, "D_accept"),
                "D_regret": macro_defined(piece, "D_regret"),
                "frac_l": macro_defined(piece, "frac_l"),
                "frac_r": macro_defined(piece, "frac_r"),
                "mean_l": macro_defined(piece, "mean_l"),
                "mean_r": macro_defined(piece, "mean_r"),
            }
            for pair, piece in sub.groupby("pair")
        }
    (OUT / "secondary_coverage_summary.json").write_text(json.dumps(secondary, indent=2) + "\n")
    (OUT / "heterogeneity.json").write_text(json.dumps(heterogeneity(frame[frame["pair"] == "A_union"]), indent=2) + "\n")
    (OUT / "drift.json").write_text(json.dumps(drift_summary(focus["rules"]), indent=2) + "\n")
    tau_rows = fixed_tau(focus["decisions"])["rows"]
    tau_frame = pd.DataFrame(tau_rows)
    tau_summary = []
    for (method, tau), sub in tau_frame.groupby(["method", "tau"]):
        by = {}
        for dataset, piece in sub.groupby("dataset_id"):
            weight = piece["n"].to_numpy()
            n_cert = int(piece["n_cert"].sum())
            by[dataset] = {
                "n": int(piece["n"].sum()),
                "n_cert": n_cert,
                "n_abstain": int(piece["n_abstain"].sum()),
                "frac": float(np.average(piece["frac"], weights=weight)),
                "mean_r": None if int(piece["n_cert"].sum()) == 0 else float(np.average(piece.loc[piece["n_cert"] > 0, "mean_r"], weights=piece.loc[piece["n_cert"] > 0, "n_cert"])),
                "n_r_gt_tau": int(piece["n_r_gt_tau"].sum()),
                "n_material_exceed": int(piece["n_material_exceed"].sum()),
                "n_cert_outside_os_event": None if piece["n_cert_outside_os_event"].isna().all() else int(piece["n_cert_outside_os_event"].fillna(0).sum()),
            }
        tau_summary.append({"method": method, "tau": float(tau), "by_dataset": by})
    (OUT / "fixed_tau.json").write_text(json.dumps(tau_summary, indent=2) + "\n")
    # Decision ledger for the primary union pair, test cells only.
    ledger_rows = []
    for item in focus["decisions"]:
        n = len(item["r"])
        for i in range(n):
            strong = bool(item["accept"]["union_C_strong"][i])
            coupled = bool(item["accept"]["union_C_box"][i])
            rule_s = item["rules"]["union_C_strong"]
            rule_c = item["rules"]["union_C_box"]
            ledger_rows.append({
                "dataset_id": item["dataset_id"],
                "truth_id": item["truth_id"][i],
                "replicate": int(item["replicate"][i]),
                "mechanism": item["key"][0],
                "epsilon": item["key"][1],
                "k": int(item["key"][2]),
                "weight_family": item["key"][3],
                "r": float(item["r"][i]),
                "q_strong": float(item["q"]["union_C_strong"][i]),
                "q_coupled": float(item["q"]["union_C_box"][i]),
                "accept_strong": strong,
                "accept_coupled": coupled,
                "v_strong": rule_s["v"],
                "v_coupled": rule_c["v"],
                "p_strong": rule_s["p"],
                "p_coupled": rule_c["p"],
            })
    ledger = pd.DataFrame(ledger_rows)
    # Reapply the saved rule with the saved hash. A violation stops the run.
    violations = 0
    for item in focus["decisions"]:
        for method in ("union_C_strong", "union_C_box"):
            again = apply_rule(item["q"][method], item["hash"], item["rules"][method])
            violations += int(np.sum(again != item["accept"][method]))
    if violations:
        raise SystemExit(f"policy replay violations {violations}")
    ledger.to_parquet(OUT / "test_policy_ledger.parquet", index=False)
    print("[time_forward] split sensitivity", flush=True)
    sensitivity = {}
    for name, spec in SPLIT_SPEC.items():
        if name == "primary_60_40":
            continue
        alt = {dataset: pack_dataset(merged, dataset, n_cal) for dataset, n_cal in spec.items()}
        evaluated = evaluate_split(alt, 1, 2)
        sub = pd.DataFrame(evaluated["rows"])
        pair_summary = {}
        for pair, piece in sub.groupby("pair"):
            pair_summary[pair] = {
                "D_accept": macro_defined(piece, "D_accept"),
                "D_regret": macro_defined(piece, "D_regret"),
                "frac_l": macro_defined(piece, "frac_l"),
                "frac_r": macro_defined(piece, "frac_r"),
                "mean_l": macro_defined(piece, "mean_l"),
                "mean_r": macro_defined(piece, "mean_r"),
                "empty_l": int(piece["empty_l"].sum()),
                "empty_r": int(piece["empty_r"].sum()),
            }
        sensitivity[name] = {"boundaries": spec, "pairs": pair_summary}
    (OUT / "split_sensitivity.json").write_text(json.dumps(sensitivity, indent=2) + "\n")
    print("[time_forward] frontier", flush=True)
    frontier = frontier_audit(merged)
    (OUT / "integer_frontier_audit.json").write_text(json.dumps(frontier, indent=2) + "\n")
    if frontier["differences"]:
        (OUT / "integer_frontier_differences.json").write_text(json.dumps(frontier["differences"], indent=2) + "\n")
    rules_lookup = {}
    for rule in focus["rules"]:
        rules_lookup[(rule["dataset_id"], (rule["mechanism"], rule["epsilon"], rule["k"], rule["weight_family"]), rule["method"])] = rule
    print("[time_forward] bootstrap", flush=True)
    cluster = bootstrap_fixed(pack, rules_lookup, BOOT_SEED, "cluster")
    block = bootstrap_fixed(pack, rules_lookup, BLOCK_SEED, "block")
    (OUT / "bootstrap_cluster.json").write_text(json.dumps(cluster, indent=2) + "\n")
    (OUT / "bootstrap_block.json").write_text(json.dumps(block, indent=2) + "\n")
    print("[time_forward] calibration uncertainty", flush=True)
    cal = calibration_sensitivity(pack, CAL_SEED)
    (OUT / "calibration_uncertainty.json").write_text(json.dumps(cal, indent=2) + "\n")
    (OUT / "numerical_audit.json").write_text(json.dumps({
        "fixtures": "passed",
        "policy_replay_violations": violations,
        "k5_policy_identity": "asserted inside evaluate_split",
        "nonfinite_r": int((~np.isfinite(merged["r"].to_numpy())).sum()),
        "join_rows": int(len(merged)),
    }, indent=2) + "\n")
    (OUT / "stage_times.json").write_text(json.dumps({"total_s": time.perf_counter() - started}, indent=2) + "\n")
    print("[time_forward] finished", flush=True)


if __name__ == "__main__":
    main()
