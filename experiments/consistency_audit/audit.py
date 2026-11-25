"""Independent consistency audit of the real-data results.

Recomputes headline quantities from the ledgers with separate code paths.
Writes no releases.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
# Opaque seed salt: its exact bytes enter the SHA-256 seed derivation.
THIN_SALT = "G5F-THIN"
BOX = ROOT / "experiments/real_box_evaluation/ledger.parquet"
OS_EVAL = ROOT / "experiments/real_order_statistic_evaluation/ledger.parquet"
MATCHED_FRONT = ROOT / "experiments/matched_rate_utility/frontier_exact.json"
TIME_FORWARD_LEDGER = ROOT / "experiments/time_forward_utility/test_policy_ledger.parquet"
TIME_FORWARD_THR = ROOT / "experiments/time_forward_utility/primary_thresholds.parquet"
TARGETS = {
    "0.10": 19440,
    "0.25": 48600,
    "0.50": 97200,
    "0.75": 145800,
    "0.90": 174960,
    "0.95": 184680,
}
METHODS = {
    "union_coupled": ("box", "union_C_box"),
    "union_strong": ("box", "union_C_strong"),
    "direct_coupled": ("box", "direct_C_box"),
    "direct_strong": ("box", "direct_C_strong"),
    "os_rank1_coupled": ("os_eval", "C_OS_rank1_coupled"),
    "os_rank1_strong": ("os_eval", "C_OS_rank1_strong"),
    "os_coupled_envelope": ("os_eval", "C_OS_coupled_envelope"),
    "os_strong_envelope": ("os_eval", "C_OS_strong_envelope"),
}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tie_keep(hash_u: np.ndarray, n_equal: int, need: int) -> np.ndarray:
    n = len(hash_u)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if need >= n_equal:
        return np.ones(n, dtype=bool)
    if need <= 0 or n_equal <= 0:
        return np.zeros(n, dtype=bool)
    quota, remainder = divmod(int(need) << 64, int(n_equal))
    if quota >= (1 << 64):
        return np.ones(n, dtype=bool)
    values = hash_u.astype(np.uint64, copy=False)
    keep = values < np.uint64(quota)
    if remainder > 0:
        keep = keep | (values == np.uint64(quota))
    return keep


def hashes_for(frame: pd.DataFrame, seed: int) -> np.ndarray:
    out = np.empty(len(frame), dtype=np.uint64)
    for i, row in enumerate(frame.itertuples(index=False)):
        payload = (
            f"{THIN_SALT}|{seed}|{row.truth_id}|{row.mechanism}|{row.epsilon}|"
            f"{int(row.k)}|{row.weight_family}|{int(row.replicate)}"
        ).encode()
        out[i] = np.uint64(int(hashlib.sha256(payload).hexdigest()[:16], 16))
    return out


def accept_mask(q: np.ndarray, h: np.ndarray, v: np.ndarray, n_equal: np.ndarray, need: np.ndarray) -> np.ndarray:
    acc = q < v
    tied = q == v
    if not np.any(tied):
        return acc
    acc = acc.copy()
    groups = pd.DataFrame({"ne": n_equal, "nd": need}).groupby(["ne", "nd"]).groups
    for (ne, nd), idx in groups.items():
        idx = np.asarray(list(idx))
        mask = tied[idx]
        if not np.any(mask):
            continue
        sub = idx[mask]
        acc[sub] = tie_keep(h[sub], int(ne), int(nd))
    return acc


def stratum_macros(frame: pd.DataFrame, acc_s: np.ndarray, acc_c: np.ndarray) -> dict:
    work = frame[["dataset_id", "mechanism", "epsilon", "k", "weight_family"]].copy()
    work["r"] = frame["r"].to_numpy()
    work["acc_s"] = acc_s
    work["acc_c"] = acc_c
    da, dr, empty = [], [], 0
    by_source = {}
    for dataset, sub in work.groupby("dataset_id", sort=True):
        d_a, d_r = [], []
        for _, piece in sub.groupby(["mechanism", "epsilon", "k", "weight_family"], sort=True):
            left = piece["acc_s"].to_numpy()
            right = piece["acc_c"].to_numpy()
            d_a.append(float(right.mean() - left.mean()))
            if left.any() and right.any():
                d_r.append(float(piece.loc[left, "r"].mean() - piece.loc[right, "r"].mean()))
            else:
                empty += 1
        by_source[dataset] = {
            "D_accept": float(np.mean(d_a)),
            "D_regret": float(np.mean(d_r)) if d_r else None,
            "n_strata": int(len(d_a)),
            "n_defined_regret": int(len(d_r)),
        }
    sources = [by_source[name] for name in ("gharchive", "tlc_yellow")]
    return {
        "by_source": by_source,
        "D_accept": float(np.mean([row["D_accept"] for row in sources])),
        "D_regret": float(np.mean([row["D_regret"] for row in sources])),
        "empty_strata": empty,
    }


def source_pooled(frame: pd.DataFrame, acc_s: np.ndarray, acc_c: np.ndarray) -> dict:
    r = frame["r"].to_numpy()
    ds = frame["dataset_id"].to_numpy()
    da, dr = [], []
    for name in ("gharchive", "tlc_yellow"):
        m = ds == name
        da.append(float(acc_c[m].mean() - acc_s[m].mean()))
        dr.append(float(r[m & acc_s].mean() - r[m & acc_c].mean()))
    return {"D_accept": float(np.mean(da)), "D_regret": float(np.mean(dr))}


def dataset_rate(flag: np.ndarray, dataset: np.ndarray, truth: np.ndarray, name: str) -> float:
    m = dataset == name
    by = pd.Series(flag[m]).groupby(truth[m]).mean()
    return float(by.mean())


def integer_inverse(scores: np.ndarray, is_tlc: np.ndarray) -> dict:
    order = np.argsort(scores, kind="mergesort")
    ordered = scores[order]
    tlc = is_tlc[order]
    change = np.empty(len(ordered), dtype=bool)
    change[0] = True
    change[1:] = ordered[1:] != ordered[:-1]
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], len(ordered))
    unique = ordered[starts]
    mass = np.empty(len(starts), dtype=np.int64)
    for i, (a, b) in enumerate(zip(starts, ends)):
        n_tlc = int(tlc[a:b].sum())
        mass[i] = 4 * n_tlc + 5 * int((b - a) - n_tlc)
    cumulative = np.cumsum(mass, dtype=np.int64)
    out = {}
    for label, target in TARGETS.items():
        index = int(np.searchsorted(cumulative, np.int64(target), side="left"))
        previous = int(cumulative[index - 1]) if index else 0
        out[label] = {
            "T": float(unique[index]),
            "numerator": int(cumulative[index]),
            "numerator_prev": previous,
            "preceding_below": previous < target,
            "reached": int(cumulative[index]) >= target,
        }
    return out


def main() -> None:
    c = pd.read_parquet(BOX)
    d = pd.read_parquet(OS_EVAL)
    key = ["truth_id", "mechanism", "epsilon", "k", "weight_family", "replicate"]
    checks = {
        "rows_c": int(len(c)),
        "rows_d": int(len(d)),
        "truths_c": int(c.truth_id.nunique()),
        "truths_d": int(d.truth_id.nunique()),
        "duplicate_c": bool(c.duplicated(key).any()),
        "duplicate_d": bool(d.duplicated(key).any()),
        "cells_per_truth": sorted(int(v) for v in c.groupby("truth_id").size().unique()),
        "tlc_truths": int(c.loc[c.dataset_id.eq("tlc_yellow"), "truth_id"].nunique()),
        "gh_truths": int(c.loc[c.dataset_id.eq("gharchive"), "truth_id"].nunique()),
    }
    merged = c.merge(d, on=key, suffixes=("_c", "_d"), indicator=True)
    checks["join_rows"] = int(len(merged))
    checks["join_both"] = bool((merged["_merge"] == "both").all())
    for column in ("dataset_id", "seed", "y_sha256", "pi_sha256"):
        checks[f"match_{column}"] = bool(merged[f"{column}_c"].equals(merged[f"{column}_d"]))
    for column in ("S", "true_regret"):
        checks[f"max_abs_{column}"] = float((merged[f"{column}_c"] - merged[f"{column}_d"]).abs().max())

    scale = c["S"].to_numpy()
    dataset = c["dataset_id"].to_numpy()
    truth = c["truth_id"].to_numpy()
    tau = 0.10
    rates = {}
    for name, col in (
        ("union_box", "union_C_box"),
        ("union_strong", "union_C_strong"),
        ("direct_box", "direct_C_box"),
        ("direct_strong", "direct_C_strong"),
        ("union_box_plus", "union_C_box_plus"),
        ("union_strong_plus", "union_C_strong_plus_decision"),
    ):
        q = c[col].to_numpy() / scale
        flag = q <= tau
        tlc = dataset_rate(flag, dataset, truth, "tlc_yellow")
        gh = dataset_rate(flag, dataset, truth, "gharchive")
        rates[name] = {"tlc": tlc, "github": gh, "macro": 0.5 * (tlc + gh)}
    rates["delta_union"] = {
        source: rates["union_box"][source] - rates["union_strong"][source]
        for source in ("tlc", "github", "macro")
    }
    rates["delta_direct"] = {
        source: rates["direct_box"][source] - rates["direct_strong"][source]
        for source in ("tlc", "github", "macro")
    }
    rates["count_decision_disagreements_union_box"] = int(np.sum((c["union_C_box"] / scale <= tau) != (c["union_C_box_plus"] / scale <= tau)))
    rates["count_decision_disagreements_union_strong"] = int(
        np.sum((c["union_C_strong"] / scale <= tau) != (c["union_C_strong_plus_decision"] / scale <= tau))
    )
    rates["max_abs_union_box_minus_plus"] = float((c["union_C_box"] - c["union_C_box_plus"]).abs().max())
    k5 = c["k"] <= 5
    rates["k5_union_max_abs"] = float((c.loc[k5, "union_C_box"] - c.loc[k5, "union_C_strong"]).abs().max())
    rates["k5_direct_max_abs"] = float((c.loc[k5, "direct_C_box"] - c.loc[k5, "direct_C_strong"]).abs().max())

    dscale = d["S"].to_numpy()
    dos = {}
    flags = {}
    for name, col in (
        ("rank1_coupled", "C_OS_rank1_coupled"),
        ("rank1_strong", "C_OS_rank1_strong"),
        ("env_coupled", "C_OS_coupled_envelope"),
        ("env_strong", "C_OS_strong_envelope"),
    ):
        flag = d[col].to_numpy() / dscale <= tau
        flags[name] = flag
        tlc = dataset_rate(flag, d["dataset_id"].to_numpy(), d["truth_id"].to_numpy(), "tlc_yellow")
        gh = dataset_rate(flag, d["dataset_id"].to_numpy(), d["truth_id"].to_numpy(), "gharchive")
        dos[name] = {"tlc": tlc, "github": gh, "macro": 0.5 * (tlc + gh)}
    dos["rank1_coupling"] = {s: dos["rank1_coupled"][s] - dos["rank1_strong"][s] for s in ("tlc", "github", "macro")}
    dos["os_coupling"] = {s: dos["env_coupled"][s] - dos["env_strong"][s] for s in ("tlc", "github", "macro")}
    dos["coupled_refine"] = {s: dos["env_coupled"][s] - dos["rank1_coupled"][s] for s in ("tlc", "github", "macro")}
    dos["strong_refine"] = {s: dos["env_strong"][s] - dos["rank1_strong"][s] for s in ("tlc", "github", "macro")}
    dos["n_new_coupled"] = int(np.sum(flags["env_coupled"] & ~flags["rank1_coupled"]))
    dos["k5_rank1_max_abs"] = float((d.loc[d.k <= 5, "C_OS_rank1_coupled"] - d.loc[d.k <= 5, "C_OS_rank1_strong"]).abs().max())
    dos["k5_envelope_max_abs"] = float((d.loc[d.k <= 5, "C_OS_coupled_envelope"] - d.loc[d.k <= 5, "C_OS_strong_envelope"]).abs().max())

    # Matched-rate comparison at c=0.50, independent lexsort.
    work = c[["dataset_id", "truth_id", "evaluation_date", "mechanism", "epsilon", "k", "weight_family", "replicate", "S", "true_regret", "union_C_box", "union_C_strong"]].copy()
    work["epsilon_key"] = work["epsilon"].map(lambda value: str(float(value)))
    keys = list(zip(work.truth_id, work.mechanism, work.epsilon_key, work.k.astype(int), work.weight_family, work.replicate.astype(int)))
    tie = np.empty(len(keys), dtype=np.uint64)
    for i, item in enumerate(keys):
        payload = f"20261128|{item[0]}|{item[1]}|{item[2]}|{item[3]}|{item[4]}|{item[5]}".encode()
        tie[i] = np.uint64(int(hashlib.sha256(payload).hexdigest()[:16], 16))
    work["tie"] = tie
    work["r"] = work["true_regret"].to_numpy() / work["S"].to_numpy()
    dates = work[["dataset_id", "truth_id", "evaluation_date"]].drop_duplicates().sort_values(["dataset_id", "evaluation_date", "truth_id"])
    dates["truth_index"] = dates.groupby("dataset_id").cumcount()
    work = work.merge(dates[["dataset_id", "truth_id", "truth_index"]], on=["dataset_id", "truth_id"])
    work["final"] = work["truth_index"].to_numpy() * 100 + work["replicate"].to_numpy()
    matched = {}
    for dataset, sub in work.groupby("dataset_id"):
        diffs = []
        means_l, means_r = [], []
        for _, piece in sub.groupby(["mechanism", "epsilon_key", "k", "weight_family"]):
            q_l = piece["union_C_strong"].to_numpy() / piece["S"].to_numpy()
            q_r = piece["union_C_box"].to_numpy() / piece["S"].to_numpy()
            r = piece["r"].to_numpy()
            m = len(piece) // 2
            left = np.lexsort((piece["final"].to_numpy(), piece["tie"].to_numpy(), q_l))[:m]
            right = np.lexsort((piece["final"].to_numpy(), piece["tie"].to_numpy(), q_r))[:m]
            ml, mr = float(r[left].mean()), float(r[right].mean())
            means_l.append(ml)
            means_r.append(mr)
            diffs.append(ml - mr)
        matched[dataset] = {
            "D": float(np.mean(diffs)),
            "mean_strong": float(np.mean(means_l)),
            "mean_coupled": float(np.mean(means_r)),
            "n_strata": int(len(diffs)),
        }
    matched["macro_D"] = 0.5 * (matched["gharchive"]["D"] + matched["tlc_yellow"]["D"])
    matched["macro_strong"] = 0.5 * (matched["gharchive"]["mean_strong"] + matched["tlc_yellow"]["mean_strong"])
    matched["macro_coupled"] = 0.5 * (matched["gharchive"]["mean_coupled"] + matched["tlc_yellow"]["mean_coupled"])

    # Dates and split counts.
    date_sets = {}
    for dataset, sub in dates.groupby("dataset_id"):
        ordered = sub.sort_values(["evaluation_date", "truth_id"])
        date_sets[dataset] = [str(value) for value in ordered["evaluation_date"]]
    split = {
        "tlc_cal": date_sets["tlc_yellow"][:27],
        "tlc_test": date_sets["tlc_yellow"][27:],
        "gh_cal": date_sets["gharchive"][:21],
        "gh_test": date_sets["gharchive"][21:],
    }
    split_bounds = {name: [values[0], values[-1], len(values)] for name, values in split.items()}

    # Frontier.
    is_tlc = c.dataset_id.eq("tlc_yellow").to_numpy()
    matched_front = json.loads(MATCHED_FRONT.read_text())
    frontier = {}
    frontier_differences = []
    for method, (source, column) in METHODS.items():
        frame = c if source == "box" else d
        scores = frame[column].to_numpy() / frame["S"].to_numpy()
        flags_tlc = frame.dataset_id.eq("tlc_yellow").to_numpy()
        got = integer_inverse(scores, flags_tlc)
        frontier[method] = got
        for label, row in got.items():
            old = matched_front[method][label]["T"]
            if old != row["T"]:
                frontier_differences.append({"method": method, "c": label, "matched_T": old, "integer_T": row["T"], "numerator": row["numerator"], "numerator_prev": row["numerator_prev"]})

    # Primary tie sensitivity.
    led = pd.read_parquet(TIME_FORWARD_LEDGER)
    thr = pd.read_parquet(TIME_FORWARD_THR)
    strong = thr[thr.method.eq("union_C_strong")][["dataset_id", "mechanism", "epsilon", "k", "weight_family", "v", "n_equal", "need"]].rename(
        columns={"v": "v_s", "n_equal": "ne_s", "need": "nd_s"}
    )
    coupled = thr[thr.method.eq("union_C_box")][["dataset_id", "mechanism", "epsilon", "k", "weight_family", "v", "n_equal", "need"]].rename(
        columns={"v": "v_c", "n_equal": "ne_c", "need": "nd_c"}
    )
    base = led.merge(strong, on=["dataset_id", "mechanism", "epsilon", "k", "weight_family"], how="left", validate="many_to_one")
    base = base.merge(coupled, on=["dataset_id", "mechanism", "epsilon", "k", "weight_family"], how="left", validate="many_to_one")
    if base[["v_s", "v_c", "ne_s", "ne_c"]].isna().any().any():
        raise SystemExit("threshold join failed")
    base = base.sort_values(["dataset_id", "mechanism", "epsilon", "k", "weight_family", "truth_id", "replicate"]).reset_index(drop=True)
    primary_seeds = []
    pooled_seeds = []
    for seed in [20270111, *range(20270115, 20270165)]:
        h = hashes_for(base, seed)
        acc_s = accept_mask(base.q_strong.to_numpy(), h, base.v_s.to_numpy(), base.ne_s.to_numpy(), base.nd_s.to_numpy())
        acc_c = accept_mask(base.q_coupled.to_numpy(), h, base.v_c.to_numpy(), base.ne_c.to_numpy(), base.nd_c.to_numpy())
        primary_seeds.append({"seed": seed, **stratum_macros(base, acc_s, acc_c)})
        pooled_seeds.append({"seed": seed, **source_pooled(base, acc_s, acc_c)})
    extra = primary_seeds[1:]
    extra_p = pooled_seeds[1:]
    tie = {
        "n_test_rows": int(len(base)),
        "official_seed": primary_seeds[0],
        "primary_stratum_equal": {
            "n_seeds": len(extra),
            "D_regret_min": min(row["D_regret"] for row in extra),
            "D_regret_max": max(row["D_regret"] for row in extra),
            "D_regret_median": float(np.median([row["D_regret"] for row in extra])),
            "D_regret_n_negative": int(sum(row["D_regret"] < 0 for row in extra)),
            "D_accept_min": min(row["D_accept"] for row in extra),
            "D_accept_max": max(row["D_accept"] for row in extra),
            "D_accept_median": float(np.median([row["D_accept"] for row in extra])),
            "D_accept_n_negative": int(sum(row["D_accept"] < 0 for row in extra)),
            "empty_strata_max": max(row["empty_strata"] for row in extra),
        },
        "source_pooled_recomputed": {
            "D_regret_min": min(row["D_regret"] for row in extra_p),
            "D_regret_max": max(row["D_regret"] for row in extra_p),
            "D_accept_min": min(row["D_accept"] for row in extra_p),
            "D_accept_max": max(row["D_accept"] for row in extra_p),
            "D_regret_n_negative": int(sum(row["D_regret"] < 0 for row in extra_p)),
            "D_accept_n_negative": int(sum(row["D_accept"] < 0 for row in extra_p)),
        },
    }
    gh = json.loads((ROOT / "experiments/real_data_panel/integrity/gh_independent_check.json").read_text())
    inside = sum(row["inside"] for row in gh["sample_accounting"])
    outside = sum(row["outside"] for row in gh["sample_accounting"])
    retention = [row["inside"] / row["events"] for row in gh["sample_accounting"]]
    result = {
        "interpreter": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "checks": checks,
        "box_tau_0_10": rates,
        "os_tau_0_10": dos,
        "matched_rate": matched,
        "split_bounds": split_bounds,
        "frontier_differences": frontier_differences,
        "frontier_t50": {method: frontier[method]["0.50"] for method in frontier},
        "tie": tie,
        "github_retention": {
            "inside": inside,
            "outside": outside,
            "min": min(retention),
            "max": max(retention),
            "n_days": len(retention),
        },
    }
    (OUT / "recompute_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "delta_union": rates["delta_union"],
        "os": {k: dos[k]["macro"] if isinstance(dos[k], dict) and "macro" in dos[k] else dos[k] for k in ("rank1_coupling", "os_coupling", "coupled_refine", "strong_refine", "n_new_coupled")},
        "matched": matched["macro_D"],
        "matched_sources": {k: matched[k]["D"] for k in ("gharchive", "tlc_yellow")},
        "split": split_bounds,
        "n_frontier_differences": len(frontier_differences),
        "t50_union_strong_num": frontier["union_strong"]["0.50"]["numerator"],
        "tie_official_accept": primary_seeds[0]["D_accept"],
        "tie_official_regret": primary_seeds[0]["D_regret"],
        "tie_primary": tie["primary_stratum_equal"],
        "tie_pooled": tie["source_pooled_recomputed"],
        "k5": rates["k5_union_max_abs"],
        "count_disagree": rates["count_decision_disagreements_union_box"],
        "retention": result["github_retention"],
    }, indent=2))


if __name__ == "__main__":
    main()
