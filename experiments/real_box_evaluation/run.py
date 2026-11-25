#!/usr/bin/env python3
"""Box-event certificates on the real-data truth panel (TLC and GitHub Archive).

Reads the hash-locked panel manifests and universes. Does not download data.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys

sys.path.insert(0, str(REPO / "src"))

from private_mobility_cert.certificates.certified_risk_frontier import (  # noqa: E402
    STAB,
    normalize_certificate,
    release_scale_U,
)
from private_mobility_cert.certificates.count_feasible import (  # noqa: E402
    matched_count_feasible_certificates,
)
from private_mobility_cert.certificates.direct_box_calibration import (  # noqa: E402
    RAW_GAP_ZERO_ATOL,
    canonicalize_k5_certificates,
    certification_decision,
    direct_box_radius_geometric,
    direct_box_radius_laplace,
)
from private_mobility_cert.certificates.release_conditioned import (  # noqa: E402
    box_half_width,
    deploy_ranking,
    exact_weighted_regret,
)
from private_mobility_cert.certificates.relaxation_closure import (  # noqa: E402
    monotone_closed_value,
)
from private_mobility_cert.certificates.weighted_ranking import (  # noqa: E402
    stable_ranking,
    weight_family,
)
from private_mobility_cert.privacy.geometric import release_geometric_histogram  # noqa: E402
from private_mobility_cert.privacy.laplace import release_laplace_histogram  # noqa: E402

PANEL = REPO / "experiments" / "real_data_panel"
OUT = Path(__file__).resolve().parent
CHUNKS = OUT / "chunks"
MASTER = 20260807
ALPHA = 0.05
TAUS = (0.05, 0.10, 0.25, 0.50, 1.00)
PRIMARY_TAU = 0.10
BOOTSTRAP_SEED = 20261124
BLOCK_SEED = 20261125
BOOTSTRAP_B = 1999
BLOCK_LENGTH = 7

EXPECTED = {
    "manifests/tlc_truth_manifest.jsonl": "942a6cff6abfd58e63b0d025b4a45da41d9a550ee64e2be4e3572b3dc7a9532c",
    "manifests/gharchive_truth_manifest.jsonl": "745752764c77212af41fdb53d45934a4ad1cbee44caca4aa403ae2b8ae90978f",
    "candidate_universes/tlc_pulocation_universe.json": "8d6604116210a0d3fa28e2f6d2874682a7b6fb90d49d730d5b7321db9d9b749f",
    "candidate_universes/gharchive_repos_top2000.json": "706126c42ce5fa26371938b0889f1710d0934f6da437f7242be7b39bfc73a56a",
}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def seed_for(truth_id: str, mech: str, eps: float, k: int, wf: str, rep: int) -> int:
    payload = f"{MASTER}|{truth_id}|{mech}|{eps}|{k}|{wf}|{rep}".encode()
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16) % (2**31 - 1)


def release(x: np.ndarray, mech: str, eps: float, seed: int) -> np.ndarray:
    if mech == "M1":
        return np.asarray(release_laplace_histogram(x, epsilon=eps, seed=seed).y_mech, dtype=np.float64)
    return np.asarray(release_geometric_histogram(x, epsilon=eps, seed=seed).y_mech, dtype=np.float64)


def radii(n: int, mech: str, eps: float) -> tuple[float, float]:
    if mech == "M1":
        return (
            float(box_half_width("M1_flat_laplace", n, eps, ALPHA)),
            float(direct_box_radius_laplace(n, eps, ALPHA)),
        )
    return (
        float(box_half_width("M2_two_sided_geometric", n, eps, ALPHA)),
        float(direct_box_radius_geometric(n, eps, ALPHA)),
    )


def load_manifest(name: str) -> list[dict]:
    path = PANEL / "manifests" / name
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_inputs() -> list[dict]:
    for rel, exp in EXPECTED.items():
        got = sha_file(PANEL / rel)
        if got != exp:
            raise SystemExit(f"hash mismatch {rel}: {got}")
    tlc = load_manifest("tlc_truth_manifest.jsonl")
    gh = load_manifest("gharchive_truth_manifest.jsonl")
    if len(tlc) != 45 or len(gh) != 36:
        raise SystemExit(f"truth counts {len(tlc)} {len(gh)}")
    tlc_days = [r["evaluation_date"] for r in tlc]
    gh_days = [r["evaluation_date"] for r in gh]
    if tlc_days != [d.isoformat() for d in _dates("2024-04-01", 45)]:
        raise SystemExit("TLC dates are not the configured window")
    if gh_days != [d.isoformat() for d in _dates("2024-02-01", 36)]:
        raise SystemExit("GitHub dates are not the configured window")
    universes = {
        "tlc_yellow": json.loads((PANEL / "candidate_universes" / "tlc_pulocation_universe.json").read_text()),
        "gharchive": json.loads((PANEL / "candidate_universes" / "gharchive_repos_top2000.json").read_text()),
    }
    rows = tlc + gh
    seen = set()
    for row in rows:
        if row["truth_id"] in seen:
            raise SystemExit(f"duplicate {row['truth_id']}")
        seen.add(row["truth_id"])
        univ = universes[row["dataset_id"]]
        if sha_file(PANEL / "candidate_universes" / (
            "tlc_pulocation_universe.json" if row["dataset_id"] == "tlc_yellow" else "gharchive_repos_top2000.json"
        )) != row["candidate_universe_sha256"]:
            raise SystemExit(f"universe hash mismatch {row['truth_id']}")
        if len(univ["ids"]) != row["vector_length"]:
            raise SystemExit(f"universe length {row['truth_id']}")
        path = REPO / row["vector_path"]
        if sha_file(path) != row["vector_file_sha256"]:
            raise SystemExit(f"file hash {row['truth_id']}")
        x = np.load(path)
        if sha_arr(x) != row["vector_raw_sha256"]:
            raise SystemExit(f"raw hash {row['truth_id']}")
        if x.shape != (row["vector_length"],) or not np.all(np.isfinite(x)) or np.any(x < 0):
            raise SystemExit(f"vector integrity {row['truth_id']}")
        if np.any(np.abs(x - np.rint(x)) > 1e-9):
            raise SystemExit(f"non-integer {row['truth_id']}")
    return rows


def _dates(start: str, n: int):
    from datetime import date, timedelta
    cur = date.fromisoformat(start)
    return [cur + timedelta(days=i) for i in range(n)]


def smoke_check() -> dict:
    expected = {}
    for name in ("tlc_smoke.json", "gharchive_smoke.json"):
        payload = json.loads((PANEL / "smoke_tests" / name).read_text())
        for rel in payload["releases"]:
            expected[(payload["truth_id"], rel["mechanism"], rel["replicate"])] = rel
    by_id = {r["truth_id"]: r for r in load_manifest("tlc_truth_manifest.jsonl") + load_manifest("gharchive_truth_manifest.jsonl")}
    out = []
    for (truth_id, mech, rep), rel in expected.items():
        row = by_id[truth_id]
        x = np.load(REPO / row["vector_path"])
        seed = seed_for(truth_id, mech, 1.0, 10, "dcg", rep)
        if seed != rel["seed"]:
            raise SystemExit(f"seed mismatch {truth_id} {mech} {rep}: {seed} != {rel['seed']}")
        y1 = release(x, mech, 1.0, seed)
        y2 = release(x, mech, 1.0, seed)
        if not np.array_equal(y1, y2):
            raise SystemExit(f"release not repeatable {truth_id}")
        digest = sha_arr(y1)
        if digest != rel["y_sha256"]:
            raise SystemExit(f"Y hash mismatch {truth_id} {mech} {rep}")
        w = weight_family("dcg", 10)
        tu, _td = radii(int(x.size), mech, 1.0)
        fam = matched_count_feasible_certificates(y1, k=10, weights=w, u_minus=tu, u_plus=tu)
        if abs(fam["C_box_original"] - rel["C_box"]) > 1e-8:
            raise SystemExit(f"C_box smoke mismatch {truth_id} {fam['C_box_original']} {rel['C_box']}")
        if abs(fam["C_sep_original"] - rel["C_sep"]) > 1e-8:
            raise SystemExit(f"C_sep smoke mismatch {truth_id} {fam['C_sep_original']} {rel['C_sep']}")
        out.append({"truth_id": truth_id, "mechanism": mech, "replicate": rep, "seed": seed, "y_sha256": digest})
    return {"n": len(out), "rows": out}


def write_protocol() -> str:
    weights = {}
    for name in ("uniform", "dcg", "reciprocal"):
        for k in (5, 10, 25):
            weights[f"{name}_k{k}"] = [float(v) for v in weight_family(name, k)]
    calibration = []
    for dataset, n in (("tlc_yellow", 261), ("gharchive", 2000)):
        for mech in ("M1", "M2"):
            for eps in (0.5, 1.0, 2.0):
                tu, td = radii(n, mech, eps)
                calibration.append({
                    "dataset_id": dataset,
                    "n": n,
                    "mechanism": mech,
                    "epsilon": eps,
                    "union_radius": tu,
                    "direct_radius": td,
                    "direct_le_union": bool(td <= tu + 1e-12),
                })
    protocol = {
        "experiment": "real_box_evaluation",
        "inputs": EXPECTED,
        "alpha": ALPHA,
        "sensitivity": 1.0,
        "mechanisms": {
            "M1": "continuous Laplace, scale 1/epsilon",
            "M2": "unit-sensitivity discrete Laplace",
        },
        "epsilon": [0.5, 1.0, 2.0],
        "k": [5, 10, 25],
        "weights": ["uniform", "dcg", "reciprocal"],
        "weight_vectors": weights,
        "replicates": 10,
        "cells_per_truth": 540,
        "master_seed": MASTER,
        "seed": "SHA-256 of MASTER|truth_id|mechanism|epsilon|k|weight_family|replicate, first 16 hex digits, modulo 2**31-1. Python default float formatting.",
        "not_seedkey": True,
        "normalization": "S = max(U(Y+), 1e-6); certify iff C/S <= tau with no additive slack",
        "primary_tau": PRIMARY_TAU,
        "secondary_tau": list(TAUS),
        "inverse_frontier": {"c": 0.50},
        "aggregation": "mean over the cells of a truth, then mean over truths in a dataset, then equal weight of the two datasets",
        "bootstrap": {
            "B": BOOTSTRAP_B,
            "seed": BOOTSTRAP_SEED,
            "unit": "truth vector, all cells kept together",
            "interval": "percentile 2.5 and 97.5",
        },
        "block_bootstrap": {
            "seed": BLOCK_SEED,
            "length": BLOCK_LENGTH,
            "design": "circular moving block within each source",
        },
        "raw_gap_atol_diagnostic_only": RAW_GAP_ZERO_ATOL,
        "calibration": calibration,
        "methods": [
            "UNION_BOUND_BOX original family",
            "DIRECT_JOINT_BOX original family",
            "count-feasible exact assignment and raw relaxations on each event",
            "monotone_closed_min_original_and_raw_plus",
        ],
        "invalid_methods_excluded": ["certificate_mswap_os"],
        "numpy": np.__version__,
        "python": sys.version.split()[0],
    }
    path = OUT / "protocol.json"
    path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    digest = sha_file(path)
    (OUT / "protocol.sha256").write_text(digest + "\n")
    return digest


def _argmin(sep: float, rank: float, pair: float, block5: float) -> str:
    vals = {"sep": sep, "rank": rank, "pair": pair, "block5": block5}
    best = min(vals.values())
    names = [name for name, value in vals.items() if abs(value - best) <= RAW_GAP_ZERO_ATOL]
    return "+".join(names)


def _pack(fam: dict, prefix: str) -> dict:
    original = {
        "C_box": fam["C_box_original"],
        "C_sep_general": fam["C_sep_original"],
        "C_rank": fam["C_rank_original"],
        "C_pair": fam["C_pair_original"],
        "C_block5": fam["C_block5_original"],
        "C_strong": fam["C_strong_original"],
    }
    # Decision copies. k<=5 replaces block5 and strong with C_box. Independents stay in `original`.
    decision = canonicalize_k5_certificates(original, k=int(fam.get("k5_identity_applied", 0) and 5 or 0) if False else 5)
    return original, decision


def evaluate_truth(row: dict) -> str:
    safe = row["truth_id"].replace(":", "_")
    dest = CHUNKS / f"{safe}.parquet"
    if dest.exists():
        old = pd.read_parquet(dest)
        if len(old) == 540 and old["truth_id"].nunique() == 1:
            return str(dest)
    x = np.load(REPO / row["vector_path"])
    n = int(x.size)
    records = []
    for mech in ("M1", "M2"):
        for eps in (0.5, 1.0, 2.0):
            tu, td = radii(n, mech, eps)
            for k in (5, 10, 25):
                for wf in ("uniform", "dcg", "reciprocal"):
                    w = weight_family(wf, k)
                    for rep in range(10):
                        seed = seed_for(row["truth_id"], mech, eps, k, wf, rep)
                        y = release(x, mech, eps, seed)
                        t0 = time.perf_counter()
                        union = matched_count_feasible_certificates(
                            y, k=k, weights=w, u_minus=tu, u_plus=tu
                        )
                        t1 = time.perf_counter()
                        direct = matched_count_feasible_certificates(
                            y, k=k, weights=w, u_minus=td, u_plus=td
                        )
                        t2 = time.perf_counter()
                        pi = deploy_ranking(y, k)
                        yplus = np.maximum(y, 0.0)
                        order = stable_ranking(yplus, min(k + 1, n))
                        boundary = float(yplus[order[k - 1]] - yplus[order[k]]) if n > k else float("nan")
                        adj = np.diff(yplus[pi]) if pi.size > 1 else np.zeros(0)
                        u_scale = release_scale_U(y, pi, w)
                        regret = float(exact_weighted_regret(x, pi, w))
                        rec = {
                            "dataset_id": row["dataset_id"],
                            "surface_id": row["surface_id"],
                            "truth_id": row["truth_id"],
                            "evaluation_date": row["evaluation_date"],
                            "n": n,
                            "mechanism": mech,
                            "epsilon": eps,
                            "k": k,
                            "weight_family": wf,
                            "replicate": rep,
                            "seed": seed,
                            "radius_union": tu,
                            "radius_direct": td,
                            "y_sha256": sha_arr(y),
                            "pi_sha256": sha_arr(pi.astype(np.int64)),
                            "U": u_scale,
                            "S": max(u_scale, STAB),
                            "normalization_floor_applied": int(u_scale < STAB),
                            "true_regret": regret,
                            "boundary_margin": boundary,
                            "min_adjacent_margin": float(np.min(-adj)) if adj.size else float("nan"),
                            "runtime_union_s": t1 - t0,
                            "runtime_direct_s": t2 - t1,
                        }
                        for prefix, fam in (("union", union), ("direct", direct)):
                            original = {
                                "C_box": float(fam["C_box_original"]),
                                "C_sep_general": float(fam["C_sep_original"]),
                                "C_rank": float(fam["C_rank_original"]),
                                "C_pair": float(fam["C_pair_original"]),
                                "C_block5": float(fam["C_block5_original"]),
                                "C_strong": float(fam["C_strong_original"]),
                            }
                            decision = canonicalize_k5_certificates(original, k=k)
                            plus_decision_strong = float(fam["C_strong"])
                            plus_decision_box = float(fam["C_box"])
                            plus_decision_block5 = float(fam["C_block5"])
                            closed = monotone_closed_value(decision["C_strong"], plus_decision_strong)
                            rec[f"{prefix}_status"] = fam["status"]
                            rec[f"{prefix}_solves"] = int(fam["n_assignment_solves"])
                            rec[f"{prefix}_C_box"] = decision["C_box"]
                            rec[f"{prefix}_C_sep"] = float(original["C_sep_general"])
                            rec[f"{prefix}_C_rank"] = float(original["C_rank"])
                            rec[f"{prefix}_C_pair"] = float(original["C_pair"])
                            rec[f"{prefix}_C_block5"] = decision["C_block5"]
                            rec[f"{prefix}_C_strong"] = decision["C_strong"]
                            rec[f"{prefix}_C_block5_independent"] = decision["C_block5_independent"]
                            rec[f"{prefix}_C_strong_independent"] = decision["C_strong_independent"]
                            rec[f"{prefix}_k5_identity"] = int(decision["k5_identity_applied"])
                            rec[f"{prefix}_C_box_plus"] = plus_decision_box
                            rec[f"{prefix}_C_sep_plus_raw"] = float(fam["C_sep_general"])
                            rec[f"{prefix}_C_rank_plus_raw"] = float(fam["C_rank"])
                            rec[f"{prefix}_C_pair_plus_raw"] = float(fam["C_pair"])
                            rec[f"{prefix}_C_block5_plus_raw"] = float(fam["C_block5_independent"])
                            rec[f"{prefix}_C_strong_plus_raw"] = float(fam["C_strong_independent"])
                            rec[f"{prefix}_C_block5_plus_decision"] = plus_decision_block5
                            rec[f"{prefix}_C_strong_plus_decision"] = plus_decision_strong
                            rec[f"{prefix}_C_strong_plus_closed"] = closed
                            rec[f"{prefix}_strong_argmin"] = _argmin(
                                original["C_sep_general"], original["C_rank"], original["C_pair"], decision["C_block5_independent"]
                            )
                            below = yplus_below(y, pi, w, fam["u_plus"] if prefix == "union" else td)
                            rec[f"{prefix}_n_deployed_below_radius"] = below
                        records.append(rec)
    frame = pd.DataFrame.from_records(records)
    if len(frame) != 540:
        raise RuntimeError(f"{row['truth_id']} produced {len(frame)} rows")
    tmp = dest.with_suffix(".parquet.part")
    frame.to_parquet(tmp, index=False)
    tmp.replace(dest)
    return str(dest)


def yplus_below(y: np.ndarray, pi: np.ndarray, w: np.ndarray, radius: float) -> int:
    released = np.maximum(y[pi], 0.0)
    positive = np.asarray(w)[: pi.size] > 0
    return int(np.count_nonzero(released[positive] < float(radius)))


def run_grid(rows: list[dict]) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    CHUNKS.mkdir(parents=True, exist_ok=True)
    pending = []
    for row in rows:
        safe = row["truth_id"].replace(":", "_")
        dest = CHUNKS / f"{safe}.parquet"
        if dest.exists():
            old = pd.read_parquet(dest, columns=["truth_id"])
            if len(old) == 540:
                continue
        pending.append(row)
    print(f"[box] pending {len(pending)} of {len(rows)}", flush=True)
    if not pending:
        return
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(evaluate_truth, row): row["truth_id"] for row in pending}
        done = 0
        for fut in as_completed(futures):
            tid = futures[fut]
            fut.result()
            done += 1
            print(f"[box] {done}/{len(pending)} {tid}", flush=True)


def _cert(value: np.ndarray, scale: np.ndarray, tau: float) -> np.ndarray:
    out = np.empty(value.shape, dtype=bool)
    for i, (v, s) in enumerate(zip(value, scale)):
        out[i] = certification_decision(float(v), float(s), tau)
    return out


def aggregate() -> None:
    frames = [pd.read_parquet(p) for p in sorted(CHUNKS.glob("*.parquet"))]
    df = pd.concat(frames, ignore_index=True)
    key = ["truth_id", "mechanism", "epsilon", "k", "weight_family", "replicate"]
    if len(df) != 43740 or df.duplicated(key).any() or df["truth_id"].nunique() != 81:
        raise SystemExit(f"ledger integrity failed: rows={len(df)} truths={df['truth_id'].nunique()}")
    if set(df.groupby("truth_id").size().unique()) != {540}:
        raise SystemExit("a truth does not have 540 cells")
    ledger = OUT / "ledger.parquet"
    df.to_parquet(ledger, index=False)
    methods = {
        "box_union": "union_C_box",
        "strong_union": "union_C_strong",
        "box_direct": "direct_C_box",
        "strong_direct": "direct_C_strong",
        "box_union_plus": "union_C_box_plus",
        "strong_union_plus_raw": "union_C_strong_plus_decision",
        "strong_union_plus_closed": "union_C_strong_plus_closed",
        "box_direct_plus": "direct_C_box_plus",
        "strong_direct_plus_raw": "direct_C_strong_plus_decision",
        "strong_direct_plus_closed": "direct_C_strong_plus_closed",
    }
    certs = {}
    for name, col in methods.items():
        for tau in TAUS:
            certs[f"{name}__{tau}"] = _cert(df[col].to_numpy(), df["S"].to_numpy(), tau)
    cert_df = pd.DataFrame(certs)
    cert_df["dataset_id"] = df["dataset_id"].to_numpy()
    cert_df["truth_id"] = df["truth_id"].to_numpy()
    cert_df["k"] = df["k"].to_numpy()
    cert_df["weight_family"] = df["weight_family"].to_numpy()
    cert_df["epsilon"] = df["epsilon"].to_numpy()
    cert_df["mechanism"] = df["mechanism"].to_numpy()
    cert_df["union_status"] = df["union_status"].to_numpy()
    cert_df["direct_status"] = df["direct_status"].to_numpy()

    def k_of(frame: pd.DataFrame, col: str) -> float:
        by = frame.groupby("truth_id")[col].mean()
        return float(by.mean()) if len(by) else float("nan")

    def dataset_k(dataset: str, col: str) -> float:
        return k_of(cert_df[cert_df["dataset_id"] == dataset], col)

    source = {}
    macro = {}
    for tau in TAUS:
        block = {}
        for dataset, label in (("tlc_yellow", "tlc"), ("gharchive", "github")):
            block[label] = {name: dataset_k(dataset, f"{name}__{tau}") for name in methods}
            block[label]["delta_union"] = block[label]["box_union"] - block[label]["strong_union"]
            block[label]["delta_direct"] = block[label]["box_direct"] - block[label]["strong_direct"]
            block[label]["delta_union_plus"] = block[label]["box_union_plus"] - block[label]["strong_union_plus_raw"]
            block[label]["delta_union_closed"] = block[label]["box_union_plus"] - block[label]["strong_union_plus_closed"]
        source[str(tau)] = block
        macro[str(tau)] = {
            name: 0.5 * (block["tlc"][name] + block["github"][name])
            for name in list(methods) + ["delta_union", "delta_direct", "delta_union_plus", "delta_union_closed"]
        }
    (OUT / "source_results.json").write_text(json.dumps(source, indent=2) + "\n")
    (OUT / "macro_results.json").write_text(json.dumps(macro, indent=2) + "\n")

    truth = (
        cert_df.groupby(["dataset_id", "truth_id"], sort=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    truth.to_parquet(OUT / "truth_level.parquet", index=False)
    numerical_audit(df)
    count_audit(df)
    heterogeneity(cert_df)
    bootstrap(truth, df)
    runtime_summary(df)
    frontier(df)
    (OUT / "ledger_rows.json").write_text(json.dumps({
        "rows": int(len(df)),
        "truths": int(df["truth_id"].nunique()),
        "ledger_sha256": sha_file(ledger),
    }, indent=2) + "\n")
    print("[box] aggregate done", len(df), flush=True)


def numerical_audit(df: pd.DataFrame) -> None:
    checks = []

    def residue(left: pd.Series, right: pd.Series, name: str) -> None:
        gap = left.to_numpy() - right.to_numpy()
        scale = 1.0 + np.abs(left.to_numpy()) + np.abs(right.to_numpy())
        rel = np.abs(gap) / scale
        violate = gap > 1e-8
        checks.append({
            "check": name,
            "n": int(len(gap)),
            "n_positive_gap_gt_1e-8": int(np.count_nonzero(violate)),
            "max_abs_gap": float(np.max(np.abs(gap))),
            "max_relative_gap": float(np.max(rel)),
            "max_positive_gap": float(np.max(gap)),
        })

    for prefix in ("union", "direct"):
        box = df[f"{prefix}_C_box"]
        for other in ("sep", "rank", "pair", "block5", "strong"):
            residue(box, df[f"{prefix}_C_{other}"], f"{prefix} box <= {other}")
        k5 = df["k"] <= 5
        gap_s = (df.loc[k5, f"{prefix}_C_strong"] - df.loc[k5, f"{prefix}_C_box"]).to_numpy()
        gap_b = (df.loc[k5, f"{prefix}_C_block5"] - df.loc[k5, f"{prefix}_C_box"]).to_numpy()
        ind = (df.loc[k5, f"{prefix}_C_strong_independent"] - df.loc[k5, f"{prefix}_C_box"]).to_numpy()
        checks.append({
            "check": f"{prefix} k<=5 decision identity",
            "n": int(k5.sum()),
            "max_abs_strong_minus_box": float(np.max(np.abs(gap_s))),
            "max_abs_block5_minus_box": float(np.max(np.abs(gap_b))),
            "max_abs_independent_strong_minus_box": float(np.max(np.abs(ind))),
        })
    residue(df["direct_C_box"], df["union_C_box"], "direct box <= union box")
    exact = (df["union_status"] == "COUNT_FEASIBLE_EXACT") & (df["direct_status"] == "COUNT_FEASIBLE_EXACT")
    residue(df.loc[exact, "direct_C_box_plus"], df.loc[exact, "union_C_box_plus"], "direct plus <= union plus on nonempty")
    for prefix in ("union", "direct"):
        exact_p = df[f"{prefix}_status"] == "COUNT_FEASIBLE_EXACT"
        residue(df.loc[exact_p, f"{prefix}_C_box_plus"], df.loc[exact_p, f"{prefix}_C_box"], f"{prefix} plus box <= original")
        for other in ("sep", "rank", "pair", "block5"):
            residue(
                df.loc[exact_p, f"{prefix}_C_box_plus"],
                df.loc[exact_p, f"{prefix}_C_{other}_plus_raw"],
                f"{prefix} plus box <= raw {other}",
            )
    finite_cols = [c for c in df.columns if c.startswith("union_C") or c.startswith("direct_C")]
    n_bad = int(np.count_nonzero(~np.isfinite(df[finite_cols].to_numpy())))
    ambiguous = 0
    for col in ("union_C_box", "union_C_strong", "direct_C_box", "direct_C_strong"):
        for tau in TAUS:
            gap = np.abs(df[col].to_numpy() - tau * df["S"].to_numpy())
            scale = np.maximum(1.0, df["S"].to_numpy())
            ambiguous += int(np.count_nonzero(gap <= 1e-8 * scale))
    report = {"checks": checks, "nonfinite_certificate_values": n_bad, "threshold_near_ties_1e-8": ambiguous}
    (OUT / "numerical_audit.json").write_text(json.dumps(report, indent=2) + "\n")


def count_audit(df: pd.DataFrame) -> None:
    out = {"by_dataset_mechanism_epsilon": []}
    for prefix in ("union", "direct"):
        for (dataset, mech, eps), sub in df.groupby(["dataset_id", "mechanism", "epsilon"]):
            out["by_dataset_mechanism_epsilon"].append({
                "event": prefix,
                "dataset_id": dataset,
                "mechanism": mech,
                "epsilon": float(eps),
                "n": int(len(sub)),
                "n_fallback": int((sub[f"{prefix}_status"] != "COUNT_FEASIBLE_EXACT").sum()),
            })
        exact = df[f"{prefix}_status"] == "COUNT_FEASIBLE_EXACT"
        worse = {}
        for name, raw_col, orig_col in (
            ("sep", f"{prefix}_C_sep_plus_raw", f"{prefix}_C_sep"),
            ("rank", f"{prefix}_C_rank_plus_raw", f"{prefix}_C_rank"),
            ("pair", f"{prefix}_C_pair_plus_raw", f"{prefix}_C_pair"),
            ("block5", f"{prefix}_C_block5_plus_raw", f"{prefix}_C_block5_independent"),
            ("strong", f"{prefix}_C_strong_plus_raw", f"{prefix}_C_strong_independent"),
        ):
            delta = df.loc[exact, raw_col].to_numpy() - df.loc[exact, orig_col].to_numpy()
            worse[name] = {
                "n_exact": int(exact.sum()),
                "n_raw_increase_gt_1e-8": int(np.count_nonzero(delta > 1e-8)),
                "max_increase": float(np.max(delta)) if len(delta) else 0.0,
                "min_increase": float(np.min(delta)) if len(delta) else 0.0,
            }
        out[f"{prefix}_raw_minus_original"] = worse
        closed_delta = df[f"{prefix}_C_strong_plus_closed"].to_numpy() - df[f"{prefix}_C_strong_plus_decision"].to_numpy()
        out[f"{prefix}_closed_minus_raw_decision"] = {
            "n_changed_gt_1e-8": int(np.count_nonzero(np.abs(closed_delta) > 1e-8)),
            "min": float(np.min(closed_delta)),
            "max": float(np.max(closed_delta)),
        }
    (OUT / "count_feasibility_audit.json").write_text(json.dumps(out, indent=2) + "\n")


def heterogeneity(cert_df: pd.DataFrame) -> None:
    col_b = "box_union__0.1"
    col_s = "strong_union__0.1"
    # pandas may stringify 0.10 as 0.1
    if col_b not in cert_df.columns:
        col_b = "box_union__0.10"
        col_s = "strong_union__0.10"
    slices = {
        "dataset": ["dataset_id"],
        "k": ["k"],
        "weight": ["weight_family"],
        "epsilon": ["epsilon"],
        "mechanism": ["mechanism"],
        "dataset_k": ["dataset_id", "k"],
        "k_weight": ["k", "weight_family"],
        "dataset_epsilon": ["dataset_id", "epsilon"],
        "mechanism_epsilon": ["mechanism", "epsilon"],
        "union_status": ["union_status"],
        "direct_status": ["direct_status"],
    }
    result = {}
    work = cert_df.copy()
    work["gain"] = work[col_b].astype(float) - work[col_s].astype(float)
    for name, cols in slices.items():
        rows = []
        group_cols = list(dict.fromkeys(cols + ["dataset_id", "truth_id"]))
        grouped = work.groupby(group_cols, sort=False)["gain"].mean().reset_index()
        # equal-weight macro within each slice cell: average truths per dataset, then average datasets present
        for key, sub in grouped.groupby(cols, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            by_ds = sub.groupby("dataset_id")["gain"].mean()
            rows.append({
                **{c: _jsonify(v) for c, v in zip(cols, key)},
                "datasets": {ds: float(val) for ds, val in by_ds.items()},
                "equal_weight_macro": float(by_ds.mean()),
            })
        result[name] = rows
    (OUT / "heterogeneity.json").write_text(json.dumps(result, indent=2) + "\n")


def _jsonify(v):
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return v


def bootstrap(truth: pd.DataFrame, df: pd.DataFrame) -> None:
    def cols(tau: float) -> list[str]:
        tag = str(tau)
        alt = str(float(tau))
        names = [
            f"box_union__{tag}", f"strong_union__{tag}",
            f"box_direct__{tag}", f"strong_direct__{tag}",
            f"box_union_plus__{tag}", f"strong_union_plus_raw__{tag}",
            f"strong_union_plus_closed__{tag}",
        ]
        if names[0] not in truth.columns:
            names = [n.replace(f"__{tag}", f"__{alt}") for n in names]
        return names

    def estimate(sample: pd.DataFrame, names: list[str]) -> dict:
        out = {}
        for dataset, sub in sample.groupby("dataset_id"):
            out[dataset] = {name: float(sub[name].mean()) for name in names}
        macro_box = 0.5 * (out["tlc_yellow"][names[0]] + out["gharchive"][names[0]])
        macro_strong = 0.5 * (out["tlc_yellow"][names[1]] + out["gharchive"][names[1]])
        macro_dbox = 0.5 * (out["tlc_yellow"][names[2]] + out["gharchive"][names[2]])
        macro_dstrong = 0.5 * (out["tlc_yellow"][names[3]] + out["gharchive"][names[3]])
        macro_pbox = 0.5 * (out["tlc_yellow"][names[4]] + out["gharchive"][names[4]])
        macro_praw = 0.5 * (out["tlc_yellow"][names[5]] + out["gharchive"][names[5]])
        macro_pclosed = 0.5 * (out["tlc_yellow"][names[6]] + out["gharchive"][names[6]])
        return {
            "delta_union": macro_box - macro_strong,
            "delta_direct": macro_dbox - macro_dstrong,
            "calibration_box": macro_box - macro_dbox,
            "coupling_interaction": (macro_dstrong - macro_dbox) - (macro_strong - macro_box),
            "count_effect_union": macro_pbox - macro_box,
            "delta_union_plus_raw": macro_pbox - macro_praw,
            "delta_union_closed": macro_pbox - macro_pclosed,
            "tlc_delta_union": out["tlc_yellow"][names[0]] - out["tlc_yellow"][names[1]],
            "github_delta_union": out["gharchive"][names[0]] - out["gharchive"][names[1]],
        }

    names = cols(0.10)
    point = estimate(truth, names)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    groups = {
        ds: truth[truth["dataset_id"] == ds].reset_index(drop=True)
        for ds in ("tlc_yellow", "gharchive")
    }
    draws = {key: [] for key in point}
    for _ in range(BOOTSTRAP_B):
        parts = []
        for ds, sub in groups.items():
            idx = rng.integers(0, len(sub), size=len(sub))
            parts.append(sub.iloc[idx])
        est = estimate(pd.concat(parts, ignore_index=True), names)
        for key, value in est.items():
            draws[key].append(value)
    intervals = {}
    for key, values in draws.items():
        arr = np.asarray(values, dtype=np.float64)
        lo, hi = np.quantile(arr, [0.025, 0.975])
        intervals[key] = {"point_from_full_sample": point[key], "lo": float(lo), "hi": float(hi)}

    # circular 7-day blocks, truths ordered by date within source
    rng_b = np.random.default_rng(BLOCK_SEED)
    ordered = {
        ds: truth[truth["dataset_id"] == ds].sort_values("truth_id").reset_index(drop=True)
        for ds in groups
    }
    block_draws = {key: [] for key in point}
    for _ in range(BOOTSTRAP_B):
        parts = []
        for ds, sub in ordered.items():
            t = len(sub)
            starts = rng_b.integers(0, t, size=int(np.ceil(t / BLOCK_LENGTH)))
            idx = []
            for s in starts:
                idx.extend((int(s) + offset) % t for offset in range(BLOCK_LENGTH))
            parts.append(sub.iloc[idx[:t]])
        est = estimate(pd.concat(parts, ignore_index=True), names)
        for key, value in est.items():
            block_draws[key].append(value)
    block_intervals = {}
    for key, values in block_draws.items():
        arr = np.asarray(values, dtype=np.float64)
        lo, hi = np.quantile(arr, [0.025, 0.975])
        block_intervals[key] = {"lo": float(lo), "hi": float(hi)}
    (OUT / "bootstrap.json").write_text(json.dumps({
        "B": BOOTSTRAP_B,
        "seed": BOOTSTRAP_SEED,
        "truth_cluster": intervals,
        "circular_7day_block": {"seed": BLOCK_SEED, "intervals": block_intervals},
    }, indent=2) + "\n")


def frontier(df: pd.DataFrame) -> None:
    """Smallest tau with aggregated K(tau) >= 0.5, on the empirical step function."""
    methods = {
        "box_union": "union_C_box",
        "strong_union": "union_C_strong",
        "box_direct": "direct_C_box",
        "strong_direct": "direct_C_strong",
    }
    result = {}
    for name, col in methods.items():
        norm = df[col].to_numpy() / df["S"].to_numpy()
        work = pd.DataFrame({"dataset_id": df["dataset_id"], "truth_id": df["truth_id"], "norm": norm})
        # candidate taus are the cell values; search the ones that can cross 0.5
        guesses = np.unique(np.quantile(norm, np.linspace(0, 1, 401)))

        def k_at(tau: float) -> float:
            flags = work["norm"] <= tau
            tmp = work.assign(flag=flags)
            by = tmp.groupby(["dataset_id", "truth_id"])["flag"].mean().groupby("dataset_id").mean()
            return float(by.mean())

        best = None
        for tau in guesses:
            if k_at(float(tau)) >= 0.5:
                best = float(tau)
                break
        result[name] = {"T_0.50": best, "K_at_T": None if best is None else k_at(best)}
    (OUT / "frontier.json").write_text(json.dumps(result, indent=2) + "\n")


def runtime_summary(df: pd.DataFrame) -> None:
    total = df["runtime_union_s"] + df["runtime_direct_s"]
    def stats(s: pd.Series) -> dict:
        return {
            "n": int(len(s)),
            "median": float(s.median()),
            "p90": float(s.quantile(0.90)),
            "p95": float(s.quantile(0.95)),
            "sum": float(s.sum()),
        }
    report = {
        "per_cell_union_family_s": stats(df["runtime_union_s"]),
        "per_cell_direct_family_s": stats(df["runtime_direct_s"]),
        "per_cell_both_s": stats(total),
        "solves_union_sum": int(df["union_solves"].sum()),
        "solves_direct_sum": int(df["direct_solves"].sum()),
        "by_n": {},
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    for n, sub in df.groupby("n"):
        report["by_n"][str(int(n))] = stats(sub["runtime_union_s"] + sub["runtime_direct_s"])
    (OUT / "runtime.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = validate_inputs()
    smoke = smoke_check()
    (OUT / "smoke_preflight.json").write_text(json.dumps(smoke, indent=2) + "\n")
    digest = write_protocol()
    print("[box] protocol", digest, "smoke", smoke["n"], flush=True)
    t0 = time.perf_counter()
    run_grid(rows)
    t1 = time.perf_counter()
    aggregate()
    t2 = time.perf_counter()
    (OUT / "stage_times.json").write_text(json.dumps({
        "grid_s": t1 - t0,
        "aggregate_s": t2 - t1,
    }, indent=2) + "\n")
    print("[box] finished", flush=True)


if __name__ == "__main__":
    main()
