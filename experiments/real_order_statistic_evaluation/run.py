#!/usr/bin/env python3
"""Order-statistic certificates on the real-data panel.

Replays the releases of the box-event evaluation (same seeds, hash-checked
ledger) and adds the rank-1 order-statistic box and the order-statistic-aware
envelope, in coupled and strong forms.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(REPO / "src"))

from private_mobility_cert.certificates.certified_risk_frontier import (  # noqa: E402
    STAB,
    release_scale_U,
)
from private_mobility_cert.certificates.os_envelope import (  # noqa: E402
    certificate_os_envelope,
    envelope_schedule,
    prefix_min_thresholds,
)
from private_mobility_cert.certificates.os_matched_envelope import (  # noqa: E402
    METHOD_OS_ENVELOPE,
    METHOD_OS_MATCHED,
    certificate_os_matched,
)
from private_mobility_cert.certificates.rank_adaptive import (  # noqa: E402
    binomial_sf,
    geometric_positive_tail_strict,
    laplace_positive_tail,
)
from private_mobility_cert.certificates.release_conditioned import (  # noqa: E402
    build_threshold_table,
    deploy_ranking,
    exact_weighted_regret,
)
from private_mobility_cert.certificates.weighted_ranking import (  # noqa: E402
    allocate_eta_weight_proportional,
    stable_ranking,
    weight_family,
)
from private_mobility_cert.privacy.geometric import release_geometric_histogram  # noqa: E402
from private_mobility_cert.privacy.laplace import release_laplace_histogram  # noqa: E402

MASTER = 20260807
ALPHA = 0.05
TAUS = (("0.05", 0.05), ("0.10", 0.10), ("0.25", 0.25), ("0.50", 0.50), ("1.00", 1.00))
BOOT_SEED = 20261126
BLOCK_SEED = 20261127
BOOT_B = 1999
MATERIAL_ABS = 1e-8
MATERIAL_REL = 1e-6
OUT = REPO / "experiments/real_order_statistic_evaluation"
CHUNKS = OUT / "chunks"
PANEL = REPO / "experiments/real_data_panel"
BOX = REPO / "experiments/real_box_evaluation"
EXPECTED = {
    "manifests/tlc_truth_manifest.jsonl": "942a6cff6abfd58e63b0d025b4a45da41d9a550ee64e2be4e3572b3dc7a9532c",
    "manifests/gharchive_truth_manifest.jsonl": "745752764c77212af41fdb53d45934a4ad1cbee44caca4aa403ae2b8ae90978f",
}
EXPECTED_UNIVERSE = {
    "candidate_universes/tlc_pulocation_universe.json": "8d6604116210a0d3fa28e2f6d2874682a7b6fb90d49d730d5b7321db9d9b749f",
    "candidate_universes/gharchive_repos_top2000.json": "706126c42ce5fa26371938b0889f1710d0934f6da437f7242be7b39bfc73a56a",
}
BOX_LEDGER = "281ca9b676184d7450bbd2c8da225b94da312bb37d3c97855b1818ecc8c956cf"
BOX_PROTOCOL = "a85d4177a4614036001fb76c5edeb26d78e3b4182fc709cb2d079414794caccc"
_TABLES: dict = {}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_arr(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def seed_for(truth_id: str, mech: str, eps: float, k: int, wf: str, rep: int) -> int:
    payload = f"{MASTER}|{truth_id}|{mech}|{eps}|{k}|{wf}|{rep}".encode()
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**31 - 1)


def release(x: np.ndarray, mech: str, eps: float, seed: int) -> np.ndarray:
    if mech == "M1":
        drawn = release_laplace_histogram(x, epsilon=eps, seed=seed)
    elif mech == "M2":
        drawn = release_geometric_histogram(x, epsilon=eps, seed=seed)
    else:
        raise RuntimeError(f"unknown mechanism {mech}")
    return np.asarray(drawn.y_mech, dtype=np.float64)


def load_manifests() -> list[dict]:
    rows = []
    for name in ("tlc_truth_manifest.jsonl", "gharchive_truth_manifest.jsonl"):
        path = PANEL / "manifests" / name
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def validate_inputs() -> list[dict]:
    for rel, expected in EXPECTED.items():
        got = sha_file(PANEL / rel)
        if got != expected:
            raise SystemExit(f"hash mismatch {rel}: {got}")
    for rel, expected in EXPECTED_UNIVERSE.items():
        got = sha_file(PANEL / rel)
        if got != expected:
            raise SystemExit(f"hash mismatch {rel}: {got}")
    if sha_file(BOX / "ledger.parquet") != BOX_LEDGER:
        raise SystemExit("box-event ledger hash mismatch")
    if sha_file(BOX / "protocol.json") != BOX_PROTOCOL:
        raise SystemExit("box-event protocol hash mismatch")
    rows = load_manifests()
    if len(rows) != 81 or len({row["truth_id"] for row in rows}) != 81:
        raise SystemExit(f"expected 81 truths, found {len(rows)}")
    for row in rows:
        vector = np.load(REPO / row["vector_path"])
        if sha_file(REPO / row["vector_path"]) != row["vector_file_sha256"]:
            raise SystemExit(f"vector file hash mismatch {row['truth_id']}")
        if sha_arr(np.asarray(vector)) != row["vector_raw_sha256"]:
            raise SystemExit(f"vector raw hash mismatch {row['truth_id']}")
        if vector.shape != (int(row["vector_length"]),) or not np.all(np.isfinite(vector)):
            raise SystemExit(f"vector shape or finiteness failed {row['truth_id']}")
        if np.any(vector < 0) or np.any(vector != np.round(vector)):
            raise SystemExit(f"vector is not a nonnegative integer count {row['truth_id']}")
    return rows


def mechanism_name(mech: str) -> str:
    if mech == "M1":
        return "M1_flat_laplace"
    if mech == "M2":
        return "M2_two_sided_geometric"
    raise RuntimeError(mech)


def get_table(n: int, mech: str, eps: float, k: int, wf: str):
    key = (int(n), mech, float(eps), int(k), wf)
    if key not in _TABLES:
        weights = weight_family(wf, k)
        _TABLES[key] = build_threshold_table(
            mechanism=mechanism_name(mech),
            n=n,
            k_max=k,
            epsilon=eps,
            alpha=ALPHA,
            weights=weights,
        )
    return _TABLES[key]


def material_gap(left: float, right: float) -> bool:
    return float(left) > float(right) + MATERIAL_REL * (1.0 + abs(float(right))) + MATERIAL_ABS


def event_membership(z: np.ndarray, u_plus: np.ndarray, u_minus: np.ndarray, k: int):
    ordered = np.sort(np.asarray(z, dtype=np.float64))
    n = int(ordered.size)
    n_viol = 0
    first_side = ""
    first_rank = 0
    for rank in range(k):
        upper = float(u_plus[rank])
        lower = float(u_minus[rank])
        if ordered[n - 1 - rank] > upper + 1e-8 * (1.0 + abs(upper)):
            n_viol += 1
            if first_rank == 0:
                first_side, first_rank = "upper", rank + 1
        if ordered[rank] < -lower - 1e-8 * (1.0 + abs(lower)):
            n_viol += 1
            if first_rank == 0:
                first_side, first_rank = "lower", rank + 1
    return n_viol == 0, n_viol, first_side, first_rank


def write_protocol(rows: list[dict]) -> str:
    weights = {
        family: {str(k): weight_family(family, k).tolist() for k in (5, 10, 25)}
        for family in ("uniform", "dcg", "reciprocal")
    }
    protocol = {
        "experiment": "real_order_statistic_evaluation",
        "population": "real_data_panel_81_truths",
        "input_hashes": {**EXPECTED, **EXPECTED_UNIVERSE},
        "box_ledger_sha256": BOX_LEDGER,
        "box_protocol_sha256": BOX_PROTOCOL,
        "n_truths": len(rows),
        "evaluation_key": [
            "truth_id", "mechanism", "epsilon", "k", "weight_family", "replicate",
        ],
        "seed": {
            "master": MASTER,
            "format": "{MASTER}|{truth_id}|{mechanism}|{epsilon}|{k}|{weight_family}|{replicate}",
            "digest": "SHA-256 first 16 hex digits modulo 2**31-1",
            "not": "SeedKey",
        },
        "alpha": ALPHA,
        "sensitivity": 1,
        "event": "E_OS = intersection_r Z_(n-r+1) <= u_r_plus and Z_(r) >= -u_r_minus",
        "alpha_allocation": "eta_r_plus = eta_r_minus = (alpha/2) * w_r / sum(w)",
        "tail_convention": {
            "M1": "P(Z > u) for Laplace(0, 1/epsilon)",
            "M2": "P(Z > u) strict geometric tail, integer u",
        },
        "os_coupled_method": METHOD_OS_ENVELOPE,
        "os_strong_method": METHOD_OS_MATCHED,
        "strong_definition": "min(C_sep, C_rank, C_pair, C_block5) at auxiliary t_s, then add delta_s once",
        "k5_rule": "decision C_strong equals C_box for k<=5 before delta_s is added",
        "normalization": "S = max(U(Y+), 1e-6); certify when C/S <= tau; no additive slack",
        "primary_tau": 0.10,
        "secondary_tau": [0.05, 0.25, 0.50, 1.00],
        "inverse_frontier": "smallest 401-quantile of C/S whose equal-weight K reaches 0.50",
        "aggregation": "mean within truth, mean within dataset, equal weight across the two datasets",
        "bootstrap": {
            "B": BOOT_B,
            "truth_cluster_seed": BOOT_SEED,
            "circular_block_seed": BLOCK_SEED,
            "block_length_days": 7,
            "unit": "truth vector",
        },
        "material_violation": {
            "relative": MATERIAL_REL,
            "absolute": MATERIAL_ABS,
            "rule": "left > right + 1e-6*(1+|right|) + 1e-8",
        },
        "membership_tolerance": "1e-8 relative on each declared order-statistic constraint",
        "calibration_stop": "stop if a binomial tail exceeds its eta by more than 1e-8",
        "weight_vectors": weights,
        "excluded_certificates": ["m-swap", "certificate_min_valid"],
        "python": sys.version,
        "numpy": np.__version__,
    }
    path = OUT / "protocol.json"
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    digest = sha_file(path)
    (OUT / "protocol.sha256").write_text(digest + "\n")
    return digest


def calibration_audit(rows: list[dict]) -> dict:
    sizes = sorted({int(row["vector_length"]) for row in rows})
    worst = 0.0
    n_tables = 0
    n_not_monotone = 0
    max_positive_step = 0.0
    records = []
    for n in sizes:
        for mech in ("M1", "M2"):
            for eps in (0.5, 1.0, 2.0):
                for k in (5, 10, 25):
                    for wf in ("uniform", "dcg", "reciprocal"):
                        weights = weight_family(wf, k)
                        eta_plus, eta_minus = allocate_eta_weight_proportional(weights, ALPHA)
                        if abs(float(eta_plus.sum() + eta_minus.sum()) - ALPHA) > 1e-12:
                            raise SystemExit("alpha shares do not sum to alpha")
                        table = get_table(n, mech, eps, k, wf)
                        if table.u_plus.size < k or table.u_minus.size < k:
                            raise SystemExit("threshold table shorter than k")
                        if not np.all(np.isfinite(table.u_plus)) or np.any(table.u_plus < 0):
                            raise SystemExit("invalid threshold table")
                        if np.any(table.u_plus > 1e5):
                            raise SystemExit("threshold solver returned a fail-closed sentinel")
                        if mech == "M2" and np.max(np.abs(table.u_plus - np.round(table.u_plus))) > 0:
                            raise SystemExit("discrete threshold is not an integer")
                        if np.max(np.abs(table.u_plus - table.u_minus)) > 0:
                            raise SystemExit("constructed table is not symmetric")
                        step = float(np.max(np.diff(table.u_plus))) if k > 1 else 0.0
                        max_positive_step = max(max_positive_step, step)
                        if step > 1e-12:
                            n_not_monotone += 1
                        excesses = []
                        for rank in range(1, k + 1):
                            threshold = float(table.u_plus[rank - 1])
                            if mech == "M1":
                                probability = laplace_positive_tail(threshold, eps)
                            else:
                                probability = geometric_positive_tail_strict(int(round(threshold)), eps)
                            survival = binomial_sf(rank, n, probability)
                            excesses.append(float(survival - eta_plus[rank - 1]))
                        excess = max(excesses)
                        worst = max(worst, excess)
                        n_tables += 1
                        records.append({
                            "n": n,
                            "mechanism": mech,
                            "epsilon": eps,
                            "k": k,
                            "weight_family": wf,
                            "max_excess": excess,
                            "min_excess": min(excesses),
                            "prefix_min_changes": bool(
                                np.max(np.abs(prefix_min_thresholds(table.u_plus) - table.u_plus[:k])) > 1e-12
                            ),
                        })
    if worst > 1e-8:
        raise SystemExit(f"material calibration excess {worst}")
    report = {
        "n_tables": n_tables,
        "max_positive_excess": worst,
        "n_tables_not_componentwise_nonincreasing": n_not_monotone,
        "max_positive_consecutive_step": max_positive_step,
        "tables": records,
    }
    (OUT / "calibration_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run_independent() -> dict:
    path = OUT / "independent_checks.py"
    spec = importlib.util.spec_from_file_location("independent_checks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.run_suite()
    (OUT / "independent_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["n_failures"]:
        raise SystemExit(f"independent checks failed: {report['failures'][:5]}")
    return report


def boundary_margin(y: np.ndarray, k: int) -> float:
    yplus = np.maximum(y, 0.0)
    if yplus.size <= k:
        return float("nan")
    order = stable_ranking(yplus, k + 1)
    return float(yplus[order[k - 1]] - yplus[order[k]])


def evaluate_truth(payload: dict) -> str:
    row = payload["truth"]
    lookup = payload["box"]
    safe = row["truth_id"].replace(":", "_")
    dest = CHUNKS / f"{safe}.parquet"
    if dest.exists():
        old = pd.read_parquet(dest, columns=["truth_id"])
        if len(old) == 540 and old["truth_id"].nunique() == 1:
            return str(dest)
    x = np.asarray(np.load(REPO / row["vector_path"]), dtype=np.float64)
    records = []
    for mech in ("M1", "M2"):
        for eps in (0.5, 1.0, 2.0):
            for k in (5, 10, 25):
                for wf in ("uniform", "dcg", "reciprocal"):
                    weights = weight_family(wf, k)
                    for rep in range(10):
                        prior = lookup[(mech, eps, k, wf, rep)]
                        seed = seed_for(row["truth_id"], mech, eps, k, wf, rep)
                        if int(seed) != int(prior["seed"]):
                            raise RuntimeError(f"seed mismatch {row['truth_id']}")
                        y = release(x, mech, eps, seed)
                        if sha_arr(y) != prior["y_sha256"]:
                            raise RuntimeError(f"Y digest mismatch {row['truth_id']} {mech} {eps} {k} {wf} {rep}")
                        pi = deploy_ranking(y, k)
                        if sha_arr(pi.astype(np.int64)) != prior["pi_sha256"]:
                            raise RuntimeError(f"ranking digest mismatch {row['truth_id']}")
                        scale_u = float(release_scale_U(y, pi, weights))
                        scale = max(scale_u, STAB)
                        regret = float(exact_weighted_regret(x, pi, weights))
                        if abs(scale - float(prior["S"])) > 1e-8 * (1.0 + abs(float(prior["S"]))):
                            raise RuntimeError(f"S mismatch {row['truth_id']}")
                        if abs(regret - float(prior["true_regret"])) > 1e-8 * (1.0 + abs(float(prior["true_regret"]))):
                            raise RuntimeError(f"regret mismatch {row['truth_id']}")
                        started = time.perf_counter()
                        table = get_table(x.size, mech, eps, k, wf)
                        lookup_s = time.perf_counter() - started
                        started = time.perf_counter()
                        outcome = certificate_os_matched(y, k=k, weights=weights, table=table)
                        row_s = time.perf_counter() - started
                        inside, n_viol, side, rank = event_membership(
                            y - x, table.u_plus, table.u_minus, k
                        )
                        if inside and material_gap(regret, outcome.coupled_envelope):
                            raise RuntimeError(
                                f"on-event validity failure {row['truth_id']} {mech} {eps} {k} {wf} {rep}"
                            )
                        if outcome.rank1_coupled < -1e-8 or outcome.strong_envelope < -1e-8:
                            raise RuntimeError("negative certificate")
                        _schedule_w, schedule_terms = envelope_schedule(
                            weights, table.u_plus, table.u_minus, k=k
                        )
                        t_rank1 = float(schedule_terms[0].half_width)
                        records.append({
                            "dataset_id": row["dataset_id"],
                            "surface_id": row["surface_id"],
                            "truth_id": row["truth_id"],
                            "evaluation_date": row["evaluation_date"],
                            "n": int(x.size),
                            "mechanism": mech,
                            "epsilon": eps,
                            "alpha": ALPHA,
                            "k": k,
                            "weight_family": wf,
                            "replicate": rep,
                            "seed": int(seed),
                            "truth_sha256": row["vector_raw_sha256"],
                            "y_sha256": prior["y_sha256"],
                            "pi_sha256": prior["pi_sha256"],
                            "S": scale,
                            "true_regret": regret,
                            "boundary_margin": boundary_margin(y, k),
                            "threshold_sha256": sha_arr(np.concatenate([table.u_plus, table.u_minus])),
                            "t_rank1": t_rank1,
                            "C_OS_rank1_coupled": outcome.rank1_coupled,
                            "C_OS_rank1_strong": outcome.rank1_strong,
                            "C_OS_coupled_envelope": outcome.coupled_envelope,
                            "C_OS_strong_envelope": outcome.strong_envelope,
                            "C_OS_strong_envelope_independent": outcome.strong_envelope_independent,
                            "chosen_s_coupled": outcome.chosen_s_coupled,
                            "chosen_s_strong": outcome.chosen_s_strong,
                            "delta_coupled": outcome.delta_coupled,
                            "delta_strong": outcome.delta_strong,
                            "half_width_coupled": outcome.half_width_coupled,
                            "half_width_strong": outcome.half_width_strong,
                            "n_box_solves": outcome.n_box_solves,
                            "n_structural_solves": outcome.n_structural_solves,
                            "n_unique_half_widths": outcome.n_unique_half_widths,
                            "n_dominated_arms": outcome.n_pruned_structural,
                            "in_os_event": int(inside),
                            "n_violated_constraints": int(n_viol),
                            "first_violated_side": side,
                            "first_violated_rank": int(rank),
                            "runtime_threshold_lookup_s": lookup_s,
                            "runtime_box_s": outcome.runtime_box_s,
                            "runtime_structural_s": outcome.runtime_structural_s,
                            "runtime_row_s": row_s,
                            "solver_status": "ok",
                            "coupled_method": outcome.coupled_method,
                            "strong_method": outcome.strong_method,
                            "box_union_C_box": float(prior["union_C_box"]),
                            "box_union_C_strong": float(prior["union_C_strong"]),
                            "box_direct_C_box": float(prior["direct_C_box"]),
                            "box_direct_C_strong": float(prior["direct_C_strong"]),
                            "k5_identity_applied": outcome.k5_identity_applied,
                        })
    frame = pd.DataFrame.from_records(records)
    if len(frame) != 540:
        raise RuntimeError(f"incomplete truth {row['truth_id']}")
    tmp = dest.with_suffix(".parquet.part")
    frame.to_parquet(tmp, index=False)
    tmp.replace(dest)
    return str(dest)


def sample_preflight(rows: list[dict], box: pd.DataFrame) -> dict:
    earliest = {}
    for row in sorted(rows, key=lambda item: item["evaluation_date"]):
        earliest.setdefault(row["dataset_id"], row)
    checked = 0
    max_reference_gap = 0.0
    for row in earliest.values():
        x = np.asarray(np.load(REPO / row["vector_path"]), dtype=np.float64)
        sub = box[box["truth_id"] == row["truth_id"]]
        for mech in ("M1", "M2"):
            for eps in (0.5, 1.0, 2.0):
                for k in (5, 10, 25):
                    for wf in ("uniform", "dcg", "reciprocal"):
                        for rep in (0, 1):
                            prior = sub[
                                (sub["mechanism"] == mech)
                                & (np.isclose(sub["epsilon"], eps))
                                & (sub["k"] == k)
                                & (sub["weight_family"] == wf)
                                & (sub["replicate"] == rep)
                            ].iloc[0]
                            seed = seed_for(row["truth_id"], mech, eps, k, wf, rep)
                            if int(seed) != int(prior["seed"]):
                                raise SystemExit("sample seed mismatch")
                            y = release(x, mech, eps, seed)
                            if sha_arr(y) != prior["y_sha256"]:
                                raise SystemExit("sample Y digest mismatch")
                            pi = deploy_ranking(y, k)
                            if sha_arr(pi.astype(np.int64)) != prior["pi_sha256"]:
                                raise SystemExit("sample ranking digest mismatch")
                            weights = weight_family(wf, k)
                            scale = max(float(release_scale_U(y, pi, weights)), STAB)
                            regret = float(exact_weighted_regret(x, pi, weights))
                            if abs(scale - float(prior["S"])) > 1e-8 * (1.0 + abs(float(prior["S"]))):
                                raise SystemExit("sample S mismatch")
                            if abs(regret - float(prior["true_regret"])) > 1e-8 * (1.0 + abs(regret)):
                                raise SystemExit("sample regret mismatch")
                            table = get_table(x.size, mech, eps, k, wf)
                            matched = certificate_os_matched(y, k=k, weights=weights, table=table)
                            reference = certificate_os_envelope(y, k=k, weights=weights, table=table)
                            gap = abs(matched.coupled_envelope - float(reference.value))
                            max_reference_gap = max(max_reference_gap, gap)
                            if gap > 1e-8 * (1.0 + abs(float(reference.value))):
                                raise SystemExit("sample reference envelope mismatch")
                            checked += 1
    report = {"n_checked": checked, "max_abs_matched_minus_reference": max_reference_gap}
    (OUT / "sample_preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run_grid(rows: list[dict], box: pd.DataFrame) -> None:
    CHUNKS.mkdir(parents=True, exist_ok=True)
    grouped = {
        truth_id: {
            (rec["mechanism"], float(rec["epsilon"]), int(rec["k"]), rec["weight_family"], int(rec["replicate"])): rec
            for rec in sub.to_dict("records")
        }
        for truth_id, sub in box.groupby("truth_id")
    }
    pending = []
    for row in rows:
        safe = row["truth_id"].replace(":", "_")
        dest = CHUNKS / f"{safe}.parquet"
        if dest.exists():
            old = pd.read_parquet(dest, columns=["truth_id"])
            if len(old) == 540:
                continue
        if len(grouped[row["truth_id"]]) != 540:
            raise SystemExit(f"box-event rows missing for {row['truth_id']}")
        pending.append({"truth": row, "box": grouped[row["truth_id"]]})
    print(f"[os] pending {len(pending)} of {len(rows)}", flush=True)
    if not pending:
        return
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(evaluate_truth, item): item["truth"]["truth_id"] for item in pending}
        done = 0
        for future in as_completed(futures):
            truth_id = futures[future]
            future.result()
            done += 1
            print(f"[os] {done}/{len(pending)} {truth_id}", flush=True)


def _rates(df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    scale = df["S"].to_numpy()
    base = df[["dataset_id", "truth_id", "evaluation_date"]].copy()
    columns = {
        "rank1_strong": "C_OS_rank1_strong",
        "rank1_coupled": "C_OS_rank1_coupled",
        "strong_envelope": "C_OS_strong_envelope",
        "coupled_envelope": "C_OS_coupled_envelope",
        "union_box": "box_union_C_box",
        "union_strong": "box_union_C_strong",
        "direct_box": "box_direct_C_box",
        "direct_strong": "box_direct_C_strong",
    }
    for label, tau in TAUS:
        flags = base.copy()
        for name, column in columns.items():
            value = df[column].to_numpy()
            flags[f"{name}__{label}"] = np.isfinite(value) & (value <= tau * scale)
        flags[f"zero_coupled__{label}"] = df["C_OS_coupled_envelope"].to_numpy() <= 1e-12
        flags[f"s1_coupled__{label}"] = df["chosen_s_coupled"].to_numpy() == 1
        pieces.append(flags)
    merged = pieces[0]
    for extra in pieces[1:]:
        merged = merged.join(extra.drop(columns=["dataset_id", "truth_id", "evaluation_date"]))
    return merged


def _macro(truth: pd.DataFrame, columns: list[str]) -> dict:
    by_dataset = truth.groupby("dataset_id")[columns].mean()
    macro = by_dataset.mean()
    return {
        "by_dataset": {ds: {col: float(by_dataset.loc[ds, col]) for col in columns} for ds in by_dataset.index},
        "equal_weight_macro": {col: float(macro[col]) for col in columns},
    }


def _gain_block(rates: dict) -> dict:
    out = {}
    for label, _tau in TAUS:
        b = rates[f"rank1_coupled__{label}"]
        a = rates[f"rank1_strong__{label}"]
        d = rates[f"coupled_envelope__{label}"]
        c = rates[f"strong_envelope__{label}"]
        out[label] = {
            "K_rank1_strong": a,
            "K_rank1_coupled": b,
            "K_strong_envelope": c,
            "K_coupled_envelope": d,
            "rank1_coupling": b - a,
            "os_coupling": d - c,
            "coupled_refinement": d - b,
            "strong_refinement": c - a,
            "interaction": (d - c) - (b - a),
            "K_union_box": rates[f"union_box__{label}"],
            "K_union_strong": rates[f"union_strong__{label}"],
            "K_direct_box": rates[f"direct_box__{label}"],
            "K_direct_strong": rates[f"direct_strong__{label}"],
        }
    return out


def aggregate() -> None:
    frames = [pd.read_parquet(path) for path in sorted(CHUNKS.glob("*.parquet"))]
    df = pd.concat(frames, ignore_index=True)
    key = ["truth_id", "mechanism", "epsilon", "k", "weight_family", "replicate"]
    if len(df) != 43740 or df["truth_id"].nunique() != 81 or df.duplicated(key).any():
        raise SystemExit(f"ledger integrity failed rows={len(df)} truths={df['truth_id'].nunique()}")
    if set(df.groupby("truth_id").size().unique()) != {540}:
        raise SystemExit("a truth does not have 540 cells")
    df.to_parquet(OUT / "ledger.parquet", index=False)
    flags = _rates(df)
    truth = flags.groupby(["dataset_id", "truth_id", "evaluation_date"], as_index=False).mean()
    truth.to_parquet(OUT / "truth_level.parquet", index=False)
    rate_cols = [col for col in truth.columns if "__" in col]
    summary = _macro(truth, rate_cols)
    source = {}
    macro = {}
    for dataset, values in summary["by_dataset"].items():
        source[dataset] = _gain_block(values)
    macro_block = _gain_block(summary["equal_weight_macro"])
    for label, _tau in TAUS:
        macro[label] = macro_block[label]
        macro[label]["zero_coupled_rate"] = summary["equal_weight_macro"][f"zero_coupled__{label}"]
        macro[label]["s1_fraction"] = summary["equal_weight_macro"][f"s1_coupled__{label}"]
    (OUT / "source_results.json").write_text(json.dumps(source, indent=2) + "\n")
    (OUT / "macro_results.json").write_text(json.dumps(macro, indent=2) + "\n")
    numerical_audit(df)
    validity_audit(df)
    crossing_audit(df)
    heterogeneity(df)
    bootstrap(truth)
    frontier(df)
    runtime_summary(df)
    (OUT / "ledger_rows.json").write_text(json.dumps({
        "rows": int(len(df)),
        "truths": int(df["truth_id"].nunique()),
    }) + "\n")


def numerical_audit(df: pd.DataFrame) -> None:
    checks = []

    def residue(left: pd.Series, right: pd.Series, name: str) -> None:
        gap = left.to_numpy() - right.to_numpy()
        scale = 1.0 + np.abs(right.to_numpy())
        checks.append({
            "check": name,
            "n": int(len(gap)),
            "n_positive_gap_gt_1e-8": int(np.count_nonzero(gap > 1e-8)),
            "n_material": int(np.count_nonzero(gap > MATERIAL_REL * scale + MATERIAL_ABS)),
            "max_positive_gap": float(np.max(gap)),
            "max_abs_gap": float(np.max(np.abs(gap))),
        })

    residue(df["C_OS_coupled_envelope"], df["C_OS_rank1_coupled"], "coupled envelope <= rank1 coupled")
    residue(df["C_OS_coupled_envelope"], df["C_OS_strong_envelope"], "coupled envelope <= strong envelope")
    residue(df["C_OS_strong_envelope"], df["C_OS_rank1_strong"], "strong envelope <= rank1 strong")
    residue(df["C_OS_rank1_coupled"], df["C_OS_rank1_strong"], "rank1 coupled <= rank1 strong")
    k5 = df["k"] <= 5
    identity = (df.loc[k5, "C_OS_strong_envelope"] - df.loc[k5, "C_OS_coupled_envelope"]).to_numpy()
    checks.append({
        "check": "k<=5 envelope identity",
        "n": int(k5.sum()),
        "max_abs_gap": float(np.max(np.abs(identity))),
    })
    ambiguous = []
    for column in (
        "C_OS_rank1_coupled",
        "C_OS_rank1_strong",
        "C_OS_coupled_envelope",
        "C_OS_strong_envelope",
    ):
        for label, tau in TAUS:
            gap = np.abs(df[column].to_numpy() - tau * df["S"].to_numpy())
            scale = np.maximum(df["S"].to_numpy(), 1.0)
            hits = np.flatnonzero(gap <= 1e-8 * scale)
            for index in hits[:20]:
                ambiguous.append({
                    "column": column,
                    "tau": label,
                    "truth_id": df.iloc[index]["truth_id"],
                    "mechanism": df.iloc[index]["mechanism"],
                    "epsilon": float(df.iloc[index]["epsilon"]),
                    "k": int(df.iloc[index]["k"]),
                    "weight_family": df.iloc[index]["weight_family"],
                    "replicate": int(df.iloc[index]["replicate"]),
                    "value": float(df.iloc[index][column]),
                    "S": float(df.iloc[index]["S"]),
                    "ratio": float(df.iloc[index][column] / df.iloc[index]["S"]),
                })
    report = {
        "checks": checks,
        "nonfinite": int(np.count_nonzero(~np.isfinite(df[[
            "C_OS_rank1_coupled", "C_OS_rank1_strong",
            "C_OS_coupled_envelope", "C_OS_strong_envelope",
        ]].to_numpy()))),
        "n_threshold_near_ties": len(ambiguous),
        "threshold_near_ties": ambiguous,
    }
    (OUT / "numerical_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(item.get("n_material", 0) for item in checks):
        raise SystemExit("material same-event dominance violation")
    if checks[-1]["max_abs_gap"] > 1e-8:
        raise SystemExit("k<=5 OS envelope identity failed")


def validity_audit(df: pd.DataFrame) -> None:
    gap = df["true_regret"].to_numpy() - df["C_OS_coupled_envelope"].to_numpy()
    scale = MATERIAL_REL * (1.0 + np.abs(df["C_OS_coupled_envelope"].to_numpy())) + MATERIAL_ABS
    material = gap > scale
    inside = df["in_os_event"].to_numpy() == 1
    grouped = []
    for keys, sub in df.groupby(["dataset_id", "mechanism", "epsilon", "k"]):
        sub_gap = sub["true_regret"].to_numpy() - sub["C_OS_coupled_envelope"].to_numpy()
        sub_inside = sub["in_os_event"].to_numpy() == 1
        grouped.append({
            "dataset_id": keys[0],
            "mechanism": keys[1],
            "epsilon": float(keys[2]),
            "k": int(keys[3]),
            "n": int(len(sub)),
            "n_outside": int((~sub_inside).sum()),
            "n_on_event_material": int((sub_inside & (sub_gap > scale[:0] if False else (
                sub_gap > MATERIAL_REL * (1.0 + np.abs(sub["C_OS_coupled_envelope"].to_numpy())) + MATERIAL_ABS
            ))).sum()),
        })
    report = {
        "n": int(len(df)),
        "n_outside": int((~inside).sum()),
        "n_on_event_material": int((inside & material).sum()),
        "n_off_event_material": int((~inside & material).sum()),
        "max_on_event_gap": float(np.max(gap[inside])) if inside.any() else None,
        "max_off_event_gap": float(np.max(gap[~inside])) if (~inside).any() else None,
        "by_dataset_mechanism_epsilon_k": grouped,
    }
    (OUT / "validity_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["n_on_event_material"]:
        raise SystemExit("on-event validity violation")


def crossing_audit(df: pd.DataFrame) -> None:
    scale = df["S"].to_numpy()
    regret = df["true_regret"].to_numpy()
    report = {}
    for label, tau in TAUS:
        a = df["C_OS_rank1_strong"].to_numpy() <= tau * scale
        b = df["C_OS_rank1_coupled"].to_numpy() <= tau * scale
        c = df["C_OS_strong_envelope"].to_numpy() <= tau * scale
        d = df["C_OS_coupled_envelope"].to_numpy() <= tau * scale
        new = d & ~b
        both_abstain_then_os = d & ~a & ~b
        selected = df.loc[new]
        report[label] = {
            "n_all_certify": int((a & b & c & d).sum()),
            "n_all_abstain": int((~a & ~b & ~c & ~d).sum()),
            "n_only_coupled_envelope": int((d & ~a & ~b & ~c).sum()),
            "n_rank1_coupling_changes": int((b & ~a).sum()),
            "n_os_coupled_but_not_rank1_coupled": int(new.sum()),
            "n_os_coupled_when_both_rank1_abstain": int(both_abstain_then_os.sum()),
            "n_strong_envelope_but_not_rank1_strong": int((c & ~a).sum()),
            "new_coupled_regret": _regret_summary(selected),
        }
    (OUT / "threshold_crossing_audit.json").write_text(json.dumps(report, indent=2) + "\n")


def _regret_summary(selected: pd.DataFrame) -> dict:
    if selected.empty:
        return {"n": 0}
    regret = selected["true_regret"].to_numpy()
    norm = regret / selected["S"].to_numpy()
    return {
        "n": int(len(selected)),
        "mean": float(np.mean(regret)),
        "median": float(np.median(regret)),
        "p90": float(np.quantile(regret, 0.90)),
        "max": float(np.max(regret)),
        "mean_normalized": float(np.mean(norm)),
        "median_normalized": float(np.median(norm)),
        "fraction_zero": float(np.mean(np.abs(regret) <= 1e-12)),
        "fraction_regret_over_S_le_0.10": float(np.mean(norm <= 0.10)),
        "by_dataset": {key: int(val) for key, val in selected.groupby("dataset_id").size().items()},
        "by_k": {str(int(key)): int(val) for key, val in selected.groupby("k").size().items()},
        "by_weight": {key: int(val) for key, val in selected.groupby("weight_family").size().items()},
        "by_epsilon": {str(float(key)): int(val) for key, val in selected.groupby("epsilon").size().items()},
        "by_mechanism": {key: int(val) for key, val in selected.groupby("mechanism").size().items()},
    }


def heterogeneity(df: pd.DataFrame) -> None:
    scale = df["S"].to_numpy()
    work = df.copy()
    work["gain_rank1"] = (work["C_OS_rank1_coupled"] <= 0.10 * work["S"]).astype(float) - (
        work["C_OS_rank1_strong"] <= 0.10 * work["S"]
    ).astype(float)
    work["gain_os"] = (work["C_OS_coupled_envelope"] <= 0.10 * work["S"]).astype(float) - (
        work["C_OS_strong_envelope"] <= 0.10 * work["S"]
    ).astype(float)
    work["gain_coupled_refine"] = (work["C_OS_coupled_envelope"] <= 0.10 * work["S"]).astype(float) - (
        work["C_OS_rank1_coupled"] <= 0.10 * work["S"]
    ).astype(float)
    work["gain_strong_refine"] = (work["C_OS_strong_envelope"] <= 0.10 * work["S"]).astype(float) - (
        work["C_OS_rank1_strong"] <= 0.10 * work["S"]
    ).astype(float)
    work["s_group"] = np.where(work["chosen_s_coupled"] == 1, "s1", "s_gt1")
    margin = work["boundary_margin"].to_numpy()
    t1 = work["t_rank1"].to_numpy()
    work["margin_group"] = np.where(margin <= 0, "nonpositive", np.where(margin <= t1, "within_rank1_threshold", "above_rank1_threshold"))
    slices = {
        "dataset": ["dataset_id"],
        "k": ["k"],
        "weight": ["weight_family"],
        "epsilon": ["epsilon"],
        "mechanism": ["mechanism"],
        "dataset_k": ["dataset_id", "k"],
        "dataset_epsilon": ["dataset_id", "epsilon"],
        "k_weight": ["k", "weight_family"],
        "mechanism_epsilon": ["mechanism", "epsilon"],
        "selected_s": ["s_group"],
        "margin": ["margin_group"],
    }
    result = {}
    for name, cols in slices.items():
        rows = []
        group_cols = list(dict.fromkeys(cols + ["dataset_id", "truth_id"]))
        for gain_name in ("gain_rank1", "gain_os", "gain_coupled_refine", "gain_strong_refine"):
            grouped = work.groupby(group_cols, sort=False)[gain_name].mean().reset_index()
            for key, sub in grouped.groupby(cols, sort=False):
                if not isinstance(key, tuple):
                    key = (key,)
                by_ds = sub.groupby("dataset_id")[gain_name].mean()
                rows.append({
                    "effect": gain_name,
                    **{col: _jsonify(value) for col, value in zip(cols, key)},
                    "datasets": {ds: float(val) for ds, val in by_ds.items()},
                    "equal_weight_macro": float(by_ds.mean()),
                })
        result[name] = rows
    (OUT / "heterogeneity.json").write_text(json.dumps(result, indent=2) + "\n")
    del scale


def _jsonify(value):
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def bootstrap(truth: pd.DataFrame) -> None:
    label = "0.10"
    columns = [
        f"rank1_strong__{label}",
        f"rank1_coupled__{label}",
        f"strong_envelope__{label}",
        f"coupled_envelope__{label}",
    ]
    datasets = {
        dataset: sub.reset_index(drop=True)
        for dataset, sub in truth.sort_values("evaluation_date").groupby("dataset_id")
    }

    def effects(frame: pd.DataFrame) -> dict:
        a = float(frame[columns[0]].mean())
        b = float(frame[columns[1]].mean())
        c = float(frame[columns[2]].mean())
        d = float(frame[columns[3]].mean())
        return {
            "rank1_coupling": b - a,
            "os_coupling": d - c,
            "coupled_refinement": d - b,
            "strong_refinement": c - a,
            "interaction": (d - c) - (b - a),
        }

    def combine(index_map: dict) -> dict:
        parts = {dataset: effects(datasets[dataset].iloc[idx]) for dataset, idx in index_map.items()}
        names = list(parts)
        macro = {
            key: float(np.mean([parts[name][key] for name in names]))
            for key in ("rank1_coupling", "os_coupling", "coupled_refinement", "strong_refinement", "interaction")
        }
        for name in names:
            for key, value in parts[name].items():
                macro[f"{name}__{key}"] = value
        return macro

    point = combine({dataset: np.arange(len(frame)) for dataset, frame in datasets.items()})
    cluster = np.random.default_rng(BOOT_SEED)
    block = np.random.default_rng(BLOCK_SEED)
    cluster_rows = []
    block_rows = []
    for _ in range(BOOT_B):
        cluster_idx = {
            dataset: cluster.integers(0, len(frame), len(frame))
            for dataset, frame in datasets.items()
        }
        block_idx = {}
        for dataset, frame in datasets.items():
            n = len(frame)
            starts = block.integers(0, n, int(np.ceil(n / 7)))
            idx = [(int(start) + offset) % n for start in starts for offset in range(7)]
            block_idx[dataset] = np.asarray(idx[:n])
        cluster_rows.append(combine(cluster_idx))
        block_rows.append(combine(block_idx))
    cluster_df = pd.DataFrame(cluster_rows)
    block_df = pd.DataFrame(block_rows)
    report = {"point": point, "truth_cluster": {}, "circular_7day": {}, "B": BOOT_B}
    for column in cluster_df.columns:
        report["truth_cluster"][column] = {
            "lo": float(np.quantile(cluster_df[column], 0.025)),
            "hi": float(np.quantile(cluster_df[column], 0.975)),
        }
        report["circular_7day"][column] = {
            "lo": float(np.quantile(block_df[column], 0.025)),
            "hi": float(np.quantile(block_df[column], 0.975)),
        }
    (OUT / "bootstrap.json").write_text(json.dumps(report, indent=2) + "\n")


def frontier(df: pd.DataFrame) -> None:
    methods = {
        "rank1_coupled": "C_OS_rank1_coupled",
        "rank1_strong": "C_OS_rank1_strong",
        "coupled_envelope": "C_OS_coupled_envelope",
        "strong_envelope": "C_OS_strong_envelope",
        "union_box": "box_union_C_box",
        "union_strong": "box_union_C_strong",
        "direct_box": "box_direct_C_box",
        "direct_strong": "box_direct_C_strong",
    }
    result = {}
    for name, column in methods.items():
        norm = df[column].to_numpy() / df["S"].to_numpy()
        work = pd.DataFrame({
            "dataset_id": df["dataset_id"].to_numpy(),
            "truth_id": df["truth_id"].to_numpy(),
            "norm": norm,
        })
        guesses = np.unique(np.quantile(norm, np.linspace(0, 1, 401)))

        def k_at(tau: float) -> float:
            flags = work["norm"] <= tau
            return float(work.assign(flag=flags).groupby(["dataset_id", "truth_id"])["flag"].mean().groupby("dataset_id").mean().mean())

        best = None
        for tau in guesses:
            if k_at(float(tau)) >= 0.5:
                best = float(tau)
                break
        result[name] = {
            "T_0.50": best,
            "K_at_T": None if best is None else k_at(best),
            "approximation": "401 empirical quantiles of cell-level C/S",
        }
    (OUT / "frontier.json").write_text(json.dumps(result, indent=2) + "\n")


def runtime_summary(df: pd.DataFrame) -> None:
    def stats(values: pd.Series) -> dict:
        return {
            "n": int(len(values)),
            "median": float(values.median()),
            "p90": float(values.quantile(0.90)),
            "p95": float(values.quantile(0.95)),
            "sum": float(values.sum()),
        }

    report = {
        "per_cell_row_s": stats(df["runtime_row_s"]),
        "per_cell_box_s": stats(df["runtime_box_s"]),
        "per_cell_structural_s": stats(df["runtime_structural_s"]),
        "per_cell_threshold_lookup_s": stats(df["runtime_threshold_lookup_s"]),
        "box_solves_sum": int(df["n_box_solves"].sum()),
        "structural_solves_sum": int(df["n_structural_solves"].sum()),
        "dominated_arms_sum": int(df["n_dominated_arms"].sum()),
        "solver_status_counts": {key: int(val) for key, val in df["solver_status"].value_counts().items()},
        "by_n_k": [],
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "rss_unit": "bytes on macOS ru_maxrss",
    }
    for (n, k), sub in df.groupby(["n", "k"]):
        report["by_n_k"].append({
            "n": int(n),
            "k": int(k),
            "row": stats(sub["runtime_row_s"]),
            "box": stats(sub["runtime_box_s"]),
            "structural": stats(sub["runtime_structural_s"]),
            "median_unique_widths": float(sub["n_unique_half_widths"].median()),
            "s1_fraction": float(np.mean(sub["chosen_s_coupled"] == 1)),
        })
    (OUT / "runtime.json").write_text(json.dumps(report, indent=2) + "\n")


def guard_sources() -> None:
    text = (REPO / "src/private_mobility_cert/certificates/os_matched_envelope.py").read_text()
    if "mswap" in text or "min_valid" in text:
        raise SystemExit("OS matched envelope references a banned certificate")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    guard_sources()
    if len(sys.argv) > 1 and sys.argv[1] == "aggregate":
        aggregate()
        return
    rows = validate_inputs()
    digest = write_protocol(rows)
    print(f"[os] protocol {digest}", flush=True)
    started = time.perf_counter()
    calibration = calibration_audit(rows)
    calibration_s = time.perf_counter() - started
    print(f"[os] calibration excess {calibration['max_positive_excess']}", flush=True)
    started = time.perf_counter()
    independent = run_independent()
    independent_s = time.perf_counter() - started
    print(f"[os] independent fixtures {independent['n_fixtures']}", flush=True)
    box = pd.read_parquet(BOX / "ledger.parquet", columns=[
        "truth_id", "mechanism", "epsilon", "k", "weight_family", "replicate",
        "seed", "y_sha256", "pi_sha256", "S", "true_regret",
        "union_C_box", "union_C_strong", "direct_C_box", "direct_C_strong",
    ])
    if len(box) != 43740:
        raise SystemExit("box-event ledger row count changed")
    started = time.perf_counter()
    sample = sample_preflight(rows, box)
    sample_s = time.perf_counter() - started
    print(f"[os] sample {sample['n_checked']}", flush=True)
    started = time.perf_counter()
    run_grid(rows, box)
    grid_s = time.perf_counter() - started
    started = time.perf_counter()
    aggregate()
    aggregate_s = time.perf_counter() - started
    (OUT / "stage_times.json").write_text(json.dumps({
        "calibration_s": calibration_s,
        "independent_s": independent_s,
        "sample_s": sample_s,
        "grid_s": grid_s,
        "aggregate_s": aggregate_s,
    }, indent=2) + "\n")
    print("[os] finished", flush=True)


if __name__ == "__main__":
    main()
