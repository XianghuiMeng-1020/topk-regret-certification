#!/usr/bin/env python3
"""Small release smoke test on a few truths. Not part of the 540-cell evaluation grid."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from private_mobility_cert.certificates.block_coupled_certificates import (  # noqa: E402
    certificate_full_coupled_box,
    matched_separable_box_value,
    strongest_nonexact_same_event,
)
from private_mobility_cert.certificates.release_conditioned import (  # noqa: E402
    box_half_width,
    deploy_ranking,
    exact_weighted_regret,
)
from private_mobility_cert.certificates.weighted_ranking import weight_family  # noqa: E402
from private_mobility_cert.privacy.geometric import release_geometric_histogram  # noqa: E402
from private_mobility_cert.privacy.laplace import release_laplace_histogram  # noqa: E402

ROOT = Path(__file__).resolve().parent
MASTER = 20260807


def seed_for(truth_id: str, mech: str, eps: float, k: int, wf: str, rep: int) -> int:
    payload = f"{MASTER}|{truth_id}|{mech}|{eps}|{k}|{wf}|{rep}".encode()
    h = hashlib.sha256(payload).hexdigest()
    return int(h[:16], 16) % (2**31 - 1)


def sha_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def release(x, mech, eps, seed):
    if mech == "M1":
        return np.asarray(release_laplace_histogram(x, epsilon=eps, seed=seed).y_mech, float)
    return np.asarray(release_geometric_histogram(x, epsilon=eps, seed=seed).y_mech, float)


def one(dataset: str) -> dict:
    manifest = ROOT / "manifests" / f"{dataset}_truth_manifest.jsonl"
    row = json.loads(manifest.read_text().splitlines()[0])
    x = np.load(REPO / row["vector_path"])
    assert hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() == row["vector_raw_sha256"]
    # npy file bytes are deterministic for this array
    tmp = ROOT / "smoke_tests" / "_resave.npy"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("wb") as f:
        np.save(f, np.ascontiguousarray(x, dtype=np.float64))
    file_repeat = hashlib.sha256(tmp.read_bytes()).hexdigest() == row["vector_file_sha256"]
    tmp.unlink()
    k = 10
    w = weight_family("dcg", k)
    outs = []
    for mech in ("M1", "M2"):
        for rep in (0, 1):
            seed = seed_for(row["truth_id"], mech, 1.0, k, "dcg", rep)
            y1 = release(x, mech, 1.0, seed)
            y2 = release(x, mech, 1.0, seed)
            mech_name = "M1_flat_laplace" if mech == "M1" else "M2_two_sided_geometric"
            t = float(box_half_width(mech_name, int(x.size), 1.0, 0.05))
            pi = deploy_ranking(y1, k)
            regret = float(exact_weighted_regret(x, pi, w))
            c_full = float(certificate_full_coupled_box(y1, k=k, weights=w, half_width=t).value)
            c_sep = float(matched_separable_box_value(y1, k=k, weights=w, half_width=t))
            c_strong, _ = strongest_nonexact_same_event(y1, k=k, weights=w, half_width=t)
            outs.append({
                "label": "SMOKE_TEST_ONLY",
                "mechanism": mech,
                "epsilon": 1.0,
                "alpha": 0.05,
                "k": k,
                "weights": "dcg",
                "replicate": rep,
                "seed": seed,
                "seed_payload": f"{MASTER}|{row['truth_id']}|{mech}|1.0|{k}|dcg|{rep}",
                "half_width_union_box": t,
                "y_sha256": sha_arr(y1),
                "repeated_y_equal": bool(np.array_equal(y1, y2)),
                "y_finite": bool(np.all(np.isfinite(y1))),
                "deployed_topk": [int(i) for i in pi],
                "true_regret": regret,
                "C_box": c_full,
                "C_sep": c_sep,
                "C_strong_raw_same_event": float(c_strong),
                "box_covers_true_regret": bool(regret <= c_full + 1e-8),
            })
    return {
        "dataset_id": row["dataset_id"],
        "truth_id": row["truth_id"],
        "selection_rule": "first manifest row, which is the earliest evaluation date",
        "vector_raw_sha256": row["vector_raw_sha256"],
        "candidate_universe_sha256": row["candidate_universe_sha256"],
        "npy_resave_matches": file_repeat,
        "seed_convention": True,
        "seedkey_used": False,
        "releases": outs,
    }


if __name__ == "__main__":
    which = sys.argv[1]
    out = one(which)
    dest = ROOT / "smoke_tests" / f"{which}_smoke.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(dest)
    print(json.dumps(out, indent=2))
