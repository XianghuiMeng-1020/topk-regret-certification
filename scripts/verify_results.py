"""Verify the committed experiment outputs without recomputing them.

Checks
  1. Every file under experiments/ matches checksums/experiment_outputs.sha256,
     and every listed file is present.
  2. Hash locks: every 64-hex constant in the real-data runners equals the
     SHA-256 of a committed file or of a pinned source object.
  3. protocol.sha256 files match their protocol.json.
  4. Panel structure: 45 + 36 truth vectors, 261 + 2000 candidates,
     43,740 ledger rows per box/OS ledger, 540 cells per truth.
  5. Source metadata in experiments/real_data_panel/source_metadata
     agrees with manifests/SOURCE_OBJECTS.json.

--write regenerates checksums/experiment_outputs.sha256 from the current tree
(run it only after a full reproduction). --out writes a JSON report.
Exit 1 on any failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
PANEL = EXP / "real_data_panel"
TABLE = ROOT / "checksums" / "experiment_outputs.sha256"
SKIP_NAMES = {".DS_Store"}
SKIP_PARTS = {"__pycache__", "chunks"}
RUNNERS = [
    "real_box_evaluation/run.py",
    "real_order_statistic_evaluation/run.py",
    "matched_rate_utility/run.py",
    "time_forward_utility/run.py",
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def experiment_files() -> dict[str, str]:
    out = {}
    for p in sorted(EXP.rglob("*")):
        if p.is_file() and p.name not in SKIP_NAMES and not SKIP_PARTS & set(p.relative_to(EXP).parts):
            out[str(p.relative_to(ROOT))] = sha(p)
    return out


def read_table() -> dict[str, str]:
    table = {}
    for line in TABLE.read_text().splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            table[rel.strip()] = digest
    return table


def check_structure(fail: list[str]) -> dict:
    tlc_vectors = sorted((PANEL / "truth_vectors/tlc").glob("*.npy"))
    gh_vectors = sorted((PANEL / "truth_vectors/gharchive").glob("*.npy"))
    tlc_u = json.loads((PANEL / "candidate_universes/tlc_pulocation_universe.json").read_text())
    gh_u = json.loads((PANEL / "candidate_universes/gharchive_repos_top2000.json").read_text())

    def n_items(u):
        return len(u["ids"]) if len(u["ids"]) == u["candidate_count"] else None

    manifests = {}
    for name in ("tlc_truth_manifest", "gharchive_truth_manifest"):
        rows = [json.loads(line) for line in (PANEL / f"manifests/{name}.jsonl").read_text().splitlines() if line.strip()]
        bad = sum(1 for row in rows if not (ROOT / row["vector_path"]).exists()
                  or sha(ROOT / row["vector_path"]) != row["vector_file_sha256"])
        manifests[name] = {"rows": len(rows), "vector_hash_mismatch": bad}
        if bad:
            fail.append(f"{name}: {bad} truth vector hash mismatch")
    lengths = {
        "tlc": sorted({int(np.load(p).shape[0]) for p in tlc_vectors}),
        "gharchive": sorted({int(np.load(p).shape[0]) for p in gh_vectors}),
    }
    ledgers = {}
    for rel in ("real_box_evaluation/ledger.parquet", "real_order_statistic_evaluation/ledger.parquet"):
        df = pd.read_parquet(EXP / rel, columns=["dataset_id", "truth_id"])
        ledgers[rel] = {
            "rows": int(len(df)),
            "truths": int(df.truth_id.nunique()),
            "cells_per_truth": sorted(int(x) for x in df.groupby("truth_id").size().unique()),
            "tlc_truths": int(df.loc[df.dataset_id.eq("tlc_yellow"), "truth_id"].nunique()),
            "gh_truths": int(df.loc[df.dataset_id.eq("gharchive"), "truth_id"].nunique()),
        }
    expect = [
        (len(tlc_vectors) == 45, "45 TLC truth vectors"),
        (len(gh_vectors) == 36, "36 GitHub truth vectors"),
        (lengths["tlc"] == [261], "TLC vectors have 261 zones"),
        (lengths["gharchive"] == [2000], "GitHub vectors have 2000 repositories"),
        (manifests["tlc_truth_manifest"]["rows"] == 45, "TLC manifest rows"),
        (manifests["gharchive_truth_manifest"]["rows"] == 36, "GitHub manifest rows"),
        (n_items(tlc_u) == 261, "TLC universe has 261 zones"),
        (n_items(gh_u) == 2000, "GitHub universe has 2000 repositories"),
    ]
    for rel, row in ledgers.items():
        expect.append((row["rows"] == 43740 and row["truths"] == 81 and row["cells_per_truth"] == [540]
                       and row["tlc_truths"] == 45 and row["gh_truths"] == 36, f"{rel} shape"))
    for ok, label in expect:
        if not ok:
            fail.append(f"structure: {label}")
    return {"tlc_truth_vectors": len(tlc_vectors), "gh_truth_vectors": len(gh_vectors),
            "truth_vector_lengths": lengths, "manifests": manifests, "ledgers": ledgers}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="regenerate checksums/experiment_outputs.sha256")
    ap.add_argument("--out", default=None, help="write a JSON report to this path")
    args = ap.parse_args()

    present = experiment_files()
    if args.write:
        TABLE.write_text("".join(f"{digest}  {rel}\n" for rel, digest in present.items()))
        print(f"wrote {TABLE.relative_to(ROOT)} ({len(present)} files)")
        return 0

    fail: list[str] = []
    table = read_table()
    mismatched = [rel for rel, digest in present.items() if rel in table and table[rel] != digest]
    unlisted = [rel for rel in present if rel not in table]
    missing = [rel for rel in table if rel not in present]
    fail += [f"checksum mismatch: {rel}" for rel in mismatched]
    fail += [f"not in checksum table: {rel}" for rel in unlisted]
    fail += [f"listed but missing: {rel}" for rel in missing]

    committed = set(present.values())
    pinned = {o["sha256"] for o in json.loads((ROOT / "manifests/SOURCE_OBJECTS.json").read_text())["objects"]}
    locks = []
    for rel in RUNNERS:
        for h in sorted(set(re.findall(r'"([0-9a-f]{64})"', (EXP / rel).read_text()))):
            ok = h in committed or h in pinned
            locks.append({"runner": rel, "sha256": h, "satisfied": ok})
            if not ok:
                fail.append(f"hash lock in {rel} not satisfied: {h}")

    protocols = []
    for p in sorted(EXP.glob("*/protocol.sha256")):
        match = p.read_text().split()[0] == sha(p.with_name("protocol.json"))
        protocols.append({"file": str(p.relative_to(ROOT)), "match": match})
        if not match:
            fail.append(f"protocol hash mismatch {p.relative_to(ROOT)}")

    structure = check_structure(fail)

    md = PANEL / "source_metadata"
    meta = {Path(m["url"]).name: m["sha256"] for m in json.loads((md / "tlc_objects.json").read_text())}
    for p in (md / "gharchive").glob("*.json"):
        m = json.loads(p.read_text())
        meta[Path(m["url"]).name] = m["sha256"]
    names = {o["name"]: o["sha256"] for o in json.loads((ROOT / "manifests/SOURCE_OBJECTS.json").read_text())["objects"]}
    if meta != names:
        fail.append("source_metadata disagrees with manifests/SOURCE_OBJECTS.json")

    report = {
        "n_experiment_files": len(present),
        "n_checksum_matches": len(present) - len(mismatched) - len(unlisted),
        "hash_locks": locks,
        "protocol_hashes": protocols,
        "structure": structure,
        "source_metadata_matches_pins": meta == names,
        "failures": fail,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"experiment files: {report['n_checksum_matches']}/{len(present)} match checksums/experiment_outputs.sha256")
    print(f"hash locks satisfied: {sum(l['satisfied'] for l in locks)}/{len(locks)}; "
          f"protocol hashes: {sum(p['match'] for p in protocols)}/{len(protocols)}")
    print(json.dumps(structure["ledgers"], indent=1))
    if fail:
        print("FAIL:", *fail[:30], sep="\n  ", file=sys.stderr)
        return 1
    print("OK: experiment outputs verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
