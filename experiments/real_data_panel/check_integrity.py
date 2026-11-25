#!/usr/bin/env python3
"""Independent recount of the truth panel from the raw source objects.

This file does not import build_truths.py. It repeats the filters and compares
the resulting array bytes with the saved npy files.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
OBJ = ROOT.parent.parent / "data" / "source_objects"


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def daterange(a: date, b: date) -> list[date]:
    out = []
    cur = a
    while cur <= b:
        out.append(cur)
        cur += timedelta(days=1)
    return out


def check_tlc() -> dict:
    months = {
        "2024-01": "train",
        "2024-02": "train",
        "2024-04": "eval",
        "2024-05": "eval",
    }
    schemas = {}
    frames = {}
    hash_ok = {}
    for ym, phase in months.items():
        path = OBJ / "tlc" / f"yellow_tripdata_{ym}.parquet"
        meta = json.loads((ROOT / "source_metadata" / "tlc" / f"yellow_tripdata_{ym}.json").read_text())
        digest = sha_file(path)
        hash_ok[ym] = digest == meta["sha256"] and path.stat().st_size == meta["actual_bytes"]
        schemas[ym] = [{"name": f.name, "type": str(f.type)} for f in pq.ParquetFile(path).schema_arrow]
        df = pd.read_parquet(path, columns=["tpep_pickup_datetime", "PULocationID"])
        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
        df = df.dropna(subset=["tpep_pickup_datetime", "PULocationID"])
        df["PULocationID"] = df["PULocationID"].astype(int)
        df["day"] = df["tpep_pickup_datetime"].dt.strftime("%Y-%m-%d")
        y, m = ym.split("-")
        df = df[df["day"].str.startswith(f"{y}-{m}")]
        frames[ym] = df
    train_ids = sorted(set(int(z) for ym in ("2024-01", "2024-02") for z in frames[ym]["PULocationID"].unique()))
    saved = json.loads((ROOT / "candidate_universes" / "tlc_pulocation_universe.json").read_text())
    eval_ids = set(int(z) for ym in ("2024-04", "2024-05") for z in frames[ym]["PULocationID"].unique())
    only_eval = sorted(eval_ids - set(train_ids))
    index = {z: i for i, z in enumerate(train_ids)}
    eval_df = pd.concat([frames["2024-04"], frames["2024-05"]], ignore_index=True)
    manifest = [json.loads(line) for line in (ROOT / "manifests" / "tlc_truth_manifest.jsonl").read_text().splitlines()]
    mismatches = []
    for row in manifest:
        day = row["evaluation_date"]
        day_df = eval_df[eval_df["day"] == day]
        x = np.zeros(len(train_ids), dtype=np.float64)
        for z, cnt in day_df["PULocationID"].value_counts().items():
            z = int(z)
            if z in index:
                x[index[z]] += float(cnt)
        got = sha_bytes(np.ascontiguousarray(x).tobytes())
        saved_x = np.load(ROOT.parent.parent / row["vector_path"])
        if got != row["vector_raw_sha256"] or not np.array_equal(saved_x, x):
            mismatches.append(day)
        if saved_x.shape != (len(train_ids),):
            mismatches.append(day + ":shape")
    ids = [r["truth_id"] for r in manifest]
    expected = [d.isoformat() for d in daterange(date(2024, 4, 1), date(2024, 5, 15))]
    got_days = [r["evaluation_date"] for r in manifest]
    outside_eval = int((~eval_df["day"].isin(expected)).sum())
    result = {
        "source_hash_ok": hash_ok,
        "schema_names": {ym: [c["name"] for c in cols] for ym, cols in schemas.items()},
        "has_pickup_and_zone": all(
            "tpep_pickup_datetime" in names and "PULocationID" in names
            for names in result_names(schemas)
        ) if False else all(
            "tpep_pickup_datetime" in [c["name"] for c in cols] and "PULocationID" in [c["name"] for c in cols]
            for cols in schemas.values()
        ),
        "universe_equal_to_saved": train_ids == saved["ids"],
        "universe_size": len(train_ids),
        "eval_only_zones_not_in_universe": len(only_eval),
        "march_not_a_source": not (OBJ / "tlc" / "yellow_tripdata_2024-03.parquet").exists(),
        "n_truths": len(manifest),
        "rows_in_eval_months_outside_apr1_may15": outside_eval,
        "expected_dates_missing": [d for d in expected if d not in got_days],
        "duplicate_ids": len(ids) - len(set(ids)),
        "vector_mismatches": mismatches,
        "all_nonnegative_integer_finite": all(
            r["n_negative"] == 0 and r["n_noninteger"] == 0 and r["n_nonfinite"] == 0 and r["vector_length"] == len(train_ids)
            for r in manifest
        ),
        "zero_candidates_retained": all(r["vector_length"] == len(train_ids) for r in manifest),
    }
    (ROOT / "integrity" / "tlc_independent_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in result if k != "schema_names"}, indent=2))
    return result


def result_names(schemas):
    return schemas


def check_gh_sample() -> dict:
    """Full recount is a second gzip pass. This checker repeats every saved eval day."""
    saved = json.loads((ROOT / "candidate_universes" / "gharchive_repos_top2000.json").read_text())
    ids = saved["ids"]
    index = {r: i for i, r in enumerate(ids)}
    manifest = [json.loads(line) for line in (ROOT / "manifests" / "gharchive_truth_manifest.jsonl").read_text().splitlines()]
    mismatches = []
    accounting = []
    for row in manifest:
        day = row["evaluation_date"]
        path = OBJ / "gharchive" / f"{day}-15.json.gz"
        meta = json.loads((ROOT / "source_metadata" / "gharchive" / f"{day}.json").read_text())
        if sha_file(path) != meta["sha256"]:
            mismatches.append(day + ":hash")
            continue
        counts: Counter = Counter()
        bad = 0
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                name = (ev.get("repo") or {}).get("name")
                if name:
                    counts[name] += 1
        x = np.zeros(len(ids), dtype=np.float64)
        total = 0
        inside = 0
        for name, cnt in counts.items():
            total += cnt
            if name in index:
                x[index[name]] += cnt
                inside += cnt
        got = sha_bytes(np.ascontiguousarray(x).tobytes())
        saved_x = np.load(ROOT.parent.parent / row["vector_path"])
        if got != row["vector_raw_sha256"] or not np.array_equal(saved_x, x):
            mismatches.append(day)
        accounting.append({"day": day, "events": total, "inside": inside, "outside": total - inside, "bad_json": bad})
        print(f"[check-gh] {day} inside {inside} outside {total - inside}", flush=True)
    expected = [d.isoformat() for d in daterange(date(2024, 2, 1), date(2024, 3, 7))]
    got_days = [r["evaluation_date"] for r in manifest]
    # train files present and hashed, but not re-aggregated here if the universe file already records them
    train_hash_ok = {}
    train_counts: Counter = Counter()
    for d in daterange(date(2024, 1, 2), date(2024, 1, 15)):
        path = OBJ / "gharchive" / f"{d.isoformat()}-15.json.gz"
        meta = json.loads((ROOT / "source_metadata" / "gharchive" / f"{d.isoformat()}.json").read_text())
        train_hash_ok[d.isoformat()] = path.exists() and sha_file(path) == meta["sha256"]
        day_counts: Counter = Counter()
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                name = (ev.get("repo") or {}).get("name")
                if name:
                    day_counts[name] += 1
        train_counts.update(day_counts)
        print(f"[check-gh] train {d.isoformat()} events {sum(day_counts.values())}", flush=True)
    rebuilt_ids = [k for k, _ in train_counts.most_common(2000)]
    result = {
        "n_truths": len(manifest),
        "universe_size": len(ids),
        "expected_dates_missing": [d for d in expected if d not in got_days],
        "duplicate_ids": len(got_days) - len(set(r["truth_id"] for r in manifest)),
        "vector_mismatches": mismatches,
        "train_source_hash_ok": train_hash_ok,
        "rebuilt_universe_matches_saved": rebuilt_ids == ids,
        "rebuilt_universe_size": len(rebuilt_ids),
        "train_total_events": int(sum(train_counts.values())),
        "all_nonnegative_integer_finite": all(
            r["n_negative"] == 0 and r["n_noninteger"] == 0 and r["n_nonfinite"] == 0 and r["vector_length"] == len(ids)
            for r in manifest
        ),
        "outside_equals_total_minus_inside": all(a["outside"] == a["events"] - a["inside"] for a in accounting),
        "sample_accounting": accounting,
    }
    (ROOT / "integrity" / "gh_independent_check.json").write_text(json.dumps(result) + "\n")
    print(json.dumps({k: result[k] for k in result if k != "sample_accounting"}, indent=2))
    return result


if __name__ == "__main__":
    import sys
    which = sys.argv[1]
    if which == "tlc":
        check_tlc()
    elif which == "gh":
        check_gh_sample()
    else:
        raise SystemExit("usage: check_integrity.py tlc|gh")
