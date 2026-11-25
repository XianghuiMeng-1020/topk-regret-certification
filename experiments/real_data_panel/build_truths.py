#!/usr/bin/env python3
"""Build the real-data truth panel from the pinned public source objects.

TLC: daily pickup-zone counts over the zones seen in the training months.
GH Archive: daily hour-15 UTC event counts over the top-2000 training repositories.
Reads data/source_objects/ and writes only under this directory.
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

ROOT = Path(__file__).resolve().parent
OBJ = ROOT.parent.parent / "data" / "source_objects"
UNIV = ROOT / "candidate_universes"
VEC = ROOT / "truth_vectors"
MAN = ROOT / "manifests"
INTEG = ROOT / "integrity"

TRAIN_MONTHS = ["2024-01", "2024-02"]
EVAL_MONTHS = ["2024-04", "2024-05"]
EVAL_START = date(2024, 4, 1)
EVAL_END = date(2024, 5, 15)
HOUR_UTC = 15
TOP_N = 2000


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def daterange(start: date, end: date) -> list[date]:
    out = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=1)
    return out


def train_days() -> list[date]:
    return daterange(date(2024, 1, 2), date(2024, 1, 15))


def eval_days() -> list[date]:
    return daterange(date(2024, 2, 1), date(2024, 2, 1) + timedelta(days=35))


def save_vec(path: Path, x: np.ndarray) -> tuple[str, str]:
    arr = np.ascontiguousarray(x, dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        np.save(f, arr)
    return sha_file(path), sha_bytes(arr.tobytes())


def filter_tlc_month(path: Path, ym: str) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_parquet(path, columns=["tpep_pickup_datetime", "PULocationID"])
    n_raw = int(len(raw))
    ts_min = str(raw["tpep_pickup_datetime"].min()) if n_raw else None
    ts_max = str(raw["tpep_pickup_datetime"].max()) if n_raw else None
    df = raw.copy()
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    null_ts = df["tpep_pickup_datetime"].isna()
    null_zone = df["PULocationID"].isna()
    n_null_ts = int(null_ts.sum())
    n_null_zone = int(null_zone.sum())
    df = df.dropna(subset=["tpep_pickup_datetime", "PULocationID"])
    numeric = pd.to_numeric(df["PULocationID"], errors="coerce")
    n_nonintegral = int(((numeric - numeric.round()).abs() > 0).sum()) if len(df) else 0
    df["PULocationID"] = df["PULocationID"].astype(int)
    df["day"] = df["tpep_pickup_datetime"].dt.strftime("%Y-%m-%d")
    y, m = ym.split("-")
    keep = df["day"].str.startswith(f"{y}-{m}")
    n_spill = int((~keep).sum())
    df = df.loc[keep, ["PULocationID", "day"]]
    account = {
        "month": ym,
        "n_raw": n_raw,
        "raw_timestamp_min": ts_min,
        "raw_timestamp_max": ts_max,
        "n_null_timestamp": n_null_ts,
        "n_null_zone": n_null_zone,
        "n_rows_after_dropna": int(n_raw - (null_ts | null_zone).sum()),
        "n_nonintegral_zone_before_cast": n_nonintegral,
        "n_outside_nominal_month": n_spill,
        "n_kept": int(len(df)),
    }
    return df, account


def build_tlc() -> dict:
    frames = {}
    accounts = []
    for ym in TRAIN_MONTHS + EVAL_MONTHS:
        path = OBJ / "tlc" / f"yellow_tripdata_{ym}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        df, account = filter_tlc_month(path, ym)
        meta = json.loads((ROOT / "source_metadata" / "tlc" / f"yellow_tripdata_{ym}.json").read_text())
        account["sha256"] = meta["sha256"]
        account["actual_bytes"] = meta["actual_bytes"]
        frames[ym] = df
        accounts.append(account)
        print(f"[tlc] {ym} kept {account['n_kept']} of {account['n_raw']}", flush=True)

    train_zones = set()
    for ym in TRAIN_MONTHS:
        train_zones.update(int(z) for z in frames[ym]["PULocationID"].unique())
    zone_ids = sorted(train_zones)
    if len(zone_ids) < 50:
        raise RuntimeError(f"TLC universe too small: {len(zone_ids)}")
    index = {z: i for i, z in enumerate(zone_ids)}
    univ = {
        "dataset_id": "tlc_yellow",
        "surface_id": "tlc_pu_zones",
        "candidate_rule": "train_2024q1_all_pulocation_ids",
        "executable_train_months": TRAIN_MONTHS,
        "march_included": False,
        "ids": zone_ids,
        "candidate_count": len(zone_ids),
        "order": "sorted integer PULocationID",
        "source_sha256": [a["sha256"] for a in accounts if a["month"] in TRAIN_MONTHS],
    }
    univ_path = UNIV / "tlc_pulocation_universe.json"
    univ_path.parent.mkdir(parents=True, exist_ok=True)
    univ_text = json.dumps(univ, indent=2, sort_keys=True) + "\n"
    univ_path.write_text(univ_text)
    univ_sha = sha_bytes(univ_text.encode())
    id_sha = sha_bytes(("\n".join(str(z) for z in zone_ids) + "\n").encode())

    eval_df = pd.concat([frames[ym] for ym in EVAL_MONTHS], ignore_index=True)
    rows = []
    missing = []
    vec_dir = VEC / "tlc"
    for cur in daterange(EVAL_START, EVAL_END):
        day_s = cur.isoformat()
        day_df = eval_df[eval_df["day"] == day_s]
        if day_df.empty:
            missing.append({"day": day_s, "reason": "no trips after filters"})
            continue
        x = np.zeros(len(zone_ids), dtype=np.float64)
        vc = day_df["PULocationID"].value_counts()
        total = int(vc.sum())
        in_univ = 0
        for z, cnt in vc.items():
            z = int(z)
            if z in index:
                x[index[z]] += float(cnt)
                in_univ += int(cnt)
        if float(x.sum()) <= 0:
            missing.append({"day": day_s, "reason": "zero mass in universe"})
            continue
        # repeat the assignment to confirm this construction is deterministic
        x2 = np.zeros(len(zone_ids), dtype=np.float64)
        for z, cnt in vc.items():
            z = int(z)
            if z in index:
                x2[index[z]] += float(cnt)
        if not np.array_equal(x, x2):
            raise RuntimeError(f"nondeterministic TLC vector {day_s}")
        path = vec_dir / f"tlc_yellow__pu_zones__{day_s}.npy"
        file_sha, raw_sha = save_vec(path, x)
        rows.append({
            "dataset_id": "tlc_yellow",
            "surface_id": "tlc_pu_zones",
            "truth_id": f"tlc_yellow::pu_zones::{day_s}",
            "evaluation_date": day_s,
            "candidate_universe_id": "tlc_pulocation_universe",
            "candidate_universe_sha256": univ_sha,
            "candidate_id_sha256": id_sha,
            "vector_path": str(path.relative_to(ROOT.parent.parent)),
            "vector_file_sha256": file_sha,
            "vector_raw_sha256": raw_sha,
            "vector_length": int(x.size),
            "count_sum": float(x.sum()),
            "nonzero": int((x > 0).sum()),
            "zero_fraction": float((x == 0).mean()),
            "n_negative": int((x < 0).sum()),
            "n_noninteger": int(np.sum(np.abs(x - np.rint(x)) > 1e-9)),
            "n_nonfinite": int((~np.isfinite(x)).sum()),
            "outside_universe_events": int(total - in_univ),
            "retained_mass": float(in_univ / total) if total else 0.0,
            "source_object_sha256": [a["sha256"] for a in accounts if a["month"] in EVAL_MONTHS],
        })
    manifest = MAN / "tlc_truth_manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "n_truths": len(rows),
        "expected_days": 45,
        "universe_size": len(zone_ids),
        "universe_sha256": univ_sha,
        "candidate_id_sha256": id_sha,
        "manifest_sha256": sha_file(manifest),
        "missing_or_skipped": missing,
        "month_accounts": accounts,
        "train_months_used": TRAIN_MONTHS,
        "march_included": False,
        "eval_dates": [r["evaluation_date"] for r in rows],
        "duplicate_truth_ids": len(rows) - len({r["truth_id"] for r in rows}),
    }
    (INTEG).mkdir(parents=True, exist_ok=True)
    (INTEG / "tlc_build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[tlc] truths {len(rows)} universe {len(zone_ids)} skipped {len(missing)}", flush=True)
    return summary


def aggregate_hour(path: Path) -> tuple[Counter, dict]:
    counts: Counter = Counter()
    n_lines = n_blank = n_bad = n_no_repo = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            n_lines += 1
            line = line.strip()
            if not line:
                n_blank += 1
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            repo = ev.get("repo") or {}
            name = repo.get("name")
            if name:
                counts[name] += 1
            else:
                n_no_repo += 1
    stats = {
        "n_lines": n_lines,
        "n_blank": n_blank,
        "n_bad_json": n_bad,
        "n_no_repo": n_no_repo,
        "n_events_with_repo": int(sum(counts.values())),
        "n_unique_repos": len(counts),
        "gzip_completed": True,
    }
    return counts, stats


def build_gh() -> dict:
    missing = []
    for d in train_days() + eval_days():
        path = OBJ / "gharchive" / f"{d.isoformat()}-{HOUR_UTC}.json.gz"
        if not path.exists() or path.stat().st_size == 0:
            missing.append(d.isoformat())
    if missing:
        summary = {
            "status": "INCOMPLETE_SOURCE",
            "missing_dates": missing,
            "note": "A partial training set was not used to name a top-2000 universe.",
        }
        (INTEG).mkdir(parents=True, exist_ok=True)
        (INTEG / "gh_build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("[gh] incomplete", missing, flush=True)
        return summary

    train_counts: Counter = Counter()
    day_stats = []
    for d in train_days():
        path = OBJ / "gharchive" / f"{d.isoformat()}-{HOUR_UTC}.json.gz"
        counts, stats = aggregate_hour(path)
        train_counts.update(counts)
        meta = json.loads((ROOT / "source_metadata" / "gharchive" / f"{d.isoformat()}.json").read_text())
        stats.update({"date": d.isoformat(), "phase": "train", "sha256": meta["sha256"], "actual_bytes": meta["actual_bytes"]})
        day_stats.append(stats)
        print(f"[gh] train {d} events {stats['n_events_with_repo']}", flush=True)

    first_seen = {name: i for i, name in enumerate(train_counts)}
    by_stable = [k for k, _ in sorted(train_counts.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))][:TOP_N]
    by_counter = [k for k, _ in train_counts.most_common(TOP_N)]
    if by_stable != by_counter:
        raise RuntimeError("Counter.most_common tie order disagrees with first-seen stable sort")
    repo_ids = by_counter
    index = {r: i for i, r in enumerate(repo_ids)}
    univ = {
        "dataset_id": "gharchive",
        "surface_id": "gh_repos",
        "candidate_rule": "train_jan2024_top2000_repos_hour15utc",
        "train_days": [d.isoformat() for d in train_days()],
        "hour_utc": HOUR_UTC,
        "top_n_requested": TOP_N,
        "ids": repo_ids,
        "candidate_count": len(repo_ids),
        "order": "Counter.most_common(2000); ties keep first-seen order",
        "tie_order_matches_stable_sort": True,
        "train_total_events": int(sum(train_counts.values())),
        "source_sha256": [s["sha256"] for s in day_stats],
    }
    univ_path = UNIV / "gharchive_repos_top2000.json"
    univ_path.parent.mkdir(parents=True, exist_ok=True)
    univ_text = json.dumps(univ, indent=2, sort_keys=True) + "\n"
    univ_path.write_text(univ_text)
    univ_sha = sha_bytes(univ_text.encode())
    id_sha = sha_bytes(("\n".join(repo_ids) + "\n").encode())

    rows = []
    skipped = []
    for d in eval_days():
        path = OBJ / "gharchive" / f"{d.isoformat()}-{HOUR_UTC}.json.gz"
        counts, stats = aggregate_hour(path)
        meta = json.loads((ROOT / "source_metadata" / "gharchive" / f"{d.isoformat()}.json").read_text())
        stats.update({"date": d.isoformat(), "phase": "eval", "sha256": meta["sha256"], "actual_bytes": meta["actual_bytes"]})
        day_stats.append(stats)
        x = np.zeros(len(repo_ids), dtype=np.float64)
        total = 0
        in_univ = 0
        for name, cnt in counts.items():
            total += cnt
            if name in index:
                x[index[name]] += cnt
                in_univ += cnt
        print(f"[gh] eval {d} events {total} in_universe {in_univ}", flush=True)
        if float(x.sum()) <= 0:
            skipped.append({"day": d.isoformat(), "reason": "zero mass in universe"})
            continue
        vec_path = VEC / "gharchive" / f"gharchive__repos__{d.isoformat()}.npy"
        file_sha, raw_sha = save_vec(vec_path, x)
        rows.append({
            "dataset_id": "gharchive",
            "surface_id": "gh_repos",
            "truth_id": f"gharchive::repos::{d.isoformat()}",
            "evaluation_date": d.isoformat(),
            "candidate_universe_id": "gharchive_repos_top2000",
            "candidate_universe_sha256": univ_sha,
            "candidate_id_sha256": id_sha,
            "vector_path": str(vec_path.relative_to(ROOT.parent.parent)),
            "vector_file_sha256": file_sha,
            "vector_raw_sha256": raw_sha,
            "vector_length": int(x.size),
            "count_sum": float(x.sum()),
            "nonzero": int((x > 0).sum()),
            "zero_fraction": float((x == 0).mean()),
            "n_negative": int((x < 0).sum()),
            "n_noninteger": int(np.sum(np.abs(x - np.rint(x)) > 1e-9)),
            "n_nonfinite": int((~np.isfinite(x)).sum()),
            "outside_universe_events": int(total - in_univ),
            "retained_mass": float(in_univ / total) if total else 0.0,
            "day_events_with_repo": int(total),
            "source_object_sha256": meta["sha256"],
        })
    manifest = MAN / "gharchive_truth_manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    (INTEG / "gh_day_stats.json").write_text(json.dumps(day_stats) + "\n")
    summary = {
        "status": "BUILT",
        "n_train_objects": len(train_days()),
        "n_eval_objects": len(eval_days()),
        "n_truths": len(rows),
        "universe_size": len(repo_ids),
        "universe_sha256": univ_sha,
        "candidate_id_sha256": id_sha,
        "manifest_sha256": sha_file(manifest),
        "train_total_events": int(sum(train_counts.values())),
        "skipped": skipped,
        "duplicate_truth_ids": len(rows) - len({r["truth_id"] for r in rows}),
        "eval_dates": [r["evaluation_date"] for r in rows],
    }
    (INTEG / "gh_build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[gh] truths {len(rows)} universe {len(repo_ids)}", flush=True)
    return summary


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "tlc"
    if which == "tlc":
        build_tlc()
    elif which == "gh":
        build_gh()
    else:
        raise SystemExit("usage: build_truths.py tlc|gh")
