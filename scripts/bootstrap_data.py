"""Fetch and verify the dated public source objects used by the real-data experiments.

Every object is pinned by URL, byte length, and SHA-256 in
manifests/SOURCE_OBJECTS.json. A present object is re-hashed. A missing object
is downloaded to a temporary file, hashed, and moved into place only when the
hash matches. Any mismatch is fatal.

Usage:
  python scripts/bootstrap_data.py                  # fetch missing, verify all
  python scripts/bootstrap_data.py --check-only     # verify, never download
  python scripts/bootstrap_data.py --check-only --allow-missing
  python scripts/bootstrap_data.py --source tlc     # tlc | gharchive
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "SOURCE_OBJECTS.json"
STATUS_OUT = ROOT / "manifests" / "SOURCE_OBJECTS_STATUS.json"
REQUIRED_SOURCES = {"tlc", "gharchive"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, expected_sha: str, expected_bytes: int) -> None:
    tmp = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "topk-regret-certification-bootstrap/1"})
    h = hashlib.sha256()
    n = 0
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
        while True:
            block = resp.read(1 << 22)
            if not block:
                break
            out.write(block)
            h.update(block)
            n += len(block)
    if n != expected_bytes or h.hexdigest() != expected_sha:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"HASH MISMATCH for {url}: got {n} bytes sha256={h.hexdigest()}, "
            f"expected {expected_bytes} bytes sha256={expected_sha}"
        )
    os.replace(tmp, dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--source", choices=["tlc", "gharchive"])
    args = ap.parse_args()

    objects = json.loads(MANIFEST.read_text())["objects"]
    if args.source:
        objects = [o for o in objects if o["source"] == args.source]
    report = []
    n_bad = n_missing_required = 0
    for obj in objects:
        path = ROOT / obj["local_path"]
        status = None
        if path.exists():
            ok = path.stat().st_size == obj["bytes"] and sha256_file(path) == obj["sha256"]
            status = "PRESENT_AND_HASH_VERIFIED" if ok else "HASH_MISMATCH"
        elif not args.check_only:
            t0 = time.time()
            print(f"[fetch] {obj['url']}", flush=True)
            download(obj["url"], path, obj["sha256"], obj["bytes"])
            status = "FETCHED_AND_HASH_VERIFIED"
            print(f"[fetch] ok {obj['name']} {time.time() - t0:.1f}s", flush=True)
        else:
            status = "MISSING_BLOCKING_REPRODUCTION" if obj["source"] in REQUIRED_SOURCES else "MISSING_OPTIONAL"
        if status == "HASH_MISMATCH":
            n_bad += 1
        if status == "MISSING_BLOCKING_REPRODUCTION":
            n_missing_required += 1
        report.append({"name": obj["name"], "source": obj["source"], "status": status})

    counts: dict[str, int] = {}
    for r in report:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    if not args.source:
        STATUS_OUT.write_text(json.dumps({"counts": counts, "objects": report}, indent=2) + "\n")
    print(json.dumps({"source_objects": counts}, indent=2))
    if n_bad:
        print(f"FAIL: {n_bad} object(s) do not match the pinned SHA-256", file=sys.stderr)
        return 2
    if n_missing_required and not args.allow_missing:
        print(
            f"FAIL: {n_missing_required} required object(s) missing; run `make data`",
            file=sys.stderr,
        )
        return 3
    if n_missing_required:
        print(f"NOTE: {n_missing_required} required object(s) missing; `make reproduce` needs them; run `make data`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
