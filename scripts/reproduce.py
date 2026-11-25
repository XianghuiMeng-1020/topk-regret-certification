"""Fresh end-to-end replay of every experiment.

Creates reproductions/<run id>/workspace/ with the same relative layout as this
repository, copies src/ and the experiment runners, links the hash-pinned
public source objects, and runs every experiment in order. Committed outputs
under experiments/ are only read, never written.

The synthetic experiments run with the synthetic interpreter
(--synthetic-python, default .venv-synthetic/bin/python) and the real-data
experiments with the current interpreter.

After each experiment, every output file is compared with the committed file
of the same path:
  BITWISE_IDENTICAL        same bytes
  SEMANTIC_EQUAL           same parsed content (JSON / JSONL / parquet / npy)
                           after excluding declared volatile fields
                           (wall-clock timings, interpreter/platform strings,
                           hashes of input files that themselves hold timings)
  NUMERIC_EQUAL            floats differ by at most 1e-12 (absolute)
  VOLATILE_ONLY            a timing file whose content differs
  DIFFERENT                anything else
  NEW_FILE / NOT_PRODUCED  present on one side only

Hash locks. The order-statistic, matched-rate and time-forward runners refuse
to run unless their upstream files carry the SHA-256 written in their source.
A fresh box-event ledger contains wall-clock runtime columns, so it never
carries the committed hash. For such a stage the orchestrator temporarily
places the committed copy of each lock-mismatched upstream file in the
workspace, runs the stage, and restores the replayed file afterwards. Every
substitution is recorded in run_manifest.json as UPSTREAM_PINNED_TO_COMMITTED.
Agreement of the replayed upstream file is established by the comparison above.

Usage:
  python scripts/reproduce.py [--run-id ID] [--experiments name,name,...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TOL = 1e-12

PANEL = "experiments/real_data_panel"
SYN = "synthetic"
REAL = "real"
STAGES = [
    ("synthetic_box_calibration", SYN, "experiments/synthetic_box_calibration/run_calibration.py", []),
    ("synthetic_box_calibration", SYN, "experiments/synthetic_box_calibration/run_canonical.py", []),
    ("synthetic_order_statistic", SYN, "experiments/synthetic_order_statistic/run.py", []),
    ("synthetic_count_feasibility", SYN, "experiments/synthetic_count_feasibility/run_feasibility.py", []),
    ("synthetic_count_feasibility", SYN, "experiments/synthetic_count_feasibility/run_closure.py", []),
    ("real_data_panel", REAL, f"{PANEL}/build_truths.py", ["tlc"]),
    ("real_data_panel", REAL, f"{PANEL}/build_truths.py", ["gh"]),
    ("real_data_panel", REAL, f"{PANEL}/check_integrity.py", ["tlc"]),
    ("real_data_panel", REAL, f"{PANEL}/check_integrity.py", ["gh"]),
    ("real_data_panel", REAL, f"{PANEL}/smoke_release.py", ["tlc"]),
    ("real_data_panel", REAL, f"{PANEL}/smoke_release.py", ["gharchive"]),
    ("real_box_evaluation", REAL, "experiments/real_box_evaluation/run.py", []),
    ("real_order_statistic_evaluation", REAL, "experiments/real_order_statistic_evaluation/run.py", []),
    ("matched_rate_utility", REAL, "experiments/matched_rate_utility/run.py", []),
    ("time_forward_utility", REAL, "experiments/time_forward_utility/run.py", []),
    ("consistency_audit", REAL, "experiments/consistency_audit/audit.py", []),
]
EXPERIMENTS = list(dict.fromkeys(name for name, *_ in STAGES))
PANEL_OUTPUT_SUBDIRS = ["candidate_universes", "manifests", "truth_vectors", "integrity", "smoke_tests"]
PANEL_INPUTS = ["panel_spec.json", "source_metadata"]
# Upstream files whose SHA-256 is written in each runner.
LOCKED_INPUTS = {
    "real_order_statistic_evaluation": [
        f"{PANEL}/manifests/tlc_truth_manifest.jsonl",
        f"{PANEL}/manifests/gharchive_truth_manifest.jsonl",
        f"{PANEL}/candidate_universes/tlc_pulocation_universe.json",
        f"{PANEL}/candidate_universes/gharchive_repos_top2000.json",
        "experiments/real_box_evaluation/ledger.parquet",
        "experiments/real_box_evaluation/protocol.json",
    ],
    "matched_rate_utility": [
        f"{PANEL}/manifests/tlc_truth_manifest.jsonl",
        f"{PANEL}/manifests/gharchive_truth_manifest.jsonl",
        "experiments/real_box_evaluation/ledger.parquet",
        "experiments/real_box_evaluation/protocol.json",
        "experiments/real_order_statistic_evaluation/ledger.parquet",
        "experiments/real_order_statistic_evaluation/protocol.json",
    ],
    "time_forward_utility": [
        f"{PANEL}/manifests/tlc_truth_manifest.jsonl",
        f"{PANEL}/manifests/gharchive_truth_manifest.jsonl",
        "experiments/real_box_evaluation/ledger.parquet",
        "experiments/real_order_statistic_evaluation/ledger.parquet",
        "experiments/matched_rate_utility/protocol.json",
        "experiments/matched_rate_utility/frontier_exact.json",
    ],
}
VOLATILE_KEY = re.compile(r"(runtime|elapsed|wall|seconds|_s$|^total_s$|^python$|^python_build$|^platform$|^machine$|^interpreter$|^ledger_sha256$|^peak_traced_bytes$|^source_sha256$|^input_sha256$)")
VOLATILE_FILES = {"stage_times.json", "runtime.json"}

LAUNCHER = r"""
import json, resource, subprocess, sys, time
t0 = time.perf_counter()
rc = subprocess.call(sys.argv[2:])
dt = time.perf_counter() - t0
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
scale = 1 if sys.platform == "darwin" else 1024
with open(sys.argv[1], "w") as fh:
    json.dump({"exit_code": rc, "wall_s": dt, "peak_rss_bytes": int(ru.ru_maxrss) * scale,
               "user_s": ru.ru_utime, "sys_s": ru.ru_stime}, fh)
sys.exit(rc)
"""


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def tree_sha(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(root)).encode() + b"\0" + sha(p).encode() + b"\n")
    return h.hexdigest()


def place_object(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        if subprocess.call(["cp", "-c", str(src), str(dst)], stderr=subprocess.DEVNULL) == 0:
            return "clone"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def keep(p: Path, root: Path) -> bool:
    parts = set(p.relative_to(root).parts)
    return p.is_file() and p.suffix != ".py" and not parts & {"__pycache__", "chunks"} and p.name != ".DS_Store"


def strip_volatile(obj, path="", dropped=None):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if VOLATILE_KEY.search(str(k)):
                if dropped is not None:
                    dropped.append(f"{path}/{k}")
                continue
            out[k] = strip_volatile(v, f"{path}/{k}", dropped)
        return out
    if isinstance(obj, list):
        return [strip_volatile(v, f"{path}/{i}", dropped) for i, v in enumerate(obj)]
    return obj


def numeric_close(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(numeric_close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(numeric_close(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= TOL or (np.isnan(float(a)) and np.isnan(float(b)))
        except (TypeError, ValueError):
            return False
    return a == b


def compare_json_like(a, b, volatile_file: bool) -> tuple[str, list]:
    if a == b:
        return "SEMANTIC_EQUAL", []
    dropped: list = []
    sa, sb = strip_volatile(a, "", dropped), strip_volatile(b, "", None)
    if sa == sb:
        return "SEMANTIC_EQUAL", sorted(set(dropped))
    if numeric_close(sa, sb):
        return "NUMERIC_EQUAL", sorted(set(dropped))
    if volatile_file:
        return "VOLATILE_ONLY", []
    return "DIFFERENT", []


def compare_file(new: Path, old: Path) -> dict:
    if sha(new) == sha(old):
        return {"status": "BITWISE_IDENTICAL"}
    suffix = new.suffix
    try:
        if suffix == ".json":
            status, dropped = compare_json_like(json.loads(new.read_text()), json.loads(old.read_text()), new.name in VOLATILE_FILES)
            return {"status": status, "volatile_fields_excluded": dropped}
        if suffix == ".jsonl":
            a = [json.loads(x) for x in new.read_text().splitlines() if x.strip()]
            b = [json.loads(x) for x in old.read_text().splitlines() if x.strip()]
            status, dropped = compare_json_like(a, b, False)
            return {"status": status, "volatile_fields_excluded": dropped}
        if suffix == ".npy":
            a, b = np.load(new), np.load(old)
            if a.dtype == b.dtype and np.array_equal(a, b):
                return {"status": "SEMANTIC_EQUAL", "note": "array values identical; file bytes differ"}
            return {"status": "DIFFERENT"}
        if suffix == ".parquet":
            a, b = pd.read_parquet(new), pd.read_parquet(old)
            if list(a.columns) != list(b.columns) or len(a) != len(b):
                return {"status": "DIFFERENT", "note": "columns or row count differ"}
            vol = [c for c in a.columns if "runtime" in c]
            exact, close, bad = True, True, []
            for col in a.columns:
                if col in vol:
                    continue
                x, y = a[col].to_numpy(), b[col].to_numpy()
                if x.dtype.kind == "f" or y.dtype.kind == "f":
                    xf, yf = x.astype(float), y.astype(float)
                    if not np.array_equal(xf, yf, equal_nan=True):
                        exact = False
                        if not np.allclose(xf, yf, rtol=0.0, atol=TOL, equal_nan=True):
                            close = False
                            bad.append(col)
                elif not (pd.Series(x).equals(pd.Series(y))):
                    exact = close = False
                    bad.append(col)
            status = "SEMANTIC_EQUAL" if exact else ("NUMERIC_EQUAL" if close else "DIFFERENT")
            return {"status": status, "rows": int(len(a)), "volatile_columns_excluded": vol, "different_columns": bad}
    except Exception as exc:  # noqa: BLE001
        return {"status": "DIFFERENT", "note": f"compare error {exc!r}"}
    if new.name in VOLATILE_FILES:
        return {"status": "VOLATILE_ONLY"}
    return {"status": "DIFFERENT"}


def compare_experiment(name: str, ws: Path) -> dict:
    rel_dir = f"experiments/{name}"
    new_root, old_root = ws / rel_dir, REPO / rel_dir
    if name == "real_data_panel":
        names = set()
        for sub in PANEL_OUTPUT_SUBDIRS:
            for root in (new_root / sub, old_root / sub):
                if root.exists():
                    names |= {str(p.relative_to(root.parents[0])) for p in root.rglob("*") if p.is_file()}
    else:
        names = {str(p.relative_to(new_root)) for p in new_root.rglob("*") if keep(p, new_root)}
        names |= {str(p.relative_to(old_root)) for p in old_root.rglob("*") if keep(p, old_root)}
    files = {}
    for name in sorted(names):
        a, b = new_root / name, old_root / name
        rel = f"{rel_dir}/{name}"
        if a.exists() and b.exists():
            files[rel] = compare_file(a, b)
            files[rel]["sha256_replay"] = sha(a)
        elif a.exists():
            files[rel] = {"status": "NEW_FILE", "sha256_replay": sha(a)}
        else:
            files[rel] = {"status": "NOT_PRODUCED"}
    counts: dict[str, int] = {}
    for v in files.values():
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    return {"counts": counts, "files": files}


def verdict(cmp: dict) -> str:
    c = cmp["counts"]
    if c.get("DIFFERENT") or c.get("NOT_PRODUCED") or c.get("NEW_FILE"):
        return "FAILED"
    if set(c) <= {"BITWISE_IDENTICAL"}:
        return "VERIFIED_BITWISE"
    return "VERIFIED_NUMERICALLY"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--experiments", default=",".join(EXPERIMENTS))
    ap.add_argument("--synthetic-python", default=str(REPO / ".venv-synthetic" / "bin" / "python"))
    args = ap.parse_args()
    wanted = args.experiments.split(",")
    unknown = sorted(set(wanted) - set(EXPERIMENTS))
    if unknown:
        raise SystemExit(f"unknown experiments {unknown}")
    synthetic_python = str(Path(args.synthetic_python).absolute())
    interpreters = {REAL: sys.executable, SYN: synthetic_python}
    if any(name.startswith("synthetic_") for name in wanted) and not Path(synthetic_python).exists():
        raise SystemExit(f"{args.synthetic_python} not found; run `make setup` or pass --synthetic-python")
    started_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"run-{started_utc}"
    run_dir = REPO / "reproductions" / run_id
    if run_dir.exists():
        raise SystemExit(f"{run_dir} exists; choose a new --run-id")
    ws = run_dir / "workspace"
    (run_dir / "logs").mkdir(parents=True)

    shutil.copytree(REPO / "src", ws / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for name in EXPERIMENTS:
        src_dir, dst_dir = REPO / "experiments" / name, ws / "experiments" / name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for script in src_dir.glob("*.py"):
            shutil.copy2(script, dst_dir / script.name)
    for item in PANEL_INPUTS:
        src, dst = REPO / PANEL / item, ws / PANEL / item
        shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)
    objects = json.loads((REPO / "manifests/SOURCE_OBJECTS.json").read_text())["objects"]
    placement = {}
    if "real_data_panel" in wanted:
        for obj in objects:
            src = REPO / obj["local_path"]
            if not src.exists() or src.stat().st_size != obj["bytes"]:
                raise SystemExit(f"missing source object {obj['local_path']}; run `make data`")
            placement[obj["name"]] = place_object(src, ws / obj["local_path"])

    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    env.update({"PYTHONPATH": str(ws / "src"), "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"})

    stage_rows, results = [], {}
    abort = None
    for i, (name, kind, rel, argv) in enumerate(STAGES):
        if name not in wanted:
            continue
        label = f"{Path(rel).stem}{'_' + argv[0] if argv else ''}"
        if abort:
            stage_rows.append({"experiment": name, "stage": label, "status": "NOT_RUN", "reason": abort})
            continue
        substitutions = []
        for lrel in LOCKED_INPUTS.get(name, []):
            committed, live = REPO / lrel, ws / lrel
            if not live.exists() or sha(live) != sha(committed):
                held = run_dir / "held_replay" / lrel
                if live.exists():
                    held.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(live), str(held))
                live.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(committed, live)
                substitutions.append({"path": lrel, "replay_sha256": sha(held) if held.exists() else None,
                                      "committed_sha256": sha(committed), "mode": "UPSTREAM_PINNED_TO_COMMITTED"})
        py = interpreters[kind]
        log = f"{i:02d}_{name}_{label}"
        metrics_path = run_dir / "logs" / f"{log}.metrics.json"
        cmd = [py, "-c", LAUNCHER, str(metrics_path), py, "-u", rel, *argv]
        print(f"[reproduce] {name}: python {rel} {' '.join(argv)}  pinned={len(substitutions)}", flush=True)
        with (run_dir / "logs" / f"{log}.stdout.log").open("w") as out, (run_dir / "logs" / f"{log}.stderr.log").open("w") as err:
            rc = subprocess.call(cmd, cwd=ws, env=env, stdout=out, stderr=err)
        for s in substitutions:
            live, held = ws / s["path"], run_dir / "held_replay" / s["path"]
            if held.exists():
                shutil.move(str(held), str(live))
        m = json.loads(metrics_path.read_text())
        row = {"experiment": name, "stage": label, "command": f"python -u {rel} {' '.join(argv)}".strip(),
               "interpreter": kind, "exit_code": rc, "wall_s": round(m["wall_s"], 3),
               "peak_rss_mib": round(m["peak_rss_bytes"] / 2**20, 1), "upstream_substitutions": substitutions}
        stage_rows.append(row)
        print(f"[reproduce] {name}: exit={rc} wall={m['wall_s']:.1f}s peak={row['peak_rss_mib']} MiB", flush=True)
        if rc != 0:
            row["status"] = "FAILED"
            row["stderr_tail"] = (run_dir / "logs" / f"{log}.stderr.log").read_text()[-2000:]
            abort = f"upstream stage {name}/{label} failed"
            continue
        row["status"] = "COMPLETED"
        if i == max(j for j, s in enumerate(STAGES) if s[0] == name):
            cmp = compare_experiment(name, ws)
            results[name] = {"verdict": verdict(cmp), **cmp}
            print(f"[reproduce] {name} comparison: {cmp['counts']} -> {results[name]['verdict']}", flush=True)

    claims = None
    needed = {"real_box_evaluation", "real_order_statistic_evaluation", "matched_rate_utility", "time_forward_utility"}
    if not abort and needed <= set(wanted):
        rc = subprocess.call([sys.executable, str(REPO / "scripts/check_claims.py"), "--root", str(ws),
                              "--out", str(run_dir / "claim_check.json")], env=env,
                             stdout=(run_dir / "logs/check_claims.stdout.log").open("w"),
                             stderr=(run_dir / "logs/check_claims.stderr.log").open("w"))
        claims = {"exit_code": rc, "path": "claim_check.json"}

    manifest = {
        "schema": "run-manifest-v1",
        "run_id": run_id,
        "src_tree_sha256": tree_sha(REPO / "src"),
        "runner_sha256": {rel: sha(REPO / rel) for _, _, rel, _ in STAGES},
        "source_objects": {o["name"]: o["sha256"] for o in objects},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "machine": platform.machine(), "system": platform.system()},
        "stages": [{k: ([{f: x for f, x in s.items() if f != "replay_sha256"} for s in v] if k == "upstream_substitutions" else v)
                    for k, v in r.items() if k not in ("wall_s", "peak_rss_mib", "stderr_tail")} for r in stage_rows],
        "verdicts": {name: v["verdict"] for name, v in results.items()},
        "claim_check": claims,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (run_dir / "stage_metrics.json").write_text(json.dumps({"started_utc": started_utc, "object_placement": placement,
                                                            "stages": stage_rows}, indent=2) + "\n")
    (run_dir / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"run_id": run_id, "verdicts": manifest["verdicts"], "claim_check": claims}, indent=2))
    if abort or any(v["verdict"] == "FAILED" for v in results.values()) or (claims and claims["exit_code"] != 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
