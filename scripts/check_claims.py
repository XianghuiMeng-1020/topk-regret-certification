"""Recompute every registered quantitative result from ledger rows.

For each target in manifests/REGRESSION_TARGETS.json:
  1. read the experiment's own summary field (when one exists),
  2. recompute the value from the row-level ledgers with code in this file,
  3. compare both with the registered value, reporting bitwise equality
     (identical float repr) separately from numeric equality (abs <= 1e-12).

--root selects the tree to check: this repository (default) or a fresh
reproduction workspace. The registered values are read only for the
comparison step. When checking this repository, the README must quote the
verified numbers at the stated rounding.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAU = 0.10
SOURCES = ("tlc_yellow", "gharchive")
TIME_FORWARD_OFFICIAL_SEED = 20270111
MATCHED_TIE_MASTER = 20261128


def load_audit_helpers():
    path = ROOT / "experiments/consistency_audit/audit.py"
    spec = importlib.util.spec_from_file_location("consistency_audit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def truth_macro(flag: np.ndarray, dataset: np.ndarray, truth: np.ndarray, name: str) -> float:
    m = dataset == name
    return float(pd.Series(flag[m]).groupby(truth[m]).mean().mean())


def rate_block(frame: pd.DataFrame, col: str) -> dict:
    flag = frame[col].to_numpy() / frame["S"].to_numpy() <= TAU
    ds, tr = frame["dataset_id"].to_numpy(), frame["truth_id"].to_numpy()
    tlc = truth_macro(flag, ds, tr, "tlc_yellow")
    gh = truth_macro(flag, ds, tr, "gharchive")
    return {"tlc": tlc, "github": gh, "macro": 0.5 * (tlc + gh), "_flag": flag}


def diff(a: dict, b: dict) -> dict:
    return {s: a[s] - b[s] for s in ("tlc", "github", "macro")}


def recompute(root: Path, helpers) -> dict:
    c = pd.read_parquet(root / "experiments/real_box_evaluation/ledger.parquet")
    d = pd.read_parquet(root / "experiments/real_order_statistic_evaluation/ledger.parquet")
    out: dict = {}
    out["panel.truths"] = int(c["truth_id"].nunique())
    out["panel.cells"] = int(len(c))

    ub, us = rate_block(c, "union_C_box"), rate_block(c, "union_C_strong")
    for s, v in diff(ub, us).items():
        out[f"box.delta_union.{s}"] = v

    r1c, r1s = rate_block(d, "C_OS_rank1_coupled"), rate_block(d, "C_OS_rank1_strong")
    ec, es = rate_block(d, "C_OS_coupled_envelope"), rate_block(d, "C_OS_strong_envelope")
    out["os.rank1_coupling.macro"] = diff(r1c, r1s)["macro"]
    out["os.os_coupling.macro"] = diff(ec, es)["macro"]
    out["os.coupled_refine.macro"] = diff(ec, r1c)["macro"]
    out["os.n_new_coupled"] = int(np.sum(ec["_flag"] & ~r1c["_flag"]))

    work = c[["dataset_id", "truth_id", "evaluation_date", "mechanism", "epsilon", "k", "weight_family",
              "replicate", "S", "true_regret", "union_C_box", "union_C_strong"]].copy()
    work["epsilon_key"] = work["epsilon"].map(lambda v: str(float(v)))
    tie = np.empty(len(work), dtype=np.uint64)
    for i, row in enumerate(zip(work.truth_id, work.mechanism, work.epsilon_key, work.k.astype(int),
                                work.weight_family, work.replicate.astype(int))):
        payload = f"{MATCHED_TIE_MASTER}|{row[0]}|{row[1]}|{row[2]}|{row[3]}|{row[4]}|{row[5]}".encode()
        tie[i] = np.uint64(int(hashlib.sha256(payload).hexdigest()[:16], 16))
    work["tie"] = tie
    work["r"] = work["true_regret"].to_numpy() / work["S"].to_numpy()
    dates = (work[["dataset_id", "truth_id", "evaluation_date"]].drop_duplicates()
             .sort_values(["dataset_id", "evaluation_date", "truth_id"]))
    dates["truth_index"] = dates.groupby("dataset_id").cumcount()
    work = work.merge(dates[["dataset_id", "truth_id", "truth_index"]], on=["dataset_id", "truth_id"])
    work["final"] = work["truth_index"].to_numpy() * 100 + work["replicate"].to_numpy()
    per_source = {}
    for dataset, sub in work.groupby("dataset_id"):
        diffs = []
        for _, piece in sub.groupby(["mechanism", "epsilon_key", "k", "weight_family"]):
            r = piece["r"].to_numpy()
            m = len(piece) // 2
            ql = piece["union_C_strong"].to_numpy() / piece["S"].to_numpy()
            qr = piece["union_C_box"].to_numpy() / piece["S"].to_numpy()
            left = np.lexsort((piece["final"].to_numpy(), piece["tie"].to_numpy(), ql))[:m]
            right = np.lexsort((piece["final"].to_numpy(), piece["tie"].to_numpy(), qr))[:m]
            diffs.append(float(r[left].mean() - r[right].mean()))
        per_source[dataset] = float(np.mean(diffs))
    out["matched_rate.macro_D"] = 0.5 * (per_source["gharchive"] + per_source["tlc_yellow"])

    led = pd.read_parquet(root / "experiments/time_forward_utility/test_policy_ledger.parquet")
    thr = pd.read_parquet(root / "experiments/time_forward_utility/primary_thresholds.parquet")
    keys = ["dataset_id", "mechanism", "epsilon", "k", "weight_family"]
    cols = keys + ["v", "n_equal", "need"]
    s_thr = thr[thr.method.eq("union_C_strong")][cols].rename(columns={"v": "v_s", "n_equal": "ne_s", "need": "nd_s"})
    c_thr = thr[thr.method.eq("union_C_box")][cols].rename(columns={"v": "v_c", "n_equal": "ne_c", "need": "nd_c"})
    base = led.merge(s_thr, on=keys, how="left", validate="many_to_one").merge(c_thr, on=keys, how="left", validate="many_to_one")
    base = base.sort_values(keys + ["truth_id", "replicate"]).reset_index(drop=True)
    h = helpers.hashes_for(base, TIME_FORWARD_OFFICIAL_SEED)
    acc_s = helpers.accept_mask(base.q_strong.to_numpy(), h, base.v_s.to_numpy(), base.ne_s.to_numpy(), base.nd_s.to_numpy())
    acc_c = helpers.accept_mask(base.q_coupled.to_numpy(), h, base.v_c.to_numpy(), base.ne_c.to_numpy(), base.nd_c.to_numpy())
    out["time_forward.accept_masks_match_ledger"] = bool(
        np.array_equal(acc_s, base.accept_strong.to_numpy().astype(bool))
        and np.array_equal(acc_c, base.accept_coupled.to_numpy().astype(bool))
    )
    fields = {"frac_l": [], "frac_r": [], "mean_l": [], "mean_r": [], "D_accept": [], "D_regret": []}
    for dataset in ("gharchive", "tlc_yellow"):
        sub_idx = base.index[base.dataset_id.eq(dataset)]
        sub = base.loc[sub_idx]
        per = {k: [] for k in fields}
        for _, piece in sub.groupby(keys[1:], sort=True):
            idx = piece.index.to_numpy()
            r = base.loc[idx, "r"].to_numpy()
            ls, rs = acc_s[idx], acc_c[idx]
            per["frac_l"].append(float(ls.mean()))
            per["frac_r"].append(float(rs.mean()))
            per["D_accept"].append(float(rs.mean() - ls.mean()))
            if ls.any():
                per["mean_l"].append(float(r[ls].mean()))
            if rs.any():
                per["mean_r"].append(float(r[rs].mean()))
            if ls.any() and rs.any():
                per["D_regret"].append(float(r[ls].mean() - r[rs].mean()))
        for k in fields:
            fields[k].append(float(np.mean(per[k])))
    for k, v in fields.items():
        out[f"time_forward.{k}"] = float(np.mean(v))

    front = json.loads((root / "experiments/matched_rate_utility/frontier_exact.json").read_text())
    differences, t50_changed = 0, False
    for method, (src, col) in helpers.METHODS.items():
        frame = c if src == "box" else d
        got = helpers.integer_inverse(frame[col].to_numpy() / frame["S"].to_numpy(),
                                      frame.dataset_id.eq("tlc_yellow").to_numpy())
        for label, row in got.items():
            if front[method][label]["T"] != row["T"]:
                differences += 1
                t50_changed |= label == "0.50"
    out["frontier.n_differences"] = differences
    out["frontier.t50_unchanged"] = not t50_changed
    return out


def dig(obj, path):
    for key in path:
        obj = obj[key]
    return obj


def compare(expected: str, got) -> dict:
    if isinstance(got, bool) or expected in ("True", "False"):
        ok = str(bool(got)) == expected
        return {"bitwise": ok, "numeric": ok}
    if isinstance(got, (int, np.integer)) and "." not in expected and "e" not in expected:
        ok = int(got) == int(expected)
        return {"bitwise": ok, "numeric": ok}
    e = float(expected)
    g = float(got)
    return {"bitwise": repr(g) == repr(e), "numeric": abs(g - e) <= TOL, "abs_diff": abs(g - e)}


def readme_checks(readme: Path, values: dict) -> list[dict]:
    text = readme.read_text(encoding="utf-8")
    checks = [
        ("PANEL_TRUTHS", "81 truths", None),
        ("PANEL_CELLS", "43,740", None),
        ("BOX_MACRO_GAIN", "+{:.2f} pp", 100.0),
        ("BOX_GITHUB_GAIN", "+{:.2f} pp", 100.0),
        ("OS_AWARE_COUPLING", "+{:.2f} pp", 100.0),
        ("MATCHED_RATE_REGRET_DIFF", "{:.3e}", 1.0),
    ]
    out = []
    for tid, fmt, scale in checks:
        token = fmt if scale is None else fmt.format(values[tid] * scale)
        out.append({"target": tid, "token": token, "present": token in text})
    return out


TOL = 1e-12


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    reg = json.loads((ROOT / "manifests/REGRESSION_TARGETS.json").read_text())
    global TOL
    TOL = float(reg["numeric_tolerance_abs"])

    got = recompute(root, load_audit_helpers())
    rows, failures = [], []
    for t in reg["targets"]:
        row = {"id": t["id"], "experiment": t["experiment"], "expected": t["expected"]}
        if t["summary"]:
            rel, path = t["summary"]
            val = dig(json.loads((root / rel).read_text()), path)
            row["summary_value"] = val
            row["summary_vs_expected"] = compare(t["expected"], val)
            if not row["summary_vs_expected"]["numeric"]:
                failures.append(f"{t['id']} summary")
        if t["recompute"]:
            val = got[t["recompute"]]
            row["recomputed_value"] = val
            row["recompute_vs_expected"] = compare(t["expected"], val)
            if not row["recompute_vs_expected"]["numeric"]:
                failures.append(f"{t['id']} recompute")
        rows.append(row)
    if not got["time_forward.accept_masks_match_ledger"]:
        failures.append("time-forward accept masks differ from stored ledger decisions")

    own_tree = root == ROOT.resolve()
    result = {
        "checked_root": "repository" if own_tree else str(root),
        "numeric_tolerance_abs": TOL,
        "targets": rows,
        "n_targets": len(rows),
        "n_bitwise_recompute": sum(1 for r in rows if r.get("recompute_vs_expected", {}).get("bitwise")),
        "n_bitwise_summary": sum(1 for r in rows if r.get("summary_vs_expected", {}).get("bitwise")),
        "time_forward_accept_masks_match_ledger": got["time_forward.accept_masks_match_ledger"],
        "failures": failures,
    }
    readme = ROOT / "README.md"
    if own_tree and readme.exists():
        values = {t["id"]: float(t["expected"]) for t in reg["targets"] if t["expected"] not in ("True", "False")}
        rc = readme_checks(readme, values)
        result["readme_number_checks"] = rc
        missing = [r["token"] for r in rc if not r["present"]]
        if missing:
            failures.append(f"README lacks verified numbers {missing}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=float) + "\n")
        print(f"written {out}")
    for r in rows:
        rc = r.get("recompute_vs_expected", {})
        sc = r.get("summary_vs_expected", {})
        print(f"{r['id']:34s} expected={r['expected']:>24s} recompute={'BITWISE' if rc.get('bitwise') else ('NUMERIC' if rc.get('numeric') else ('-' if not rc else 'FAIL'))} summary={'BITWISE' if sc.get('bitwise') else ('NUMERIC' if sc.get('numeric') else ('-' if not sc else 'FAIL'))}")
    if failures:
        print("FAIL:", failures, file=sys.stderr)
        return 1
    print(f"OK: {len(rows)} registered targets reproduced from ledger rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
