"""Clean-room import check.

Starts a child interpreter in isolated mode (-I: PYTHONPATH and user site are
ignored) from a temporary working directory. The child puts only this
repository's src/, scripts/, and tests/ on sys.path, imports every module of
private_mobility_cert and every experiment entry point, then checks that each
loaded project module lives inside this repository.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def entry_points() -> list[str]:
    files = sorted(ROOT.glob("experiments/*/*.py")) + sorted(ROOT.glob("tests/*_reference.py"))
    return [str(p.relative_to(ROOT)) for p in files]


CHILD = r"""
import importlib, importlib.util, json, pkgutil, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
outside_root = root.parent
entries = json.loads(sys.argv[2])
sys.path[:0] = [str(root / "src"), str(root / "scripts"), str(root / "tests")]
sys.dont_write_bytecode = True
import private_mobility_cert as pkg
mods = [pkg.__name__]
for info in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
    mods.append(info.name)
errors = {}
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as exc:
        errors[m] = repr(exc)
for i, rel in enumerate(entries):
    name = "_entry_%d" % i
    spec = importlib.util.spec_from_file_location(name, root / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        errors[rel] = repr(exc)
project_dirs = [root / d for d in ("src", "scripts", "tests", "experiments")]
prefixes = {sys.prefix, sys.base_prefix, sys.exec_prefix}
leaks, inside = [], 0
for name, mod in list(sys.modules.items()):
    f = getattr(mod, "__file__", None)
    if not f:
        continue
    p = Path(f).resolve()
    if any(p.is_relative_to(d) for d in project_dirs):
        inside += 1
    elif p.is_relative_to(outside_root) and not p.is_relative_to(root) and not any(
        p.is_relative_to(Path(prefix).resolve()) for prefix in prefixes
    ):
        leaks.append([name, str(p.relative_to(outside_root))])
print(json.dumps({"n_package_modules": len(mods), "n_entry_points": len(entries),
                  "n_project_modules_inside_repo": inside, "import_errors": errors,
                  "modules_from_outside_repo": leaks,
                  "package_file": str(Path(pkg.__file__).resolve().relative_to(root))}))
"""


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            [sys.executable, "-I", "-B", "-c", CHILD, str(ROOT), json.dumps(entry_points())],
            cwd=tmp,
            capture_output=True,
            text=True,
        )
    if proc.returncode != 0:
        print(proc.stdout, proc.stderr, file=sys.stderr)
        return 1
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    print(json.dumps(res, indent=2))
    if res["import_errors"] or res["modules_from_outside_repo"]:
        print("FAIL: clean-room import check", file=sys.stderr)
        return 1
    print("OK: every project module resolves inside the repository")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
