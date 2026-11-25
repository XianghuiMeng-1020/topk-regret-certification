"""The repository must be import-closed.

Every import in every Python file must resolve to the standard library, a
package pinned in the lock files, or a file inside the repository. A
clean-room child interpreter must load the package from src/.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_DIRS = ["src", "scripts", "tests", "experiments"]
THIRD_PARTY = {"numpy", "scipy", "pandas", "pyarrow", "yaml", "pytest", "dateutil"}
SKIP_PARTS = {".venv", ".venv-synthetic", "__pycache__"}


def repo_py_files() -> list[Path]:
    out = []
    for d in CODE_DIRS:
        out += [p for p in (ROOT / d).rglob("*.py") if not SKIP_PARTS & set(p.parts)]
    return sorted(out)


def module_exists(name: str) -> bool:
    base = ROOT / "src" / Path(*name.split("."))
    return base.with_suffix(".py").exists() or (base / "__init__.py").exists()


def local_names(path: Path) -> set[str]:
    names = {p.stem for p in path.parent.glob("*.py")}
    names |= {p.stem for p in (ROOT / "scripts").glob("*.py")}
    names |= {p.stem for p in (ROOT / "tests").glob("*.py")}
    names |= {p.stem for p in (ROOT / "experiments" / "synthetic_box_calibration").glob("*.py")}
    return names


def test_every_import_resolves_inside_repo_or_to_pinned_packages():
    problems = []
    for path in repo_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        local = local_names(path)
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module]
                mods += [f"{node.module}.{a.name}" for a in node.names if node.module.startswith("private_mobility_cert")]
            for mod in mods:
                top = mod.split(".")[0]
                if top in sys.stdlib_module_names or top in THIRD_PARTY or top == "__future__":
                    continue
                if top == "private_mobility_cert":
                    parent = mod.rsplit(".", 1)[0]
                    if not (module_exists(mod) or module_exists(parent)):
                        problems.append(f"{path.relative_to(ROOT)}: {mod}")
                    continue
                if top in local:
                    continue
                problems.append(f"{path.relative_to(ROOT)}: unresolved import {mod}")
    assert problems == []


def test_file_loaders_point_inside_repo():
    basenames = {p.name for p in repo_py_files()}
    missing = []
    for path in repo_py_files():
        rel = str(path.relative_to(ROOT))
        text = path.read_text(encoding="utf-8")
        if "spec_from_file_location" not in text and ".read_text(" not in text and "sys.path.insert" not in text:
            continue
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.endswith(".py") and len(node.value) > 3 and "*" not in node.value:
                if Path(node.value).name not in basenames:
                    missing.append(f"{rel} -> {node.value}")
    assert missing == []


def test_no_absolute_user_paths_in_code():
    bad = []
    rx = re.compile("(" + "/" + "Users/|/home/[a-z]|[A-Z]:" + r"\\\\" + "Users)")
    for path in repo_py_files():
        if rx.search(path.read_text(encoding="utf-8")):
            bad.append(str(path.relative_to(ROOT)))
    assert bad == []


def test_no_symlink_escapes_repo():
    escapes = []
    for p in ROOT.rglob("*"):
        if SKIP_PARTS & set(p.relative_to(ROOT).parts) or ".git" in p.relative_to(ROOT).parts:
            continue
        if p.is_symlink() and not p.resolve().is_relative_to(ROOT.resolve()):
            escapes.append(str(p.relative_to(ROOT)))
    assert escapes == []


def test_clean_room_import_loads_package_from_src():
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/verify_imports.py")], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = proc.stdout[: proc.stdout.rindex("}") + 1]
    res = json.loads(payload)
    assert res["modules_from_outside_repo"] == []
    assert res["package_file"] == "src/private_mobility_cert/__init__.py"
