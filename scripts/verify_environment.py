"""Check the interpreter and installed packages against a lock file.

Exit 0 when every pin matches. Exit 1 on any pin mismatch or when the
interpreter is outside CPython 3.11, which numpy 1.24.4 requires.
Usage: python scripts/verify_environment.py [--lock requirements-synthetic.lock]
Writes nothing.
"""

from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_lock(name: str) -> dict[str, str]:
    pins = {}
    for line in (ROOT / name).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, version = line.split("==")
            pins[name.strip()] = version.strip()
    return pins


def main() -> int:
    lock = sys.argv[sys.argv.index("--lock") + 1] if "--lock" in sys.argv else "requirements.lock"
    pins = read_lock(lock)
    rows = []
    bad = []
    for name, want in pins.items():
        try:
            got = metadata.version(name)
        except metadata.PackageNotFoundError:
            got = None
        ok = got == want
        rows.append({"package": name, "locked": want, "installed": got, "match": ok})
        if not ok:
            bad.append(name)
    py_ok = sys.version_info[:2] == (3, 11)
    venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    out = {
        "lock_file": lock,
        "python": sys.version.split()[0],
        "python_build": sys.version,
        "python_is_3_11": py_ok,
        "virtualenv": venv,
        "executable_inside_repo": Path(sys.executable).resolve().is_relative_to(ROOT.resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": rows,
        "mismatched": bad,
    }
    print(json.dumps(out, indent=2))
    if not py_ok:
        print("FAIL: Python 3.11.x required", file=sys.stderr)
        return 1
    if bad:
        print(f"FAIL: package pins differ: {bad}", file=sys.stderr)
        return 1
    if sys.version.split()[0] != "3.11.4":
        print("NOTE: the committed outputs were produced with CPython 3.11.4; protocol.json files that embed sys.version will hash differently.")
    print(f"OK: environment matches {lock}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
