#!/usr/bin/env python
"""Lock the LOCKED regression suite (tests/locked/).

    python scripts/lock_tests.py                       # write SUITE.lock (first time)
    python scripts/lock_tests.py --check               # exit 1 if any locked file changed
    python scripts/lock_tests.py --update --reason "why the required behaviour changed"

--update rewrites SUITE.lock and appends date, reason and changed files to
tests/locked/LOCK_HISTORY.md. It refuses to run without --reason.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import os
import sys

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCKED_DIR = os.path.join(BACKEND, "tests", "locked")
LOCK_FILE = os.path.join(LOCKED_DIR, "SUITE.lock")
HISTORY_FILE = os.path.join(LOCKED_DIR, "LOCK_HISTORY.md")
_EXCLUDED = {"SUITE.lock", "LOCK_HISTORY.md"}

HEADER = (
    "# SUITE.lock - sha256 of every file in tests/locked/.\n"
    "# The locked tests define REQUIRED behaviour of the attestation product.\n"
    "# They may only change with a deliberate, reviewed reason:\n"
    "#   python scripts/lock_tests.py --update --reason \"<why>\"\n"
    "# (the reason and changed files are recorded in tests/locked/LOCK_HISTORY.md).\n"
)


def current_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for root, dirs, files in os.walk(LOCKED_DIR):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in files:
            if name in _EXCLUDED or name.endswith((".pyc", ".pyo")):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, BACKEND).replace(os.sep, "/")
            with open(path, "rb") as fh:
                out[rel] = hashlib.sha256(fh.read()).hexdigest()
    return dict(sorted(out.items()))


def read_lock() -> dict[str, str] | None:
    if not os.path.exists(LOCK_FILE):
        return None
    out: dict[str, str] = {}
    with open(LOCK_FILE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sha, path = line.split(None, 1)
            out[path] = sha
    return out


def diff(locked: dict[str, str], now: dict[str, str]) -> dict[str, list[str]]:
    return {
        "changed": sorted(p for p in now if p in locked and locked[p] != now[p]),
        "added": sorted(p for p in now if p not in locked),
        "removed": sorted(p for p in locked if p not in now),
    }


def check() -> tuple[bool, str]:
    """(ok, message) — the logic behind --check, also used by test_lock_integrity."""
    locked = read_lock()
    if locked is None:
        return False, f"{os.path.relpath(LOCK_FILE, BACKEND)} is missing; run scripts/lock_tests.py"
    d = diff(locked, current_hashes())
    if not any(d.values()):
        return True, "locked suite unchanged"
    lines = ["Locked tests changed without a lock update:"]
    for kind, paths in d.items():
        lines += [f"  {kind}: {p}" for p in paths]
    lines.append('If the change is deliberate and reviewed: '
                 'python scripts/lock_tests.py --update --reason "<why>"')
    return False, "\n".join(lines)


def write_lock(hashes: dict[str, str]) -> None:
    with open(LOCK_FILE, "w", encoding="utf-8") as fh:
        fh.write(HEADER)
        for path, sha in hashes.items():
            fh.write(f"{sha}  {path}\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--reason", default="")
    args = ap.parse_args(argv)

    if args.check:
        ok, msg = check()
        print(msg, file=sys.stdout if ok else sys.stderr)
        return 0 if ok else 1

    locked = read_lock()
    now = current_hashes()
    if locked is not None and not args.update:
        print("SUITE.lock exists; use --check, or --update --reason \"<why>\"", file=sys.stderr)
        return 2
    if args.update and not args.reason.strip():
        print("refusing --update without --reason: say why the required behaviour changed",
              file=sys.stderr)
        return 2

    d = diff(locked or {}, now)
    write_lock(now)
    if args.update:
        new = not os.path.exists(HISTORY_FILE)
        with open(HISTORY_FILE, "a", encoding="utf-8") as fh:
            if new:
                fh.write("# Locked suite history\n\nEvery change to tests/locked/ with its reviewed reason.\n")
            fh.write(f"\n## {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            fh.write(f"Reason: {args.reason.strip()}\n\n")
            for kind, paths in d.items():
                for p in paths:
                    fh.write(f"- {kind}: {p}\n")
            if not any(d.values()):
                fh.write("- (no file changes; lock rewritten)\n")
    print(f"wrote {os.path.relpath(LOCK_FILE, BACKEND)} ({len(now)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
