"""Tamper lock for the pipeline test suite.

A fix must change the CODE until the tests pass — never the tests until they
pass. This script makes that rule mechanical:

    python scripts/test_lock.py --verify     # run before every suite run (run_suite.py does)
    python scripts/test_lock.py --add FILE --reason "..." --approver NAME
                                             # lock a NEW test file (existing locks untouched)
    python scripts/test_lock.py --relock FILE --reason "..." --approver NAME
                                             # deliberately change a locked file (reviewed)

--verify fails if any locked file is missing or its SHA-256 differs. New,
unlocked files under tests/suite are reported (allowed: adding tests is fine;
editing them is not). Every --add/--relock is appended to the lock's history
with who/why/when, so the PR diff of SUITE.lock shows exactly what changed and
who approved it. CODEOWNERS makes that diff need a reviewer's approval.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
SUITE = BACKEND / "tests" / "suite"
LOCK = SUITE / "SUITE.lock"
# The guard protects itself and the runner, too.
ALWAYS = ["scripts/test_lock.py", "scripts/run_suite.py"]


def _sha(path: Path) -> str:
    # Normalise line endings so a git autocrlf checkout does not look tampered.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _rel(path: Path) -> str:
    return path.resolve().relative_to(BACKEND).as_posix()


def _load() -> dict:
    if not LOCK.is_file():
        return {"files": {}, "history": []}
    return json.loads(LOCK.read_text(encoding="utf-8"))


def _save(lock: dict) -> None:
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def digest(lock: dict) -> str:
    blob = json.dumps(lock["files"], sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def verify(quiet: bool = False) -> int:
    lock = _load()
    if not lock["files"]:
        print("TEST LOCK: no lock file — run with --init first", file=sys.stderr)
        return 2
    bad, missing = [], []
    for rel, want in lock["files"].items():
        p = BACKEND / rel
        if not p.is_file():
            missing.append(rel)
        elif _sha(p) != want:
            bad.append(rel)
    unlocked = sorted(
        _rel(p) for p in SUITE.rglob("*.py")
        if _rel(p) not in lock["files"] and "__pycache__" not in p.parts
    )
    if bad or missing:
        print("TEST FILES TAMPERED — refusing to run the suite.", file=sys.stderr)
        for r in bad:
            print(f"  modified: {r}", file=sys.stderr)
        for r in missing:
            print(f"  deleted:  {r}", file=sys.stderr)
        print("Fix the code, not the tests. A deliberate, reviewed test change uses "
              "--relock with --reason and --approver.", file=sys.stderr)
        return 1
    if not quiet:
        print(f"TEST LOCK OK: {len(lock['files'])} locked files, digest {digest(lock)}")
        for r in unlocked:
            print(f"  note: new unlocked test file {r} (lock it with --add once final)")
    return 0


def _record(lock: dict, action: str, files: list[str], reason: str, approver: str) -> None:
    if not reason or not approver:
        sys.exit("--reason and --approver are required")
    for rel in files:
        lock["files"][rel] = _sha(BACKEND / rel)
    lock["history"].append({
        "when": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "action": action, "files": files, "reason": reason, "approver": approver,
    })
    _save(lock)
    print(f"{action}: {', '.join(files)} -> digest {digest(lock)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--verify", action="store_true")
    g.add_argument("--init", action="store_true", help="first lock of every suite file")
    g.add_argument("--add", nargs="+", metavar="FILE")
    g.add_argument("--relock", nargs="+", metavar="FILE")
    ap.add_argument("--reason", default="")
    ap.add_argument("--approver", default="")
    a = ap.parse_args()

    if a.verify:
        return verify()
    lock = _load()
    if a.init:
        if lock["files"]:
            sys.exit("already locked; use --add for new files or --relock for reviewed changes")
        files = sorted({_rel(p) for p in SUITE.rglob("*.py") if "__pycache__" not in p.parts}
                       | set(ALWAYS))
        _record(lock, "init", files, a.reason, a.approver)
        return 0
    targets = [_rel(BACKEND / f if not Path(f).is_absolute() else Path(f)) for f in (a.add or a.relock)]
    if a.add:
        already = [t for t in targets if t in lock["files"]]
        if already:
            sys.exit(f"already locked (use --relock, reviewed): {already}")
        _record(lock, "add", targets, a.reason, a.approver)
    else:
        unknown = [t for t in targets if t not in lock["files"]]
        if unknown:
            sys.exit(f"not locked yet (use --add): {unknown}")
        _record(lock, "relock", targets, a.reason, a.approver)
    return 0


if __name__ == "__main__":
    sys.exit(main())
