"""One command to run the pipeline test suite and get a PR-ready summary.

    python scripts/run_suite.py            # fast: unit + flow (+ gateway unit), ~1 min
    python scripts/run_suite.py --full     # + real OCR round trips and the real dataset
    python scripts/run_suite.py --only test_gateway.py --full

Always verifies the test lock first: if any locked test file was edited, the
suite does not run and the exit code is non-zero.

Output: live step logs in the terminal, then
    backend/test-reports/suite_summary.md   (paste into the PR)
    backend/test-reports/suite_results.json
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="include real OCR + dataset tests")
    ap.add_argument("--only", nargs="*", default=None, help="test file names under tests/suite")
    ap.add_argument("--no-lock", action="store_true", help=argparse.SUPPRESS)
    a, extra = ap.parse_known_args()

    if not a.no_lock:
        rc = subprocess.call([sys.executable, str(BACKEND / "scripts" / "test_lock.py"), "--verify"])
        if rc:
            return rc

    targets = ([f"tests/suite/{f}" for f in a.only] if a.only else ["tests/suite"])
    cmd = [sys.executable, "-m", "pytest", *targets, "-p", "no:cacheprovider",
           "-o", "console_output_style=classic", "-rfxXs", "--tb=short", "-v", "-s"]
    if not a.full:
        cmd += ["-m", "not realocr and not dataset"]
    cmd += extra

    # UTF-8 so the summary (µ, ², em dashes in gap reasons) prints on a Windows console.
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    print("$ " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=BACKEND, env=env)
    summary = BACKEND / "test-reports" / "suite_summary.md"
    if summary.is_file():
        print(f"\nPR summary written to {summary}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
