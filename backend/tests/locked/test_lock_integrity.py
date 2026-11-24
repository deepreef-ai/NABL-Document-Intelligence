"""LOCKED: the locked suite itself cannot drift silently."""
from __future__ import annotations

import importlib.util
import os

_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "lock_tests.py")


def test_locked_files_match_suite_lock():
    """Why this must never break: these tests are the product's definition of
    correct. Weakening one to make a regression pass must be a visible, reviewed
    decision (scripts/lock_tests.py --update --reason ...), never a quiet edit."""
    spec = importlib.util.spec_from_file_location("lock_tests", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ok, message = mod.check()
    assert ok, message
