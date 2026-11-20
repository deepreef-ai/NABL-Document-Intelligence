#!/usr/bin/env python3
"""Detect — and undo — files vanishing from the working tree.

Why this exists
---------------
MEASURED, twice in one day. First `App.css` came back with every `rg-*` rule
gone and `ReviewPage.tsx` reverted to a version predating the review form.
Then, far worse, `app/graph/` lost 26 of its 31 source files: the whole
extraction engine except five modules, plus a router, three imports and five
test modules. The application would not import at all.

`apply_integration.py` reported "integration intact" throughout. It checks that
certain EDITS are present INSIDE files, so a file that no longer exists has no
edit to be missing — the safety net had a hole exactly the shape of the
disaster. This closes it.

What it detects
---------------
For every tracked file, comparing a recorded manifest against the disk:

    expected file exists    the manifest says it should be here
    actual file exists      it is here
    file added              on disk, not in the manifest
    file deleted            in the manifest, not on disk
    file renamed            gone from its path, but its exact content is
                            present under a new one
    file size changed       same path, different length
    file hash changed       same path, same length, different content

Renames are reported as renames rather than as a delete plus an add, because
the two call for opposite responses: a delete wants restoring, a rename wants
recording.

Why it stores content
---------------------
Detection alone would have told us the engine was gone and left us with
nothing to put back. Every tracked file's bytes are kept in a
content-addressed store, so `--restore` returns a deleted or altered file
exactly as it was. The store is keyed by hash, so a file that never changes
costs one copy however many times it is snapshotted.

    python scripts/file_manifest.py --snapshot   # record the tree as correct
    python scripts/file_manifest.py --check      # report drift, exit 1 if any
    python scripts/file_manifest.py --restore    # put deleted/changed files back
    python scripts/file_manifest.py --restore --deleted-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT = os.path.abspath(os.path.join(ROOT, ".."))
STORE = os.path.join(ROOT, "scripts", "integrity_store")
MANIFEST = os.path.join(STORE, "manifest.json")

#: Directories worth protecting, relative to the project root. Everything the
#: team writes; nothing it merely installs.
TRACKED_DIRS = (
    os.path.join("backend", "app"),
    os.path.join("backend", "tests"),
    os.path.join("backend", "scripts"),
    os.path.join("frontend", "src"),
)

#: Only source. A .pyc is a build artefact — and after the graph package was
#: lost, bytecode was the ONLY surviving copy, which is precisely the state
#: this tool exists to make impossible.
TRACKED_SUFFIXES = (".py", ".ts", ".tsx", ".css", ".json", ".md", ".mmd")

EXCLUDED_PARTS = frozenset({"__pycache__", "node_modules", ".venv", "integrity_store", ".git"})


def _is_tracked(path: str) -> bool:
    if not path.endswith(TRACKED_SUFFIXES):
        return False
    return not (EXCLUDED_PARTS & set(path.replace("\\", "/").split("/")))


def _walk() -> list[str]:
    """Every tracked file, as a path relative to the project root."""
    out: list[str] = []
    for rel_dir in TRACKED_DIRS:
        base = os.path.join(PROJECT, rel_dir)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDED_PARTS]
            for name in filenames:
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, PROJECT).replace("\\", "/")
                if _is_tracked(rel):
                    out.append(rel)
    return sorted(out)


def _digest(path: str) -> tuple[str, int]:
    """(sha256, size) read in blocks, so a large file does not sit in memory."""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 16):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


@dataclass
class Drift:
    """Everything that differs between the manifest and the disk."""

    added: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    renamed: list[tuple[str, str]] = field(default_factory=list)   # (was, now)
    size_changed: list[tuple[str, int, int]] = field(default_factory=list)
    hash_changed: list[str] = field(default_factory=list)

    def any(self) -> bool:
        return bool(self.added or self.deleted or self.renamed
                    or self.size_changed or self.hash_changed)

    @property
    def total(self) -> int:
        return (len(self.added) + len(self.deleted) + len(self.renamed)
                + len(self.size_changed) + len(self.hash_changed))


def load_manifest() -> dict[str, dict]:
    if not os.path.exists(MANIFEST):
        return {}
    with open(MANIFEST, encoding="utf-8") as fh:
        return json.load(fh).get("files", {})


def compare(manifest: dict[str, dict]) -> Drift:
    """Manifest versus disk, classified.

    Order matters: renames are resolved BEFORE deletions and additions are
    reported, so one moved file does not show up as two unrelated problems.
    """
    drift = Drift()
    on_disk = _walk()
    disk_info: dict[str, tuple[str, int]] = {}
    for rel in on_disk:
        try:
            disk_info[rel] = _digest(os.path.join(PROJECT, rel))
        except OSError as exc:
            print(f"  !! cannot read {rel}: {exc}", file=sys.stderr)

    missing = [rel for rel in manifest if rel not in disk_info]
    extra = [rel for rel in disk_info if rel not in manifest]

    # A missing path whose exact content turned up elsewhere was moved, not
    # lost. Match on hash: same bytes under a new name.
    unmatched_extra = list(extra)
    for gone in list(missing):
        want = manifest[gone].get("sha256")
        for candidate in list(unmatched_extra):
            if disk_info[candidate][0] == want:
                drift.renamed.append((gone, candidate))
                missing.remove(gone)
                unmatched_extra.remove(candidate)
                break

    drift.deleted = sorted(missing)
    drift.added = sorted(unmatched_extra)

    for rel, recorded in manifest.items():
        if rel not in disk_info:
            continue
        sha, size = disk_info[rel]
        if size != recorded.get("size"):
            drift.size_changed.append((rel, recorded.get("size", -1), size))
        elif sha != recorded.get("sha256"):
            # Same length, different bytes: a real edit, not a truncation.
            drift.hash_changed.append(rel)
    drift.size_changed.sort()
    drift.hash_changed.sort()
    return drift


def snapshot() -> int:
    """Record the tree as correct, and keep a copy of every file's bytes."""
    os.makedirs(STORE, exist_ok=True)
    files: dict[str, dict] = {}
    stored = 0
    for rel in _walk():
        full = os.path.join(PROJECT, rel)
        sha, size = _digest(full)
        files[rel] = {"sha256": sha, "size": size}
        blob = os.path.join(STORE, "blobs", sha[:2], sha)
        if not os.path.exists(blob):
            os.makedirs(os.path.dirname(blob), exist_ok=True)
            shutil.copy2(full, blob)
            stored += 1

    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"version": 1, "files": files}, fh, indent=2, sort_keys=True)
    print(f"recorded {len(files)} file(s); {stored} new blob(s) stored")
    print(f"manifest: {os.path.relpath(MANIFEST, PROJECT)}")
    return 0


def report(drift: Drift, manifest: dict[str, dict]) -> None:
    print(f"tracked in manifest : {len(manifest)}")
    print(f"present on disk     : {len(_walk())}")
    if not drift.any():
        print("\nno drift — every expected file exists, unchanged")
        return

    print(f"\n{drift.total} difference(s):\n")
    if drift.deleted:
        print(f"  FILE DELETED ({len(drift.deleted)}) — expected, but not on disk:")
        for rel in drift.deleted:
            print(f"     {rel}")
    if drift.renamed:
        print(f"  FILE RENAMED ({len(drift.renamed)}) — same content, new path:")
        for was, now in drift.renamed:
            print(f"     {was}  ->  {now}")
    if drift.hash_changed:
        print(f"  FILE HASH CHANGED ({len(drift.hash_changed)}) — same size, different content:")
        for rel in drift.hash_changed:
            print(f"     {rel}")
    if drift.size_changed:
        print(f"  FILE SIZE CHANGED ({len(drift.size_changed)}):")
        for rel, was, now in drift.size_changed:
            print(f"     {rel}  {was} -> {now} bytes")
    if drift.added:
        print(f"  FILE ADDED ({len(drift.added)}) — on disk, not in the manifest:")
        for rel in drift.added:
            print(f"     {rel}")


def restore(drift: Drift, manifest: dict[str, dict], deleted_only: bool) -> int:
    """Put files back from the content store.

    Deleted files are always restorable. Changed ones are restored only when
    asked for without `--deleted-only`, because a change is usually deliberate
    work and overwriting it would be the same kind of loss this tool exists to
    prevent.
    """
    targets = list(drift.deleted)
    if not deleted_only:
        targets += drift.hash_changed + [rel for rel, _, _ in drift.size_changed]

    if not targets:
        print("nothing to restore")
        return 0

    done, failed = 0, 0
    for rel in targets:
        sha = manifest[rel]["sha256"]
        blob = os.path.join(STORE, "blobs", sha[:2], sha)
        if not os.path.exists(blob):
            print(f"  !! no stored copy for {rel} ({sha[:12]})", file=sys.stderr)
            failed += 1
            continue
        dest = os.path.join(PROJECT, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(blob, dest)
        print(f"  restored {rel}")
        done += 1

    print(f"\nrestored {done} file(s)" + (f", {failed} had no stored copy" if failed else ""))
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--snapshot", action="store_true",
                      help="record the current tree as correct and store every file's bytes")
    mode.add_argument("--check", action="store_true",
                      help="report drift against the manifest; exit 1 if any (the default)")
    mode.add_argument("--restore", action="store_true",
                      help="put deleted (and, without --deleted-only, changed) files back")
    ap.add_argument("--deleted-only", action="store_true",
                    help="with --restore: replace only files that are gone, never edited ones")
    a = ap.parse_args()

    if a.snapshot:
        return snapshot()

    manifest = load_manifest()
    if not manifest:
        print("no manifest yet — run with --snapshot first", file=sys.stderr)
        return 2

    drift = compare(manifest)
    if a.restore:
        return restore(drift, manifest, a.deleted_only)

    report(drift, manifest)
    return 1 if drift.any() else 0


if __name__ == "__main__":
    raise SystemExit(main())
