"""The integrity net that would have caught the graph package disappearing.

MEASURED: `app/graph/` lost 26 of its 31 source files and the application
stopped importing, while `apply_integration.py --check` reported "integration
intact" throughout. That tool verifies EDITS INSIDE files, so a file that no
longer exists has no edit to be missing — the hole in the net was exactly the
shape of the failure.

These tests pin the seven conditions the manifest has to distinguish, and the
one that matters most: a deleted file must come BACK, not merely be reported.
"""
import json
import os
import shutil

import pytest

from scripts import file_manifest as fm


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A miniature project: two tracked dirs, a store, no real repo touched."""
    project = tmp_path / "proj"
    (project / "backend" / "app").mkdir(parents=True)
    (project / "frontend" / "src").mkdir(parents=True)
    (project / "backend" / "app" / "engine.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    (project / "backend" / "app" / "helper.py").write_text("X = 2\n", encoding="utf-8")
    (project / "frontend" / "src" / "App.css").write_text(".a { color: red; }\n", encoding="utf-8")
    # Never tracked: a build artefact, and the bytecode that was the only
    # survivor last time.
    (project / "backend" / "app" / "__pycache__").mkdir()
    (project / "backend" / "app" / "__pycache__" / "engine.cpython-312.pyc").write_bytes(b"\x00bytecode")

    store = tmp_path / "store"
    monkeypatch.setattr(fm, "PROJECT", str(project))
    monkeypatch.setattr(fm, "STORE", str(store))
    monkeypatch.setattr(fm, "MANIFEST", str(store / "manifest.json"))
    monkeypatch.setattr(fm, "TRACKED_DIRS", (os.path.join("backend", "app"),
                                             os.path.join("frontend", "src")))
    return project


def snapshot_then(tree, mutate):
    fm.snapshot()
    mutate(tree)
    manifest = fm.load_manifest()
    return fm.compare(manifest), manifest


class TestTheSevenConditions:
    def test_expected_file_exists_and_actual_file_exists(self, tree):
        drift, manifest = snapshot_then(tree, lambda _: None)
        assert not drift.any()
        assert set(manifest) == {
            "backend/app/engine.py", "backend/app/helper.py", "frontend/src/App.css",
        }

    def test_file_deleted(self, tree):
        drift, _ = snapshot_then(tree, lambda p: (p / "backend" / "app" / "engine.py").unlink())
        assert drift.deleted == ["backend/app/engine.py"]
        assert drift.added == []          # not also reported as something else

    def test_file_added(self, tree):
        drift, _ = snapshot_then(
            tree, lambda p: (p / "backend" / "app" / "new.py").write_text("Y = 1\n", encoding="utf-8"))
        assert drift.added == ["backend/app/new.py"]
        assert drift.deleted == []

    def test_file_renamed(self, tree):
        # A move is ONE event. Reported as a delete plus an add it invites
        # restoring the old path and leaving the tree with both.
        def rename(p):
            src = p / "backend" / "app" / "engine.py"
            src.rename(p / "backend" / "app" / "core.py")

        drift, _ = snapshot_then(tree, rename)
        assert drift.renamed == [("backend/app/engine.py", "backend/app/core.py")]
        assert drift.deleted == []
        assert drift.added == []

    def test_file_size_changed(self, tree):
        drift, _ = snapshot_then(
            tree, lambda p: (p / "backend" / "app" / "helper.py").write_text("X = 22222\n", encoding="utf-8"))
        assert [r for r, _, _ in drift.size_changed] == ["backend/app/helper.py"]
        assert drift.hash_changed == []   # size already explains it

    def test_file_hash_changed(self, tree):
        # Same length, different bytes — the quiet one. A revert that swaps a
        # value for another of equal length changes no size at all.
        drift, _ = snapshot_then(
            tree, lambda p: (p / "backend" / "app" / "helper.py").write_text("X = 3\n", encoding="utf-8"))
        assert drift.hash_changed == ["backend/app/helper.py"]
        assert drift.size_changed == []


class TestRestore:
    """Detection without recovery would have told us the engine was gone and
    left us nothing to put back."""

    def test_a_deleted_file_comes_back_byte_for_byte(self, tree):
        original = (tree / "backend" / "app" / "engine.py").read_bytes()
        drift, manifest = snapshot_then(tree, lambda p: (p / "backend" / "app" / "engine.py").unlink())
        fm.restore(drift, manifest, deleted_only=True)
        assert (tree / "backend" / "app" / "engine.py").read_bytes() == original

    def test_a_whole_package_comes_back(self, tree):
        # The real failure: most of a package gone at once.
        def wipe(p):
            for name in ("engine.py", "helper.py"):
                (p / "backend" / "app" / name).unlink()

        drift, manifest = snapshot_then(tree, wipe)
        assert len(drift.deleted) == 2
        fm.restore(drift, manifest, deleted_only=True)
        assert fm.compare(manifest).any() is False

    def test_deleted_only_leaves_edited_files_alone(self, tree):
        # An edit is usually deliberate work. Overwriting it would be the same
        # kind of loss this tool exists to prevent.
        def both(p):
            (p / "backend" / "app" / "engine.py").unlink()
            (p / "backend" / "app" / "helper.py").write_text("X = 999\n", encoding="utf-8")

        drift, manifest = snapshot_then(tree, both)
        fm.restore(drift, manifest, deleted_only=True)
        assert (tree / "backend" / "app" / "engine.py").exists()
        assert (tree / "backend" / "app" / "helper.py").read_text(encoding="utf-8") == "X = 999\n"

    def test_restoring_everything_also_reverts_an_edit(self, tree):
        drift, manifest = snapshot_then(
            tree, lambda p: (p / "backend" / "app" / "helper.py").write_text("X = 999\n", encoding="utf-8"))
        fm.restore(drift, manifest, deleted_only=False)
        assert (tree / "backend" / "app" / "helper.py").read_text(encoding="utf-8") == "X = 2\n"


class TestWhatIsTracked:
    def test_bytecode_is_never_tracked(self, tree):
        # After the graph package was lost, .pyc was the only surviving copy.
        # Storing build output would let that pass for a healthy tree.
        assert all(".pyc" not in rel for rel in fm.load_manifest() or fm._walk())

    def test_pycache_is_skipped_entirely(self, tree):
        assert not any("__pycache__" in rel for rel in fm._walk())

    def test_dependencies_are_not_tracked(self, tree):
        (tree / "backend" / "app" / "node_modules").mkdir()
        (tree / "backend" / "app" / "node_modules" / "dep.ts").write_text("x", encoding="utf-8")
        assert not any("node_modules" in rel for rel in fm._walk())


class TestManifestFile:
    def test_it_records_a_hash_and_a_size_per_file(self, tree):
        fm.snapshot()
        with open(fm.MANIFEST, encoding="utf-8") as fh:
            data = json.load(fh)
        entry = data["files"]["backend/app/engine.py"]
        assert len(entry["sha256"]) == 64
        # Measured from the file, not from the string that was written:
        # write_text translates newlines on Windows, so the bytes on disk
        # are not the bytes handed to it.
        assert entry["size"] == (tree / "backend" / "app" / "engine.py").stat().st_size

    def test_identical_content_is_stored_once(self, tree):
        (tree / "frontend" / "src" / "copy.ts").write_text("X = 2\n", encoding="utf-8")  # same as helper.py
        fm.snapshot()
        blobs = [f for _, _, fs in os.walk(os.path.join(fm.STORE, "blobs")) for f in fs]
        assert len(blobs) == len(set(blobs)) == 3   # engine, helper/copy (shared), App.css
