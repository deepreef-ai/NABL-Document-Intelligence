"""Recursive dataset discovery + stable document_id assignment.

document_id is content-hash-based, not path-based: the same bytes get the
same id even if the file is renamed or moved within the dataset, which is a
strictly stronger guarantee than "same path" and costs nothing extra (the
file has to be read into memory to process it either way).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def discover_files(root: Path) -> list[Path]:
    """Every supported file under `root`, recursively. Sorted for a
    deterministic processing order across runs (helps reading logs/progress,
    not required for correctness — id assignment is hash-based, not order-based)."""
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class IdRegistry:
    """Persistent content_hash -> "LR_NNNNNN" mapping, so:
      - the SAME source file (by content) gets the SAME document_id on a
        later run, even if it's been renamed/moved within the dataset;
      - a fresh run doesn't reassign ids that an earlier run already handed
        out to OTHER files, since new ids only ever count up from
        whatever's already in the registry.
    """

    def __init__(self, registry_path: Path):
        self.path = registry_path
        self._by_hash: dict[str, str] = {}
        self._next_seq = 1
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._by_hash = data.get("by_hash", {})
            self._next_seq = data.get("next_seq", 1)

    def get_or_assign(self, file_hash: str) -> str:
        existing = self._by_hash.get(file_hash)
        if existing:
            return existing
        doc_id = f"LR_{self._next_seq:06d}"
        self._by_hash[file_hash] = doc_id
        self._next_seq += 1
        self._save()
        return doc_id

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"by_hash": self._by_hash, "next_seq": self._next_seq}, indent=2),
            encoding="utf-8",
        )
