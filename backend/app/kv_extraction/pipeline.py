"""Runs the numbered-table heuristic over every normalized document
produced by dataset_normalization, writing results to a SEPARATE file per
document (normalized/<document_id>/table_rows.json) rather than editing
document.json — stage 1's output stays exactly as stage 1 left it."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from app.kv_extraction.table_rows import extract_numbered_table_rows


@dataclass
class KvRunStats:
    documents_scanned: int = 0
    documents_with_rows: int = 0
    total_rows: int = 0


def run(normalized_dir: Path) -> KvRunStats:
    """`normalized_dir` is dataset_normalization's own output dir (the one
    containing normalized/<document_id>/document.json — NOT the dataset root)."""
    stats = KvRunStats()
    doc_dirs = sorted((normalized_dir / "normalized").glob("LR_*"))

    for doc_dir in doc_dirs:
        doc_json_path = doc_dir / "document.json"
        if not doc_json_path.exists():
            continue
        doc = json.loads(doc_json_path.read_text(encoding="utf-8"))
        if doc.get("status") != "processed":
            continue

        stats.documents_scanned += 1
        pages_out = []
        doc_row_count = 0
        for page in doc["pages"]:
            rows = extract_numbered_table_rows(page["text"])
            doc_row_count += len(rows)
            pages_out.append({"page_number": page["page_number"], "rows": [asdict(r) for r in rows]})

        if doc_row_count > 0:
            stats.documents_with_rows += 1
        stats.total_rows += doc_row_count

        output = {
            "document_id": doc["document_id"],
            "original_filename": doc["original_filename"],
            "pages": pages_out,
        }
        (doc_dir / "table_rows.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    return stats


def print_summary(stats: KvRunStats) -> None:
    print()
    print("=== Key-Value Extraction Summary ===")
    print(f"Documents scanned:      {stats.documents_scanned}")
    print(f"Documents with rows:    {stats.documents_with_rows}")
    print(f"Total key-value rows:   {stats.total_rows}")
