"""Orchestrates Step 5: for every normalized document (all of them — Step 3
sampled for cheap schema discovery, but every document needs its own label),
classify domain + extract fields/tests in one LLM call, guided by Step 4's
master schema, and write one label JSON per document plus a dataset index.

Resumable and per-document failure-isolated, same conventions as
dataset_normalization/pipeline.py and schema_discovery/pipeline.py: an
existing labels/<document_id>.json is reused unless --force, and one failed
LLM call is recorded as extraction_status="failed" (never silently dropped)
rather than aborting the whole batch.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.labeling.extraction import extract_label
from app.labeling.models import DocumentLabel, LabelIndexEntry
from app.labeling.schema_hints import build_domain_hints, load_master_schema
from app.schema_discovery.sampling import load_normalized_documents


@dataclass
class RunStats:
    total_documents: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)


def _label_path(output_dir: Path, document_id: str) -> Path:
    return output_dir / "labels" / f"{document_id}.json"


def _document_ocr_info(doc_json: dict) -> tuple[bool, float | None]:
    pages = doc_json.get("pages", [])
    ocr_used = any(page.get("ocr_used") for page in pages)
    confidences = [page["ocr_confidence"] for page in pages if page.get("ocr_confidence") is not None]
    avg_confidence = round(sum(confidences) / len(confidences), 4) if confidences else None
    return ocr_used, avg_confidence


def run(normalized_dir: Path, master_schema_dir: Path, output_dir: Path, force: bool = False) -> RunStats:
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    hints = build_domain_hints(load_master_schema(master_schema_dir))
    records = load_normalized_documents(normalized_dir)
    stats = RunStats(total_documents=len(records))

    index_entries: list[LabelIndexEntry] = []

    for record in records:
        label_path = _label_path(output_dir, record.document_id)

        if label_path.exists() and not force:
            cached = json.loads(label_path.read_text(encoding="utf-8"))
            stats.skipped += 1
            index_entries.append(LabelIndexEntry(
                document_id=record.document_id, original_filename=record.original_filename,
                domain=cached.get("domain", "other"),
                annotation_status=cached.get("annotation_status", "pending"),
                extraction_status=cached.get("extraction_status", "ok"),
                field_count=len(cached.get("fields", {})), test_count=len(cached.get("tests", [])),
                error=cached.get("error"),
            ))
            continue

        # page_count/OCR provenance come from the ORIGINAL normalized
        # document.json — sampling.py's NormalizedDocRecord only carries the
        # concatenated text, not per-page metadata.
        doc_json_path = normalized_dir / "normalized" / record.document_id / "document.json"
        doc_json = json.loads(doc_json_path.read_text(encoding="utf-8"))
        page_count = doc_json.get("page_count", len(doc_json.get("pages", [])))
        ocr_used, ocr_confidence = _document_ocr_info(doc_json)

        try:
            label = extract_label(
                document_id=record.document_id, original_filename=record.original_filename,
                text=record.text, page_count=page_count, source_ocr_used=ocr_used,
                source_ocr_confidence=ocr_confidence, domain_hints=hints,
            )
        except Exception as exc:  # noqa: BLE001 — one bad LLM call must never stop the batch
            stats.failed += 1
            stats.failures.append((record.document_id, f"{type(exc).__name__}: {exc}"))
            label = DocumentLabel(
                document_id=record.document_id, original_filename=record.original_filename,
                domain="other", page_count=page_count, source_ocr_used=ocr_used,
                source_ocr_confidence=ocr_confidence, fields={}, tests=[],
                annotation_status="pending", extraction_status="failed", error=str(exc),
            )

        label_path.write_text(json.dumps(label.to_json_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        if label.extraction_status == "ok":
            stats.processed += 1
        index_entries.append(LabelIndexEntry(
            document_id=label.document_id, original_filename=label.original_filename, domain=label.domain,
            annotation_status=label.annotation_status, extraction_status=label.extraction_status,
            field_count=len(label.fields), test_count=len(label.tests), error=label.error,
        ))

    with (output_dir / "label_index.jsonl").open("w", encoding="utf-8") as index_file:
        for entry in index_entries:
            index_file.write(json.dumps(entry.to_json_dict(), ensure_ascii=False) + "\n")

    return stats


def print_summary(stats: RunStats) -> None:
    print()
    print("=== Labeling Summary ===")
    print(f"Total normalized documents: {stats.total_documents}")
    print(f"Processed (new LLM calls):  {stats.processed}")
    print(f"Skipped (already cached):   {stats.skipped}")
    print(f"Failed:                     {stats.failed}")
    if stats.failures:
        print()
        print("--- Failures ---")
        for document_id, reason in stats.failures:
            print(f"  {document_id}: {reason}")
