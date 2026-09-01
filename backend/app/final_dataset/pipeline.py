"""Orchestrates Step 7: assembles the FINAL labeled dataset from ONLY
approved annotations (Step 6's gate), using the master-schema canonical
field names Step 5 already wrote — this stage does no re-extraction, no
re-classification beyond a deterministic document_type derivation, and
never invents a value.

Produces ONE combined dataset across every domain (no train/validation/test
split, per the task) in two forms:
  - final_labeled_dataset.jsonl — one full-fidelity JSON record per document
  - final_labeled_dataset.csv   — the same records, tabular: one row per
    document, one column per field name seen anywhere in the dataset
    (a common field like "sample_id" shares one column across every domain
    that has it; a domain-specific field is simply blank for documents that
    don't have it), and `tests` serialized as a JSON string column — CSV has
    no native way to represent a variable-length nested list without a
    second file or a lossy flattening, so the JSONL file is the full-fidelity
    source of truth and the CSV is a best-effort tabular derivative.

Every assembled record is validated (validation.py) before being written; a
record that fails is EXCLUDED from the output and logged in
dataset_summary.json's validation_errors, never written anyway.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from app.final_dataset.document_type import derive_document_type
from app.final_dataset.models import DatasetSummary, FinalDocumentRecord
from app.final_dataset.validation import validate_record


def _load_labels(labeled_dir: Path) -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted((labeled_dir / "labels").glob("*.json"))]


def _load_normalized_documents(normalized_dir: Path) -> dict[str, dict]:
    docs = {}
    for path in sorted((normalized_dir / "normalized").glob("*/document.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        docs[data.get("document_id", path.parent.name)] = data
    return docs


def build_final_dataset(labeled_dir: Path, normalized_dir: Path) -> tuple[list[FinalDocumentRecord], DatasetSummary]:
    labels = _load_labels(labeled_dir)
    doc_jsons = _load_normalized_documents(normalized_dir)

    # The one hard gate this whole stage exists to enforce: only "approved"
    # labels — not "pending", not "reviewed" — ever reach the final dataset.
    approved = [label for label in labels if label.get("annotation_status") == "approved"]

    records: list[FinalDocumentRecord] = []
    validation_errors: list[dict] = []

    for label in approved:
        document_id = label.get("document_id")
        doc_json = doc_jsons.get(document_id, {})
        original_filename = label.get("original_filename", "")
        record = FinalDocumentRecord(
            document_id=document_id,
            original_filename=original_filename,
            domain=label.get("domain"),
            document_type=derive_document_type(original_filename),
            source_format=doc_json.get("source_format") or Path(original_filename).suffix.lstrip(".").lower(),
            source_path=doc_json.get("source_path", ""),
            page_count=label.get("page_count"),
            fields=label.get("fields", {}),
            tests=label.get("tests", []),
            annotation_status=label.get("annotation_status"),
        )

        errors = validate_record(record)
        if errors:
            validation_errors.append({"document_id": document_id, "errors": errors})
            continue
        records.append(record)

    summary = DatasetSummary(
        approved_annotations=len(approved),
        excluded_by_validation=len(validation_errors),
        validation_errors=validation_errors,
    )
    for record in records:
        summary.documents_by_domain[record.domain] = summary.documents_by_domain.get(record.domain, 0) + 1
        summary.documents_by_format[record.source_format] = summary.documents_by_format.get(record.source_format, 0) + 1
        page_key = str(record.page_count)
        summary.documents_by_page_count[page_key] = summary.documents_by_page_count.get(page_key, 0) + 1
        summary.total_fields += len(record.fields)
        summary.total_test_records += len(record.tests)
        summary.missing_null_fields += sum(1 for value in record.fields.values() if value is None)
    summary.total_documents = len(records)

    return records, summary


def _all_field_names(records: list[FinalDocumentRecord]) -> list[str]:
    names: dict[str, None] = {}
    for record in records:
        for key in record.fields:
            names.setdefault(key, None)
    return sorted(names)


def write_final_dataset(records: list[FinalDocumentRecord], summary: DatasetSummary, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "final_labeled_dataset.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_json_dict(), ensure_ascii=False) + "\n")

    field_columns = _all_field_names(records)
    base_columns = [
        "document_id", "original_filename", "domain", "document_type",
        "source_format", "source_path", "page_count", "annotation_status",
    ]
    with (output_dir / "final_labeled_dataset.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(base_columns + [f"field__{name}" for name in field_columns] + ["tests_json"])
        for record in records:
            row = [
                record.document_id, record.original_filename, record.domain, record.document_type,
                record.source_format, record.source_path, record.page_count, record.annotation_status,
            ]
            row.extend(record.fields.get(name) for name in field_columns)
            row.append(json.dumps(record.tests, ensure_ascii=False))
            writer.writerow(row)

    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary.to_json_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def print_summary(summary: DatasetSummary) -> None:
    print()
    print("=== Final Dataset Summary ===")
    print(f"Total documents (written):  {summary.total_documents}")
    print(f"Approved annotations seen:  {summary.approved_annotations}")
    print(f"Excluded by validation:     {summary.excluded_by_validation}")
    print(f"Documents by domain:        {summary.documents_by_domain}")
    print(f"Documents by format:        {summary.documents_by_format}")
    print(f"Documents by page count:    {summary.documents_by_page_count}")
    print(f"Total fields:               {summary.total_fields}")
    print(f"Total test records:         {summary.total_test_records}")
    print(f"Missing/null fields:        {summary.missing_null_fields}")
    if summary.validation_errors:
        print()
        print("--- Validation exclusions ---")
        for entry in summary.validation_errors:
            print(f"  {entry['document_id']}: {entry['errors']}")
