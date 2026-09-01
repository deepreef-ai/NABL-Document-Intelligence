"""Orchestrates the final-dataset validation stage: loads the compiled
final_labeled_dataset.jsonl, validates every record, and writes
final_dataset_qc.json. Read-only end to end — no label, no dataset file,
and no annotation_status is ever modified here.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.final_dataset_qc.models import FinalDatasetQCReport
from app.final_dataset_qc.validate import (
    load_id_registry,
    load_jsonl_records,
    load_master_schema_keys,
    load_normalized_documents,
    validate_record,
)


def run(final_dataset_dir: Path, normalized_dir: Path, master_schema_dir: Path) -> FinalDatasetQCReport:
    jsonl_path = final_dataset_dir / "final_labeled_dataset.jsonl"
    records, parse_errors = load_jsonl_records(jsonl_path)

    schema_keys = load_master_schema_keys(master_schema_dir)
    doc_jsons = load_normalized_documents(normalized_dir)
    id_registry = load_id_registry(normalized_dir)

    report = FinalDatasetQCReport(total_documents=len(records))
    for entry in parse_errors:
        report.invalid_documents += 1
        report.errors.append({
            "document_id": None, "check": "jsonl_valid", "severity": "hard",
            "message": f"line {entry['line']}: {entry['message']}",
        })

    id_counts: dict[str, int] = {}
    for record in records:
        doc_id = record.get("document_id")
        id_counts[doc_id] = id_counts.get(doc_id, 0) + 1
    duplicate_ids = {doc_id for doc_id, count in id_counts.items() if count > 1}

    for record in records:
        document_id = record.get("document_id")
        is_valid, issues = validate_record(record, schema_keys, doc_jsons, id_registry)

        if document_id in duplicate_ids:
            is_valid = False
            report.duplicate_documents += 1
            issues.append({
                "document_id": document_id, "check": "unique_document_id", "severity": "hard",
                "message": f"document_id {document_id!r} appears {id_counts[document_id]} times in the dataset",
            })

        report.errors.extend(issues)
        if record.get("annotation_status") != "approved":
            report.invalid_annotations += 1
        report.unexpected_keys += sum(1 for i in issues if i["check"] == "no_unexpected_keys")
        report.missing_values += sum(1 for i in issues if i["check"] == "missing_values_are_null")

        if is_valid:
            report.valid_documents += 1
        else:
            report.invalid_documents += 1

    (final_dataset_dir / "final_dataset_qc.json").write_text(
        json.dumps(report.to_json_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def print_summary(report: FinalDatasetQCReport) -> None:
    print()
    print("=== Final Dataset Validation Summary ===")
    print(f"Total documents:      {report.total_documents}")
    print(f"Valid documents:      {report.valid_documents}")
    print(f"Invalid documents:    {report.invalid_documents}")
    print(f"Duplicate documents:  {report.duplicate_documents}")
    print(f"Invalid annotations:  {report.invalid_annotations}")
    print(f"Unexpected keys:      {report.unexpected_keys}")
    print(f"Missing values:       {report.missing_values}")
    hard_errors = [e for e in report.errors if e["severity"] == "hard"]
    if hard_errors:
        print()
        print(f"--- {len(hard_errors)} hard error(s) ---")
        for entry in hard_errors[:50]:
            print(f"  {entry['document_id']} [{entry['check']}]: {entry['message']}")
        if len(hard_errors) > 50:
            print(f"  ... and {len(hard_errors) - 50} more (see final_dataset_qc.json)")
