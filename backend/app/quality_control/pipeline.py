"""Orchestrates Step 6: runs every check in checks.py against every label
Step 5 produced, cross-referencing each label's source normalized document
and the dataset as a whole (for duplicate detection), then writes the
verdict back — never the field/test content itself, only annotation_status
and a `qc` block — and produces one aggregate qc_report.json.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.labeling.schema_hints import build_domain_hints, load_master_schema
from app.quality_control.checks import ALL_CHECKS, HARD_ISSUE_CODES, check_valid_document_id
from app.quality_control.models import QCContext, QCIssue, QCResult
from app.schema_discovery.domains import CANONICAL_DOMAINS

REQUIRED_FIELD_COUNT = 3
DEFAULT_OCR_CONFIDENCE_THRESHOLD = 0.5

_OCR_FAILURE_CODES = {"ocr_failed", "ocr_low_confidence"}
_EXTRACTION_FAILURE_CODES = {"extraction_failed", "normalization_failed"}


def _load_labels(labeled_dir: Path) -> dict[str, dict]:
    labels = {}
    for path in sorted((labeled_dir / "labels").glob("*.json")):
        labels[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return labels


def load_normalized_documents(normalized_dir: Path) -> dict[str, dict]:
    docs = {}
    for path in sorted((normalized_dir / "normalized").glob("*/document.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        docs[data.get("document_id", path.parent.name)] = data
    return docs


def _find_duplicate_content(doc_jsons: dict[str, dict]) -> dict[str, list[str]]:
    """Groups document_ids whose SOURCE FILE bytes hash identically — a
    genuine duplicate in the raw dataset, not just a filename coincidence.
    Skips any document whose source file isn't readable (already flagged by
    check_readable_source_file)."""
    by_hash: dict[str, list[str]] = defaultdict(list)
    for document_id, doc_json in doc_jsons.items():
        source_path = doc_json.get("source_path")
        if not source_path:
            continue
        path = Path(source_path)
        if not path.is_file():
            continue
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
        by_hash[digest].append(document_id)

    duplicate_of: dict[str, list[str]] = {}
    for ids in by_hash.values():
        if len(ids) > 1:
            for document_id in ids:
                duplicate_of[document_id] = [i for i in ids if i != document_id]
    return duplicate_of


@dataclass
class QCReport:
    total_documents: int = 0
    approved: int = 0
    pending: int = 0
    rejected: int = 0
    ocr_failures: int = 0
    extraction_failures: int = 0
    missing_fields: int = 0
    duplicate_documents: int = 0
    duplicate_document_ids: int = 0
    domain_distribution: dict[str, int] = field(default_factory=dict)

    def to_json_dict(self) -> dict:
        return asdict(self)


def run(
    labeled_dir: Path,
    normalized_dir: Path,
    master_schema_dir: Path,
    ocr_confidence_threshold: float = DEFAULT_OCR_CONFIDENCE_THRESHOLD,
) -> tuple[QCReport, list[QCResult]]:
    labels = _load_labels(labeled_dir)
    doc_jsons = load_normalized_documents(normalized_dir)
    hints = build_domain_hints(load_master_schema(master_schema_dir))

    # document_id uniqueness: structurally guaranteed by one file per id, but
    # a label's OWN document_id field could still disagree with its filename
    # (checked per-document) or, if labels were ever hand-merged from
    # multiple runs, two files could exist for the same id under different
    # names — check the field value itself for duplicates too.
    id_field_counts: dict[str, int] = defaultdict(int)
    for label in labels.values():
        doc_id = label.get("document_id")
        if doc_id:
            id_field_counts[doc_id] += 1
    duplicate_ids = {doc_id for doc_id, count in id_field_counts.items() if count > 1}

    duplicate_content = _find_duplicate_content(doc_jsons)

    results: list[QCResult] = []
    domain_distribution: dict[str, int] = defaultdict(int)

    for document_id, label in labels.items():
        domain = label.get("domain") if label.get("domain") in CANONICAL_DOMAINS else "other"
        domain_distribution[domain] += 1

        ctx = QCContext(
            doc_json=doc_jsons.get(document_id),
            required_fields=hints.get(domain, {}).get("fields", [])[:REQUIRED_FIELD_COUNT],
            ocr_confidence_threshold=ocr_confidence_threshold,
            duplicate_document_id=label.get("document_id") in duplicate_ids,
            duplicate_content_with=duplicate_content.get(document_id, []),
        )

        issues: list[QCIssue] = check_valid_document_id(label, document_id, ctx)
        for check in ALL_CHECKS:
            issues.extend(check(label, ctx))

        passed = not any(issue.code in HARD_ISSUE_CODES for issue in issues)
        previous_status = label.get("annotation_status", "pending")
        # A gate that reflects CURRENT truth every run, not a one-way
        # ratchet: failing is ALWAYS forced to "rejected", even overriding a
        # prior "approved" (enforcing "never let an invalid annotation stay
        # approved"); passing preserves "approved" (a human's prior sign-off
        # still holds) but otherwise resolves to "pending" — including
        # recovering a label that was "rejected" on an earlier run but now
        # passes (e.g. the underlying data was fixed, or the checks were
        # recalibrated) rather than leaving "rejected" as a permanent trap.
        new_status = "rejected" if not passed else ("approved" if previous_status == "approved" else "pending")

        label["annotation_status"] = new_status
        label["qc"] = {
            "passed": passed,
            "issues": [issue.to_json_dict() for issue in issues],
        }
        (labeled_dir / "labels" / f"{document_id}.json").write_text(
            json.dumps(label, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        results.append(QCResult(document_id=document_id, passed=passed, issues=issues,
                                 previous_status=previous_status, new_status=new_status))

    report = QCReport(total_documents=len(labels), domain_distribution=dict(sorted(domain_distribution.items())))
    for result in results:
        if result.new_status == "approved":
            report.approved += 1
        elif result.new_status == "pending":
            report.pending += 1
        elif result.new_status == "rejected":
            report.rejected += 1

        issue_codes = {issue.code for issue in result.issues}
        if issue_codes & _OCR_FAILURE_CODES:
            report.ocr_failures += 1
        if issue_codes & _EXTRACTION_FAILURE_CODES:
            report.extraction_failures += 1
        if "missing_required_fields" in issue_codes:
            report.missing_fields += 1
        if "duplicate_document" in issue_codes:
            report.duplicate_documents += 1
        if "duplicate_document_id" in issue_codes:
            report.duplicate_document_ids += 1

    (labeled_dir / "qc_report.json").write_text(
        json.dumps(report.to_json_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report, results


def print_summary(report: QCReport) -> None:
    print()
    print("=== Quality Control Summary ===")
    print(f"Total documents:      {report.total_documents}")
    print(f"Approved:             {report.approved}")
    print(f"Pending:              {report.pending}")
    print(f"Rejected:             {report.rejected}")
    print(f"OCR failures:         {report.ocr_failures}")
    print(f"Extraction failures:  {report.extraction_failures}")
    print(f"Missing fields:       {report.missing_fields}")
    print(f"Duplicate documents:  {report.duplicate_documents}")
    print(f"Duplicate document IDs: {report.duplicate_document_ids}")
    print(f"Domain distribution:  {report.domain_distribution}")
