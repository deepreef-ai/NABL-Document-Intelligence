"""Individual QC checks — each a pure function of (label dict, QCContext)
returning a list of QCIssue (empty if that check passes). Read-only: no
check ever mutates the label's `fields`/`tests` content, only reports on it.
"""
from __future__ import annotations

import re
from pathlib import Path

from app.quality_control.models import QCContext, QCIssue
from app.schema_discovery.domains import CANONICAL_DOMAINS

_DOCUMENT_ID_PATTERN = re.compile(r"^LR_\d{6}$")
_RANGE_LIKE_PATTERN = re.compile(r"^\s*[\d.]+\s*-\s*[\d.]+\s*$")
_PLACEHOLDER_SENTINELS = {"n/a", "na", "null", "none", "-", "--", "tbd"}


def check_valid_document_id(label: dict, expected_id: str, ctx: QCContext) -> list[QCIssue]:
    doc_id = label.get("document_id")
    issues = []
    if not doc_id or not isinstance(doc_id, str) or not _DOCUMENT_ID_PATTERN.match(doc_id):
        issues.append(QCIssue("invalid_document_id", f"document_id {doc_id!r} is missing or not in the expected LR_NNNNNN form"))
    elif doc_id != expected_id:
        issues.append(QCIssue("document_id_filename_mismatch", f"document_id {doc_id!r} does not match its filename ({expected_id!r})"))
    return issues


def check_readable_source_file(label: dict, ctx: QCContext) -> list[QCIssue]:
    if ctx.doc_json is None:
        return [QCIssue("normalized_document_missing", "no corresponding normalized document.json found for this label")]
    source_path = ctx.doc_json.get("source_path")
    if not source_path or not Path(source_path).is_file():
        return [QCIssue("source_file_unreadable", f"source file not found or unreadable: {source_path!r}")]
    return []


def check_page_count(label: dict, ctx: QCContext) -> list[QCIssue]:
    if ctx.doc_json is None:
        return []  # already covered by check_readable_source_file
    expected = ctx.doc_json.get("page_count")
    actual_pages = len(ctx.doc_json.get("pages", []))
    label_count = label.get("page_count")
    if label_count != expected or expected != actual_pages:
        return [QCIssue(
            "page_count_mismatch",
            f"label page_count={label_count}, normalized document page_count={expected}, actual page entries={actual_pages}",
        )]
    return []


def check_valid_domain(label: dict, ctx: QCContext) -> list[QCIssue]:
    domain = label.get("domain")
    if domain not in CANONICAL_DOMAINS:
        return [QCIssue("invalid_domain", f"domain {domain!r} is not one of {CANONICAL_DOMAINS}")]
    return []


def check_valid_schema(label: dict, ctx: QCContext) -> list[QCIssue]:
    issues = []
    if not isinstance(label.get("fields"), dict):
        issues.append(QCIssue("invalid_schema", "'fields' is missing or not an object"))
    tests = label.get("tests")
    if not isinstance(tests, list):
        issues.append(QCIssue("invalid_schema", "'tests' is missing or not a list"))
        return issues
    for i, row in enumerate(tests):
        if not isinstance(row, dict):
            issues.append(QCIssue("invalid_schema", f"tests[{i}] is not an object"))
            continue
        missing_keys = [k for k in ("test_name", "result", "unit", "reference_range") if k not in row]
        if missing_keys:
            issues.append(QCIssue("invalid_schema", f"tests[{i}] is missing key(s) {missing_keys}"))
        elif not row.get("test_name"):
            issues.append(QCIssue("invalid_schema", f"tests[{i}] has an empty test_name"))
    return issues


def check_missing_required_fields(label: dict, ctx: QCContext) -> list[QCIssue]:
    if not ctx.required_fields:
        return []
    fields = label.get("fields")
    if not isinstance(fields, dict):
        return []  # already flagged by check_valid_schema
    missing = [f for f in ctx.required_fields if fields.get(f) is None]
    if missing:
        return [QCIssue("missing_required_fields", f"required fields missing/null: {missing}")]
    return []


def check_invalid_values(label: dict, ctx: QCContext) -> list[QCIssue]:
    tests = label.get("tests")
    if not isinstance(tests, list):
        return []
    problems = []
    for i, row in enumerate(tests):
        if not isinstance(row, dict):
            continue
        result = row.get("result")
        if isinstance(result, str):
            stripped = result.strip()
            if stripped == "":
                problems.append(f"tests[{i}].result is an empty string instead of null")
            elif stripped.lower() in _PLACEHOLDER_SENTINELS:
                problems.append(f"tests[{i}].result looks like a placeholder sentinel: {result!r}")
    if problems:
        return [QCIssue("invalid_values", "; ".join(problems))]
    return []


def check_table_mappings(label: dict, ctx: QCContext) -> list[QCIssue]:
    tests = label.get("tests")
    if not isinstance(tests, list):
        return []
    problems = []
    last_result_by_name: dict[str, object] = {}
    for i, row in enumerate(tests):
        if not isinstance(row, dict):
            continue
        name = row.get("test_name")
        unit = row.get("unit")
        reference_range = row.get("reference_range")
        if isinstance(unit, str) and _RANGE_LIKE_PATTERN.match(unit) and not reference_range:
            problems.append(f"tests[{i}] ({name!r}): unit {unit!r} looks like a reference range — possible column swap")
        if name in last_result_by_name and last_result_by_name[name] != row.get("result"):
            problems.append(f"tests[{i}]: duplicate test_name {name!r} with a different result than an earlier row")
        if name is not None:
            last_result_by_name[name] = row.get("result")
    if problems:
        return [QCIssue("incorrect_table_mapping", "; ".join(problems))]
    return []


def check_extraction_and_ocr_failures(label: dict, ctx: QCContext) -> list[QCIssue]:
    issues = []
    if label.get("extraction_status") == "failed":
        issues.append(QCIssue("extraction_failed", label.get("error") or "labeling extraction failed"))
    if ctx.doc_json is not None and ctx.doc_json.get("status") == "failed":
        error_text = (ctx.doc_json.get("error") or "").lower()
        if "ocr" in error_text or "deepreef" in error_text:
            issues.append(QCIssue("ocr_failed", ctx.doc_json.get("error")))
        else:
            issues.append(QCIssue("normalization_failed", ctx.doc_json.get("error")))
    confidence = label.get("source_ocr_confidence")
    if label.get("source_ocr_used") and confidence is not None and confidence < ctx.ocr_confidence_threshold:
        issues.append(QCIssue("ocr_low_confidence", f"OCR confidence {confidence} below threshold {ctx.ocr_confidence_threshold}"))
    return issues


def check_duplicates(label: dict, ctx: QCContext) -> list[QCIssue]:
    issues = []
    if ctx.duplicate_document_id:
        issues.append(QCIssue("duplicate_document_id", f"document_id {label.get('document_id')!r} is used by more than one label"))
    if ctx.duplicate_content_with:
        issues.append(QCIssue("duplicate_document", f"identical source content to: {ctx.duplicate_content_with}"))
    return issues


ALL_CHECKS = (
    check_readable_source_file,
    check_page_count,
    check_valid_domain,
    check_valid_schema,
    check_missing_required_fields,
    check_invalid_values,
    check_table_mappings,
    check_extraction_and_ocr_failures,
    check_duplicates,
)

# Issue codes that mean the label is structurally broken or untrustworthy —
# these are what "invalid" means for the "never let an invalid annotation
# become approved" rule, and force annotation_status to "rejected".
#
# Everything else (missing_required_fields, invalid_values,
# incorrect_table_mapping, ocr_low_confidence) is a completeness/quality
# SIGNAL worth a human's attention during review, not proof the label is
# wrong — a document that genuinely doesn't mention a patient's name (e.g.
# redacted for privacy) isn't a broken label, and treating "some field is
# null" as automatic grounds for rejection would reject the majority of a
# real dataset for reasons that are properties of the source document, not
# defects in the extraction. These softer issues still show up in
# label["qc"]["issues"] and are still counted in the QC report (ocr_failures,
# missing_fields, etc.) — they just don't by themselves flip the gate.
HARD_ISSUE_CODES = frozenset({
    "invalid_document_id",
    "document_id_filename_mismatch",
    "normalized_document_missing",
    "source_file_unreadable",
    "page_count_mismatch",
    "invalid_domain",
    "invalid_schema",
    "extraction_failed",
    "normalization_failed",
    "ocr_failed",
    "duplicate_document",
    "duplicate_document_id",
})
