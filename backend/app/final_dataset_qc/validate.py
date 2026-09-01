"""Loading and per-record validation for the final-dataset QC stage.

Checks are split hard/soft, same convention as app/quality_control/checks.py:
HARD (makes a document count as invalid — a structural/integrity defect):
unique document_id, valid domain, valid source_format, annotation actually
approved, well-formed fields/tests objects, no duplicate JSON keys, test
rows keeping test_name/result/unit/reference_range together, multi-page
structure agreeing with the source normalized document, and the source
file being traceable back to this document_id via Step 2's content-hash
registry.
SOFT (always reported and counted, never by itself invalidating a
document): a field key not found in Step 4's master schema for that
domain, and a field value that's an empty string or placeholder text
instead of null. These are soft because the master schema was built from a
23-document SAMPLE (Step 3) — a field a real document genuinely has that
the sample never saw is expected, not a defect, and forcing every novel
field name to have been foreseen would reject good data for reasons that
are properties of the schema-discovery sample, not the label.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.schema_discovery.domains import CANONICAL_DOMAINS

VALID_SOURCE_FORMATS = {"pdf", "jpg", "jpeg", "png", "tif", "tiff"}
_PLACEHOLDER_SENTINELS = {"n/a", "na", "null", "none", "-", "--", "tbd"}
_DUPLICATE_KEYS_MARKER = "__duplicate_keys__"


def _dup_key_hook(pairs: list[tuple[str, object]]) -> dict:
    """Python's json module silently keeps only the LAST value for a
    duplicate object key — this hook instead records which keys were
    duplicated so validate_record can report it instead of hiding it."""
    seen: set[str] = set()
    duplicates: list[str] = []
    result: dict = {}
    for key, value in pairs:
        if key in seen:
            duplicates.append(key)
        seen.add(key)
        result[key] = value
    if duplicates:
        result[_DUPLICATE_KEYS_MARKER] = duplicates
    return result


def load_jsonl_records(jsonl_path: Path) -> tuple[list[dict], list[dict]]:
    records = []
    parse_errors = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line, object_pairs_hook=_dup_key_hook))
            except json.JSONDecodeError as exc:
                parse_errors.append({"line": line_number, "message": str(exc)})
    return records, parse_errors


def load_master_schema_keys(master_schema_dir: Path) -> dict[str, set[str]]:
    path = master_schema_dir / "master_schema.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {domain: set(info.get("keys", [])) for domain, info in data.get("domains", {}).items()}


def load_normalized_documents(normalized_dir: Path) -> dict[str, dict]:
    docs = {}
    for path in sorted((normalized_dir / "normalized").glob("*/document.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        docs[data.get("document_id", path.parent.name)] = data
    return docs


def load_id_registry(normalized_dir: Path) -> dict[str, str]:
    path = normalized_dir / "id_registry.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("by_hash", {})


def _content_hash(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def validate_record(
    record: dict,
    schema_keys_by_domain: dict[str, set[str]],
    doc_jsons: dict[str, dict],
    id_registry_by_hash: dict[str, str],
) -> tuple[bool, list[dict]]:
    document_id = record.get("document_id")
    issues: list[dict] = []
    hard_failed = False

    def hard(check: str, message: str) -> None:
        nonlocal hard_failed
        hard_failed = True
        issues.append({"document_id": document_id, "check": check, "severity": "hard", "message": message})

    def soft(check: str, message: str) -> None:
        issues.append({"document_id": document_id, "check": check, "severity": "soft", "message": message})

    if _DUPLICATE_KEYS_MARKER in record:
        hard("no_duplicate_keys", f"duplicate JSON key(s) at the document's top level: {record[_DUPLICATE_KEYS_MARKER]}")

    domain = record.get("domain")
    if domain not in CANONICAL_DOMAINS:
        hard("valid_domain", f"domain {domain!r} is not one of {CANONICAL_DOMAINS}")

    source_format = record.get("source_format")
    if source_format not in VALID_SOURCE_FORMATS:
        hard("valid_source_format", f"source_format {source_format!r} is not one of {sorted(VALID_SOURCE_FORMATS)}")

    if record.get("annotation_status") != "approved":
        hard("annotation_approved", f"annotation_status is {record.get('annotation_status')!r}, not 'approved'")

    fields = record.get("fields")
    if not isinstance(fields, dict):
        hard("valid_schema", "'fields' is missing or not an object")
    else:
        if _DUPLICATE_KEYS_MARKER in fields:
            hard("no_duplicate_keys", f"duplicate JSON key(s) in 'fields': {fields[_DUPLICATE_KEYS_MARKER]}")
        expected = schema_keys_by_domain.get(domain, set())
        for key, value in fields.items():
            if key == _DUPLICATE_KEYS_MARKER:
                continue
            if expected and key not in expected:
                soft("no_unexpected_keys", f"field {key!r} is not in the master schema for domain {domain!r}")
            if value is None:
                soft("missing_values_are_null", f"field {key!r} is null")
            elif isinstance(value, str) and (value.strip() == "" or value.strip().lower() in _PLACEHOLDER_SENTINELS):
                soft("missing_values_are_null", f"field {key!r} has a placeholder/empty value {value!r} instead of null")

    tests = record.get("tests")
    if not isinstance(tests, list):
        hard("valid_schema", "'tests' is missing or not a list")
    else:
        for i, row in enumerate(tests):
            if not isinstance(row, dict):
                hard("test_row_relationship", f"tests[{i}] is not an object")
                continue
            missing_keys = [k for k in ("test_name", "result", "unit", "reference_range") if k not in row]
            if missing_keys:
                hard("test_row_relationship", f"tests[{i}] is missing key(s) {missing_keys}")
            elif not row.get("test_name"):
                hard("test_row_relationship", f"tests[{i}] has an empty test_name")

    page_count = record.get("page_count")
    doc_json = doc_jsons.get(document_id)
    if doc_json is None:
        hard("source_traceable", f"no corresponding normalized document found for {document_id!r}")
    else:
        expected_page_count = doc_json.get("page_count")
        actual_pages = len(doc_json.get("pages", []))
        if page_count != expected_page_count or page_count != actual_pages:
            hard(
                "multi_page_preserved",
                f"page_count mismatch: record={page_count}, normalized={expected_page_count}, actual page entries={actual_pages}",
            )
        source_path = doc_json.get("source_path")
        if not source_path or not Path(source_path).is_file():
            hard("source_traceable", f"source file not found or unreadable: {source_path!r}")
        else:
            digest = _content_hash(Path(source_path))
            mapped_id = id_registry_by_hash.get(digest) if digest else None
            if mapped_id != document_id:
                hard(
                    "source_traceable",
                    f"source file content does not map back to {document_id!r} in id_registry.json (maps to {mapped_id!r})",
                )

    return not hard_failed, issues
