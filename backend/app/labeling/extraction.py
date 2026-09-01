"""One LLM call per document: classify its domain AND extract structured
fields/tests in the same pass, guided (not constrained) by Step 4's master
schema hints. Reuses the project's existing multi-provider LLM chain
(app/llm/factory.py) exactly as documents/classifier.py, documents/extractor.py,
and schema_discovery/llm_discovery.py already do; no new provider code.
"""
from __future__ import annotations

from typing import Any

from app.labeling.models import DocumentLabel, TestRow
from app.labeling.schema_hints import format_hints_block
from app.llm.factory import get_llm_chain
from app.schema_discovery.domains import CANONICAL_DOMAINS

_MAX_TEXT_CHARS = 6000

_JSON_INSTRUCTION = (
    'Respond with ONLY a JSON object: {"domain": "<one of: '
    f'{", ".join(CANONICAL_DOMAINS)}>", "fields": {{<document-level '
    'attributes as snake_case keys, values exactly as written>}}, "tests": '
    '[{"test_name": "<parameter name>", "result": "<value as written>", '
    '"unit": "<unit as written, or null>", "reference_range": "<range as '
    'written, or null>"}, ...]}'
)


def _system_prompt(hints_block: str) -> str:
    return (
        "You classify a laboratory/testing report into exactly one domain and "
        "extract structured data from it, based ONLY on what is actually "
        "written in the text given to you. Never invent, guess, or infer a "
        "value that is not present in the text — if you are not sure a field "
        "is really there, leave it out rather than fabricate it. Preserve "
        "every value exactly as written (numbers, decimals, units, symbols) — "
        "do not round numbers, convert units, or reformat text. Every "
        "distinct test/parameter row in the document becomes its own entry "
        "in \"tests\", preserving which result/unit/reference_range belongs "
        "to which test — do not merge different tests' values together.\n\n"
        "Known field names seen in similar documents for each domain (use "
        "these names when they match; still report any other clearly-labeled "
        "field or test parameter actually present, even if not listed here):\n"
        f"{hints_block}"
    )


def extract_label(
    document_id: str,
    original_filename: str,
    text: str,
    page_count: int,
    source_ocr_used: bool,
    source_ocr_confidence: float | None,
    domain_hints: dict[str, dict[str, list[str]]],
) -> DocumentLabel:
    hints_block = format_hints_block(domain_hints)
    prompt = f"Document text (may be truncated):\n\n{text[:_MAX_TEXT_CHARS]}\n\n{_JSON_INSTRUCTION}"
    result = get_llm_chain().generate_json(system=_system_prompt(hints_block), user_text=prompt)

    domain = _normalize_domain(result.get("domain"))
    expected_fields = domain_hints.get(domain, {}).get("fields", [])

    raw_fields = result.get("fields")
    raw_fields = raw_fields if isinstance(raw_fields, dict) else {}

    # Every hinted document-level field is always present in the output —
    # with the model's value if it found one, else null (never omitted, per
    # the task's "if a field is not present, use null" requirement) — plus
    # any other field the model reported that wasn't on the hint list at all.
    fields: dict[str, Any] = {key: _normalize_scalar(raw_fields.get(key)) for key in expected_fields}
    for key, value in raw_fields.items():
        if key in fields:
            continue
        normalized = _normalize_scalar(value)
        if normalized is not None:
            fields[key] = normalized

    tests = _normalize_tests(result.get("tests"))

    return DocumentLabel(
        document_id=document_id, original_filename=original_filename, domain=domain,
        page_count=page_count, source_ocr_used=source_ocr_used, source_ocr_confidence=source_ocr_confidence,
        fields=fields, tests=tests, annotation_status="pending", extraction_status="ok",
    )


def _normalize_domain(raw: object) -> str:
    if isinstance(raw, str) and raw.strip().lower() in CANONICAL_DOMAINS:
        return raw.strip().lower()
    return "other"


def _normalize_scalar(value: object) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)  # a malformed (dict/list) reply — stringify rather than crash


def _normalize_tests(raw: object) -> list[TestRow]:
    if not isinstance(raw, list):
        return []
    tests = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        test_name = row.get("test_name")
        if not test_name:
            continue
        tests.append(TestRow(
            test_name=str(test_name).strip(),
            result=_normalize_scalar(row.get("result")),
            unit=_normalize_scalar(row.get("unit")),
            reference_range=_normalize_scalar(row.get("reference_range")),
        ))
    return tests
