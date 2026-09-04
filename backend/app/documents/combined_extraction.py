"""One LLM call per chunk that returns EVERYTHING applicable: open-ended
key/values, structured results-table rows, and the named schema slots the
active NABL form needs.

This is the budget-optimised consolidation. Before it, a single upload made
up to three separate calls over the same text — classify, then
`extract_fields` for FIELD_SETS slots, then `unified_extraction` for open
fields/tables — even though all three read the same page. Combining them is
the single biggest call reduction available, and it is safe because the
outputs don't interact: the model is asked for three named parts of one JSON
object.

The `tests` array keeps its own shape (test_name / result / unit /
reference_range) and is never flattened into unrelated fields, so a lab
report's results table survives extraction as a table.
"""
from __future__ import annotations

import logging

from app.documents.unified_extraction import DocumentPayload

log = logging.getLogger(__name__)

_BASE = (
    "You are an information-extraction system for laboratory/testing reports "
    "and for the supporting documents of a NABL laboratory-accreditation "
    "application (calibration certificates, staff CVs, PT/ILC reports, "
    "registration proofs).\n\n"
    "Extract ONLY what is actually present. Never invent, guess or infer a "
    "value that isn't really there; if unsure, omit the field. A value you "
    "cannot establish from the source must be omitted, not filled with a "
    "placeholder.\n\n"
    "Each page is given to you as text AND as an image. The text is a "
    "machine transcription and its reading order may be wrong — a form often "
    "lists every LABEL first and every VALUE afterwards. Do NOT pair a label "
    "with a value because they are adjacent in the text; pair them only when "
    "the image shows them on the same row. If you cannot tell which value "
    "belongs to a label, omit that field."
)

_SHAPE = (
    "\n\nRespond with ONLY a JSON object of this exact shape:\n"
    '{"fields": {"<snake_case_key>": "<value exactly as written>", ...}, '
    '"tests": [{"test_name": "<name>", "result": "<value as written>", '
    '"unit": "<unit or null>", "reference_range": "<range or null>"}, ...], '
    '"schema_fields": {"<requested.field.path>": "<value>", ...}}\n\n'
    '"fields" — every header/metadata value (names, dates, addresses, report '
    "numbers, sample details, letterhead/lab identity, accreditation numbers, "
    "page markers, sampling details, footnotes), keyed by the document's own "
    "printed labels in snake_case.\n"
    '"tests" — every row of a results table. Empty list if there is no table. '
    "Keep one entry per row; never merge rows or split a row across entries.\n"
    '"schema_fields" — ONLY the requested field paths listed below, using each '
    "path verbatim as the key. Omit any you cannot find. Use an empty object "
    "if none were requested."
)


def build_system_prompt(schema_field_paths: list[str] | None = None) -> str:
    prompt = _BASE + _SHAPE
    if schema_field_paths:
        # Asked for in the SAME call rather than a second one. Capped so a
        # 95-field whole-form request can't crowd out the document text.
        paths = schema_field_paths[:120]
        prompt += (
            "\n\nRequested schema field paths for \"schema_fields\" "
            f"({len(paths)}): {', '.join(paths)}"
        )
    return prompt


def extract(chain, payload: DocumentPayload, schema_field_paths: list[str] | None = None) -> dict:
    """Returns {"fields": {...}, "tests": [...], "schema_fields": {...}}.

    Falls back to a TEXT-ONLY retry when the call carried images and every
    provider refused it. MEASURED 2026-09-03: with Nova unavailable, an
    image-bearing chunk failed outright because every other provider rejects
    multi-image requests ("gemini takes one image per request, got 9
    pages"), so a 17-page document extracted ZERO fields — even though 15 of
    its 17 pages had exact PyMuPDF text that any text provider could have
    read. Degrading to text-only loses the scanned pages' visual evidence,
    which is a real loss, but it is strictly better than losing the whole
    document, and the pages that needed vision are recoverable afterwards
    (see documents/recovery.py's escalation).
    """
    system = build_system_prompt(schema_field_paths)
    try:
        result = chain.generate_json(
            system,
            payload.user_text,
            image_media_type=payload.media_type,
            images=payload.images or None,
        )
    except Exception:
        if not payload.images:
            raise
        log.warning(
            "combined extraction failed with %d image(s); retrying text-only",
            len(payload.images),
        )
        result = chain.generate_json(system, payload.user_text)
    if not isinstance(result, dict):
        return {"fields": {}, "tests": [], "schema_fields": {}}

    def clean(obj) -> dict:
        # An empty value is not a field — see unified_extraction.extract on
        # why the model legitimately emits keys it has no value for.
        if not isinstance(obj, dict):
            return {}
        return {k: v for k, v in obj.items() if isinstance(v, (str, int, float)) and str(v).strip()}

    tests = result.get("tests")
    return {
        "fields": clean(result.get("fields")),
        "tests": [t for t in tests if isinstance(t, dict)] if isinstance(tests, list) else [],
        "schema_fields": clean(result.get("schema_fields")),
    }
