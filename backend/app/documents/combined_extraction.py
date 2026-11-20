"""One LLM call per chunk that returns EVERYTHING applicable: header/metadata
key-values, the results-table rows, and the named NABL schema slots the active
form needs.

This is the budget-optimised consolidation. Before it a single chunk cost up
to three separate calls over the same text — classify, then `extract_fields`
for FIELD_SETS slots, then the lab-report pass, then a second letterhead pass
— even though all four read identical text. Combining them is the single
biggest call reduction available, and it is safe because the outputs don't
interact: the model is asked for three named parts of one JSON object.

The `tests` array keeps its own shape (test_name / result / unit /
reference_range) and is never flattened into unrelated fields, so a lab
report's results table survives extraction as a table.

Ported from the feat/token branch (commit 70c2b2e), with two deliberate
adaptations:

1. ONE image per payload, not a list. That branch also widened
   llm/chain.py to take `images=[...]`; this branch's chain still takes a
   single `image=`, and the chunking here only ever has one page raster to
   send anyway (see documents/pipeline.py). Porting multi-image support is a
   separate change to every provider.
2. Confidence is computed HERE rather than asked for. The ported prompt asks
   for no confidence at all, which would have dropped the OCR-verbatim check
   (documents/lab_report.py's _verified_against_source) that MEASURED ~3x the
   error recall of the model's own self-reported confidence. That check is a
   pure function of the value and the source text, so it needs no help from
   the model — see attach_confidence below.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class DocumentPayload:
    """What gets sent to the model for one chunk."""

    text_blocks: list[str] = field(default_factory=list)
    image: bytes | None = None
    media_type: str = "image/png"
    # Non-fatal per-page problems (e.g. OCR unavailable for one page) — the
    # caller decides whether to surface them; extraction still runs.
    warnings: list[str] = field(default_factory=list)

    @property
    def user_text(self) -> str:
        if not self.text_blocks:
            return "(No extractable text layer here — read the attached page image.)"
        return "\n\n".join(self.text_blocks)


_BASE = (
    "You are an information-extraction system for laboratory/testing reports "
    "and for the supporting documents of a NABL laboratory-accreditation "
    "application (calibration certificates, staff CVs, PT/ILC reports, "
    "registration proofs).\n\n"
    "Extract ONLY what is actually present. Never invent, guess or infer a "
    "value that isn't really there; if unsure, omit the field. A value you "
    "cannot establish from the source must be omitted, not filled with a "
    "placeholder.\n\n"
    "The text is a machine transcription and its reading order may be wrong — "
    "a form often lists every LABEL first and every VALUE afterwards. Do NOT "
    "pair a label with a value because they are adjacent in the text; pair "
    "them only when the layout shows them on the same row. If you cannot tell "
    "which value belongs to a label, omit that field.\n\n"
    "LETTERHEAD AND FOOTER ARE NOT OPTIONAL. A results table dominates a page "
    "and models routinely transcribe it while skipping the masthead — "
    "MEASURED on a 48-document run, ~180 of 363 never-extracted fields were "
    "exactly these: laboratory_accreditation_no, lab_phone, lab_email, "
    "lab_website, cin, udyam_no, lab_address, lab_name, footer, page. Read "
    "the very TOP of the page (masthead, often small/faint/grey or set inside "
    "a logo) and the very BOTTOM (footer strip, signature block, rubber-stamp "
    "text) before you start on the table, and include what you find."
)

_SHAPE = (
    "\n\nRespond with ONLY a JSON object of this exact shape:\n"
    '{"fields": {"<snake_case_key>": "<value exactly as written>", ...}, '
    '"tests": [{"test_name": "<name>", "result": "<value as written>", '
    '"unit": "<unit or null>", "reference_range": "<range or null>", '
    '"method": "<method or null>", "sample_id": "<sample id or null>"}, ...], '
    '"schema_fields": {"<requested.field.path>": "<value>", ...}}\n\n'
    '"fields" — every header/metadata value (names, dates, addresses, report '
    "numbers, sample details, letterhead/lab identity, accreditation numbers, "
    "page markers, sampling details, footnotes, signatory names and titles), "
    "keyed by the document's own printed labels in snake_case. Prefer these "
    "exact keys where the document has that field: lab_name, lab_address, "
    "lab_phone, lab_email, lab_website, laboratory_accreditation_no, cin, "
    "udyam_no, report_number, ulr_no, report_issue_date, document_title, page, "
    "footer, note, client_name, client_address, sample_description, "
    "sample_number, sample_condition, sample_quantity, sample_collected_by, "
    "date_of_receipt, start_date_of_analysis, end_date_of_analysis, "
    "signatory_name, signatory_title, quality_manager, remarks.\n"
    "  THAT LIST IS NAMING GUIDANCE, NOT A LIMIT. It says what to CALL a field\n"
    "  you find; it does NOT say which fields to look for. Extract EVERY\n"
    "  labelled value the page prints, including ones the list never names,\n"
    "  keyed by its own printed label in snake_case.\n"
    "  MEASURED on a 53-document run: 367 of 395 never-extracted values were\n"
    "  fields the list does not name — the enumeration was read as the whole\n"
    "  task and everything else was skipped, even though the value was sitting\n"
    "  in the text. Among them: the tagline or slogan under a lab's name, the\n"
    "  ROLE printed beside a signature, the accreditation scope and the\n"
    "  discipline/group line, the registered office in the footer, serving size\n"
    "  and RDA notes on a food label, abbreviation keys, disclaimers,\n"
    "  conformity and results statements, section headings above a table, and\n"
    "  'tested at' / 'sample drawn by' lines.\n"
    "  If the page prints a label and a value, that is a field — return it.\n"
    '"tests" — every row of a results table. Empty list if there is no table. '
    "Keep one entry per row; never merge rows or split a row across entries. "
    "NEVER convert units — transcribe exactly what is printed.\n"
    "  test_name is the parameter name AS PRINTED ON ITS OWN LINE — copy that\n"
    "  line verbatim, INCLUDING any parenthetical the lab prints inside it.\n"
    "  \"TRIGLYCERIDES (Lipase)\" stays \"TRIGLYCERIDES (Lipase)\": the\n"
    "  parenthetical is part of the printed name, even when it names a method.\n"
    "  What does NOT belong is a SEPARATE LINE wrapping underneath the name.\n"
    "  Where the cell reads \"Haemoglobin\" and then, on the next line,\n"
    "  \"Method: Non Cyanmeth Hb\", that second line is the method column's\n"
    "  value and must NOT be appended to test_name. Same for a sample id, a\n"
    "  footnote marker or an instrument name printed on its own line under the\n"
    "  parameter.\n"
    "  MEASURED both ways: merging a wrapped line left every row of a 14-row\n"
    "  table unmatchable, and stripping an inline parenthetical broke a 4-row\n"
    "  table that had been scoring perfectly.\n"
    "  REFERENCE RANGE IS NOT OPTIONAL. Most results tables print an\n"
    "  acceptable-range column and it is the single most-skipped value:\n"
    "  MEASURED, 76% of reference-range errors were the column left empty on a\n"
    "  row whose name and result were read correctly. It is headed many ways —\n"
    "  Reference Range, Biological Reference Interval, Bio. Ref. Interval,\n"
    "  Normal Range, Normal Value, Limits, Requirement, Specification,\n"
    "  Permissible Limit, As per FSSR/FSSAI ... — and sometimes has no heading\n"
    "  at all, sitting to the right of the result. Whatever it is called, put it\n"
    "  in reference_range. Use null ONLY when the table genuinely has no such\n"
    "  column.\n"
    "  Copy the range cell WHOLE, exactly as printed: if it reads\n"
    "  \"12 - 15 g/dL\" then reference_range is \"12 - 15 g/dL\". Ranges written as\n"
    "  \"<14\", \"NMT 5.0\", \"Min 3.0\", \"Shall be Absent\" or several lines of\n"
    "  thresholds are values too — keep them verbatim.\n"
    "  reference_range and unit are INDEPENDENT columns and you fill BOTH. A\n"
    "  unit appearing inside the range cell does not remove it from the unit\n"
    "  column: for a row printing unit \"g/dL\" and range \"12 - 15 g/dL\", return\n"
    "  unit \"g/dL\" AND reference_range \"12 - 15 g/dL\". Leaving unit empty\n"
    "  because the range already mentions it is WRONG — MEASURED, that mistake\n"
    "  alone lost 52 units.\n"
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
        prompt += f'\n\nRequested schema field paths for "schema_fields" ({len(paths)}): {", ".join(paths)}'
    return prompt


def extract(chain, payload: DocumentPayload, schema_field_paths: list[str] | None = None) -> dict:
    """Returns {"fields": {...}, "tests": [...], "schema_fields": {...}}.

    Falls back to a TEXT-ONLY retry when the call carried an image and every
    provider refused it. MEASURED 2026-09-03 on the source branch: with Nova
    unavailable an image-bearing chunk failed outright because other providers
    reject some image payloads, so a 17-page document extracted ZERO fields —
    even though most of its pages had exact PyMuPDF text any text provider
    could have read. Degrading to text-only loses the visual evidence, which
    is a real loss, but it is strictly better than losing the whole document.
    """
    system = build_system_prompt(schema_field_paths)
    try:
        result = chain.generate_json(
            system, payload.user_text, image=payload.image, image_media_type=payload.media_type,
        )
    except Exception:
        if payload.image is None:
            raise
        log.warning("combined extraction failed with an image; retrying text-only")
        result = chain.generate_json(system, payload.user_text)

    if not isinstance(result, dict):
        return {"fields": {}, "tests": [], "schema_fields": {}}

    def clean(obj) -> dict:
        # An empty value is not a field: the model legitimately emits keys it
        # has no value for when a label is present but the box is blank.
        if not isinstance(obj, dict):
            return {}
        return {k: v for k, v in obj.items() if isinstance(v, (str, int, float)) and str(v).strip()}

    tests = result.get("tests")
    return {
        "fields": clean(result.get("fields")),
        "tests": [t for t in tests if isinstance(t, dict)] if isinstance(tests, list) else [],
        "schema_fields": clean(result.get("schema_fields")),
    }


def attach_confidence(result: dict, source_text: str) -> dict:
    """Adds the `field_confidence` / `field_verified` maps documents/
    lab_report.py's flatten_for_review expects.

    The prompt above asks for no confidence — deliberately, since a model's
    self-reported confidence MEASURED as barely discriminative (1.0 for 96% of
    values, including 81 wrong ones). The OCR-verbatim check is the far
    stronger signal (~3x the error recall) and is a pure function of the value
    and the source text, so it is computed here instead of requested.

    A value that IS verbatim in the source gets 1.0; one that is not is capped
    below the review threshold so a human is asked. With no source text the
    check cannot run at all — which is NOT the same claim as "these were
    invented" — so nothing is flagged.
    """
    from app.documents.lab_report import UNVERIFIED_CONFIDENCE_CAP, _normalize_for_search, _verified_against_source

    source_norm = _normalize_for_search(source_text)
    fields = result.get("fields") or {}

    if not source_norm:
        return {**result, "field_confidence": {k: 1.0 for k in fields}, "field_verified": {}}

    verified = {k: _verified_against_source(v, source_norm) for k, v in fields.items()}
    confidence = {k: (1.0 if ok else UNVERIFIED_CONFIDENCE_CAP) for k, ok in verified.items()}

    tests = []
    for row in result.get("tests") or []:
        ok = _verified_against_source(row.get("result"), source_norm)
        tests.append({**row, "result_verified": ok, "confidence": 1.0 if ok else UNVERIFIED_CONFIDENCE_CAP})

    return {**result, "field_confidence": confidence, "field_verified": verified, "tests": tests}
