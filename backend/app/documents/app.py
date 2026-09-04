"""Extracts a lab/testing report's fields + test-results table via one LLM
call. Used today by scripts/generate_predictions_and_score.py (the accuracy
benchmark); the same function is the entry point for a real upload flow to
call once this document type is wired into the wizard.

This is a different document type and schema from documents/extractor.py's
FIELD_SETS-per-doc_type extraction (which reads NABL *application* supporting
certificates — equipment calibration, staff CVs, PT/ILC results — into a
fixed field list). A lab test report's schema is open-ended (whatever fields
and test rows the document actually has), so it gets its own module and
prompt rather than being folded into extractor.py's fixed-schema contract.
"""
import re

from app.llm.chain import LlmChain

SYSTEM_PROMPT = (
    "You are a precise information-extraction agent for laboratory and testing reports "
    "(milk, food, chemical, medical, environmental, and similar documents). "
    "You will receive: (1) a document image and (2) OCR-extracted text. "
    "Use the image as the primary source of truth; use OCR text only to help with spelling and ordering. "
    "Extract ONLY information that is visibly present in the document.\n\n"

    "HARD RULES:\n"
    "- Do not invent, guess, infer, or normalize any value. If a field is missing or ambiguous, omit it.\n"
    "- A label with no value (blank, redacted, or empty box) means no value — omit that field entirely.\n"
    "- Never use a label’s own text as the value (e.g. do not output \"patient_name\": \"PATIENT NAME\").\n"
    "- Copy values exactly as written (units, symbols, case, punctuation, spacing).\n"
    "- Output ONLY a single JSON object. No markdown, no code fences, no explanations.\n"
    "- If the same label appears multiple times, include each occurrence separately in 'tests' or 'fields' as appropriate.\n"
    "- For multi-page tables, extract all rows across pages.\n"
    "- For checkboxes/radio buttons, report only the option that is visibly marked/selected.\n"
    "- Ignore decorative watermarks (e.g. diagonal \"SAMPLE\") and marginal handwritten scribbles.\n"
    "- Do include letterhead, signature blocks, rubber-stamp text, and footer disclaimers as fields.\n\n"

    "COVERAGE CHECKLIST — scan the whole page in this order and extract all visible fields:\n"
    "1. Letterhead/masthead: lab_name, lab_address, lab_phone, lab_fax, lab_email, lab_website,\n"
    "   cin, udyam_no, laboratory_accreditation_no, document_title.\n"
    "2. Header detail block: report/sample/client identifiers and dates.\n"
    "3. Results tables: all rows go in 'tests', never in 'fields'.\n"
    "4. Notes/remarks under the table: remarks, note, conformity or decision-rule statements, limit footnotes.\n"
    "5. Signature block: signatory_name, signatory_title, quality_manager, reviewed/approved names,\n"
    "   including names printed inside rubber stamps. Illegible handwritten signatures with no printed name yield nothing.\n"
    "6. Footer strip: page (e.g. \"Page 1 of 2\"), footer disclaimer sentence, any second address/contact line.\n\n"

    "OUTPUT SCHEMA (JSON ONLY):\n"
    "{\n"
    '  "fields": {\n'
    '    "<snake_case_key>": "<value exactly as written>",\n'
    '    "...": "..."\n'
    "  },\n"
    '  "field_confidence": {\n'
    '    "<snake_case_key>": <0.0-1.0>,\n'
    '    "...": "..."\n'
    "  },\n"
    '  "tests": [\n'
    "    {\n"
    '      "test_name": "<name exactly as written>",\n'
    '      "result": "<value exactly as written>",\n'
    '      "unit": "<unit or null>",\n'
    '      "reference_range": "<range or null>",\n'
    '      "method": "<method or null>",\n'
    '      "sample_id": "<sample identifier or null>",\n'
    '      "confidence": <0.0-1.0>\n'
    "    }\n"
    "  ]\n"
    "}\n\n"

    "FIELD NAMING:\n"
    "- Use these exact keys whenever the document has that field:\n"
    "  lab_name, lab_address, lab_phone, lab_fax, lab_email, lab_website,\n"
    "  laboratory_accreditation_no, cin, udyam_no,\n"
    "  report_number, ulr_no, report_issue_date, document_title, page, footer, note,\n"
    "  quantity_and_condition,\n"
    "  client_name, client_address, contact_person, customer_reference,\n"
    "  sample_name, sample_description, sample_number, sample_code, sample_condition, sample_quantity,\n"
    "  sampling_location, sample_collected_by, sampling_protocol, date_of_pick_up, date_of_receipt,\n"
    "  start_date_of_analysis, end_date_of_analysis, batch_no, manufacturing_date, expiry_date,\n"
    "  discipline, group, sample_sub_group, nabl_scope, testing_lab_address,\n"
    "  quality_manager, signatory_name, signatory_title, remarks.\n"
    "- For any other visible field not in this list, create a concise snake_case key from the label.\n"
    "- Keep values verbatim; do not reformat dates or numbers.\n"
    "- Never put results-table columns (test_name, result, unit, reference_range, method, sample_id) into 'fields'.\n\n"

    "CONFIDENCE:\n"
    "- 'field_confidence' must have exactly one entry per key in 'fields', each a number from 0.0 to 1.0.\n"
    "- Each row in 'tests' carries its own 'confidence' (0.0–1.0).\n"
    "- 1.0 = clearly printed/typed text, unambiguous. Lower it for handwriting, blurry/low-quality scans, "
    "partially obscured or cut-off values, or any reading you are not fully sure of.\n"
    "- Confidence reflects how sure you are of the extracted value, not whether to include it. "
    "If a field has no visible value, omit it entirely.\n\n"

    "TESTS:\n"
    "- 'tests' contains every row of every results table: analyte/parameter name, observed value, unit, "
    "reference range, method, sample ID if present.\n"
    "- If a column is missing (e.g., no units), set that key to null.\n"
    "- If there are no tabular results, set 'tests' to an empty list [].\n\n"

    "QUALITY & SAFETY:\n"
    "- Prioritize accuracy over completeness. It is better to omit a field than to hallucinate.\n"
    "- If OCR and image disagree, trust the image.\n"
    "- Do not include any text outside the JSON object."
)


def _normalize_confidence(value) -> float:
    """A reply can omit a confidence entirely or return something
    non-numeric — that's "the model gave a value but no usable confidence",
    which is treated as uncertain (0.5) rather than as absent, the same
    defensive default extractor.py's normalize_llm_fields applies at its
    own LLM boundary."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.5
    return float(value) if 0.0 <= value <= 1.0 else 0.5


# Dash variants (en/em dash, minus sign, ...) that OCR and the model use
# interchangeably with a plain hyphen.
_DASH_VARIANTS = re.compile("[‐‑‒–—―−]")
_WHITESPACE = re.compile(r"\s+")


def _normalize_for_search(value) -> str:
    """Folds case, all whitespace, and dash variants so a value can be
    searched for inside the OCR text without spacing/punctuation noise
    producing a false "not found" — OCR writes "40 - 129" where the model
    returns "40-129", and neither spelling is an error.

    Deliberately NOT app/benchmark/compare.py's _normalize_value, despite
    the overlap: that one exists to test two extracted values for EQUALITY
    and maps placeholders ("N/A", "-", "none") to None, which is exactly
    wrong here — this one is searching for a substring inside a large text
    blob, where those are ordinary characters to match on, not "no value".
    """
    if value is None:
        return ""
    text = _DASH_VARIANTS.sub("-", str(value).strip().lower())
    return _WHITESPACE.sub("", text)


def _verified_against_source(value, source_norm: str) -> bool:
    """True when `value` appears verbatim (after normalization) in the OCR
    text that was sent to the model — i.e. the model READ it rather than
    inferring it. MEASURED 2026-09-03 over 331 values across 10 documents:
    every value that failed this check (34/34) was wrong, with zero correct
    values wrongly flagged, catching ~37% of all errors — roughly 3x the
    error recall of the model's own self-reported confidence, which sat at
    1.0 for 96% of values including 81 wrong ones.

    Not a completeness check: a value that IS in the OCR text but got
    attached to the wrong field/row still passes here (that's a layout
    attribution problem, not a hallucination). And now that an image is
    sent alongside the text, a value correctly read from the IMAGE that OCR
    missed entirely would fail this check despite being right — not once in
    that 331-value sample, but the failure mode to watch as image reading
    improves."""
    value_norm = _normalize_for_search(value)
    return bool(value_norm) and value_norm in source_norm


def extract_lab_report(chain: LlmChain, text: str, image: bytes | None = None, image_media_type: str | None = None) -> dict:
    """Returns {"fields", "field_confidence", "field_verified", "tests"}.

    Two trust signals ride alongside every extracted value:
      - "field_confidence" / a row's "confidence": what the MODEL says
        about itself. MEASURED barely discriminative (see
        _verified_against_source) — carried for completeness, not
        recommended as a gate on its own.
      - "field_verified" / a row's "result_verified": whether the value
        actually appears in the OCR text we sent. The stronger signal by
        ~3x; see _verified_against_source for the measurements.

    Both are carried the same way for the same reason: "fields" stays a
    flat {key: value} dict with its per-key signals in PARALLEL dicts keyed
    identically — rather than nesting {"value": ..., "confidence": ...}
    under each key — because the ground-truth files and
    app/benchmark/compare.py's compare_fields both read "fields" as flat
    scalars and predate both signals; nesting would break every one of
    them. A "tests" row instead carries its signals inline, since tests
    were already a list of objects and extra keys there are ignored by the
    comparison logic.

    Never raises on a malformed LLM reply shape — callers get
    empty/defaulted values instead of a KeyError/TypeError.

    `image`/`image_media_type` are optional: SYSTEM_PROMPT is written
    assuming both an image and OCR text arrive together, but a caller with
    no single image for the whole document (see
    scripts/generate_predictions_and_score.py's multi-page-PDF case) can
    still call this text-only."""
    result = chain.generate_json(SYSTEM_PROMPT, text, image=image, image_media_type=image_media_type)
    if not isinstance(result, dict):
        return {"fields": {}, "field_confidence": {}, "field_verified": {}, "tests": []}

    fields = result.get("fields")
    fields = fields if isinstance(fields, dict) else {}

    raw_confidence = result.get("field_confidence")
    raw_confidence = raw_confidence if isinstance(raw_confidence, dict) else {}
    # Driven by the keys actually present in "fields", so a confidence entry
    # for a field the model didn't return is dropped, and a field it
    # returned without one still gets a usable number.
    field_confidence = {key: _normalize_confidence(raw_confidence.get(key)) for key in fields}

    # No source text means the check couldn't run at all — which is NOT the
    # same claim as "these values were invented", so nothing is flagged
    # rather than flagging everything and burying the real signal in noise.
    source_norm = _normalize_for_search(text)
    field_verified = (
        {key: _verified_against_source(value, source_norm) for key, value in fields.items()}
        if source_norm else {}
    )

    raw_tests = result.get("tests")
    tests = []
    for row in raw_tests if isinstance(raw_tests, list) else []:
        if not isinstance(row, dict):
            continue
        row = {**row, "confidence": _normalize_confidence(row.get("confidence"))}
        if source_norm:
            row["result_verified"] = _verified_against_source(row.get("result"), source_norm)
        tests.append(row)

    return {"fields": fields, "field_confidence": field_confidence, "field_verified": field_verified, "tests": tests}
