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
    "- SPELL THE KEY OUT IN FULL - never abbreviate a word to save space. Write\n"
    "  sample_collection_datetime, NOT samp_coll_dt; worksheet_datetime, NOT\n"
    "  work_sht_dttm; date_of_birth, NOT dob; quantity, NOT qty.\n"
    "- Keep values verbatim; do not reformat dates or numbers. A result cell that\n"
    "  prints an out-of-range flag beside the number (\"0.35 High\", \"4.2 L\") keeps\n"
    "  the flag - it is part of what the page says.\n"
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
    "- NEVER CONVERT UNITS — transcribe exactly what is printed. If the page says\n"
    "  \"mg/dl\" do not return \"umol/l\"; if it says \"U/l\" do not return \"IU/l\"; if it\n"
    "  says \"g/dl\" do not return \"g/l\". This applies to the result too: report the\n"
    "  number as printed, never rescaled to suit a converted unit.\n"
    "- If a report prints BOTH a conventional and an S.I. column for the same\n"
    "  analyte, use the conventional (first) one.\n"
    "- If there are no tabular results, set 'tests' to an empty list [].\n"
    "- DATES OR SAMPLES AS COLUMNS: if the table has one row per analyte and\n"
    "  several DATE (or sample) columns of results — a trend/comparison table —\n"
    "  emit ONE ROW PER CELL, not one per analyte. 12 analytes x 5 date columns\n"
    "  is 60 rows. Put the column's date/sample heading in 'sample_id', and\n"
    "  repeat the shared 'unit'/'reference_range' on every row from that\n"
    "  analyte. Returning only the newest column silently drops most of the\n"
    "  table — MEASURED: one such report lost 48 of its 60 result rows.\n\n"

    "QUALITY & SAFETY:\n"
    "- Prioritize accuracy over completeness. It is better to omit a field than to hallucinate.\n"
    "- If OCR and image disagree, trust the image.\n"
    "- Do not include any text outside the JSON object."
)


LETTERHEAD_PROMPT = (
    "You extract ONLY the lab-identity and page-furniture fields from a laboratory "
    "test report. Ignore the results table entirely — a separate pass handles it.\n\n"

    "Look at the very TOP of the page (the masthead/letterhead) and the very BOTTOM "
    "(the footer strip and signature block). This text is often small, faint, grey, "
    "or set inside a logo — read it anyway. These are the fields most often missed, "
    "which is the whole reason this pass exists.\n\n"

    "Extract every one of these that appears anywhere on the page:\n"
    "  lab_name, lab_address, lab_phone, lab_fax, lab_email, lab_website,\n"
    "  laboratory_accreditation_no (often beside a NABL mark or QR code, e.g. \"TC-6820\"),\n"
    "  cin, udyam_no, laboratories_at, testing_lab_address,\n"
    "  document_title (e.g. \"TEST REPORT\", \"CERTIFICATE OF ANALYSIS\"),\n"
    "  page (exactly as printed, e.g. \"Page 1 of 2\"),\n"
    "  footer (the disclaimer sentence(s) at the bottom, verbatim),\n"
    "  signatory_name, signatory_title, quality_manager, reviewed_by, approved_by\n"
    "    (a name printed inside a rubber stamp counts; an illegible handwritten\n"
    "     signature with no printed name does not, but its printed TITLE does),\n"
    "  copyright, limits_note, note, conformity_statement,\n"
    "  end_of_report (the closing marker a report prints after its last\n"
    "    result — \"***END OF REPORT***\", \"-- End of Report --\",\n"
    "    \"End of Test Report\"; copy it verbatim including any asterisks\n"
    "    or dashes).\n\n"

    "RULES:\n"
    "- Copy values exactly as written (punctuation, case, spacing, separators).\n"
    "- Omit any field you cannot actually see. Never return a label's own text as "
    "its value (no \"lab_name\": \"LAB NAME\").\n"
    "- A label whose value is blank or redacted is omitted entirely.\n"
    "- Invent a snake_case key from the page's own label only for a lab-identity or "
    "page-furniture field genuinely not listed above.\n"
    "- NEVER name a key after its own value. A masthead prints many values with no\n"
    "  label beside them; name those by WHAT THEY ARE, using the list above:\n"
    "    a bare web address -> lab_website (a second one -> lab_website_2)\n"
    "    a bare email address -> lab_email\n"
    "    a bare phone/fax number -> lab_phone / lab_fax\n"
    "    a postal address block -> lab_address\n"
    "    a company registration number (CIN/UDYAM/GST) -> cin / udyam_no\n"
    "  Returning {\"www_example_com\": \"www.example.com\"} is WRONG twice: the real\n"
    "  field is missed and a meaningless one is invented.\n\n"

    "Respond with ONLY this JSON, no other text:\n"
    "{\"fields\": {\"<snake_case_key>\": \"<value exactly as written>\", \"...\": \"...\"}}"
)


def extract_letterhead(
    chain: LlmChain, text: str, image: bytes | None = None, image_media_type: str | None = None
) -> dict:
    """Second, FOCUSED pass for the lab letterhead and footer block.

    MEASURED 2026-09-04 on the 48-document gold run: ~180 of the 363
    never-extracted fields were exactly these — laboratory_accreditation_no
    (19 documents), lab_phone (19), lab_email (19), lab_website (19), cin
    (17), udyam_no (17), laboratories_at (17), footer (21), lab_address (13),
    lab_name (11), page (10). SYSTEM_PROMPT already carries an explicit
    whole-page coverage checklist naming the letterhead and footer, and the
    model skipped them anyway: one call asked to transcribe a 60-row results
    table AND comb the masthead loses the masthead, because the table
    dominates its attention.

    So this pass asks for nothing else. Returns {"fields": {...}} only — no
    confidence, no tests — and never raises: it is additive, so a failure here
    must leave the main extraction untouched rather than sink the document.
    """
    try:
        result = chain.generate_json(LETTERHEAD_PROMPT, text, image=image, image_media_type=image_media_type)
    except Exception:  # noqa: BLE001 — a bonus pass must never fail the document
        return {"fields": {}}
    if not isinstance(result, dict):
        return {"fields": {}}
    fields = result.get("fields")
    return {"fields": fields if isinstance(fields, dict) else {}}


def merge_letterhead(primary: dict, letterhead: dict) -> dict:
    """Folds the letterhead pass into the main result, ADDITIVELY: a field the
    main pass already returned is never overwritten.

    Deliberately conservative. The main pass sees the whole page in context,
    so where the two disagree it is the better source; this pass exists to
    fill gaps, and letting it overwrite would trade a measured recall gain for
    an unmeasured precision risk. Added fields get confidence 0.5 — "the model
    gave a value but no usable confidence", the same default
    _normalize_confidence applies elsewhere — and are run through the same
    OCR-verbatim check as everything else.
    """
    merged = dict(primary)
    fields = dict(primary.get("fields") or {})
    confidence = dict(primary.get("field_confidence") or {})
    verified = dict(primary.get("field_verified") or {})
    source_norm = primary.get("_source_norm") or ""

    added = 0
    for key, value in (letterhead.get("fields") or {}).items():
        if key in fields or not isinstance(value, str) or not value.strip():
            continue
        fields[key] = value
        confidence[key] = 0.5
        if source_norm:
            verified[key] = _verified_against_source(value, source_norm)
        added += 1

    merged["fields"] = fields
    merged["field_confidence"] = confidence
    merged["field_verified"] = verified
    merged["letterhead_fields_added"] = added
    return merged


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


def extract_lab_report(
    chain: LlmChain, text: str, image: bytes | None = None,
    image_media_type: str | None = None, images: list[tuple[bytes, str]] | None = None,
) -> dict:
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
    no image at all can still call this text-only.

    `images` is the multi-page form: one call covering a WINDOW of pages
    gets that window's page rasters, so the text and the pixels it is
    checked against describe the same pages. See
    scripts/generate_predictions_and_score.py's page-window builder."""
    # `images` only when a caller actually sent several, so a chain (or test
    # double) that predates the multi-image parameter keeps working — same
    # reasoning as LlmChain.generate_json's own pass-through.
    extra = {"images": images} if images else {}
    result = chain.generate_json(
        SYSTEM_PROMPT, text, image=image, image_media_type=image_media_type, **extra,
    )
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

    return {
        "fields": fields,
        "field_confidence": field_confidence,
        "field_verified": field_verified,
        "tests": tests,
        # Kept so merge_letterhead can run the same OCR-verbatim check on
        # anything the second pass adds; stripped before the result is saved.
        "_source_norm": source_norm,
    }


# --- Adapting the {fields, tests} shape to the review pipeline ---------------
# extract_lab_report's output is the shape the accuracy benchmark scores and
# the ground-truth files are written in, so it stays authoritative. The review
# pipeline (documents/pipeline.py) instead stores a flat list of
# {field, value, confidence} rows, one per ExtractedField. Converting HERE,
# next to the function that produces the shape, is what lets the app and the
# benchmark share ONE prompt and one extractor instead of maintaining two.

# Columns of a results row worth storing, in the order they should appear.
_TEST_COLUMNS = ("test_name", "result", "unit", "reference_range", "method", "sample_id")

# Ceiling on rows kept from one results table — a backstop against a
# degenerate/looping model response writing junk rows to extracted_fields.
#
# Set above the largest measured real document: config.py records
# high-protein-paneer.pdf at 272 test rows, so 300 clears it. Note this is
# rarely the binding limit — the provider's max-output-tokens cap (see
# config.py's nova_max_tokens) usually fires first, returning a reply cut off
# mid-JSON that llm/json_utils.py rejects and pipeline.py skips with a
# warning. It CAN bind on a table of many short rows, which packs more rows
# into the same token budget; a table over 300 such rows is truncated here.
MAX_TEST_ROWS = 300

# A value that does NOT appear verbatim in the OCR text we sent is capped to
# this confidence, which puts it under the review threshold (0.85) so a human
# is asked to confirm it. MEASURED 2026-09-03 over 331 values across 10
# documents: every value failing that check (34/34) was wrong, with zero
# correct values wrongly flagged — roughly 3x the error recall of the model's
# own self-reported confidence, which sat at 1.0 for 96% of values including
# 81 wrong ones. See _verified_against_source.
UNVERIFIED_CONFIDENCE_CAP = 0.5


def _cell(value) -> str | None:
    """Everything downstream treats a value as text — documents/grounding.py's
    ground() calls .strip() on it, and ExtractedField.value is a String
    column. A bare number ("s_no": 1) arrives as int and used to raise
    AttributeError, failing the whole document."""
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    return text if text.strip() else None


def _confidence(raw, verified) -> float:
    return min(_normalize_confidence(raw), UNVERIFIED_CONFIDENCE_CAP) if verified is False else _normalize_confidence(raw)


def flatten_for_review(result: dict) -> list[dict]:
    """extract_lab_report's {fields, tests} -> the flat
    [{field, value, confidence}] contract every other extractor in
    documents/extractor.py returns.

    A results table becomes one field per CELL, keyed "tests[i].column" — the
    same convention documents/compiler.py already uses for repeating records.
    Per cell rather than one JSON blob per table so each value is separately
    editable and confirmable in the review UI, and — the part a blob can never
    have — groundable to its own bounding box on the page, since
    documents/pipeline.py grounds each field's value independently.
    """
    flat: list[dict] = []

    confidences = result.get("field_confidence") or {}
    verified = result.get("field_verified") or {}
    for key, value in (result.get("fields") or {}).items():
        cell = _cell(value)
        if cell is not None:
            flat.append({"field": key, "value": cell, "confidence": _confidence(confidences.get(key), verified.get(key))})

    rows = result.get("tests") or []
    for i, row in enumerate(rows[:MAX_TEST_ROWS]):
        if not isinstance(row, dict):
            continue
        confidence = _confidence(row.get("confidence"), row.get("result_verified"))
        for column in _TEST_COLUMNS:
            cell = _cell(row.get(column))
            if cell is not None:
                flat.append({"field": f"tests[{i}].{column}", "value": cell, "confidence": confidence})
    return flat
