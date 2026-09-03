import json
import logging
import re
from typing import get_args, get_origin

from pydantic import BaseModel

from app.llm.factory import get_chunked_extraction_chain, get_llm_chain

log = logging.getLogger(__name__)

# Which fields to pull per doc_type, and where they land in the compiled form
# (see documents/compiler.py). Empty list = no structured extraction attempted
# for that type (quality_manual_sop / other are reference material, not
# form-fillable records) — flagged in the plan as a known MVP limitation.
FIELD_SETS: dict[str, list[str]] = {
    "legal_proof": [
        "organisation.laboratory_name",
        "organisation.address",
        "organisation.telephone",
        "organisation.email",
        "organisation.legal_entity_registration_number",
        "organisation.legal_entity_registering_authority",
        "organisation.gst_number",
        "organisation.pan_number",
        "organisation.tan_number",
    ],
    "equipment_calibration_certificate": [
        "name",
        "make_model",
        "serial_number",
        "range_and_accuracy",
        "calibration_date",
        "calibration_due_date",
        "calibrated_by",
    ],
    "staff_cv_certificate": [
        "name",
        "designation",
        "qualification",
        "relevant_experience_years",
    ],
    "pt_ilc_result": [
        "record_type",
        "materials_or_products",
        "parameter_or_measurand",
        "test_or_calibration_method",
        "participation_date",
        "provider",
        "provider_accreditation_body_or_country",
        "performance_metric",
        "performance_value",
        "corrective_action_taken",
    ],
    "lab_contact_info": [
        "laboratory_name",
        "country",
        "state",
        "district",
        "address",
        "pincode",
        "mobile",
        "email",
        "technical_head_or_lab_manager",
    ],
    "recognition_equipment_certificate": [
        "discipline",
        "name",
        "in_house_or_traceability",
        "external_lab_name",
    ],
    "reference_material_certificate": [
        "name",
        "source",
        "expiry_date",
        "traceability",
    ],
    "recognition_reference_material_certificate": [
        "discipline",
        "name",
        "traceability",
        "source",
        "expiry_date",
    ],
    "authorized_signatory_declaration": [
        "department_or_section",
        "name",
        "designation",
        "qualification_with_specialization",
        "relevant_experience_years",
        "authorized_area",
    ],
    "recognition_pt_report": [
        "discipline_group_subgroup",
        "test_parameter",
        "provider",
        "performance",
        "date_of_report",
    ],
    "project_reference_document": [
        "project_name",
        "project_size",
        "project_duration",
        "project_reference",
    ],
    "regulatory_recognition_certificate": [
        "last_recognition_certificate_number",
        "last_recognition_issuing_authority",
        "last_recognition_validity",
    ],
    "product_list_declaration": [
        "product_name",
        "product_standard",
    ],
    "shareholder_declaration": [
        "name",
        "shareholding_percent",
        "relations_with_other_directors",
        "remarks",
    ],
    "quality_manual_sop": [],
    "other": [],
}

_SYSTEM = (
    "You extract structured field values from a NABL application supporting "
    "document. For each requested field, return the value found in the text "
    "(or null if absent) and your confidence 0-1 that the value is correct. "
    "Do not guess: prefer null with low confidence over a fabricated value."
)

_JSON_INSTRUCTION = (
    'Respond with ONLY a JSON object, no other text: {"fields": ['
    '{"field": "<field name>", "value": "<value or null>", "confidence": <0.0-1.0>}, ...]}'
    " — one entry per requested field, in the order given. Use the exact field"
    " name from the list above, verbatim, for \"field\" — do not translate,"
    " rename, or paraphrase it."
)


def normalize_llm_fields(raw: list) -> list[dict]:
    """A capable cloud model (Gemini/Groq/...) reliably includes every key
    the prompt asks for. A small local model under Ollama's generic
    `format: json` grammar (which only enforces "this is valid JSON", not a
    specific schema) sometimes doesn't — e.g. omitting "confidence" entirely
    on some entries, observed in practice with qwen2.5:3b. Every downstream
    consumer (documents/pipeline.py, compiler.py) indexes "field"/"value"/
    "confidence" unconditionally, so normalize once here, right at the LLM
    boundary, instead of pushing defensive .get()s through every call site."""
    normalized = []
    for item in raw:
        if not isinstance(item, dict) or not item.get("field"):
            continue
        confidence = item.get("confidence")
        if not isinstance(confidence, (int, float)):
            confidence = 0.5  # the model gave a value but skipped confidence — treat as uncertain, not absent
        normalized.append({"field": item["field"], "value": item.get("value"), "confidence": confidence})
    return normalized


def extract_fields(doc_type: str, text: str) -> list[dict]:
    fields = FIELD_SETS.get(doc_type, [])
    if not fields:
        return []
    prompt = f"Fields to extract: {', '.join(fields)}\n\nDocument text:\n\n{text[:8000]}\n\n{_JSON_INSTRUCTION}"
    result = get_llm_chain().generate_json(system=_SYSTEM, user_text=prompt)
    return _filter_to_requested_fields(doc_type, fields, normalize_llm_fields(result["fields"]))


def extract_fields_vision(doc_type: str, image_bytes: bytes, media_type: str) -> list[dict]:
    """English/Latin scanned-page fallback: deepreef-ocr has no model for it
    (see ocr_client.SUPPORTED_SCRIPTS), so read the page directly via the
    LLM chain's image input instead of OCR text."""
    fields = FIELD_SETS.get(doc_type, [])
    if not fields:
        return []
    prompt = f"Fields to extract: {', '.join(fields)}\n\n{_JSON_INSTRUCTION}"
    result = get_llm_chain().generate_json(system=_SYSTEM, user_text=prompt, image=image_bytes, image_media_type=media_type)
    return _filter_to_requested_fields(doc_type, fields, normalize_llm_fields(result["fields"]))


def _filter_to_requested_fields(doc_type: str, requested: list[str], returned: list[dict]) -> list[dict]:
    """The LLM occasionally paraphrases a field name instead of returning it
    verbatim (e.g. "lab_name" for "laboratory_name"). compiler.py matches
    field_path by exact name, so a renamed field would otherwise be stored
    under a bogus path and silently vanish from the compiled form later —
    dropping it here instead, loudly, keeps the failure visible in logs."""
    requested_set = set(requested)
    kept = []
    for item in returned:
        if item.get("field") in requested_set:
            kept.append(item)
        else:
            log.warning(
                "extractor: dropping field %r for doc_type=%r — not in the requested set %r",
                item.get("field"), doc_type, requested,
            )
    return kept


# --- Open-ended extraction ---------------------------------------------------
# Runs unconditionally, alongside extract_fields/extract_fields_vision (never
# instead of), for every document regardless of doc_type — including "other"
# and "quality_manual_sop", whose FIELD_SETS are empty and would otherwise
# never produce a single field. Unlike extract_fields, nothing here is
# filtered against a known field-name list: the model invents its own names,
# and documents/compiler.py routes the result into the compiled form's
# `extra_fields` bucket rather than a named schema attribute.
_OPEN_SYSTEM = (
    "You extract every distinct, meaningful piece of information from a document "
    "as key/value pairs. Skip boilerplate/noise. Do not guess a value that isn't "
    "actually present.\n\n"
    "NAMING — take the field name from the DOCUMENT'S OWN printed label, "
    "lowercased and snake_cased, and nothing more:\n"
    '- a line reading "Sample Name: CASTOR OIL" is {"field": "sample_name"}, '
    'not "product_name" or "name_of_sample".\n'
    '- a line reading "Date: 28-Mar-2023" is {"field": "date"} — do NOT add a '
    'qualifier the label does not have (not "prescription_date", not '
    '"report_date").\n'
    "- only invent a name when the value genuinely has no printed label.\n\n"
    "REPEATING TABLES — keep one field for the whole table, whose value is a "
    "JSON array of row objects. Do NOT flatten rows into numbered fields:\n"
    '- CORRECT: {"field": "medicines", "value": [{"medicine": "X", "dose": '
    '"1-0-1"}, {"medicine": "Y", "dose": "0-0-1"}]}\n'
    '- WRONG: medicine_1_name, medicine_1_dose, medicine_2_name, ...'
)

_OPEN_JSON_INSTRUCTION = (
    'Respond with ONLY a JSON object, no other text: {"fields": ['
    '{"field": "<snake_case of the document\'s own label>", '
    '"value": "<value, or a JSON array of row objects for a table>", '
    '"confidence": <0.0-1.0>}, ...]}'
)

# Guardrail against a runaway response, not a budget: the hand-labeled
# reference documents average ~25 fields and top out at 61, and the old cap
# of 40 was already truncating real documents mid-extraction (observed on
# 001_Lab-report). Set well clear of the largest real document instead of
# just above the average.
MAX_OPEN_FIELDS = 150
# Generous enough to hold a JSON-encoded repeating table (a 4-row
# medication table already runs past 500 chars, and such a value is
# dropped rather than truncated — see _sanitize_open_fields), while
# still bounding a runaway single value.
MAX_OPEN_VALUE_CHARS = 4000


def extract_open_fields(text: str) -> list[dict]:
    prompt = f"Document text:\n\n{text[:8000]}\n\n{_OPEN_JSON_INSTRUCTION}"
    result = get_llm_chain().generate_json(system=_OPEN_SYSTEM, user_text=prompt)
    return _sanitize_open_fields(normalize_llm_fields(result["fields"]))


def extract_open_fields_vision(image_bytes: bytes, media_type: str) -> list[dict]:
    result = get_llm_chain().generate_json(
        system=_OPEN_SYSTEM, user_text=_OPEN_JSON_INSTRUCTION, image=image_bytes, image_media_type=media_type
    )
    return _sanitize_open_fields(normalize_llm_fields(result["fields"]))


def _sanitize_open_fields(fields: list[dict]) -> list[dict]:
    """A local guardrail against a runaway/garbage model response — not a
    schema check (see extract_open_fields' docstring: there is no known field
    list to check against here). Drops duplicate field names (first wins),
    truncates absurdly long values, and caps the total row count.

    A repeating table arrives as a LIST of row objects (see _OPEN_SYSTEM) and
    is JSON-encoded to a string here, because everything downstream of this
    point — FieldResult.value, documents/grounding.py's ground(), and the
    ExtractedField String column — handles text, not nested structures. A
    structure too long to store is DROPPED rather than truncated: cutting a
    JSON array mid-string yields something that neither parses as JSON nor
    reads as a value, which is worse than not having the field."""
    seen: set[str] = set()
    sanitized: list[dict] = []
    for item in fields:
        name = item.get("field")
        if not name or name in seen:
            continue
        value = item.get("value")

        if isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False)
            if len(value) > MAX_OPEN_VALUE_CHARS:
                log.warning(
                    "extractor: dropping open field %r — encoded table is %d chars, over the %d limit",
                    name, len(value), MAX_OPEN_VALUE_CHARS,
                )
                continue
        elif isinstance(value, str):
            value = value[:MAX_OPEN_VALUE_CHARS]
        elif value is not None:
            # A bare number ("qty": 180) arrives as int/float. Everything
            # downstream treats a value as text — documents/grounding.py's
            # ground() calls .strip() on it — so an un-stringified number
            # raised AttributeError and failed the whole document
            # (observed on 2 of 53: 'int' object has no attribute 'strip').
            value = str(value)

        seen.add(name)
        sanitized.append({"field": name, "value": value, "confidence": item.get("confidence", 0.5)})
        if len(sanitized) >= MAX_OPEN_FIELDS:
            log.warning("extractor: open-ended extraction hit the %d-field cap, truncating", MAX_OPEN_FIELDS)
            break
    return sanitized


# --- Whole-form extraction ---------------------------------------------------
# A "completed_application_form" upload IS (or covers many sections of) the
# target NABL form itself — an applicant's own filled copy, or a multi-section
# draft — rather than one narrow supporting certificate. FIELD_SETS's one-
# doc-type-to-one-entity model doesn't fit that, so this path introspects the
# actual Pydantic schema for the application's form_type and asks for
# EVERY section at once, using an "attr[i].subfield" convention for repeating
# tables (equipment, staff, scope, ...) so it still fits the same flat
# {field, value, confidence} shape every other extractor returns — no schema
# change needed in documents/compiler.py beyond a matching merge function.
_INDEX_PATTERN = re.compile(r"\[\d+\]")


def _flatten_schema(model_cls: type[BaseModel], prefix: str = "", depth: int = 0, max_depth: int = 2) -> list[str]:
    paths: list[str] = []
    for name, field in model_cls.model_fields.items():
        annotation = field.annotation
        origin = get_origin(annotation)
        args = get_args(annotation)

        # Unwrap `X | None` (== Optional[X] == Union[X, None]) to X.
        if origin is not None and type(None) in args:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                annotation = non_none[0]
                origin = get_origin(annotation)
                args = get_args(annotation)

        if origin is list:
            item_type = args[0] if args else None
            if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                for sub in _flatten_schema(item_type, max_depth=1):
                    paths.append(f"{name}[i].{sub}")
            # list[str]/list[enum] fields (e.g. disciplines, trainings) aren't
            # structured enough to ask for generically — skipped on purpose.
        elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
            if depth < max_depth:
                paths.extend(_flatten_schema(annotation, prefix=f"{prefix}{name}.", depth=depth + 1, max_depth=max_depth))
        else:
            paths.append(f"{prefix}{name}")
    return paths


_FULL_FORM_SYSTEM = (
    "This document is a FILLED COPY of a NABL laboratory application form itself "
    "(or a draft covering many of its sections at once) — not a single narrow "
    "supporting certificate. You may be shown the whole document, or just one "
    "page/chunk of a much longer one — extract whatever sections are actually "
    "present in the text you were given; don't expect every section to be there. "
    "For a field written as \"something[i].subfield\", repeat it once per record "
    "you find, substituting 0, 1, 2, ... for i in the order the records appear in "
    "THIS text (e.g. two equipment rows produce equipment[0].name and "
    "equipment[1].name — a second chunk that finds two more equipment rows should "
    "likewise start again from equipment[0], not continue the numbering; the "
    "caller re-indexes chunks against each other). Only emit indices for records "
    "actually present in the text — do not invent empty ones. Do not guess: "
    "prefer omitting a field over a fabricated value. IMPORTANT: only include a "
    "field in your response if you actually found a value for it in the text — "
    "if a requested field is not present, leave it out of the JSON entirely; do "
    "NOT include it with a null value. (The caller already knows the full list "
    "of requested fields and fills in the ones you omitted — restating them with "
    "null wastes your output on fields you already know are absent.)"
)

_FULL_FORM_JSON_INSTRUCTION = (
    'Respond with ONLY a JSON object, no other text: {"fields": ['
    '{"field": "<field name, e.g. organisation.gst_number or equipment[0].name>", '
    '"value": "<the value you found>", "confidence": <0.0-1.0>}, ...]} — include '
    "ONLY fields you actually found a value for; omit every field you didn't."
)

# A real filled NABL form runs 15-30 pages and the first several are NABL's
# own boilerplate instructions/amendment history, not applicant data — a
# small truncation would silently cut off before the actual content starts.
# extract_full_form_fields_chunked (below) avoids this by processing the
# document page-by-page instead of relying on truncation; this cap only
# matters for extract_full_form_fields's single-shot fallback.
_MAX_SINGLE_SHOT_CHARS = 150000


def extract_full_form_fields(form_type: str, text: str) -> list[dict]:
    """Single-shot whole-form extraction: everything in one prompt. Prefer
    extract_full_form_fields_chunked for a real multi-page upload — this is
    for a document short enough to not need chunking (or direct/test use)."""
    field_templates = form_field_templates(form_type)
    return _extract_full_form_chunk(field_templates, text[:_MAX_SINGLE_SHOT_CHARS])


def extract_full_form_fields_chunked(form_type: str, chunks: list[str]) -> tuple[list[dict], list[str]]:
    """The real entry point for a multi-page filled-form upload: one LLM call
    per page/chunk instead of one giant prompt — keeps every call well within
    any provider's context/payload limits (crucial for Ollama's small default
    context and for Groq/HF's hard request-size caps alike; see
    llm/factory.py's get_chunked_extraction_chain for the chunked-extraction
    provider order), and lets each chunk find only what's actually on it.

    Each chunk's LLM call re-starts every repeating entity's index from 0 (it
    has no idea what earlier chunks found), so results are re-indexed against
    a running offset per list attribute before being returned — the caller
    (documents/compiler.py) sees one globally-consistent set of indices, same
    as if a single giant call had produced them.

    One chunk's LLM chain being exhausted (every provider currently
    rate-limited/down) must not discard every OTHER chunk's already-extracted
    fields — that chunk is skipped and noted in the returned warnings list
    rather than raising and losing the whole document's extraction."""
    field_templates = form_field_templates(form_type)
    offsets: dict[str, int] = {}
    all_fields: list[dict] = []
    warnings: list[str] = []
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        try:
            chunk_fields = _extract_full_form_chunk(field_templates, chunk)
        except Exception as exc:  # noqa: BLE001 — see docstring above
            log.warning("extract_full_form_fields_chunked: chunk %d failed, skipping: %s", i, exc)
            warnings.append(f"chunk {i} could not be extracted: {exc}")
            continue
        all_fields.extend(_renumber_chunk_fields(chunk_fields, offsets))
    return all_fields, warnings


def form_field_templates(form_type: str) -> list[str]:
    from app.schemas.forms import FORM_MODEL, NablFormType  # local import: forms.py doesn't depend on this module

    model_cls = FORM_MODEL[NablFormType(form_type)]
    return _flatten_schema(model_cls)


def extract_section_fields(field_templates: list[str], texts: list[str]) -> list[dict]:
    """documents/retrieval.py's entry point: same JSON contract as
    _extract_full_form_chunk, but for a SET of pre-retrieved chunks covering
    one schema section, instead of the whole document or one arbitrary page —
    only the pages actually relevant to this section are sent.

    The LLM is told (see _FULL_FORM_SYSTEM) to omit any field it didn't find,
    rather than spelling out a null entry for every one it was asked about —
    for a slow local model (llm/providers.py's OllamaProvider) generating
    that JSON is the expensive part, and most of a section's requested fields
    are typically absent on any given document, so this measurably cuts
    output size and generation time. But pipeline.py's verification retry
    pass (_process_completed_application_form) needs to see an explicit
    None-valued entry for every flat field it didn't get a value for, to
    know what's worth a second, targeted look — so those get backfilled
    here, in Python, at effectively zero cost, restoring the exact
    one-entry-per-requested-field contract this function has always
    returned. Indexed "[i]." fields are repeating records, not backfilled —
    there's no such thing as a 'missing' record to invent."""
    combined = "\n\n--- next page ---\n\n".join(texts)
    raw = _extract_full_form_chunk(field_templates, combined)
    return _fill_missing_flat_fields(field_templates, raw)


def _fill_missing_flat_fields(field_templates: list[str], returned: list[dict]) -> list[dict]:
    returned_names = {f["field"] for f in returned}
    filled = list(returned)
    for template in field_templates:
        if "[i]" in template or template in returned_names:
            continue
        filled.append({"field": template, "value": None, "confidence": 0.0})
    return filled


def _extract_full_form_chunk(field_templates: list[str], text: str) -> list[dict]:
    prompt = (
        f"Fields to extract (repeat '[i].' fields once per record found): {', '.join(field_templates)}\n\n"
        f"Document text:\n\n{text}\n\n{_FULL_FORM_JSON_INSTRUCTION}"
    )
    result = get_chunked_extraction_chain().generate_json(system=_FULL_FORM_SYSTEM, user_text=prompt)
    return _filter_to_requested_indexed_fields(field_templates, normalize_llm_fields(result["fields"]))


_INDEXED_FIELD = re.compile(r"^(\w+)\[(\d+)\]\.(.+)$")


def _renumber_chunk_fields(chunk_fields: list[dict], offsets: dict[str, int]) -> list[dict]:
    """Remap one chunk's own equipment[0]/equipment[1]/... indices onto a
    running global offset per list attribute, mutating `offsets` in place so
    the next chunk continues from where this one left off."""
    renumbered = []
    local_max: dict[str, int] = {}
    for f in chunk_fields:
        match = _INDEXED_FIELD.match(f.get("field", ""))
        if not match:
            renumbered.append(f)
            continue
        attr, local_index, sub_path = match.group(1), int(match.group(2)), match.group(3)
        global_index = offsets.get(attr, 0) + local_index
        local_max[attr] = max(local_max.get(attr, -1), local_index)
        renumbered.append({**f, "field": f"{attr}[{global_index}].{sub_path}"})
    for attr, max_local_index in local_max.items():
        offsets[attr] = offsets.get(attr, 0) + max_local_index + 1
    return renumbered


def _filter_to_requested_indexed_fields(templates: list[str], returned: list[dict]) -> list[dict]:
    template_set = set(templates)
    kept = []
    for item in returned:
        field = item.get("field", "")
        if _INDEX_PATTERN.sub("[i]", field) in template_set:
            kept.append(item)
        else:
            log.warning("extractor: dropping out-of-schema field %r for whole-form extraction", field)
    return kept
