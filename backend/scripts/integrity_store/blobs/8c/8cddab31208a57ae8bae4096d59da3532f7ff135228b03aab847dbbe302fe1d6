"""Targeted second-pass retry for fields the main per-section extraction
pass came back with nothing for (missing/null) — used right after the main
pass in the completed_application_form pipeline.

Batched, not one call per field: every field missing from a given section
shares that section's exact same retrieved source text (see pipeline.py's
field_source_text), so re-asking about each one individually was N separate
LLM calls re-reading the identical text N times. A form with ~90 schema
fields and only ~20 ever populated could turn into 60-70+ retry calls this
way — one call per group of fields sharing the same source text (in
practice, one call per section with any gaps) gets the same information for
a fraction of the calls."""
from app.documents.extractor import normalize_llm_fields
from app.llm.factory import get_chunked_extraction_chain

_SYSTEM = (
    "You are re-attempting to extract fields that a first pass could not "
    "find a value for. Read the source text again carefully. For each field "
    "listed, return it ONLY if the text actually supports a value — omit any "
    "field you still can't find, do not guess or invent one."
)

_JSON_INSTRUCTION = (
    'Respond with ONLY a JSON object, no other text: {"fields": ['
    '{"field": "<name>", "value": "<the value you found>", "confidence": <0.0-1.0>}, '
    "...]} — include ONLY fields you actually found a value for; omit every field you didn't."
)


def verify_fields(field_paths: list[str], source_text: str) -> list[dict]:
    prompt = (
        f"Fields to re-check: {', '.join(field_paths)}\n\n"
        f"Source text:\n\n{source_text[:4000]}\n\n{_JSON_INSTRUCTION}"
    )
    result = get_chunked_extraction_chain().generate_json(system=_SYSTEM, user_text=prompt)
    return normalize_llm_fields(result["fields"])
