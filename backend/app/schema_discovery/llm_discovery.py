"""One LLM call per sampled document: read the text Step 2 already extracted
(pymupdf or the existing OCR — this module never touches a PDF/image/OCR
engine itself) and identify which field/parameter names are PRESENT — never
their values, never a field invented from a generic template. Reuses the
project's existing multi-provider LLM chain (app/llm/factory.py) exactly as
documents/classifier.py and documents/extractor.py already do; no new
provider code.
"""
from __future__ import annotations

import re

from app.llm.factory import get_llm_chain
from app.schema_discovery.domains import CANONICAL_DOMAINS
from app.schema_discovery.models import DocumentKeys

_SYSTEM = (
    "You perform schema discovery on OCR/text-extracted laboratory test "
    "reports and certificates spanning many different domains. Your ONLY "
    "job is to identify which field/parameter names are PRESENT in the "
    "given document — never their values, and never a field that isn't "
    "actually there. Do not pad the answer with fields from a generic "
    "template that this specific document doesn't mention.\n\n"
    "Classify the document into exactly one domain: "
    f"{', '.join(CANONICAL_DOMAINS)} (use 'other' when it does not clearly "
    "fit any of the rest).\n\n"
    "For every label/heading/parameter name you see in the text (for "
    "example 'Patient Name', 'Fat %', 'SNF', 'Batch No.', 'pH', "
    "'Moisture (%)', 'Ash Content'), convert it into a concise lowercase "
    "snake_case identifier (for example patient_name, fat_percent, snf, "
    "batch_number, ph, moisture_percent, ash_content)."
)

_JSON_INSTRUCTION = (
    'Respond with ONLY a JSON object, no other text: {"candidate_domain": '
    f'"<one of: {", ".join(CANONICAL_DOMAINS)}>", "keys": ["<snake_case_key>", ...]}}'
)

_MAX_TEXT_CHARS = 6000  # matches documents/classifier.py's own truncation


def discover_document_keys(document_id: str, text: str) -> DocumentKeys:
    prompt = f"Document text (may be truncated):\n\n{text[:_MAX_TEXT_CHARS]}\n\n{_JSON_INSTRUCTION}"
    result = get_llm_chain().generate_json(system=_SYSTEM, user_text=prompt)

    return DocumentKeys(
        document_id=document_id,
        candidate_domain=_normalize_domain(result.get("candidate_domain")),
        keys=_normalize_keys(result.get("keys")),
    )


def _normalize_domain(raw: object) -> str:
    if isinstance(raw, str) and raw.strip().lower() in CANONICAL_DOMAINS:
        return raw.strip().lower()
    return "other"


def _normalize_keys(raw: object) -> list[str]:
    """Defensive cleanup for whatever a free-tier model actually returns —
    not every provider reliably emits clean snake_case (see classifier.py's
    _parse_classification for the same concern with 'confidence')."""
    if not isinstance(raw, list):
        return []
    seen: dict[str, None] = {}
    for item in raw:
        if not isinstance(item, str):
            continue
        cleaned = _to_snake_case(item)
        if cleaned:
            seen.setdefault(cleaned, None)
    return list(seen)


def _to_snake_case(raw: str) -> str:
    lowered = re.sub(r"[^a-z0-9]+", "_", raw.strip().lower())
    return lowered.strip("_")
