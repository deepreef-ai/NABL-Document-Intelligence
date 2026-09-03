"""Shared field/result types + the source-grounding helper.

Split out from pipeline.py so the new rule_extraction.py / chunking.py /
retrieval.py modules can produce and ground FieldResults without importing
pipeline.py itself (which imports them) — avoids a circular import.
"""
from dataclasses import dataclass, field

from app.documents.geometry import Rect


@dataclass
class FieldResult:
    field: str
    value: str | None
    confidence: float
    source_page: int | None = None
    source_bbox: Rect | None = None
    # "llm" | "rule_based" | "verification" | "open_extraction" — see
    # extraction_report.py; "open_extraction" fields (documents/extractor.py's
    # extract_open_fields*) are routed by compiler.py into the compiled
    # form's extra_fields bucket rather than a named schema attribute.
    source: str = "llm"


@dataclass
class PipelineResult:
    doc_type: str
    doc_confidence: float
    # "born_digital_pdf" | "ocr_pdf" | "mixed_pdf" | "docx" | "ocr:<script>" | "vision_llm"
    extraction_source: str
    fields: list[FieldResult]
    # Non-fatal: e.g. one schema section's LLM call exhausted every provider
    # (all rate-limited/down) while other sections succeeded. The document
    # still gets saved as "extracted" with whatever fields DID come back —
    # see documents/pipeline.py's _process_completed_application_form and
    # routers/documents.py's _run_pipeline — rather than losing every
    # already-successful section over one section's outage.
    extraction_warnings: list[str] = field(default_factory=list)
    # How many pages were actually read and extracted from. Surfaced all the
    # way to the review UI on purpose: with no page cap any more, "did it
    # really read all 17 pages of this scan" is the first question worth
    # answering when an extraction looks thin, and it's invisible otherwise.
    page_count: int | None = None


def ground(value: str, candidates: list[tuple[str, Rect]]) -> Rect | None:
    """Best-effort: find which OCR line / PDF text span an extracted value
    came from, so the review UI can highlight it. Case-insensitive substring
    match either direction; exact match wins over partial."""
    if not value:
        return None
    needle = value.strip().lower()
    if not needle:
        return None
    for cand_text, rect in candidates:
        if cand_text.strip().lower() == needle:
            return rect
    for cand_text, rect in candidates:
        cand_norm = cand_text.strip().lower()
        if needle in cand_norm or cand_norm in needle:
            return rect
    return None
