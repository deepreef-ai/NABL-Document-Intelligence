"""inspect -> classify -> retrieve -> chunk -> extract -> validate -> recover,
under a hard per-document LLM call budget.

Replaces the ad-hoc call sequence in documents/pipeline.py, where each stage
independently decided to call Nova and nothing knew the total. Every call
here goes through CallBudget, so a document cannot cost more than
max_total_llm_calls no matter which branches it takes, and the exact spend is
reported per document.

What it does NOT change: the shape of what the pipeline returns. It emits
FieldResults exactly as before, so compiler.py, the review UI and the
form-fill path are unaffected.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.config import get_settings
from app.documents import adaptive_chunking
from app.documents import call_budget as cb
from app.documents import (
    classifier,
    combined_extraction,
    deterministic_validation,
    page_inspection,
    recovery,
    retrieval,
)
from app.documents.grounding import FieldResult, ground
from app.documents.ocr_client import OcrClient
from app.documents.unified_extraction import DocumentPayload
from app.llm.factory import get_llm_chain

log = logging.getLogger(__name__)


@dataclass
class OrchestratedResult:
    doc_type: str = "other"
    doc_confidence: float = 0.0
    classification_method: str = "local"
    extraction_source: str = "born_digital_pdf"
    page_count: int = 0
    fields: list[FieldResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    call_log: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    relevant_pages: list[int] = field(default_factory=list)
    chunk_labels: list[str] = field(default_factory=list)


def _classify(inspected, form_type: str, filename: str, budget: cb.CallBudget) -> tuple[str, float, str]:
    """Local first, always. Nova only when local is not confident enough AND
    the budget allows — this is where most of the saving on short documents
    comes from, since every 1-page upload used to pay for a call here."""
    text = inspected.full_text
    doc_type, confidence = classifier.classify_locally_scored(
        text, inspected.page_count, form_type, filename
    )
    if confidence >= get_settings().local_classification_min_confidence:
        return doc_type, confidence, "local"

    if not budget.spend(cb.CLASSIFICATION):
        # Keep the local guess rather than failing: a low-confidence guess is
        # strictly better than none, and doc_type only selects which schema
        # slots to ASK for.
        return doc_type, confidence, "local_budget_capped"
    try:
        if text.strip():
            llm_type, llm_conf = classifier.classify_text(text)
        else:
            first = next((p.image for p in inspected.pages if p.image), None)
            if first is None:
                return doc_type, confidence, "local"
            llm_type, llm_conf = classifier.classify_image(first, "image/png")
        return llm_type, llm_conf, "llm"
    except Exception as exc:  # noqa: BLE001 — classification failure must not sink the upload
        log.warning("orchestrator: LLM classification failed, keeping local guess: %s", exc)
        return doc_type, confidence, "local_after_llm_failure"


def _schema_paths(doc_type: str, form_type: str) -> list[str]:
    """The named slots folded into the extraction call (spec section 6) — so
    the separate FIELD_SETS call disappears rather than being replaced."""
    from app.documents.extractor import FIELD_SETS, form_field_templates

    if doc_type == "completed_application_form":
        return form_field_templates(form_type)
    return list(FIELD_SETS.get(doc_type, []))


def run(
    data: bytes,
    suffix: str,
    filename: str = "",
    script: str = "english",
    ocr_client: OcrClient | None = None,
    form_type: str = "NABL_151",
    document_id: str = "unknown",
) -> OrchestratedResult:
    settings = get_settings()
    budget = cb.CallBudget.from_settings(document_id)

    # 1. Inspect — per page, no LLM.
    inspected = page_inspection.inspect(data, suffix, document_id, filename, script, ocr_client)
    out = OrchestratedResult(
        page_count=inspected.page_count,
        extraction_source=inspected.document.source_type,
        warnings=list(inspected.warnings),
    )

    # 2. Classify — local unless genuinely unsure.
    out.doc_type, out.doc_confidence, out.classification_method = _classify(
        inspected, form_type, filename, budget
    )
    schema_paths = _schema_paths(out.doc_type, form_type)

    # 3. Retrieve — which pages are plausible evidence.
    if len(inspected.pages) <= 1 or not schema_paths:
        relevant = [p.page_number for p in inspected.pages]
    else:
        relevant = retrieval.select_relevant_pages(
            document_id, schema_paths, [p.page for p in inspected.pages]
        )
    out.relevant_pages = sorted(relevant)

    # 4. Chunk — as few calls as the budget and context allow.
    chunks = adaptive_chunking.build_chunks(inspected.pages, relevant)
    out.chunk_labels = [c.label for c in chunks]

    # 5. Extract — one combined call per chunk.
    chain = get_llm_chain()
    fields: dict[str, str] = {}
    schema_fields: dict[str, str] = {}
    tests: list[dict] = []
    for chunk in chunks:
        if not budget.spend(cb.EXTRACTION):
            out.warnings.append(f"stopped before chunk {chunk.label}: {budget.stop_reason}")
            break
        payload = DocumentPayload(
            text_blocks=[chunk.text], images=chunk.images, media_type="image/png"
        )
        try:
            result = combined_extraction.extract(chain, payload, schema_paths)
        except Exception as exc:  # noqa: BLE001 — one chunk failing keeps the others
            log.warning("orchestrator: chunk %s extraction failed: %s", chunk.label, exc)
            out.warnings.append(f"chunk {chunk.label} extraction failed: {exc}")
            continue
        for k, v in result["fields"].items():
            fields.setdefault(k, v)
        for k, v in result["schema_fields"].items():
            schema_fields.setdefault(k, v)
        tests.extend(result["tests"])

    # 6. Validate — deterministic, no call.
    evidence = "\n".join(p.text or "" for p in inspected.pages)
    required = schema_paths[:5] if out.doc_type == "completed_application_form" else schema_paths
    validation = deterministic_validation.validate(
        {**fields, **schema_fields}, tests, evidence, required_fields=required
    )
    out.validation = validation.to_dict()

    # 7. Recover — one batched call for every gap, vision only if justified.
    if not validation.ok:
        vision_pages = recovery.needs_vision(
            inspected.pages, out.relevant_pages, has_suspicious=bool(validation.suspicious)
        )
        request = recovery.build_request(
            validation.missing,
            validation.suspicious,
            inspected.pages,
            out.relevant_pages,
            max_chars=settings.max_chunk_chars,
            include_images=bool(vision_pages),
        )
        recovered = recovery.recover(chain, request, budget)
        for k, v in recovered.items():
            if k in schema_fields or k in schema_paths:
                schema_fields[k] = v
            else:
                fields[k] = v
        if recovered:
            revalidated = deterministic_validation.validate(
                {**fields, **schema_fields}, tests, evidence, required_fields=required
            )
            out.validation = revalidated.to_dict()
            out.validation["recovered_fields"] = sorted(recovered)

    # 8. Emit FieldResults, grounded to a page/span for evidence.
    candidates = [(t, r, p.page_number) for p in inspected.pages for t, r in p.spans]
    text_by_page = {p.page_number: (p.text or "") for p in inspected.pages}
    ocr_conf_by_page = {p.page_number: p.page.ocr_confidence for p in inspected.pages}

    def emit(path: str, value, source: str) -> FieldResult:
        rect, page_no, source_text = None, None, None
        if value:
            match = ground(str(value), [(t, r) for t, r, _ in candidates])
            if match is not None:
                rect = match
                page_no = next(p for t, r, p in candidates if r is match)
                source_text = next((t for t, r, _ in candidates if r is match), None)
            else:
                needle = str(value).strip().lower()
                page_no = next(
                    (pn for pn, txt in text_by_page.items() if needle and needle in txt.lower()), None
                )
        return FieldResult(
            path,
            None if value is None else str(value),
            0.8,
            page_no,
            rect,
            source=source,
            source_text=source_text,
            ocr_confidence=ocr_conf_by_page.get(page_no) if page_no else None,
        )

    for path, value in schema_fields.items():
        out.fields.append(emit(path, value, "llm"))
    for key, value in fields.items():
        out.fields.append(emit(key, value, "open_extraction"))
    for i, row in enumerate(tests):
        for attr in ("test_name", "result", "unit", "reference_range"):
            if row.get(attr) is not None:
                out.fields.append(emit(f"tests[{i}].{attr}", row[attr], "open_extraction"))

    out.call_log = budget.as_dict()
    return out
