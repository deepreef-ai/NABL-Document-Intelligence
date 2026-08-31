import logging
import mimetypes

from app.documents import classifier, extractor, local_ocr, pdf_utils
from app.documents.docx_utils import extract_text as extract_docx_text
from app.documents.geometry import Rect
from app.documents.grounding import FieldResult, PipelineResult, ground
from app.documents.ocr_client import OcrClient, OcrResult, SUPPORTED_SCRIPTS

log = logging.getLogger(__name__)

# DOCX has no real "page" boundary — approximate one so a long filled-form
# DOCX still gets the chunked whole-form treatment (see
# extract_full_form_fields_chunked) instead of one giant single-shot prompt.
DOCX_CHUNK_CHARS = 4000

# Pages beyond this are still stored but not OCR'd/extracted from — keeps a
# pathological 200-page upload from burning the whole request budget. Applies
# to the per-page rasterize-and-vision path (each page is its own LLM call,
# so the cap has to stay small).
MAX_PAGES = 5

# A born-digital PDF's text is cheap to pull out (PyMuPDF, no LLM call), and a
# document that's actually a filled copy of the NABL application form itself
# can legitimately run 20-30 pages — 5 pages of a real one is just the
# amendment sheet and table of contents, not the applicant's data. Much
# larger cap for this path only.
MAX_TEXT_PAGES = 40


def _guess_kind(filename: str, content_type: str) -> str:
    content_type = content_type or mimetypes.guess_type(filename)[0] or ""
    if "pdf" in content_type or filename.lower().endswith(".pdf"):
        return "pdf"
    if "wordprocessingml" in content_type or filename.lower().endswith(".docx"):
        return "docx"
    return "image"


def process_document(
    data: bytes,
    filename: str,
    content_type: str,
    script: str = "english",
    ocr_client: OcrClient | None = None,
    form_type: str = "NABL_151",
    document_id: str = "unknown",
) -> PipelineResult:
    ocr_client = ocr_client or OcrClient()
    kind = _guess_kind(filename, content_type)

    if kind == "pdf":
        return _process_pdf(data, script, ocr_client, form_type, document_id)
    if kind == "docx":
        return _process_docx(data, form_type)
    return _process_image(data, content_type or "image/jpeg", script, ocr_client)


def _classify_and_extract_text(
    text: str, chunks: list[str], form_type: str
) -> tuple[str, float, list[dict], list[str]]:
    """Used by the DOCX path only — the PDF path (_process_pdf) does its own
    classify/branch inline since completed_application_form there needs real
    page objects/bboxes for the RAG pipeline, which a DOCX doesn't have."""
    doc_type, doc_confidence = classifier.classify_text(text)
    warnings: list[str] = []
    if doc_type == "completed_application_form":
        fields, warnings = extractor.extract_full_form_fields_chunked(form_type, chunks)
    else:
        fields = extractor.extract_fields(doc_type, text)
    return doc_type, doc_confidence, fields, warnings


def _process_pdf(data: bytes, script: str, ocr_client: OcrClient, form_type: str, document_id: str) -> PipelineResult:
    if pdf_utils.has_text_layer(data):
        pages = pdf_utils.extract_text_and_boxes(data)[:MAX_TEXT_PAGES]
        full_text = "\n".join(p.text for p in pages)
        local_result = classifier.classify_locally(full_text, len(pages), form_type)
        doc_type, doc_confidence = local_result or classifier.classify_text(full_text)

        warnings: list[str] = []
        if doc_type == "completed_application_form":
            fields, warnings = _process_completed_application_form(data, pages, form_type, document_id)
        else:
            raw_fields = extractor.extract_fields(doc_type, full_text)
            candidates: list[tuple[str, Rect, int]] = [
                (t, r, page.page_number) for page in pages for t, r in page.spans
            ]
            fields = []
            for f in raw_fields:
                rect, page_no = None, None
                if f["value"]:
                    match = ground(f["value"], [(t, r) for t, r, _ in candidates])
                    if match:
                        rect = match
                        page_no = next(p for t, r, p in candidates if r is match)
                fields.append(FieldResult(f["field"], f["value"], f["confidence"], page_no, rect))
        return PipelineResult(doc_type, doc_confidence, "born_digital_pdf", fields, warnings)

    # Scanned PDF: rasterize each page and route it like a standalone image.
    n_pages = min(pdf_utils.page_count(data), MAX_PAGES)
    all_fields: list[FieldResult] = []
    doc_type, doc_confidence, source = "other", 0.0, "vision_llm"
    for page_no in range(n_pages):
        png = pdf_utils.rasterize_page(data, page_no)
        result = _process_image(png, "image/png", script, ocr_client)
        if page_no == 0:
            doc_type, doc_confidence, source = result.doc_type, result.doc_confidence, result.extraction_source
        for f in result.fields:
            f.source_page = page_no
        all_fields.extend(result.fields)
    return PipelineResult(doc_type, doc_confidence, source, all_fields)


#  Below this combined character count across every page, per-section
#  retrieval buys nothing — every remaining field already fits comfortably
#  in one prompt, so a 1-3 page upload gets exactly ONE extraction call
#  instead of one per schema section (~8-10 calls for a typical NABL form).
#  Splitting by section exists specifically so a real 15-30 page filled form
#  doesn't have to cram everything into a single call/context window; a short
#  document has no such problem for the split to solve, and the per-section
#  free-tier LLM calls are exactly what exhausts a tiny daily rate-limit
#  budget fastest for no benefit.
_SINGLE_CALL_MAX_CHARS = 12000


def _process_completed_application_form(
    data: bytes, pages: list[pdf_utils.PageText], form_type: str, document_id: str
) -> tuple[list[FieldResult], list[str]]:
    """The RAG-assisted whole-form pipeline: rule-based identifiers first
    (no LLM call), then either one single extraction call for the whole
    document (short upload — see _SINGLE_CALL_MAX_CHARS) or per-SECTION
    semantic retrieval so only the pages actually relevant to a section
    reach the LLM (long upload), then a targeted retry pass for any field
    the main pass came back with nothing for.

    The per-section path costs one LLM call per schema section (using that
    section's top-K retrieved pages combined), not one per page — cheaper,
    and repeating entities (equipment[0], equipment[1], ...) are numbered
    correctly within that single call with no cross-call re-indexing needed,
    unlike the older per-page extract_full_form_fields_chunked path this
    replaces for PDFs.
    """
    from app.documents import chunking, retrieval, rule_extraction, verification

    chunks = chunking.build_chunks(data, pages)
    chunks_by_page = {c.page_number: c for c in chunks}

    rule_fields = rule_extraction.extract_identifiers(chunks)
    rule_field_paths = {f.field for f in rule_fields}

    field_templates = extractor.form_field_templates(form_type)
    remaining_templates = [t for t in field_templates if t not in rule_field_paths]
    combined_doc_text = "\n\n".join(c.text for c in chunks)

    single_call = len(combined_doc_text) <= _SINGLE_CALL_MAX_CHARS
    if single_call:
        # Short document: skip retrieval/indexing entirely too (embedding +
        # Qdrant would be pure overhead when every chunk is going in the one
        # call anyway) — one bucket covering every remaining field, using
        # every chunk.
        sections: dict[str, list[str]] = {"__all__": remaining_templates} if remaining_templates else {}
        section_chunks: dict[str, list] | None = {"__all__": chunks}
    else:
        retrieval.index_document_chunks(document_id, chunks)
        sections = retrieval.group_templates_by_section(field_templates)
        section_chunks = None  # retrieved per section below instead

    all_fields: list[FieldResult] = list(rule_fields)
    warnings: list[str] = []
    # Which section's retrieved-chunk text a field was (attempted to be)
    # extracted from — the retry pass below needs this to re-ask about a
    # missing field, since a field with no value never got grounded to a
    # source_page in the loop below.
    field_source_text: dict[str, str] = {}

    for section, templates in sections.items():
        remaining = [t for t in templates if t not in rule_field_paths]
        if not remaining:
            continue

        if section_chunks is not None:
            relevant_chunks = section_chunks[section]
        else:
            relevant_chunks = retrieval.retrieve_chunks_for_section(document_id, section, remaining, chunks_by_page)
            if not relevant_chunks:
                # No index yet (e.g. a single-page document) — fall back to
                # everything rather than silently skipping the section.
                relevant_chunks = list(chunks_by_page.values())

        texts = [c.text for c in relevant_chunks]
        combined_text = "\n\n".join(texts)
        try:
            raw_fields = extractor.extract_section_fields(remaining, texts)
        except Exception as exc:  # noqa: BLE001 — one section's LLM chain being
            # exhausted (e.g. every provider currently rate-limited) must not
            # discard every OTHER section's already-extracted fields, or the
            # rule-based pass's. Skip just this section and keep going; the
            # document is still saved as "extracted" with a note about what's
            # missing (see routers/documents.py's _run_pipeline) rather than
            # failing the whole upload over one section's outage.
            log.warning("completed_application_form: section %r extraction failed, skipping: %s", section, exc)
            warnings.append(f"section {section!r} could not be extracted: {exc}")
            continue

        candidates: list[tuple[str, Rect, int]] = [
            (t, r, c.page_number) for c in relevant_chunks for t, r in c.spans
        ]
        for f in raw_fields:
            field_source_text[f["field"]] = combined_text
            rect, page_no = None, None
            if f["value"]:
                match = ground(f["value"], [(t, r) for t, r, _ in candidates])
                if match:
                    rect = match
                    page_no = next(p for t, r, p in candidates if r is match)
            all_fields.append(FieldResult(f["field"], f["value"], f["confidence"], page_no, rect, source="llm"))

    # Targeted second-pass retry: only fields the main pass came back with
    # NOTHING for (missing/null) — one small single-field call each against
    # that field's section text, re-reading it rather than reprocessing the
    # whole document. Skipped entirely in single-call mode: every remaining
    # field was already asked about in the one call that just ran against
    # the *entire* document, so a missing field there is overwhelmingly
    # "genuinely not in this document" rather than "the model missed it
    # among too much else" — re-asking one-by-one would multiply right back
    # up to many calls, defeating the whole point of the single-call path.
    if single_call:
        return all_fields, warnings

    # Group missing fields by their shared source text (in practice, one
    # group per section that has any gaps) instead of one LLM call per
    # field — every field missing from the same section shares the exact
    # same retrieved text, so re-asking about each one individually was N
    # separate calls re-reading identical text N times. See verification.py.
    missing_by_source: dict[str, list[str]] = {}
    for f in all_fields:
        if f.source != "llm" or f.value:
            continue
        source_text = field_source_text.get(f.field)
        if not source_text:
            continue
        missing_by_source.setdefault(source_text, []).append(f.field)

    fields_by_path = {f.field: f for f in all_fields}
    for source_text, missing_paths in missing_by_source.items():
        try:
            results = verification.verify_fields(missing_paths, source_text)
        except Exception:  # noqa: BLE001 — this retry is a bonus pass; a failure here shouldn't sink extraction
            continue
        for r in results:
            f = fields_by_path.get(r["field"])
            if f is not None and r.get("value"):
                f.value = r["value"]
                f.confidence = r.get("confidence", f.confidence)
                f.source = "verification"

    return all_fields, warnings


def _chunk_text(text: str, chunk_size: int = DOCX_CHUNK_CHARS) -> list[str]:
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks or [text]


def _process_docx(data: bytes, form_type: str) -> PipelineResult:
    text = extract_docx_text(data)
    doc_type, doc_confidence, raw_fields, warnings = _classify_and_extract_text(text, _chunk_text(text), form_type)
    # No pixel geometry for a DOCX source — the review UI falls back to a
    # page-level (whole-document) highlight for these fields.
    fields = [FieldResult(f["field"], f["value"], f["confidence"]) for f in raw_fields]
    return PipelineResult(doc_type, doc_confidence, "docx", fields, warnings)


def _process_ocr_result(ocr_result: OcrResult, source: str) -> PipelineResult:
    doc_type, doc_confidence = classifier.classify_text(ocr_result.text)
    raw_fields = extractor.extract_fields(doc_type, ocr_result.text)
    candidates = list(zip(ocr_result.lines, ocr_result.boxes))
    fields = []
    for f in raw_fields:
        rect = ground(f["value"], candidates) if f["value"] else None
        fields.append(FieldResult(f["field"], f["value"], f["confidence"], 0 if rect else None, rect))
    return PipelineResult(doc_type, doc_confidence, source, fields)


def _process_image(data: bytes, content_type: str, script: str, ocr_client: OcrClient) -> PipelineResult:
    if script in SUPPORTED_SCRIPTS:
        return _process_ocr_result(ocr_client.extract(data, script), f"ocr:{script}")

    if script == "english":
        # Local RapidOCR (the same engine deepreef-ocr's Lambda runs, using
        # its own bundled English/Latin default model instead of one of
        # deepreef-ocr's baked non-English .onnx files) — real per-line
        # bounding boxes, no cloud call, no rate limit, no AWS dependency for
        # this script. Only falls through to the vision LLM below if it
        # itself isn't usable (not installed, corrupt image, etc.).
        try:
            return _process_ocr_result(local_ocr.extract_english(data), "rapidocr:english")
        except local_ocr.LocalOcrError:
            pass

    # Any other script deepreef-ocr doesn't support, or a local-OCR failure:
    # read the page directly with a vision LLM call. No per-field bbox
    # available this way — documented limitation, see the plan's "English-OCR
    # gap" section (now only reached when local OCR itself fails).
    media_type = content_type if content_type.startswith("image/") else "image/jpeg"
    doc_type, doc_confidence = classifier.classify_image(data, media_type)
    raw_fields = extractor.extract_fields_vision(doc_type, data, media_type)
    fields = [FieldResult(f["field"], f["value"], f["confidence"]) for f in raw_fields]
    return PipelineResult(doc_type, doc_confidence, "vision_llm", fields)
