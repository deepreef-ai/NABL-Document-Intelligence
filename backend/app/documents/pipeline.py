import logging
import mimetypes

from app.documents import classifier, extractor, local_ocr, pdf_utils, unified_extraction
from app.documents.docx_utils import extract_text as extract_docx_text
from app.documents.geometry import Rect
from app.documents.grounding import FieldResult, PipelineResult, ground
from app.documents.ocr_client import OcrClient, OcrResult, SUPPORTED_SCRIPTS
from app.llm.factory import get_llm_chain

log = logging.getLogger(__name__)

# DOCX has no real "page" boundary — approximate one so a long filled-form
# DOCX still gets the chunked whole-form treatment (see
# extract_full_form_fields_chunked) instead of one giant single-shot prompt.
DOCX_CHUNK_CHARS = 4000

# DOCX ONLY. PDFs and images go through documents/unified_extraction.py
# instead, which sends every page's text AND its image. A DOCX has no page
# raster to send, so it keeps this text-only path — chunked because
# extract_open_fields truncates its input to 8000 chars internally, which on
# a long document silently ignores everything past its first page or two.
OPEN_EXTRACTION_CHUNK_CHARS = 7000


# Open-ended/table extraction has no schema to validate a value against, so
# every field it produces lands just under the review UI's
# confidence_threshold (0.85) on purpose — a reviewer should confirm these,
# not have them silently auto-accepted. The shared prompt (see
# unified_extraction.SYSTEM_PROMPT) deliberately does not ask the model to
# self-report a confidence: that number is the model's opinion of its own
# output, and the whole point of the review step is not to take that on
# trust.
UNIFIED_FIELD_CONFIDENCE = 0.8


def _unified_field_results(
    data: bytes, suffix: str, script: str, ocr_client: OcrClient | None, ground_fn, warnings: list[str]
) -> list[FieldResult]:
    """Runs documents/unified_extraction.py — the same module, prompt and
    page handling scripts/generate_predictions_and_score.py uses — and
    converts its {fields, tests} result into FieldResults.

    Table rows are flattened to "tests[i].result" etc., the same "attr[i]."
    convention compiler.py already understands, and everything is tagged
    source="open_extraction" so compiler.py routes it to the compiled form's
    extra_fields bucket rather than trying to match it to a named schema
    slot it was never meant for.
    """
    try:
        payload = unified_extraction.build_payload(data, suffix, script=script, ocr_client=ocr_client)
        result = unified_extraction.extract(get_llm_chain(), payload)
    except Exception as exc:  # noqa: BLE001 — the schema-guided fields already
        # extracted must survive this call failing (every provider rate-limited,
        # an unrenderable page, ...). The document is still saved as extracted.
        log.warning("unified extraction failed, keeping schema-guided fields only: %s", exc)
        warnings.append(f"open-ended/table extraction unavailable: {exc}")
        return []

    warnings.extend(payload.warnings)

    raw: list[dict] = [
        {"field": key, "value": value, "confidence": UNIFIED_FIELD_CONFIDENCE}
        for key, value in result["fields"].items()
    ]
    for i, row in enumerate(result["tests"]):
        for attr in ("test_name", "result", "unit", "reference_range"):
            if row.get(attr) is not None:
                raw.append({"field": f"tests[{i}].{attr}", "value": row[attr], "confidence": UNIFIED_FIELD_CONFIDENCE})

    return ground_fn(raw, "open_extraction")


def _extract_open_fields_chunked(text: str) -> list[dict]:
    seen: set[str] = set()
    combined: list[dict] = []
    for chunk in _chunk_text(text, OPEN_EXTRACTION_CHUNK_CHARS):
        if not chunk.strip():
            continue
        for f in extractor.extract_open_fields(chunk):
            if f["field"] in seen:
                continue
            seen.add(f["field"])
            combined.append(f)
    return combined

# No page cap on either PDF path, by design. There used to be two (40 pages
# for born-digital text, 5 for scanned) and both silently truncated real
# documents: a 17-page report read as 5 pages still produces a confident-
# looking result from a fraction of the input, which is worse than being
# slow. Cost is bounded by the extraction strategy instead — text pulling is
# free (PyMuPDF), per-page OCR is local/free, and the LLM calls are grouped
# per schema section rather than per page (see
# _process_completed_application_form), so page count drives OCR time but
# NOT the number of LLM calls.

# A page whose text layer yields fewer than this many characters is treated
# as having no usable text and gets OCR'd instead. Same threshold
# documents/chunking.py uses, so a page OCR'd here won't be re-OCR'd there.
MIN_PAGE_TEXT_CHARS = 20


def _image_suffix(content_type: str) -> str:
    """unified_extraction.build_payload routes on a file extension; an upload
    only carries a MIME type, so map one to the other."""
    return ".jpg" if "jpeg" in (content_type or "") or "jpg" in (content_type or "") else ".png"


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


def _read_pdf_pages(
    data: bytes, script: str, ocr_client: OcrClient
) -> tuple[list[pdf_utils.PageText], str, list[str]]:
    """EVERY page of the PDF, as text, whatever it takes to get there: a real
    text layer where one exists, OCR of the rasterized page where it doesn't.

    This is deliberately ONE path for born-digital, scanned, and mixed PDFs.
    The scanned case used to be handled separately — rasterize each page and
    route it through _process_image as if it were a standalone photo — which
    meant a scanned filled-in form could never reach the whole-form
    extraction path at all: each page got classified on its own and matched
    against one narrow doc_type's FIELD_SETS, so a 20-page scanned
    application form came back with a handful of certificate fields instead
    of the form's actual sections. Getting every page to text FIRST means a
    scanned document is extracted exactly as well as a born-digital one.

    Returns (pages, extraction_source, warnings). extraction_source
    distinguishes born_digital_pdf / ocr_pdf / mixed_pdf so the review UI can
    show where a value actually came from.
    """
    pages_by_number = {p.page_number: p for p in pdf_utils.extract_text_and_boxes(data)}
    count = pdf_utils.page_count(data)
    pages: list[pdf_utils.PageText] = []
    warnings: list[str] = []
    used_text = used_ocr = False

    for i in range(count):
        page = pages_by_number.get(i)
        if page is not None and len(page.text.strip()) >= MIN_PAGE_TEXT_CHARS:
            pages.append(page)
            used_text = True
            continue

        try:
            png = pdf_utils.rasterize_page(data, i)
            ocr = (
                local_ocr.extract_english(png)
                if script not in SUPPORTED_SCRIPTS
                else ocr_client.extract(png, script)
            )
            pages.append(pdf_utils.PageText(
                page_number=i, text=ocr.text, spans=list(zip(ocr.lines, ocr.boxes)),
            ))
            used_ocr = True
        except Exception as exc:  # noqa: BLE001 — one unreadable page must not lose the other 19
            log.warning("page %d could not be OCR'd, keeping it empty: %s", i, exc)
            warnings.append(f"page {i + 1} could not be read (no text layer and OCR failed): {exc}")
            pages.append(page or pdf_utils.PageText(page_number=i, text="", spans=[]))

    source = "mixed_pdf" if (used_text and used_ocr) else ("ocr_pdf" if used_ocr else "born_digital_pdf")
    return pages, source, warnings


def _process_pdf(data: bytes, script: str, ocr_client: OcrClient, form_type: str, document_id: str) -> PipelineResult:
    pages, source, warnings = _read_pdf_pages(data, script, ocr_client)
    full_text = "\n".join(p.text for p in pages)

    local_result = classifier.classify_locally(full_text, len(pages), form_type)
    doc_type, doc_confidence = local_result or classifier.classify_text(full_text)

    candidates: list[tuple[str, Rect, int]] = [
        (t, r, page.page_number) for page in pages for t, r in page.spans
    ]

    def _ground_and_wrap(raw: list[dict], field_source: str) -> list[FieldResult]:
        wrapped = []
        for f in raw:
            rect, page_no = None, None
            if f["value"]:
                match = ground(f["value"], [(t, r) for t, r, _ in candidates])
                if match:
                    rect = match
                    page_no = next(p for t, r, p in candidates if r is match)
            wrapped.append(FieldResult(f["field"], f["value"], f["confidence"], page_no, rect, source=field_source))
        return wrapped

    if doc_type == "completed_application_form":
        fields, extraction_warnings = _process_completed_application_form(data, pages, form_type, document_id)
        warnings = warnings + extraction_warnings
    else:
        fields = _ground_and_wrap(extractor.extract_fields(doc_type, full_text), "llm")

    # Open-ended extraction runs unconditionally, on top of whatever the
    # branch above already found (including "other"/quality_manual_sop,
    # whose FIELD_SETS are empty — this is their only source of fields).
    # See documents/extractor.py's extract_open_fields and
    # documents/compiler.py's routing of source="open_extraction" fields
    # into the compiled form's extra_fields bucket.
    fields = fields + _unified_field_results(data, ".pdf", script, ocr_client, _ground_and_wrap, warnings)

    return PipelineResult(doc_type, doc_confidence, source, fields, warnings, page_count=len(pages))


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
    fields += [
        FieldResult(f["field"], f["value"], f["confidence"], source="open_extraction")
        for f in _extract_open_fields_chunked(text)
    ]
    return PipelineResult(doc_type, doc_confidence, "docx", fields, warnings)  # DOCX has no page geometry


def _process_ocr_result(
    ocr_result: OcrResult, source: str, image_bytes: bytes, suffix: str,
    script: str = "english", ocr_client: OcrClient | None = None,
) -> PipelineResult:
    doc_type, doc_confidence = classifier.classify_text(ocr_result.text)
    candidates = list(zip(ocr_result.lines, ocr_result.boxes))

    def _ground_and_wrap(raw: list[dict], field_source: str) -> list[FieldResult]:
        wrapped = []
        for f in raw:
            rect = ground(f["value"], candidates) if f["value"] else None
            wrapped.append(FieldResult(f["field"], f["value"], f["confidence"], 0 if rect else None, rect, source=field_source))
        return wrapped

    fields = _ground_and_wrap(extractor.extract_fields(doc_type, ocr_result.text), "llm")
    warnings: list[str] = []
    fields += _unified_field_results(image_bytes, suffix, script, ocr_client, _ground_and_wrap, warnings)
    return PipelineResult(doc_type, doc_confidence, source, fields, warnings, page_count=1)


def _process_image(data: bytes, content_type: str, script: str, ocr_client: OcrClient) -> PipelineResult:
    if script in SUPPORTED_SCRIPTS:
        return _process_ocr_result(ocr_client.extract(data, script), f"ocr:{script}", data, _image_suffix(content_type), script, ocr_client)

    if script == "english":
        # Local RapidOCR (the same engine deepreef-ocr's Lambda runs, using
        # its own bundled English/Latin default model instead of one of
        # deepreef-ocr's baked non-English .onnx files) — real per-line
        # bounding boxes, no cloud call, no rate limit, no AWS dependency for
        # this script. Only falls through to the vision LLM below if it
        # itself isn't usable (not installed, corrupt image, etc.).
        try:
            return _process_ocr_result(local_ocr.extract_english(data), "rapidocr:english", data, _image_suffix(content_type), script, ocr_client)
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
    warnings: list[str] = []
    fields += _unified_field_results(
        data, _image_suffix(media_type), script, ocr_client, lambda raw, src: [
            FieldResult(f["field"], f["value"], f["confidence"], source=src) for f in raw
        ], warnings,
    )
    return PipelineResult(doc_type, doc_confidence, "vision_llm", fields, warnings, page_count=1)
