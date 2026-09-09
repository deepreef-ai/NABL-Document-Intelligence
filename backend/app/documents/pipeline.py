import logging
import mimetypes

from app.documents import call_budget as cb
from app.documents import combined_extraction
from app.documents import classifier, extractor, lab_report, local_ocr, pdf_utils
from app.documents.docx_utils import extract_text as extract_docx_text
from app.documents.geometry import Rect
from app.documents.grounding import FieldResult, PipelineResult, ground
from app.documents.ocr_client import OcrClient, OcrResult, SUPPORTED_SCRIPTS
from app.config import get_settings
from app.llm.factory import get_llm_chain

log = logging.getLogger(__name__)

# DOCX has no real "page" boundary — approximate one so a long filled-form
# DOCX still gets the chunked whole-form treatment (see
# extract_full_form_fields_chunked) instead of one giant single-shot prompt.
DOCX_CHUNK_CHARS = 4000

# A COST lever, not a correctness one. A born-digital PDF's text is free to
# pull out (PyMuPDF, no LLM call), but that text is then chunked by character
# count and each chunk costs an extraction call — so this bounds calls per
# upload, nothing else. A filled NABL form legitimately runs 20-30 pages,
# hence the generous value; 40 pages is ~15 chunks.
MAX_TEXT_PAGES = 50


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
    # One budget per document, created here and passed explicitly. NOT a
    # module-level counter: routers/documents.py runs uploads concurrently in
    # a thread pool, so shared mutable state would attribute one document's
    # calls to another.
    budget = cb.CallBudget.from_settings(document_id)

    if kind == "pdf":
        result = _process_pdf(data, script, ocr_client, form_type, document_id, filename, budget)
    elif kind == "docx":
        result = _process_docx(data, form_type, budget)
    else:
        result = _process_image(data, content_type or "image/jpeg", script, ocr_client, budget)

    if budget.stop_reason:
        # Recorded, never silent: "we stopped at the ceiling" must not read as
        # "the document genuinely had nothing more in it".
        result.extraction_warnings.append(budget.stop_reason)
    log.info("document %s: %s", document_id, budget.as_dict())
    return result


def _classify_and_extract_text(
    text: str, chunks: list[str], form_type: str, budget: cb.CallBudget
) -> tuple[str, float, list[dict], list[str]]:
    """Used by the DOCX path only — the PDF path (_process_pdf) does its own
    classify/branch inline since completed_application_form there needs real
    page objects/bboxes for the RAG pipeline, which a DOCX doesn't have."""
    doc_type, doc_confidence = classifier.classify(text, 1, form_type, budget=budget)
    warnings: list[str] = []
    if doc_type == "completed_application_form":
        fields, warnings = extractor.extract_full_form_fields_chunked(form_type, chunks)
    else:
        fields = extractor.extract_fields(doc_type, text)
    return doc_type, doc_confidence, fields, warnings


def _first_page_raster(data: bytes) -> bytes | None:
    """Page one as a PNG, or None if it cannot be rendered.

    Best-effort by design: a document whose first page will not rasterize must
    still extract from its text rather than fail. Page one only — the masthead
    is there, and one image keeps the payload within what every provider in
    the chain accepts (llm/chain.py takes a single image).
    """
    try:
        return pdf_utils.rasterize_page(data, 0)
    except Exception as exc:  # noqa: BLE001 — a raster failure must not sink extraction
        log.warning("could not rasterize page 1 for the vision payload: %s", exc)
        return None


def _process_pdf(
    data: bytes, script: str, ocr_client: OcrClient, form_type: str, document_id: str,
    filename: str = "", budget: cb.CallBudget | None = None,
) -> PipelineResult:
    budget = budget or cb.CallBudget.from_settings(document_id)
    if pdf_utils.has_text_layer(data):
        pages = pdf_utils.extract_text_and_boxes(data)[:MAX_TEXT_PAGES]
        full_text = "\n".join(p.text for p in pages)
        doc_type, doc_confidence = classifier.classify(full_text, len(pages), form_type, filename, budget)

        warnings: list[str] = []
        if doc_type == "completed_application_form":
            # Deliberately NOT given the open-ended pass below. That path
            # already asks for every one of the form's ~90 schema fields via
            # per-section retrieval, so there is little left for an open pass
            # to find — while a real 15-30 page form would add ~15 more LLM
            # calls (one per 7000-char chunk) to an already call-heavy path,
            # on a rate-limited free tier. Move this call outside the if/else
            # to match feat/token's "every document, no exceptions" behaviour.
            fields, warnings = _process_completed_application_form(data, pages, form_type, document_id, budget)
        else:
            # No separate extract_fields call: the schema slots are asked
            # for inside the same combined call below.
            raw_fields = []
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
            # A born-digital PDF still gets its FIRST PAGE rasterized and sent
            # alongside the text. MEASURED on the 53-document benchmark: a
            # lab's masthead and footer are frequently a graphic, so
            # lab_email / lab_address / cin / lab_telefax / lab_tagline are
            # absent from the PyMuPDF text entirely — on one Chennai Testing
            # Laboratory report, 5 of 9 golden letterhead fields were simply
            # not in the 2,679 characters we were sending. Text-only made them
            # unreachable rather than hard. Costs no extra LLM call: the image
            # rides the call that was already being made.
            open_results = _lab_report_fields(
                full_text, budget=budget, schema_field_paths=extractor.FIELD_SETS.get(doc_type),
                warnings=warnings, letterhead_image=_first_page_raster(data),
            )
            _ground_fields(open_results, candidates)
            fields.extend(open_results)
        return PipelineResult(doc_type, doc_confidence, "born_digital_pdf", fields, warnings)

    # Scanned PDF: rasterize each page and route it like a standalone image.
    total_pages = pdf_utils.page_count(data)
    n_pages = min(total_pages, get_settings().max_scanned_pages)
    all_fields: list[FieldResult] = []
    warnings: list[str] = []
    if n_pages < total_pages:
        # Recorded rather than silent: a truncated scan otherwise returns a
        # confident-looking result with no sign the remaining pages exist.
        warnings.append(f"only the first {n_pages} of {total_pages} pages were read (max_scanned_pages)")
        log.warning("scanned PDF truncated: read %d of %d pages", n_pages, total_pages)
    doc_type, doc_confidence, source = "other", 0.0, "vision_llm"
    for page_no in range(n_pages):
        png = pdf_utils.rasterize_page(data, page_no)
        result = _process_image(png, "image/png", script, ocr_client, budget)
        if page_no == 0:
            doc_type, doc_confidence, source = result.doc_type, result.doc_confidence, result.extraction_source
        for f in result.fields:
            f.source_page = page_no
        all_fields.extend(result.fields)
    return PipelineResult(doc_type, doc_confidence, source, all_fields, warnings)


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
    data: bytes, pages: list[pdf_utils.PageText], form_type: str, document_id: str, budget: cb.CallBudget
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
        if not budget.spend(cb.EXTRACTION):
            warnings.append(f"stopped before section {section!r}: {budget.stop_reason}")
            log.info("whole-form extraction stopped early: %s", budget.stop_reason)
            break
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
        # RECOVERY, not EXTRACTION: a bonus retry pass must never be able
        # to eat the allowance the main extraction needs.
        if not budget.spend(cb.RECOVERY):
            warnings.append(f"retry pass stopped: {budget.stop_reason}")
            break
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


# extract_lab_report does NOT truncate its input — but one call carrying a
# whole 40-page document would blow past the provider's context/output
# budget and return a reply cut off mid-JSON. Fed in chunks and merged
# instead, first occurrence of a repeated header field winning.
# Sized so the whole corpus fits inside max_initial_extraction_calls (4):
# the largest document is 32,465 chars (high-protein-paneer.pdf), which needs
# 4 chunks at 9,000 and 5 at 7,000. Input cost is trivial (~2.2k tokens); it
# is the OUTPUT budget (config.py's nova_max_tokens) that bounds a chunk.
LAB_REPORT_CHUNK_CHARS = 9000


# A chunk holding real text that yields almost no values is a FAILED
# extraction, not an empty page. Both thresholds are deliberately generous:
# the retry must fire on a collapse and stay quiet on a genuinely sparse
# document (a one-line certificate, a mostly-blank continuation page).
EMPTY_RETRY_MIN_CHARS = 400
EMPTY_RETRY_MIN_VALUES = 3


def _extraction_yield(extracted: dict) -> int:
    """How many values one chunk's extraction actually produced."""
    fields = len(extracted.get("fields") or {})
    schema = len(extracted.get("schema_fields") or {})
    cells = sum(
        1
        for row in (extracted.get("tests") or [])
        if isinstance(row, dict)
        for value in row.values()
        if str(value if value is not None else "").strip()
    )
    return fields + schema + cells


def _looks_collapsed(chunk: str, extracted: dict) -> bool:
    """True when this much source text should have produced more than this."""
    return (
        len(chunk.strip()) >= EMPTY_RETRY_MIN_CHARS
        and _extraction_yield(extracted) < EMPTY_RETRY_MIN_VALUES
    )


def _lab_report_fields(
    text: str, image: bytes | None = None, media_type: str | None = None,
    budget: cb.CallBudget | None = None, schema_field_paths: list[str] | None = None,
    warnings: list[str] | None = None, letterhead_image: bytes | None = None,
) -> list[FieldResult]:
    """Open-ended extraction for a document no FIELD_SETS entry covers —
    "other" and "quality_manual_sop", i.e. every lab/test report, which is not
    one of the NABL supporting-certificate types. Without this such a document
    produced nothing at all.

    Uses documents/lab_report.py's extract_lab_report — THE SAME function and
    prompt scripts/generate_predictions_and_score.py scores. That matters: the
    app and the benchmark previously ran two different prompts over the same
    kind of document, so every measured accuracy number described a code path
    users never hit.

    Never raises: this is additive, so a failure must leave the doc_type's own
    extraction untouched rather than sink the document.
    """
    budget = budget or cb.CallBudget.from_settings()
    chain = get_llm_chain()
    seen: set[str] = set()
    results: list[FieldResult] = []
    # Running per-attribute row offset across chunks. Each chunk's call
    # numbers its own rows from 0 (it cannot see what earlier chunks found),
    # so without re-indexing chunk 2's tests[0] collides with chunk 1's and is
    # dropped by the dedup below — MEASURED on a 5-chunk document: 10 rows
    # found, 2 kept.
    offsets: dict[str, int] = {}
    # One retry per DOCUMENT, not per chunk: RECOVERY holds 2 calls and the
    # letterhead pass needs one of them.
    retries_left = 1

    # An image with no text at all is a legitimate payload (the vision-only
    # fallback: no OCR text to pair with the page). Skipping blank chunks
    # would otherwise make that path extract nothing at all.
    # Carried out of the loop for the letterhead pass below, which needs the
    # page-one text and raster and must not depend on how many chunks ran.
    first_chunk_text, first_chunk_image, first_chunk_media = text, image, media_type

    chunks = _chunk_text(text, LAB_REPORT_CHUNK_CHARS)
    if image is not None and not any(c.strip() for c in chunks):
        chunks = [""]

    for chunk_index, chunk in enumerate(chunks):
        if not chunk.strip() and image is None:
            continue
        if not budget.spend(cb.EXTRACTION):
            log.info("stopping extraction early: %s", budget.stop_reason)
            break
        payload = combined_extraction.DocumentPayload(
            text_blocks=[chunk], image=image, media_type=media_type or "image/png",
        )
        try:
            extracted = combined_extraction.extract(chain, payload, schema_field_paths)
        except Exception as exc:  # noqa: BLE001 — see docstring
            log.warning("combined extraction failed on a chunk, skipping: %s", exc)
            if warnings is not None:
                # Otherwise the document reports status="extracted" with no
                # error and zero fields — a failed call looks exactly like a
                # document that genuinely had nothing in it.
                warnings.append(f"chunk {chunk_index} could not be extracted: {exc}")
            continue
        if (
            retries_left
            and _looks_collapsed(chunk, extracted)
            and get_settings().extraction_empty_retry
            and budget.spend(cb.RECOVERY)
        ):
            retries_left -= 1
            log.info(
                "chunk %d yielded %d values from %d chars — retrying once",
                chunk_index, _extraction_yield(extracted), len(chunk.strip()),
            )
            try:
                retried = combined_extraction.extract(chain, payload, schema_field_paths)
            except Exception as exc:  # noqa: BLE001 — a failed retry must not sink the chunk
                log.warning("empty-yield retry failed on chunk %d: %s", chunk_index, exc)
            else:
                if _extraction_yield(retried) > _extraction_yield(extracted):
                    extracted = retried
            if _looks_collapsed(chunk, extracted) and warnings is not None:
                # Surfaced only when the RETRY also came back empty: that is a
                # real quality signal for a reviewer. A retry that worked is
                # not an error and must not show up as one.
                warnings.append(
                    f"chunk {chunk_index} extracted almost nothing from "
                    f"{len(chunk.strip())} characters, twice"
                )

        extracted = combined_extraction.attach_confidence(extracted, chunk)

        # Named NABL schema slots keep source="llm" — they land in real schema
        # attributes via compiler.py. Everything else is open-ended.
        for path, value in (extracted.get("schema_fields") or {}).items():
            if path not in seen:
                seen.add(path)
                results.append(FieldResult(path, str(value), 1.0, source="llm"))

        for f in extractor.renumber_chunk_fields(lab_report.flatten_for_review(extracted), offsets):
            if f["field"] in seen:
                continue
            seen.add(f["field"])
            results.append(FieldResult(f["field"], f["value"], f["confidence"], source="open_extraction"))
        # Only the first chunk carries the page image; a later chunk's text
        # belongs to a different page, and pairing it with page 1's pixels
        # would invite the model to read values off the wrong page.
        if chunk_index == 0:
            first_chunk_text, first_chunk_image, first_chunk_media = chunk, image, media_type
        image, media_type = None, None

    # The letterhead pass gets pixels even when the main call did not: a
    # born-digital PDF's masthead is often a graphic, so its lab_email /
    # lab_address / cin are absent from the text layer entirely, while the
    # SAME image attached to the main call costs table rows (MEASURED: row
    # recall 0.983 -> 0.883). Different payloads for different jobs.
    settings = get_settings()
    results += _letterhead_fields(
        chain, budget, seen, first_chunk_text,
        (letterhead_image or first_chunk_image) if settings.letterhead_vision else None,
        first_chunk_media or "image/png",
    )
    return results


def _letterhead_fields(
    chain, budget: cb.CallBudget, seen: set[str],
    text: str, image: bytes | None, media_type: str | None,
) -> list[FieldResult]:
    """A second, FOCUSED call for the masthead and footer block.

    MEASURED on the 53-document benchmark: with the letterhead folded into the
    main combined call, 311 of 543 empty fields (52%) were lab-identity keys —
    lab_name, lab_email, lab_address, lab_phone, lab_website,
    lab_accreditation_no, cin_no, lab_tagline. The original finding holds: one
    call asked to transcribe a results table AND comb the masthead loses the
    masthead, because the table dominates its attention. Asking for nothing
    else is what recovers them.

    Runs ONCE per document, on the first chunk only — the masthead is on page
    one, so repeating it per chunk would multiply cost for nothing (the
    original per-chunk version did).

    Charged to RECOVERY rather than EXTRACTION: it is exactly that, and
    keeping it out of the extraction pool means a long document never trades
    a page of content for its letterhead.

    Never raises — additive, so a failure here must not cost the document.
    """
    if not get_settings().lab_report_letterhead_pass:
        return []
    if not (text or "").strip() and image is None:
        return []
    if not budget.spend(cb.RECOVERY):
        log.info("letterhead pass skipped: %s", budget.stop_reason)
        return []

    try:
        found = lab_report.extract_letterhead(chain, text, image=image, image_media_type=media_type)
    except Exception as exc:  # noqa: BLE001 — see docstring
        log.warning("letterhead pass failed, skipping: %s", exc)
        return []

    # Additive only: a field the main pass already returned is never
    # overwritten. It sees the whole page in context, so where the two
    # disagree it is the better source; this pass exists to fill gaps.
    merged = lab_report.merge_letterhead(
        {"fields": {}, "field_confidence": {}, "field_verified": {}, "tests": []}, found
    )
    merged = combined_extraction.attach_confidence(
        {"fields": merged.get("fields", {}), "tests": []}, text
    )

    added = []
    for f in lab_report.flatten_for_review(merged):
        if f["field"] in seen:
            continue
        seen.add(f["field"])
        added.append(FieldResult(f["field"], f["value"], f["confidence"], source="open_extraction"))
    if added:
        log.info("letterhead pass recovered %d field(s)", len(added))
    return added


def _ground_fields(open_results: list[FieldResult], candidates: list[tuple[str, Rect, int]]) -> None:
    """Give open fields the same page/bbox grounding every other field gets,
    so they highlight on the page in the review UI. In place."""
    for f in open_results:
        if not f.value:
            continue
        match = ground(f.value, [(text, rect) for text, rect, _ in candidates])
        if match:
            f.source_bbox = match
            f.source_page = next(page for _t, rect, page in candidates if rect is match)


def _process_docx(data: bytes, form_type: str, budget: cb.CallBudget | None = None) -> PipelineResult:
    budget = budget or cb.CallBudget.from_settings()
    text = extract_docx_text(data)
    doc_type, doc_confidence, raw_fields, warnings = _classify_and_extract_text(text, _chunk_text(text), form_type, budget)
    # No pixel geometry for a DOCX source — the review UI falls back to a
    # page-level (whole-document) highlight for these fields.
    fields = [FieldResult(f["field"], f["value"], f["confidence"]) for f in raw_fields]
    fields += _lab_report_fields(text, budget=budget, warnings=warnings)
    return PipelineResult(doc_type, doc_confidence, "docx", fields, warnings)


def _process_ocr_result(
    ocr_result: OcrResult, source: str, image_bytes: bytes, media_type: str, budget: cb.CallBudget,
) -> PipelineResult:
    doc_type, doc_confidence = classifier.classify(ocr_result.text, 1, "NABL_151", budget=budget)
    # Schema slots are asked for inside the combined call below.
    raw_fields = []
    candidates = list(zip(ocr_result.lines, ocr_result.boxes))
    fields = []
    for f in raw_fields:
        rect = ground(f["value"], candidates) if f["value"] else None
        fields.append(FieldResult(f["field"], f["value"], f["confidence"], 0 if rect else None, rect))
    warnings: list[str] = []
    for f in _lab_report_fields(
        ocr_result.text, image_bytes, media_type, budget, extractor.FIELD_SETS.get(doc_type),
        warnings=warnings,
    ):
        rect = ground(f.value, candidates) if f.value else None
        f.source_bbox, f.source_page = rect, 0 if rect else None
        fields.append(f)
    return PipelineResult(doc_type, doc_confidence, source, fields, warnings)


def _process_image(
    data: bytes, content_type: str, script: str, ocr_client: OcrClient,
    budget: cb.CallBudget | None = None,
) -> PipelineResult:
    budget = budget or cb.CallBudget.from_settings()
    media_type = content_type if content_type.startswith("image/") else "image/jpeg"

    if script in SUPPORTED_SCRIPTS:
        return _process_ocr_result(ocr_client.extract(data, script), f"ocr:{script}", data, media_type, budget)

    if script == "english":
        # Local RapidOCR (the same engine deepreef-ocr's Lambda runs, using
        # its own bundled English/Latin default model instead of one of
        # deepreef-ocr's baked non-English .onnx files) — real per-line
        # bounding boxes, no cloud call, no rate limit, no AWS dependency for
        # this script. Only falls through to the vision LLM below if it
        # itself isn't usable (not installed, corrupt image, etc.).
        try:
            return _process_ocr_result(local_ocr.extract_english(data), "rapidocr:english", data, media_type, budget)
        except local_ocr.LocalOcrError:
            pass

    # Any other script deepreef-ocr doesn't support, or a local-OCR failure:
    # there's no OCR text at all here, so read the page directly with a
    # vision-only LLM call. No per-field bbox available this way —
    # documented limitation, see the plan's "English-OCR gap" section (now
    # only reached when local OCR itself fails).
    doc_type, doc_confidence = (
        classifier.classify_image(data, media_type) if budget.spend(cb.CLASSIFICATION) else ("other", 0.0)
    )
    # Schema slots are asked for inside the combined call below.
    raw_fields = []
    fields = [FieldResult(f["field"], f["value"], f["confidence"]) for f in raw_fields]
    # No OCR text on this path — extract_lab_report reads the image directly.
    warnings: list[str] = []
    fields += _lab_report_fields(
        "", data, media_type, budget, extractor.FIELD_SETS.get(doc_type), warnings=warnings,
    )
    return PipelineResult(doc_type, doc_confidence, "vision_llm", fields, warnings)
