from types import SimpleNamespace

from app.documents import chunking, classifier, extractor, local_ocr, pdf_utils, pipeline, retrieval, rule_extraction, verification
from app.documents.geometry import Rect
from app.documents.ocr_client import OcrResult
from app.documents.pipeline import _guess_kind, process_document


def test_guess_kind_by_content_type_and_extension():
    assert _guess_kind("cert.pdf", "application/pdf") == "pdf"
    assert _guess_kind("cert.PDF", "") == "pdf"
    assert _guess_kind("cv.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document") == "docx"
    assert _guess_kind("photo.jpg", "image/jpeg") == "image"


def test_born_digital_pdf_path_grounds_extracted_value_to_its_span(monkeypatch):
    fake_page = SimpleNamespace(
        page_number=0,
        text="GST Number: 27ABCDE1234F1Z5",
        spans=[("GST Number: 27ABCDE1234F1Z5", Rect(x=1, y=2, w=3, h=4))],
    )
    monkeypatch.setattr(pdf_utils, "has_text_layer", lambda data: True)
    monkeypatch.setattr(pdf_utils, "extract_text_and_boxes", lambda data: [fake_page])
    monkeypatch.setattr(classifier, "classify_text", lambda text: ("legal_proof", 0.9))
    monkeypatch.setattr(
        extractor,
        "extract_fields",
        lambda doc_type, text: [
            {"field": "organisation.gst_number", "value": "GST Number: 27ABCDE1234F1Z5", "confidence": 0.95}
        ],
    )

    result = process_document(b"%PDF-fake", "cert.pdf", "application/pdf", script="english")

    assert result.doc_type == "legal_proof"
    assert result.extraction_source == "born_digital_pdf"
    assert result.fields[0].source_bbox == Rect(x=1, y=2, w=3, h=4)
    assert result.fields[0].source_page == 0


def test_docx_path_has_no_bbox(monkeypatch):
    monkeypatch.setattr("app.documents.pipeline.extract_docx_text", lambda data: "Name: Jane Doe")
    monkeypatch.setattr(classifier, "classify_text", lambda text: ("staff_cv_certificate", 0.8))
    monkeypatch.setattr(
        extractor, "extract_fields", lambda doc_type, text: [{"field": "name", "value": "Jane Doe", "confidence": 0.7}]
    )

    result = process_document(b"docxbytes", "cv.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

    assert result.extraction_source == "docx"
    assert result.fields[0].source_bbox is None


def test_supported_script_image_calls_the_ocr_client_not_vision(monkeypatch):
    calls = {"vision": 0}
    fake_ocr = SimpleNamespace(
        extract=lambda data, script: OcrResult(
            text="राम", lines=["राम"], confidence=0.8, boxes=[Rect(x=0, y=0, w=5, h=5)], model_used="devanagari_rec.onnx", region_count=1
        )
    )
    monkeypatch.setattr(classifier, "classify_text", lambda text: ("other", 0.5))
    monkeypatch.setattr(extractor, "extract_fields", lambda doc_type, text: [])
    monkeypatch.setattr(extractor, "extract_fields_vision", lambda *a, **k: calls.__setitem__("vision", calls["vision"] + 1))

    result = process_document(b"imgbytes", "board.jpg", "image/jpeg", script="devanagari", ocr_client=fake_ocr)

    assert result.extraction_source == "ocr:devanagari"
    assert calls["vision"] == 0


def test_completed_application_form_routes_to_rag_pipeline(monkeypatch):
    monkeypatch.setattr(pipeline, "_SINGLE_CALL_MAX_CHARS", 0)  # force the per-section path
    fake_pages = [
        SimpleNamespace(page_number=0, text="page one text", spans=[]),
        SimpleNamespace(page_number=1, text="page two text", spans=[]),
    ]
    monkeypatch.setattr(pdf_utils, "has_text_layer", lambda data: True)
    monkeypatch.setattr(pdf_utils, "extract_text_and_boxes", lambda data: fake_pages)
    monkeypatch.setattr(classifier, "classify_text", lambda text: ("completed_application_form", 0.9))

    fake_chunks = [
        chunking.Chunk(page_number=0, text="page one text", spans=[]),
        chunking.Chunk(page_number=1, text="page two text", spans=[]),
    ]
    monkeypatch.setattr(chunking, "build_chunks", lambda data, pages: fake_chunks)
    monkeypatch.setattr(rule_extraction, "extract_identifiers", lambda chunks: [])
    monkeypatch.setattr(retrieval, "index_document_chunks", lambda document_id, chunks: None)
    monkeypatch.setattr(
        retrieval,
        "group_templates_by_section",
        lambda templates: {"organisation": ["organisation.gst_number"], "equipment": ["equipment[0].name"]},
    )
    monkeypatch.setattr(
        retrieval,
        "retrieve_chunks_for_section",
        lambda document_id, section, templates, chunks_by_page: list(chunks_by_page.values()),
    )

    def fake_extract_section(field_templates, texts):
        if "organisation.gst_number" in field_templates:
            return [{"field": "organisation.gst_number", "value": "27ABCDE1234F1Z5", "confidence": 0.9}]
        return [{"field": "equipment[0].name", "value": "Digital Multimeter", "confidence": 0.85}]

    monkeypatch.setattr(extractor, "extract_section_fields", fake_extract_section)
    monkeypatch.setattr(
        extractor,
        "extract_fields",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should use the RAG whole-form path, not extract_fields")),
    )

    result = process_document(
        b"%PDF-fake", "filled-application.pdf", "application/pdf", form_type="NABL_151", document_id="doc1"
    )

    assert result.doc_type == "completed_application_form"
    assert {f.field for f in result.fields} == {"organisation.gst_number", "equipment[0].name"}
    assert {f.source for f in result.fields} == {"llm"}


def test_rag_pipeline_keeps_other_sections_when_one_sections_llm_chain_is_exhausted(monkeypatch):
    """A section whose LLM chain runs out of providers (e.g. everything
    rate-limited at that moment) must not wipe out fields another section
    already extracted successfully — see pipeline.py's
    _process_completed_application_form and the try/except around
    extract_section_fields."""
    from app.llm.base import LlmProviderError

    monkeypatch.setattr(pipeline, "_SINGLE_CALL_MAX_CHARS", 0)  # force the per-section path
    fake_pages = [
        SimpleNamespace(page_number=0, text="page one text", spans=[]),
        SimpleNamespace(page_number=1, text="page two text", spans=[]),
    ]
    monkeypatch.setattr(pdf_utils, "has_text_layer", lambda data: True)
    monkeypatch.setattr(pdf_utils, "extract_text_and_boxes", lambda data: fake_pages)
    monkeypatch.setattr(classifier, "classify_text", lambda text: ("completed_application_form", 0.9))

    fake_chunks = [
        chunking.Chunk(page_number=0, text="page one text", spans=[]),
        chunking.Chunk(page_number=1, text="page two text", spans=[]),
    ]
    monkeypatch.setattr(chunking, "build_chunks", lambda data, pages: fake_chunks)
    monkeypatch.setattr(rule_extraction, "extract_identifiers", lambda chunks: [])
    monkeypatch.setattr(retrieval, "index_document_chunks", lambda document_id, chunks: None)
    monkeypatch.setattr(
        retrieval,
        "group_templates_by_section",
        lambda templates: {"organisation": ["organisation.gst_number"], "equipment": ["equipment[0].name"]},
    )
    monkeypatch.setattr(
        retrieval,
        "retrieve_chunks_for_section",
        lambda document_id, section, templates, chunks_by_page: list(chunks_by_page.values()),
    )

    def flaky_extract_section(field_templates, texts):
        if "equipment[0].name" in field_templates:
            raise LlmProviderError("every configured LLM provider failed: groq: 429 | ollama: timeout")
        return [{"field": "organisation.gst_number", "value": "27ABCDE1234F1Z5", "confidence": 0.9}]

    monkeypatch.setattr(extractor, "extract_section_fields", flaky_extract_section)

    result = process_document(
        b"%PDF-fake", "filled-application.pdf", "application/pdf", form_type="NABL_151", document_id="doc1"
    )

    # The organisation section's field survived even though equipment's
    # extraction blew up entirely.
    assert {f.field for f in result.fields} == {"organisation.gst_number"}
    assert len(result.extraction_warnings) == 1
    assert "equipment" in result.extraction_warnings[0]


def test_rag_pipeline_retries_only_fields_missing_a_value_not_low_confidence_ones(monkeypatch):
    monkeypatch.setattr(pipeline, "_SINGLE_CALL_MAX_CHARS", 0)  # force the per-section path (single-call skips retry)
    fake_pages = [SimpleNamespace(page_number=0, text="page one text", spans=[])]
    monkeypatch.setattr(pdf_utils, "has_text_layer", lambda data: True)
    monkeypatch.setattr(pdf_utils, "extract_text_and_boxes", lambda data: fake_pages)
    monkeypatch.setattr(classifier, "classify_text", lambda text: ("completed_application_form", 0.9))

    fake_chunks = [chunking.Chunk(page_number=0, text="page one text", spans=[])]
    monkeypatch.setattr(chunking, "build_chunks", lambda data, pages: fake_chunks)
    monkeypatch.setattr(rule_extraction, "extract_identifiers", lambda chunks: [])
    monkeypatch.setattr(retrieval, "index_document_chunks", lambda document_id, chunks: None)
    monkeypatch.setattr(
        retrieval,
        "group_templates_by_section",
        lambda templates: {"organisation": ["organisation.gst_number", "organisation.pan_number"]},
    )
    monkeypatch.setattr(
        retrieval,
        "retrieve_chunks_for_section",
        lambda document_id, section, templates, chunks_by_page: list(chunks_by_page.values()),
    )
    monkeypatch.setattr(
        extractor,
        "extract_section_fields",
        lambda field_templates, texts: [
            # Low confidence but a real value present — must NOT be retried.
            {"field": "organisation.gst_number", "value": "27ABCDE1234F1Z5", "confidence": 0.2},
            # No value at all — must be retried.
            {"field": "organisation.pan_number", "value": None, "confidence": 0.0},
        ],
    )

    verify_calls = []

    def fake_verify(field_paths, source_text):
        verify_calls.append(field_paths)
        return [{"field": "organisation.pan_number", "value": "ABCDE1234F", "confidence": 0.95}]

    monkeypatch.setattr(verification, "verify_fields", fake_verify)

    result = process_document(
        b"%PDF-fake", "filled-application.pdf", "application/pdf", form_type="NABL_151", document_id="doc1"
    )

    assert verify_calls == [["organisation.pan_number"]]  # only the missing field was retried, in ONE batched call
    by_field = {f.field: f for f in result.fields}
    assert by_field["organisation.gst_number"].confidence == 0.2  # untouched despite low confidence
    assert by_field["organisation.gst_number"].source == "llm"
    assert by_field["organisation.pan_number"].value == "ABCDE1234F"
    assert by_field["organisation.pan_number"].source == "verification"


def test_retry_pass_batches_every_missing_field_in_a_section_into_one_call(monkeypatch):
    """The actual fix: N missing fields sharing the same section's source
    text must cost ONE verify_fields call, not N individual ones — this is
    what turns a 60-70 call retry pass into single digits on a real form."""
    # Long enough to exceed pipeline._SINGLE_CALL_MAX_CHARS, so this goes
    # through the per-section (retry-eligible) path rather than the
    # short-document single-call fast path, which skips retry entirely.
    long_text = "page one text. " * 1000
    fake_pages = [SimpleNamespace(page_number=0, text=long_text, spans=[])]
    monkeypatch.setattr(pdf_utils, "has_text_layer", lambda data: True)
    monkeypatch.setattr(pdf_utils, "extract_text_and_boxes", lambda data: fake_pages)
    monkeypatch.setattr(classifier, "classify_text", lambda text: ("completed_application_form", 0.9))

    fake_chunks = [chunking.Chunk(page_number=0, text=long_text, spans=[])]
    monkeypatch.setattr(chunking, "build_chunks", lambda data, pages: fake_chunks)
    monkeypatch.setattr(rule_extraction, "extract_identifiers", lambda chunks: [])
    monkeypatch.setattr(retrieval, "index_document_chunks", lambda document_id, chunks: None)
    monkeypatch.setattr(
        retrieval,
        "group_templates_by_section",
        lambda templates: {
            "organisation": [
                "organisation.gst_number", "organisation.pan_number",
                "organisation.tan_number", "organisation.telephone",
            ]
        },
    )
    monkeypatch.setattr(
        retrieval,
        "retrieve_chunks_for_section",
        lambda document_id, section, templates, chunks_by_page: list(chunks_by_page.values()),
    )
    # Only one of the four requested fields has a real value; the other
    # three come back None (simulating extract_section_fields's own
    # backfill of fields the LLM didn't find — see extractor.py's
    # _fill_missing_flat_fields, not exercised here since this mock
    # replaces the real function entirely) and should all retry together.
    monkeypatch.setattr(
        extractor, "extract_section_fields",
        lambda field_templates, texts: [
            {"field": "organisation.gst_number", "value": "27ABCDE1234F1Z5", "confidence": 0.9},
            {"field": "organisation.pan_number", "value": None, "confidence": 0.0},
            {"field": "organisation.tan_number", "value": None, "confidence": 0.0},
            {"field": "organisation.telephone", "value": None, "confidence": 0.0},
        ],
    )

    verify_calls = []

    def fake_verify(field_paths, source_text):
        verify_calls.append(sorted(field_paths))
        return [{"field": "organisation.pan_number", "value": "ABCDE1234F", "confidence": 0.9}]

    monkeypatch.setattr(verification, "verify_fields", fake_verify)

    process_document(b"%PDF-fake", "filled-application.pdf", "application/pdf", form_type="NABL_151", document_id="doc1")

    assert len(verify_calls) == 1  # ONE call, not three
    assert verify_calls[0] == ["organisation.pan_number", "organisation.tan_number", "organisation.telephone"]


def test_short_document_uses_a_single_extraction_call_for_every_section(monkeypatch):
    """A 1-3 page upload has no need for the per-section split — everything
    fits in one prompt, so it should cost exactly one extract_section_fields
    call covering every remaining field, with retrieval/indexing and the
    per-field retry pass skipped entirely (see pipeline._SINGLE_CALL_MAX_CHARS)."""
    fake_pages = [SimpleNamespace(page_number=0, text="short page text", spans=[])]
    monkeypatch.setattr(pdf_utils, "has_text_layer", lambda data: True)
    monkeypatch.setattr(pdf_utils, "extract_text_and_boxes", lambda data: fake_pages)
    monkeypatch.setattr(classifier, "classify_text", lambda text: ("completed_application_form", 0.9))

    fake_chunks = [chunking.Chunk(page_number=0, text="short page text", spans=[])]
    monkeypatch.setattr(chunking, "build_chunks", lambda data, pages: fake_chunks)
    monkeypatch.setattr(rule_extraction, "extract_identifiers", lambda chunks: [])

    def fail(*a, **k):
        raise AssertionError("retrieval/indexing must be skipped entirely for a short document")

    monkeypatch.setattr(retrieval, "index_document_chunks", fail)
    monkeypatch.setattr(retrieval, "group_templates_by_section", fail)
    monkeypatch.setattr(retrieval, "retrieve_chunks_for_section", fail)

    calls = []

    def fake_extract_section(field_templates, texts):
        calls.append(field_templates)
        return [{"field": "organisation.gst_number", "value": "27ABCDE1234F1Z5", "confidence": 0.9}]
        # Every OTHER requested field is left missing on purpose — the retry
        # pass must NOT fire for any of them in single-call mode.

    monkeypatch.setattr(extractor, "extract_section_fields", fake_extract_section)
    monkeypatch.setattr(
        verification,
        "verify_fields",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("single-call mode must skip the retry pass")),
    )

    result = process_document(
        b"%PDF-fake", "filled-application.pdf", "application/pdf", form_type="NABL_151", document_id="doc1"
    )

    assert len(calls) == 1  # exactly one extraction call for the whole document
    assert "organisation.gst_number" in calls[0]
    assert "equipment[i].name" in calls[0]  # every remaining field across every section was in that one call
    by_field = {f.field: f for f in result.fields}
    assert by_field["organisation.gst_number"].value == "27ABCDE1234F1Z5"


def test_english_image_prefers_local_rapidocr_over_vision(monkeypatch):
    calls = {"vision": 0}
    monkeypatch.setattr(
        local_ocr,
        "extract_english",
        lambda data: OcrResult(
            text="GST Number 27ABCDE1234F1Z5",
            lines=["GST Number 27ABCDE1234F1Z5"],
            confidence=0.9,
            boxes=[Rect(x=0, y=0, w=10, h=5)],
            model_used="rapidocr-default-en",
            region_count=1,
        ),
    )
    monkeypatch.setattr(classifier, "classify_text", lambda text: ("legal_proof", 0.9))
    monkeypatch.setattr(
        extractor,
        "extract_fields",
        lambda doc_type, text: [{"field": "organisation.gst_number", "value": "27ABCDE1234F1Z5", "confidence": 0.9}],
    )
    monkeypatch.setattr(extractor, "extract_fields_vision", lambda *a, **k: calls.__setitem__("vision", calls["vision"] + 1))

    result = process_document(b"imgbytes", "board.jpg", "image/jpeg", script="english")

    assert result.extraction_source == "rapidocr:english"
    assert calls["vision"] == 0
    assert result.fields[0].source_bbox == Rect(x=0, y=0, w=10, h=5)


def test_english_image_falls_back_to_vision_when_rapidocr_fails(monkeypatch):
    monkeypatch.setattr(
        local_ocr,
        "extract_english",
        lambda data: (_ for _ in ()).throw(local_ocr.LocalOcrError("rapidocr not installed")),
    )
    monkeypatch.setattr(classifier, "classify_image", lambda data, media_type: ("legal_proof", 0.6))
    monkeypatch.setattr(
        extractor,
        "extract_fields_vision",
        lambda doc_type, data, media_type: [{"field": "organisation.laboratory_name", "value": "Acme Labs", "confidence": 0.6}],
    )

    fake_ocr = SimpleNamespace(extract=lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not call deepreef-ocr for english")))

    result = process_document(b"imgbytes", "board.jpg", "image/jpeg", script="english", ocr_client=fake_ocr)

    assert result.extraction_source == "vision_llm"
    assert result.fields[0].value == "Acme Labs"
    assert result.fields[0].source_bbox is None


def test_english_image_falls_back_to_vision_when_bytes_arent_a_real_image(monkeypatch):
    # No monkeypatch on local_ocr at all — real PIL genuinely can't decode
    # garbage bytes (rejected before RapidOCR's engine is ever invoked, so
    # this doesn't need rapidocr installed either), proving the fallback
    # works without needing to mock local_ocr itself.
    monkeypatch.setattr(classifier, "classify_image", lambda data, media_type: ("other", 0.3))
    monkeypatch.setattr(extractor, "extract_fields_vision", lambda *a, **k: [])

    result = process_document(b"imgbytes", "board.jpg", "image/jpeg", script="english")

    assert result.extraction_source == "vision_llm"
