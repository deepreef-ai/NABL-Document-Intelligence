"""Unit tests for the deterministic nodes.

These are the tests that matter most. The deterministic nodes are where the
business rules live — never skip a page, never fill an unsupported value,
never resolve a conflict on confidence alone — and they can all be asserted
without a model in the loop.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.graph.nodes.chunking import create_chunks
from app.graph.nodes.completeness import check_completeness
from app.graph.nodes.evidence import validate_evidence
from app.graph.nodes.form_fill import fill_form
from app.graph.nodes.helpers import (
    is_placeholder,
    normalise_for_match,
    similarity,
    snake_case,
)
from app.graph.nodes.preprocess import preprocess_document
from app.graph.nodes.quality import run_quality_control
from app.graph.nodes.response_validation import validate_responses
from app.graph.nodes.validation import validate_document
from app.graph.schemas import (
    ChunkRecord,
    ChunkStatus,
    EvidenceStatus,
    ExtractedField,
    ExtractionStatus,
    FinalStatus,
    FormMapping,
    MappingStatus,
    PageRecord,
    PageStatus,
)
from app.graph.state import GraphState


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


class TestHelpers:
    @pytest.mark.parametrize("a,b", [
        ("40 - 129", "40-129"),
        ("02 Nov 2020", "02Nov2020"),
        ("IN 123", "in123"),
        ("4.2 %", "4.2%"),
    ])
    def test_ocr_spacing_differences_are_not_real_differences(self, a, b):
        assert normalise_for_match(a) == normalise_for_match(b)

    def test_genuinely_different_values_stay_different(self):
        assert normalise_for_match("4.2") != normalise_for_match("4.3")

    def test_similarity_is_symmetric_and_bounded(self):
        assert similarity("abc", "abc") == 1.0
        assert 0.0 <= similarity("hello world", "hello there") <= 1.0
        assert similarity("", "") == 1.0

    @pytest.mark.parametrize("v", ["", "N/A", "n/a", "null", "NONE", "-", "not available", None])
    def test_sentinels_are_placeholders(self, v):
        assert is_placeholder(v)

    def test_real_values_are_not_placeholders(self):
        assert not is_placeholder("0")
        assert not is_placeholder("Cow Milk")

    def test_snake_case(self):
        assert snake_case("Report No.") == "report_no"
        assert snake_case("  GST   Number ") == "gst_number"
        assert snake_case("!!!") == "unnamed_field"


# --------------------------------------------------------------------------
# B. validation
# --------------------------------------------------------------------------


class TestDocumentValidation:
    def test_accepts_a_real_pdf(self, simple_pdf):
        out = validate_document(GraphState(document_id="d", file_path=simple_pdf))
        assert out.get("final_status") is None
        assert out["file_type"] == "pdf"
        assert out["total_pages"] == 2

    def test_missing_file_is_rejected(self):
        out = validate_document(GraphState(document_id="d", file_path="/nope/none.pdf"))
        assert out["final_status"] == FinalStatus.REJECTED
        assert out["errors"][0].error_type == "INVALID_FILE"

    def test_empty_file_is_rejected(self, empty_file):
        out = validate_document(GraphState(document_id="d", file_path=empty_file))
        assert out["final_status"] == FinalStatus.REJECTED
        assert out["errors"][0].error_type == "EMPTY_DOCUMENT"

    def test_corrupt_pdf_is_rejected(self, corrupt_pdf):
        out = validate_document(GraphState(document_id="d", file_path=corrupt_pdf))
        assert out["final_status"] == FinalStatus.REJECTED
        assert out["errors"][0].error_type == "CORRUPTED_FILE"

    def test_unsupported_type_is_rejected(self, unsupported_file):
        out = validate_document(GraphState(document_id="d", file_path=unsupported_file))
        assert out["errors"][0].error_type == "UNSUPPORTED_FILE_TYPE"

    def test_password_protected_pdf_is_rejected(self, encrypted_pdf):
        out = validate_document(GraphState(document_id="d", file_path=encrypted_pdf))
        assert out["final_status"] == FinalStatus.REJECTED
        assert out["errors"][0].error_type == "PASSWORD_PROTECTED"

    def test_oversized_file_is_rejected(self, simple_pdf, monkeypatch):
        from app.graph.config import get_graph_settings

        monkeypatch.setattr(get_graph_settings(), "max_file_size_mb", 0, raising=False)
        out = validate_document(GraphState(document_id="d", file_path=simple_pdf))
        assert out["errors"][0].error_type == "FILE_TOO_LARGE"


# --------------------------------------------------------------------------
# C. preprocessing
# --------------------------------------------------------------------------


class TestPreprocessing:
    def test_every_page_gets_a_record(self, three_page_pdf):
        state = GraphState(document_id="d", file_path=three_page_pdf, file_type="pdf")
        out = preprocess_document(state)
        assert out["total_pages"] == 3
        assert set(out["page_metadata"]) == {1, 2, 3}

    def test_blank_page_is_recorded_not_dropped(self, three_page_pdf):
        state = GraphState(document_id="d", file_path=three_page_pdf, file_type="pdf")
        out = preprocess_document(state)
        blank = out["page_metadata"][3]
        assert blank.char_count == 0
        assert blank.status in (PageStatus.EMPTY, PageStatus.OCR_FAILED, PageStatus.OCR)
        assert 3 in out["page_metadata"], "a blank page must still be accounted for"

    def test_text_is_extracted(self, simple_pdf):
        state = GraphState(document_id="d", file_path=simple_pdf, file_type="pdf")
        out = preprocess_document(state)
        assert "LR-2024-0195" in out["page_text"][1]


# --------------------------------------------------------------------------
# E. chunking
# --------------------------------------------------------------------------


def _state_with_pages(pages: dict[int, str]) -> GraphState:
    return GraphState(
        document_id="doc",
        file_type="pdf",
        total_pages=len(pages),
        page_text=pages,
        page_metadata={
            p: PageRecord(
                page_number=p, text=t, char_count=len(t),
                status=PageStatus.NATIVE_TEXT if t.strip() else PageStatus.EMPTY,
            )
            for p, t in pages.items()
        },
    )


class TestChunking:
    def test_every_page_lands_in_a_chunk(self):
        state = _state_with_pages({1: "alpha " * 50, 2: "beta " * 50, 3: ""})
        out = create_chunks(state)
        covered = {p for c in out["chunks"] for p in c.page_numbers}
        assert covered == {1, 2, 3}

    def test_empty_page_gets_its_own_accounted_chunk(self):
        state = _state_with_pages({1: "text", 2: ""})
        out = create_chunks(state)
        empties = [c for c in out["chunks"] if c.status == ChunkStatus.SKIPPED_EMPTY]
        assert len(empties) == 1
        assert empties[0].page_numbers == [2]

    def test_oversized_page_is_split_never_dropped(self, monkeypatch):
        from app.graph.config import get_graph_settings

        monkeypatch.setattr(get_graph_settings(), "max_chunk_chars", 2000, raising=False)
        state = _state_with_pages({1: "word " * 3000})
        out = create_chunks(state)
        assert len(out["chunks"]) > 1
        assert all(c.page_numbers == [1] for c in out["chunks"])

    def test_chunks_respect_the_size_limit(self, monkeypatch):
        from app.graph.config import get_graph_settings

        monkeypatch.setattr(get_graph_settings(), "max_chunk_chars", 3000, raising=False)
        state = _state_with_pages({i: "sample text " * 60 for i in range(1, 11)})
        out = create_chunks(state)
        assert all(c.char_count <= 3200 for c in out["chunks"]), \
            "chunking must never merge past the limit to save calls"


# --------------------------------------------------------------------------
# G. response validation
# --------------------------------------------------------------------------


def _f(**kw) -> ExtractedField:
    base = dict(
        field_name="Report No", normalized_field_name="report_no", value="X-1",
        normalized_value="X-1", data_type="string", page_number=1, chunk_id="c1",
        exact_source_evidence="Report No: X-1", confidence_score=0.9,
        extraction_status=ExtractionStatus.VERIFIED, field_uid="uid1",
    )
    base.update(kw)
    return ExtractedField(**base)


class TestExtractedFieldNameCoercion:
    """MEASURED: 'VU3 cytology (positive) CLN' and
    '1616659113-Test-Result-on-Beta-Casein-Certification' both crashed their
    whole document with AttributeError: 'int' object has no attribute
    'strip' — the LLM emitted a bare number as a field_name (e.g. a numbered
    cytology finding) and the old validator called .strip() on it directly."""

    def test_numeric_field_name_is_coerced_to_a_string_not_crashed_on(self):
        f = _f(field_name=3)
        assert f.field_name == "3"

    def test_float_field_name_is_coerced_to_a_string(self):
        f = _f(field_name=2024.0)
        assert f.field_name == "2024.0"

    def test_blank_field_name_is_still_rejected(self):
        with pytest.raises(ValidationError):
            _f(field_name="   ")

    def test_none_field_name_is_still_rejected(self):
        with pytest.raises(ValidationError):
            _f(field_name=None)


class TestResponseValidation:
    def _state(self, fields, chunk_text="Report No: X-1"):
        return GraphState(
            document_id="d",
            page_text={1: chunk_text},
            page_metadata={1: PageRecord(page_number=1, text=chunk_text,
                                         char_count=len(chunk_text),
                                         status=PageStatus.NATIVE_TEXT)},
            chunks=[ChunkRecord(chunk_id="c1", page_numbers=[1], text=chunk_text,
                                char_count=len(chunk_text), status=ChunkStatus.PROCESSED)],
            extracted_fields=fields,
            processed_chunks=["c1"],
        )

    def test_unknown_chunk_id_is_dropped(self):
        out = validate_responses(self._state([_f(chunk_id="ghost")]))
        assert out["normalized_fields"] == []

    def test_nonexistent_page_is_dropped(self):
        out = validate_responses(self._state([_f(page_number=99)]))
        assert out["normalized_fields"] == []

    def test_identical_duplicate_rows_are_collapsed(self):
        out = validate_responses(self._state([_f(field_uid="a"), _f(field_uid="b")]))
        assert len(out["normalized_fields"]) == 1

    def test_distinct_table_rows_survive(self):
        rows = [
            _f(field_uid="a", value="row1", exact_source_evidence="serial row1", occurrence_index=0),
            _f(field_uid="b", value="row2", exact_source_evidence="serial row2", occurrence_index=1),
        ]
        out = validate_responses(self._state(rows))
        assert len(out["normalized_fields"]) == 2, "repeated occurrences must not be deduplicated"

    def test_prose_masquerading_as_a_field_name_is_dropped(self):
        out = validate_responses(self._state([_f(field_name="x" * 200)]))
        assert out["normalized_fields"] == []


# --------------------------------------------------------------------------
# I. evidence validation
# --------------------------------------------------------------------------


class TestEvidenceValidation:
    def _state(self, fields, page_text="Report No: LR-2024-0195 and Fat Content: 4.2 %"):
        return GraphState(
            document_id="d",
            page_text={1: page_text, 2: "unrelated content on the second page"},
            page_metadata={
                1: PageRecord(page_number=1, text=page_text, char_count=len(page_text),
                              status=PageStatus.NATIVE_TEXT),
                2: PageRecord(page_number=2, text="unrelated", char_count=9,
                              status=PageStatus.NATIVE_TEXT),
            },
            normalized_fields=fields,
        )

    def test_real_quote_containing_the_value_is_supported(self):
        f = _f(value="LR-2024-0195", exact_source_evidence="Report No: LR-2024-0195")
        out = validate_evidence(self._state([f]))
        assert out["evidence_validation_results"][0].status == EvidenceStatus.SUPPORTED
        assert out["normalized_fields"][0].extraction_status == ExtractionStatus.VERIFIED

    def test_invented_quote_is_unsupported_and_downgraded(self):
        f = _f(value="9999", exact_source_evidence="Total Bacterial Count: 9999 cfu/ml")
        out = validate_evidence(self._state([f]))
        assert out["evidence_validation_results"][0].status == EvidenceStatus.UNSUPPORTED
        assert out["normalized_fields"][0].extraction_status != ExtractionStatus.VERIFIED

    def test_hallucination_is_kept_for_review_not_deleted(self):
        f = _f(value="9999", exact_source_evidence="Total Bacterial Count: 9999 cfu/ml")
        out = validate_evidence(self._state([f]))
        assert len(out["normalized_fields"]) == 1, "rejected values are kept, flagged, never deleted"

    def test_wrong_page_is_corrected_not_discarded(self):
        f = _f(value="LR-2024-0195",
               exact_source_evidence="Report No: LR-2024-0195", page_number=2)
        out = validate_evidence(self._state([f]))
        result = out["evidence_validation_results"][0]
        assert result.status == EvidenceStatus.WRONG_PAGE
        assert out["normalized_fields"][0].page_number == 1

    def test_genuine_quote_not_containing_the_value_is_partial(self):
        f = _f(value="7.7", exact_source_evidence="Report No: LR-2024-0195")
        out = validate_evidence(self._state([f]))
        assert out["evidence_validation_results"][0].status == EvidenceStatus.PARTIALLY_SUPPORTED

    def test_missing_evidence_is_flagged(self):
        f = _f(exact_source_evidence="")
        out = validate_evidence(self._state([f]))
        assert out["evidence_validation_results"][0].status == EvidenceStatus.NO_EVIDENCE


# --------------------------------------------------------------------------
# L. completeness
# --------------------------------------------------------------------------


class TestCompleteness:
    def test_page_missing_from_every_chunk_is_critical(self):
        state = GraphState(
            document_id="d",
            page_text={1: "a", 2: "b"},
            page_metadata={
                p: PageRecord(page_number=p, text="x", char_count=1,
                              status=PageStatus.NATIVE_TEXT) for p in (1, 2)
            },
            chunks=[ChunkRecord(chunk_id="c1", page_numbers=[1], status=ChunkStatus.PROCESSED)],
        )
        out = check_completeness(state)
        report = out["completeness_report"]
        assert report.pages_not_in_any_chunk == [2]
        assert report.critical_failure

    def test_failed_chunk_blocks_all_chunks_processed(self):
        state = GraphState(
            document_id="d",
            page_text={1: "a"},
            page_metadata={1: PageRecord(page_number=1, text="a", char_count=1,
                                         status=PageStatus.NATIVE_TEXT)},
            chunks=[ChunkRecord(chunk_id="c1", page_numbers=[1], status=ChunkStatus.FAILED)],
        )
        out = check_completeness(state)
        assert not out["completeness_report"].all_chunks_processed
        assert out["completeness_report"].chunks_failed == ["c1"]


# --------------------------------------------------------------------------
# N. form filling
# --------------------------------------------------------------------------


class TestFormFilling:
    def _state(self, mappings, fields=None):
        return GraphState(document_id="d", form_mappings=mappings,
                          normalized_fields=fields or [])

    def test_mapped_value_is_filled(self):
        out = fill_form(self._state([
            FormMapping(source_field="report_no", target_field="report.number",
                        mapped_value="X-1", mapping_confidence=0.95,
                        mapping_status=MappingStatus.MAPPED),
        ]))
        assert out["filled_form"] == {"report": {"number": "X-1"}}

    def test_conflicted_field_is_never_filled(self):
        out = fill_form(self._state([
            FormMapping(source_field="report_no", target_field="report.number",
                        mapped_value="X-1", mapping_confidence=0.99,
                        mapping_status=MappingStatus.CONFLICTED),
        ]))
        assert out["filled_form"] == {}
        assert "report.number" in out["missing_fields"]

    def test_not_found_is_reported_as_missing(self):
        out = fill_form(self._state([
            FormMapping(source_field="", target_field="report.date",
                        mapping_status=MappingStatus.NOT_FOUND),
        ]))
        assert out["missing_fields"] == ["report.date"]

    def test_higher_confidence_wins_and_is_not_overwritten(self):
        out = fill_form(self._state([
            FormMapping(source_field="a", target_field="x", mapped_value="good",
                        mapping_confidence=0.9, mapping_status=MappingStatus.MAPPED),
            FormMapping(source_field="b", target_field="x", mapped_value="worse",
                        mapping_confidence=0.4, mapping_status=MappingStatus.MAPPED),
        ]))
        assert out["filled_form"]["x"] == "good"

    def test_repeating_target_accumulates(self):
        out = fill_form(self._state([
            FormMapping(source_field="s", target_field="equipment[].serial",
                        mapped_value="S1", mapping_confidence=0.9,
                        mapping_status=MappingStatus.MAPPED),
            FormMapping(source_field="s", target_field="equipment[].serial",
                        mapped_value="S2", mapping_confidence=0.9,
                        mapping_status=MappingStatus.MAPPED),
        ]))
        serials = out["filled_form"]["equipment"][-1]["serial"]
        assert serials == ["S1", "S2"]

    def test_placeholder_value_is_not_filled(self):
        out = fill_form(self._state([
            FormMapping(source_field="a", target_field="x", mapped_value="N/A",
                        mapping_confidence=0.9, mapping_status=MappingStatus.MAPPED),
        ]))
        assert out["filled_form"] == {}


# --------------------------------------------------------------------------
# O. quality control
# --------------------------------------------------------------------------


class TestQualityControl:
    def test_value_in_form_with_no_source_field_is_critical(self):
        state = GraphState(
            document_id="d",
            page_text={1: "a"},
            page_metadata={1: PageRecord(page_number=1, text="a", char_count=1,
                                         status=PageStatus.NATIVE_TEXT)},
            chunks=[ChunkRecord(chunk_id="c1", page_numbers=[1], status=ChunkStatus.PROCESSED)],
            normalized_fields=[_f(value="real")],
            filled_form={"target": "invented value nobody extracted"},
        )
        out = run_quality_control(state)
        assert out["quality_control_report"].critical_issues
        assert not out["quality_control_report"].passed


# --------------------------------------------------------------------------
# Regression: page-number base at the rasterize boundary
# --------------------------------------------------------------------------


class TestOcrPageIndexing:
    """rasterize_page indexes doc[n] and is ZERO-based; page numbers in the
    graph are one-based. Getting this wrong OCR'd the next page for every page
    and threw on the last one — and the wrong-page reads were silent."""

    def test_ocr_rasterises_the_page_it_was_asked_for(self, monkeypatch):
        from app.graph.nodes import preprocess

        seen: list[int] = []

        def fake_rasterize(pdf_bytes, page_number, dpi=200):
            seen.append(page_number)
            return b"png"

        class _Result:
            text = "some recognised text on the page"
            confidence = 0.9

        monkeypatch.setattr("app.documents.pdf_utils.rasterize_page", fake_rasterize)
        monkeypatch.setattr("app.documents.local_ocr.extract_english", lambda b: _Result())

        preprocess._ocr_page(b"%PDF-fake", 1, 200)
        preprocess._ocr_page(b"%PDF-fake", 2, 200)

        assert seen == [0, 1], (
            "one-based page N must rasterise zero-based index N-1; "
            f"got {seen}"
        )

    def test_last_page_does_not_overflow(self, three_page_pdf):
        """The real end-to-end check: a 3-page PDF must not raise on page 3."""
        from app.graph.nodes.preprocess import preprocess_document

        state = GraphState(document_id="d", file_path=three_page_pdf, file_type="pdf")
        out = preprocess_document(state)

        overflow = [
            e for e in out["errors"]
            if "not in document" in (e.error_message or "")
        ]
        assert not overflow, f"page index overflowed the document: {overflow}"
