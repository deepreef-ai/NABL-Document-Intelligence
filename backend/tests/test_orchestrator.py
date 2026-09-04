"""The budgeted extraction path: documents/orchestrator.py and the modules
it coordinates. Every LLM boundary is mocked — these tests assert the
ROUTING and BUDGET decisions, which is the whole point of the optimisation.
"""
from types import SimpleNamespace

import pytest

from app.dataset_normalization.models import NormalizedPage
from app.documents import adaptive_chunking
from app.documents import call_budget as cb
from app.documents import (
    classifier,
    combined_extraction,
    deterministic_validation,
    orchestrator,
    page_inspection,
    recovery,
)


# A realistic lab-report page: it carries all four signals
# classify_locally_scored needs (lab vocabulary + column headers +
# >=3 result rows), so these tests exercise the confident-local path
# instead of silently falling through to a real Nova call.
_LAB_PAGE = """LABORATORY TEST REPORT
Patient Name: John Doe    Specimen: Serum    Sample ID: S-1
Test              Result   Unit     Reference Range
Hemoglobin        13.2     g/dL     13-17
Glucose           92       mg/dL    70-100
Urea              20       mg/dL    15-40"""


def _page(n, text=_LAB_PAGE, method="pymupdf", conf=None, table=True, image=b"png"):
    np = NormalizedPage(
        page_number=n, text=text, extraction_method=method,
        ocr_used=(method == "ocr"), ocr_confidence=conf, has_table=table,
    )
    return page_inspection.InspectedPage(page=np, spans=[], image=image)


def _inspected(pages, source_type="born_digital_pdf"):
    from app.dataset_normalization.models import NormalizedDocument

    doc = NormalizedDocument(
        document_id="d1", original_filename="f.pdf", source_path="f.pdf",
        source_format="pdf", source_type=source_type, page_count=len(pages),
        status="processed", pages=[p.page for p in pages],
    )
    return page_inspection.InspectedDocument(document=doc, pages=pages)


@pytest.fixture(autouse=True)
def _stub_llm(monkeypatch):
    """No test here may reach a real provider, a real embedding model or a
    real Qdrant index."""
    monkeypatch.setattr(orchestrator, "get_llm_chain", lambda: SimpleNamespace(
        generate_json=lambda *a, **k: {"fields": {}, "tests": [], "schema_fields": {}}
    ))
    monkeypatch.setattr(orchestrator.retrieval, "select_relevant_pages",
                         lambda document_id, templates, pages, **k: [p.page_number for p in pages])


# --------------------------------------------------------------- call budget

def test_budget_refuses_beyond_the_total_and_records_why():
    b = cb.CallBudget(max_classification=1, max_extraction=4, max_recovery=2, max_total=3)
    assert b.spend(cb.EXTRACTION)
    assert b.spend(cb.EXTRACTION)
    assert b.spend(cb.EXTRACTION)
    assert not b.spend(cb.EXTRACTION)  # 4th exceeds max_total=3
    assert "total LLM call budget reached" in b.stop_reason
    assert b.total_calls == 3


def test_budget_meters_categories_separately():
    b = cb.CallBudget(max_classification=1, max_extraction=4, max_recovery=2, max_total=10)
    assert b.spend(cb.CLASSIFICATION)
    assert not b.spend(cb.CLASSIFICATION)  # category cap, not total
    assert "classification call budget reached" in b.stop_reason
    assert b.spend(cb.EXTRACTION)  # a different category still has room


def test_vision_shares_the_recovery_allowance():
    """A vision call IS a recovery attempt — giving it its own pool would let
    one document quietly spend double what the total suggests."""
    b = cb.CallBudget(max_classification=1, max_extraction=4, max_recovery=1, max_total=10)
    assert b.spend(cb.RECOVERY)
    assert not b.spend(cb.VISION)


def test_call_log_shape():
    b = cb.CallBudget(document_id="d1", max_classification=1, max_extraction=4, max_recovery=2, max_total=6)
    b.spend(cb.EXTRACTION)
    b.record_usage(1053, 831)
    log = b.as_dict()
    assert log["document_id"] == "d1"
    assert log["extraction_calls"] == 1
    assert log["total_llm_calls"] == 1
    assert log["total_tokens"] == 1884


# ------------------------------------------------------------ classification

def test_confident_local_classification_spends_no_call(monkeypatch):
    monkeypatch.setattr(classifier, "classify_text",
                         lambda text: pytest.fail("must not call the LLM when local is confident"))
    b = cb.CallBudget.from_settings()
    doc_type, conf, method = orchestrator._classify(
        _inspected([_page(1)]), "NABL_151", "lab-report.pdf", b
    )
    assert method == "local"
    assert conf >= 0.7
    assert b.counts[cb.CLASSIFICATION] == 0


def test_unconfident_local_classification_falls_back_to_one_llm_call(monkeypatch):
    calls = []

    def fake_classify(text):
        calls.append(text)
        return "legal_proof", 0.9

    monkeypatch.setattr(classifier, "classify_text", fake_classify)
    b = cb.CallBudget.from_settings()
    doc_type, conf, method = orchestrator._classify(
        _inspected([_page(1, text="Dear sir, please find enclosed.", table=False)]),
        "NABL_151", "x.pdf", b,
    )
    assert (doc_type, method) == ("legal_proof", "llm")
    assert len(calls) == 1
    assert b.counts[cb.CLASSIFICATION] == 1


def test_classification_keeps_the_local_guess_when_the_budget_is_spent(monkeypatch):
    monkeypatch.setattr(classifier, "classify_text",
                         lambda text: pytest.fail("no budget left, must not call"))
    b = cb.CallBudget(max_classification=0, max_extraction=4, max_recovery=2, max_total=6)
    _, _, method = orchestrator._classify(
        _inspected([_page(1, text="ambiguous", table=False)]), "NABL_151", "x.pdf", b
    )
    assert method == "local_budget_capped"


# ------------------------------------------------------------------ end-to-end

def test_one_page_report_costs_one_extraction_call_and_no_classification(monkeypatch):
    """Target behaviour: a simple 1-page lab report = 1 Nova call total."""
    monkeypatch.setattr(page_inspection, "inspect", lambda *a, **k: _inspected([_page(1)]))
    seen = []

    def fake_extract(chain, payload, schema_paths=None):
        seen.append(payload)
        return {"fields": {"patient_name": "John Doe"}, "tests": [], "schema_fields": {}}

    monkeypatch.setattr(combined_extraction, "extract", fake_extract)
    monkeypatch.setattr(deterministic_validation, "validate",
                         lambda *a, **k: deterministic_validation.ValidationResult())

    result = orchestrator.run(b"%PDF", ".pdf", filename="lab-report.pdf", document_id="d1")

    assert len(seen) == 1
    assert result.call_log["classification_calls"] == 0
    assert result.call_log["extraction_calls"] == 1
    assert result.call_log["total_llm_calls"] == 1
    assert any(f.field == "patient_name" for f in result.fields)


def test_twenty_page_document_stays_within_the_extraction_budget(monkeypatch):
    pages = [_page(i, text=f"Page {i} " + "x" * 4000) for i in range(1, 21)]
    monkeypatch.setattr(page_inspection, "inspect", lambda *a, **k: _inspected(pages))
    calls = []
    monkeypatch.setattr(combined_extraction, "extract",
                         lambda chain, payload, schema_paths=None: calls.append(1) or
                         {"fields": {}, "tests": [], "schema_fields": {}})
    monkeypatch.setattr(deterministic_validation, "validate",
                         lambda *a, **k: deterministic_validation.ValidationResult())

    result = orchestrator.run(b"%PDF", ".pdf", filename="big.pdf", document_id="d1")

    # 20 pages must NOT become 20 calls.
    assert len(calls) <= 4
    assert result.call_log["total_llm_calls"] <= 6
    assert result.page_count == 20


def test_table_rows_survive_extraction_as_rows(monkeypatch):
    """A results table must not be flattened into unrelated fields."""
    monkeypatch.setattr(page_inspection, "inspect", lambda *a, **k: _inspected([_page(1)]))
    monkeypatch.setattr(combined_extraction, "extract", lambda chain, payload, schema_paths=None: {
        "fields": {}, "schema_fields": {},
        "tests": [{"test_name": "Hb", "result": "13.2", "unit": "g/dL", "reference_range": "13-17"}],
    })
    monkeypatch.setattr(deterministic_validation, "validate",
                         lambda *a, **k: deterministic_validation.ValidationResult())

    result = orchestrator.run(b"%PDF", ".pdf", document_id="d1")

    by_field = {f.field: f.value for f in result.fields}
    assert by_field["tests[0].test_name"] == "Hb"
    assert by_field["tests[0].result"] == "13.2"
    assert by_field["tests[0].unit"] == "g/dL"
    assert by_field["tests[0].reference_range"] == "13-17"


def test_validation_failure_triggers_exactly_one_batched_recovery_call(monkeypatch):
    monkeypatch.setattr(page_inspection, "inspect", lambda *a, **k: _inspected([_page(1), _page(2)]))
    monkeypatch.setattr(combined_extraction, "extract", lambda chain, payload, schema_paths=None: {
        "fields": {"hemoglobin": "18.2"}, "tests": [], "schema_fields": {},
    })
    recovery_calls = []

    def fake_recover(chain, request, budget):
        recovery_calls.append(sorted(request.fields))
        budget.spend(cb.RECOVERY)
        return {"hemoglobin": "13.2"}

    monkeypatch.setattr(recovery, "recover", fake_recover)

    result = orchestrator.run(b"%PDF", ".pdf", document_id="d1")

    # One call, carrying every gap — not one call per chunk.
    assert len(recovery_calls) == 1
    assert "hemoglobin" in recovery_calls[0]
    assert result.call_log["recovery_calls"] == 1
    assert result.validation.get("recovered_fields") == ["hemoglobin"]


def test_a_chunk_extraction_failure_keeps_the_other_chunks(monkeypatch):
    pages = [_page(i, text=f"Page {i} " + "x" * 6000) for i in range(1, 5)]
    monkeypatch.setattr(page_inspection, "inspect", lambda *a, **k: _inspected(pages))
    state = {"n": 0}

    def flaky(chain, payload, schema_paths=None):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("every provider failed")
        return {"fields": {f"found_{state['n']}": "v"}, "tests": [], "schema_fields": {}}

    monkeypatch.setattr(combined_extraction, "extract", flaky)
    monkeypatch.setattr(deterministic_validation, "validate",
                         lambda *a, **k: deterministic_validation.ValidationResult())

    result = orchestrator.run(b"%PDF", ".pdf", document_id="d1")

    assert any(f.field.startswith("found_") for f in result.fields)
    assert any("extraction failed" in w for w in result.warnings)


def test_mixed_pdf_does_not_cost_extra_calls_for_changing_method(monkeypatch):
    """Spec: do not make a separate call just because the extraction method
    changed between pages."""
    pages = [
        _page(1, method="pymupdf"),
        _page(2, method="ocr", conf=0.93),
        _page(3, method="pymupdf"),
    ]
    monkeypatch.setattr(page_inspection, "inspect",
                         lambda *a, **k: _inspected(pages, source_type="mixed_pdf"))
    calls = []
    monkeypatch.setattr(combined_extraction, "extract",
                         lambda chain, payload, schema_paths=None: calls.append(1) or
                         {"fields": {}, "tests": [], "schema_fields": {}})
    monkeypatch.setattr(deterministic_validation, "validate",
                         lambda *a, **k: deterministic_validation.ValidationResult())

    result = orchestrator.run(b"%PDF", ".pdf", document_id="d1")

    assert len(calls) == 1
    assert result.extraction_source == "mixed_pdf"


# --------------------------------------------------------- adaptive chunking

def test_chunking_skips_irrelevant_pages_entirely():
    pages = [_page(i, text="x" * 2000) for i in range(1, 21)]
    chunks = adaptive_chunking.build_chunks(pages, relevant_page_numbers=[3, 4], max_chars=12000, max_chunks=4)
    assert [c.page_numbers for c in chunks] == [[3, 4]]


def test_chunking_never_exceeds_the_call_budget():
    pages = [_page(i, text="x" * 11000) for i in range(1, 21)]
    chunks = adaptive_chunking.build_chunks(pages, max_chars=12000, max_chunks=3)
    assert len(chunks) == 3


def test_chunk_label_does_not_claim_a_range_it_does_not_contain():
    pages = [_page(1), _page(2), _page(7)]
    chunks = adaptive_chunking.build_chunks(pages, max_chars=100000, max_chunks=1)
    assert chunks[0].label == "p1,p2,p7"


# ------------------------------------------------------------ table detection

def test_detect_table_on_a_results_table_and_not_on_prose():
    table = (
        "Parameter        Result     Unit    Reference\n"
        "Hemoglobin       13.2       g/dL    13-17\n"
        "Glucose          92         mg/dL   70-100\n"
        "Creatinine       0.9        mg/dL   0.6-1.2"
    )
    assert page_inspection.detect_table(table) is True
    assert page_inspection.detect_table("Dear Sir, please find enclosed the certificate.") is False
    assert page_inspection.detect_table("") is False
