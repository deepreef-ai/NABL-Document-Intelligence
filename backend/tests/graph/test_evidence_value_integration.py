"""validate_evidence must check values strictly for numbers, tolerantly for text."""
import pytest

from app.graph.nodes.evidence import validate_evidence
from app.graph.schemas import EvidenceStatus, ExtractedField, ExtractionStatus
from app.graph.state import GraphState

PAGE = (
    "Hemoglobin 13.2 g/dL  Creatinine 0.15 mg/dL  Temp -5  "
    "Fees 1,00,000  Date 12-03-2024  Lab Name: Sunrise Diagnostics Pvt Ltd"
)


def _run(value, evidence, status=ExtractionStatus.VERIFIED):
    f = ExtractedField(
        field_name="f", normalized_field_name="f", value=value, page_number=1,
        chunk_id="c1", exact_source_evidence=evidence, confidence_score=0.99,
        extraction_status=status, field_uid="u1",
    )
    state = GraphState(document_id="d", page_text={1: PAGE}, normalized_fields=[f])
    return validate_evidence(state)["normalized_fields"][0]


def test_exact_decimal_supported():
    f = _run("13.2", "Hemoglobin 13.2 g/dL")
    assert f.evidence_status == EvidenceStatus.SUPPORTED
    assert f.extraction_status == ExtractionStatus.VERIFIED


def test_wrong_decimal_not_supported_and_downgraded():
    f = _run("1.5", "Creatinine 0.15 mg/dL")
    assert f.evidence_status != EvidenceStatus.SUPPORTED
    assert f.evidence_status == EvidenceStatus.PARTIALLY_SUPPORTED
    assert f.extraction_status == ExtractionStatus.UNCERTAIN


def test_sign_dropped_not_supported():
    f = _run("5", "Temp -5")
    assert f.evidence_status != EvidenceStatus.SUPPORTED


def test_indian_grouping_supported():
    f = _run("100000", "Fees 1,00,000")
    assert f.evidence_status == EvidenceStatus.SUPPORTED


def test_plain_text_value_different_case_and_spacing_supported():
    f = _run("sunrise   DIAGNOSTICS pvt ltd", "Lab Name: Sunrise Diagnostics Pvt Ltd")
    assert f.evidence_status == EvidenceStatus.SUPPORTED


def test_date_separator_difference_supported():
    f = _run("12/03/2024", "Date 12-03-2024")
    assert f.evidence_status == EvidenceStatus.SUPPORTED


def test_wrong_digit_never_supported_via_fuzzy_shortcut():
    # One digit off in a long, otherwise near-identical value: fuzzy similarity
    # would be very high, but numbers must not take that shortcut.
    f = _run("Hemoglobin 13.3 g/dL", "Hemoglobin 13.2 g/dL")
    assert f.evidence_status != EvidenceStatus.SUPPORTED
    assert f.extraction_status == ExtractionStatus.UNCERTAIN
