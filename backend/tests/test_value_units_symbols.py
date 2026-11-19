"""Unit/glyph spelling equivalence in strict value matching (no conversion)."""
import pytest

from app.documents.value_matching import (
    normalize_for_search, numbers_supported, strict_value_key, value_in_source,
)
from app.graph.nodes.evidence import validate_evidence
from app.graph.schemas import EvidenceStatus, ExtractedField, ExtractionStatus
from app.graph.state import GraphState


@pytest.mark.parametrize("a,b", [
    ("5 mg/m2", "5 mg/m²"), ("5 mg/m²", "5 mg/m^2"), ("5 mg/m2", "5 mg/m^2"),
    ("5 µg", "5 μg"), ("5 μg", "5 ug"), ("5 µg/mL", "5 ug/ml"),
    ("4.5 ×10^3/µL", "4.5 ×10³/μL"),
    ("５", "5"), ("１２.５ mg", "12.5 mg"),
    ("−5", "-5"), ("5 %", "5 ％"), ("H₂O", "h2o"),
    ("m⁻²", "m^-2"),
])
def test_equivalent_spellings_share_a_key(a, b):
    assert strict_value_key(a) == strict_value_key(b)
    assert value_in_source(a, normalize_for_search(b))
    assert value_in_source(b, normalize_for_search(a))


@pytest.mark.parametrize("a,b", [
    ("1.5", "0.15"), ("5", "-5"), ("<0.01", "0.01"), ("5 mg", "50 mg"),
    ("5 mg/dL", "5 mg/L"), ("5 mg", "5 µg"), ("5 mg/m2", "5 mg/m3"),
    ("10^3", "103"), ("10³", "103"), ("5 µg", "5 mg"),
])
def test_different_values_or_units_stay_different(a, b):
    assert strict_value_key(a) != strict_value_key(b)
    assert not value_in_source(a, normalize_for_search(b))


def test_numbers_supported_sees_through_glyphs():
    src = normalize_for_search("Dose ５ mg/m² daily")
    assert numbers_supported("5 mg/m2", src)
    assert not numbers_supported("50 mg/m2", src)


def test_original_text_is_not_altered():
    text = "5 µg/m²"
    normalize_for_search(text)
    assert text == "5 µg/m²"


def _run(page, value, quote):
    f = ExtractedField(
        field_name="f", normalized_field_name="f", value=value, page_number=1,
        chunk_id="c1", exact_source_evidence=quote, confidence_score=0.99,
        extraction_status=ExtractionStatus.VERIFIED, field_uid="u1",
    )
    st = GraphState(document_id="d", page_text={1: page}, normalized_fields=[f])
    return validate_evidence(st)["normalized_fields"][0]


def test_validate_evidence_accepts_unit_glyph_difference():
    page = "Dose: 5 mg/m² and Vitamin 4 µg daily"
    assert _run(page, "5 mg/m2", "Dose: 5 mg/m^2").evidence_status == EvidenceStatus.SUPPORTED
    assert _run(page, "4 ug", "Vitamin 4 µg daily").evidence_status == EvidenceStatus.SUPPORTED
    assert _run(page, "5 mg/m3", "Dose: 5 mg/m²").evidence_status != EvidenceStatus.SUPPORTED
    assert _run(page, "50 mg/m2", "Dose: 5 mg/m²").evidence_status != EvidenceStatus.SUPPORTED
