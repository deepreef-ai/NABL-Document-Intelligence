"""Protection and detection tests.

Two failure directions are covered deliberately:

* **Under-redaction** — an identifier survives. Caught by verification, and
  the file is marked FAILED_REVIEW, so it is loud.
* **Over-redaction** — a haemoglobin value or a test name is destroyed. Silent,
  permanent in the copy, and the reason the protected-vocabulary layer exists.

The over-redaction cases are the ones worth staring at.
"""
from __future__ import annotations

import pytest

from anonymizer.config import DetectionConfig
from anonymizer.detect import detect_page
from anonymizer.protect import (
    ProtectedVocabulary,
    looks_like_measurement,
    looks_like_reference_range,
)
from anonymizer.textsource import PageWords, Word


def line(text: str, page: int = 0, line_id: int = 1, y: float = 100.0) -> list[Word]:
    """Build a line of words with plausible, non-overlapping boxes."""
    words, x = [], 50.0
    for order, token in enumerate(text.split()):
        width = len(token) * 5.0
        words.append(Word(text=token, bbox=(x, y, x + width, y + 11),
                          page=page, line=line_id, order=order, source="text_layer"))
        x += width + 5.0
    return words


def page_of(*lines: str) -> PageWords:
    words: list[Word] = []
    for i, text in enumerate(lines, start=1):
        words.extend(line(text, line_id=i, y=100.0 + i * 20))
    return PageWords(page_number=0, words=words, source="text_layer", width=600, height=800)


@pytest.fixture
def vocab():
    return ProtectedVocabulary()


@pytest.fixture
def cfg():
    return DetectionConfig()


class TestProtectedVocabulary:
    @pytest.mark.parametrize("term", [
        "Haemoglobin", "haemoglobin", "g/dL", "mg/dl", "Serum",
        "Reference Range", "Test", "Result", "Negative", "Not Detected",
    ])
    def test_laboratory_vocabulary_is_protected(self, vocab, term):
        assert vocab.is_protected_term(term)

    def test_a_person_name_is_not_protected(self, vocab):
        assert not vocab.is_protected_term("Ramesh Kumar")

    def test_qualified_analyte_is_protected(self, vocab):
        assert vocab.is_protected_term("Glucose - Fasting")

    @pytest.mark.parametrize("value", ["12.5-16.0", "3.5 – 5.1", "< 200", "Up to 40"])
    def test_reference_ranges_recognised(self, value):
        assert looks_like_reference_range(value)

    @pytest.mark.parametrize("value", ["5.2", "12.4 g/dl", "1,200", "<0.01"])
    def test_measurements_recognised(self, value):
        assert looks_like_measurement(value)

    def test_name_field_holding_a_date_is_rejected(self, vocab):
        protected, reason = vocab.is_protected("PATIENT_NAME", "12/03/2024")
        assert protected and "date" in reason

    def test_name_field_with_no_letters_is_rejected(self, vocab):
        protected, _ = vocab.is_protected("PATIENT_NAME", "12345")
        assert protected


class TestPatternDetection:
    def test_finds_aadhaar_pan_email_phone(self, cfg, vocab):
        page = page_of(
            "Aadhaar: 2341 2345 6783",
            "PAN: ABCPK1234F",
            "Email: ramesh.kumar@example.com",
            "Mobile: 9876543210",
        )
        kinds = {d.kind for d in detect_page(page, cfg, vocab)}
        assert {"AADHAAR", "PAN", "EMAIL", "PHONE"} <= kinds

    def test_specimen_number_is_not_taken_for_aadhaar(self, cfg, vocab):
        page = page_of("Specimen No: 987654321011")
        assert not [d for d in detect_page(page, cfg, vocab) if d.kind == "AADHAAR"]

    def test_gstin_is_off_by_default(self, cfg, vocab):
        page = page_of("GSTIN: 27ABCPK1234F1Z5")
        assert not [d for d in detect_page(page, cfg, vocab) if d.kind == "GSTIN"]


class TestLabelDetection:
    def test_patient_name_is_detected(self, cfg, vocab):
        page = page_of("Patient Name: Ramesh Kumar")
        names = [d for d in detect_page(page, cfg, vocab) if d.kind == "PATIENT_NAME"]
        assert names and "Ramesh" in names[0].text

    def test_test_name_is_not_treated_as_a_patient_name(self, cfg, vocab):
        # The single most important over-redaction guard: "Test Name" is a
        # column heading on almost every lab report in existence.
        page = page_of("Test Name: Complete Blood Count")
        assert not [d for d in detect_page(page, cfg, vocab) if d.kind == "PATIENT_NAME"]

    @pytest.mark.parametrize("prefix", ["Sample", "Method", "Product", "Instrument"])
    def test_other_qualified_name_columns_are_ignored(self, cfg, vocab, prefix):
        page = page_of(f"{prefix} Name: Something Here")
        assert not [d for d in detect_page(page, cfg, vocab) if d.kind == "PATIENT_NAME"]

    def test_blank_field_is_not_redacted(self, cfg, vocab):
        page = page_of("Patient Name: N/A")
        assert not [d for d in detect_page(page, cfg, vocab) if d.kind == "PATIENT_NAME"]

    def test_value_stops_at_the_next_label(self, cfg, vocab):
        page = page_of("Patient Name: Ramesh Kumar Age: 45")
        names = [d for d in detect_page(page, cfg, vocab) if d.kind == "PATIENT_NAME"]
        assert names
        assert "45" not in names[0].text

    def test_patient_id_detected(self, cfg, vocab):
        page = page_of("UHID: UH0099123")
        assert [d for d in detect_page(page, cfg, vocab) if d.kind == "PATIENT_ID"]

    def test_lab_result_line_produces_no_detections(self, cfg, vocab):
        page = page_of(
            "Haemoglobin 12.4 g/dL 13.0-17.0",
            "Glucose Fasting 92 mg/dL 70-100",
        )
        assert detect_page(page, cfg, vocab) == []

    def test_age_sex_is_off_by_default(self, cfg, vocab):
        page = page_of("Age/Sex: 45/M")
        assert not [d for d in detect_page(page, cfg, vocab) if d.kind == "AGE_SEX"]


class TestDeduplication:
    def test_pattern_hit_wins_over_label_hit_on_the_same_words(self, cfg, vocab):
        # "UHID: ABCPK1234F" is both a patient id by label and a PAN by shape.
        page = page_of("UHID: ABCPK1234F")
        detections = detect_page(page, cfg, vocab)
        covering = [d for d in detections if "ABCPK1234F" in d.text]
        assert len(covering) == 1
        assert covering[0].kind == "PAN"
