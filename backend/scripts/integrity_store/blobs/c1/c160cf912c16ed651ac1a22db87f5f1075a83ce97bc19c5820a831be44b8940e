"""One row per fact on the review screen.

A multi-page lab report restates its header on every page and the model picks
the same value up under more than one name, so the raw field list says the same
thing three ways. MEASURED on a real cytology report: 29 rows for 21 facts,
which meant a reviewer confirming the accession number twice with no way to
tell which of the two mattered.

The line these tests defend is the one between a REPEAT and a DISAGREEMENT.
Collapsing a repeat is a kindness; collapsing a disagreement invents agreement
the document does not contain, and hides the exact thing a reviewer is there
to catch.
"""
import pytest

from app.documents.grounding import FieldResult
from app.graph.adapter import _absorb_open_duplicates, _collapse_repeats
from app.graph.schemas import EvidenceStatus, ExtractedField, ExtractionStatus


def field(name, value, *, confidence=0.9, page=1, section="",
          extraction=ExtractionStatus.VERIFIED, evidence=EvidenceStatus.SUPPORTED):
    return ExtractedField(
        field_name=name.replace("_", " ").title(),
        normalized_field_name=name,
        value=value,
        normalized_value=value,
        confidence_score=confidence,
        page_number=page,
        section_name=section,
        extraction_status=extraction,
        evidence_status=evidence,
    )


def names(rows):
    return [(r.field, r.value) for r in rows]


class TestRepeatsCollapse:
    def test_the_same_header_restated_on_every_page_is_one_row(self):
        rows = _collapse_repeats([
            field("accession_no", "UH0000000", page=1),
            field("accession_no", "UH0000000", page=2),
            field("accession_no", "UH0000000", page=3),
        ])
        assert names(rows) == [("accession_no", "UH0000000")]

    def test_a_value_that_swallowed_the_next_field_loses_to_the_short_reading(self):
        # "No therapy" is the neighbouring `therapy` field; the long reading of
        # clinical_data reached across the boundary between them.
        rows = _collapse_repeats([
            field("clinical_data", "No clinical data specified"),
            field("clinical_data", "No clinical data specified No therapy"),
            field("therapy", "No therapy"),
        ])
        assert names(rows) == [
            ("clinical_data", "No clinical data specified"),
            ("therapy", "No therapy"),
        ]

    def test_the_short_reading_wins_even_when_it_arrives_second(self):
        rows = _collapse_repeats([
            field("clinical_data", "No clinical data specified No therapy", confidence=0.95),
            field("clinical_data", "No clinical data specified", confidence=0.60),
        ])
        assert names(rows) == [("clinical_data", "No clinical data specified")]

    def test_one_number_read_under_two_related_names_is_one_row(self):
        rows = _collapse_repeats([
            field("isomorphic_erythrocytes", "760", confidence=0.95),
            field("erythrocytes_per_10_hpfs_isomorphic", "760", confidence=0.80),
        ])
        assert names(rows) == [("isomorphic_erythrocytes", "760")]


class TestDisagreementsSurvive:
    """These must NOT collapse. Each one is a real difference on the page."""

    def test_two_pages_carrying_different_chart_numbers_stay_separate(self):
        rows = _collapse_repeats([
            field("chart_no", "1400000", page=1),
            field("chart_no", "1400001", page=2),
        ])
        assert names(rows) == [("chart_no", "1400000"), ("chart_no", "1400001")]

    def test_unrelated_fields_that_happen_to_share_a_value_stay_separate(self):
        # Nothing about "M" says these are the same reading, and a form that
        # silently dropped one would be missing a field the document has.
        rows = _collapse_repeats([
            field("sex", "M"),
            field("blood_group", "M"),
        ])
        assert names(rows) == [("sex", "M"), ("blood_group", "M")]

    def test_two_different_analytes_reading_the_same_number_stay_separate(self):
        rows = _collapse_repeats([
            field("glucose", "25"),
            field("blood", "25"),
        ])
        assert names(rows) == [("glucose", "25"), ("blood", "25")]


class TestRejectedValuesNeverReachTheForm:
    @pytest.mark.parametrize("evidence", [
        EvidenceStatus.UNSUPPORTED,
        EvidenceStatus.NO_EVIDENCE,
    ])
    def test_a_value_the_evidence_gate_refused_is_dropped(self, evidence):
        rows = _collapse_repeats([field("referring_physician", "SAMPLE", evidence=evidence)])
        assert rows == []

    def test_a_conflicted_value_is_dropped(self):
        rows = _collapse_repeats([
            field("reported_date", "01/14/13", extraction=ExtractionStatus.CONFLICTED),
        ])
        assert rows == []


class TestOrdering:
    def test_rows_come_back_in_document_order_not_confidence_order(self):
        # Confidence decides which duplicate survives; it must not decide where
        # the survivor appears, or the form stops matching the page.
        rows = _collapse_repeats([
            field("patient_name", "SAMPLE, PATIENT", confidence=0.55),
            field("sex", "M", confidence=0.99),
            field("dob", "01/01/1900", confidence=0.70),
        ])
        assert [r.field for r in rows] == ["patient_name", "sex", "dob"]


class TestSectionIsCarried:
    def test_the_documents_own_heading_reaches_the_form(self):
        rows = _collapse_repeats([field("specimen_type", "Voided Urine", section="CYTOLOGIC DIAGNOSIS")])
        assert rows[0].section == "CYTOLOGIC DIAGNOSIS"


class TestPageNumbering:
    """The graph counts pages from one; the API counts from zero.

    Passing the graph's number through unconverted labelled a page-1 field
    "p2" on the review screen and jumped the document viewer one page past the
    value — and on the last page, past the end of the document.
    """

    def test_page_one_becomes_index_zero(self):
        rows = _collapse_repeats([field("accession_no", "UH0000000", page=1)])
        assert rows[0].source_page == 0

    def test_page_two_becomes_index_one(self):
        rows = _collapse_repeats([field("comments", "Examination of Feulgen slides", page=2)])
        assert rows[0].source_page == 1

    def test_a_missing_page_stays_missing(self):
        rows = _collapse_repeats([field("therapy", "No therapy", page=None)])
        assert rows[0].source_page is None


class TestMappedSlotsAbsorbTheirOpenDuplicate:
    """A named form slot and the open field it came from are ONE reading.

    The letterhead's lab name arrives twice: once as the open
    `laboratory_name`, once as `organisation.laboratory_name` filling the NABL
    slot. Shown as two rows, a reviewer confirms the same fact twice with no
    way to tell which mattered.
    """

    def test_the_open_row_is_absorbed_into_the_slot(self):
        rows = [
            FieldResult(field="laboratory_name", value="ALPHA LABS", confidence=0.9,
                        source_page=0, section="Letterhead", group="lab_info",
                        source="open_extraction"),
            FieldResult(field="organisation.laboratory_name", value="ALPHA LABS",
                        confidence=0.9, source="llm", group="lab_info"),
        ]
        _absorb_open_duplicates(rows, mapped_start=1)
        assert [r.field for r in rows] == ["organisation.laboratory_name"]

    def test_the_surviving_slot_keeps_the_open_rows_place_on_the_page(self):
        # Without this the merged row loses its sub-heading and drops to the
        # bottom of the form under no heading at all.
        rows = [
            FieldResult(field="laboratory_name", value="ALPHA LABS", confidence=0.9,
                        source_page=3, section="Letterhead", group="lab_info",
                        source="open_extraction"),
            FieldResult(field="organisation.laboratory_name", value="ALPHA LABS",
                        confidence=0.9, source="llm"),
        ]
        _absorb_open_duplicates(rows, mapped_start=1)
        assert (rows[0].section, rows[0].group, rows[0].source_page) == ("Letterhead", "lab_info", 3)

    def test_an_unrelated_field_sharing_a_value_is_not_absorbed(self):
        rows = [
            FieldResult(field="sex", value="M", confidence=0.9, source="open_extraction"),
            FieldResult(field="organisation.grade", value="M", confidence=0.9, source="llm"),
        ]
        _absorb_open_duplicates(rows, mapped_start=1)
        assert [r.field for r in rows] == ["sex", "organisation.grade"]

    def test_one_open_row_is_absorbed_by_at_most_one_slot(self):
        # Two slots fed by one value must not both claim it and delete the row
        # twice over, nor leave the second slot silently unpaired.
        rows = [
            FieldResult(field="email", value="a@b.c", confidence=0.9,
                        section="Letterhead", source="open_extraction"),
            FieldResult(field="organisation.email", value="a@b.c", confidence=0.9, source="llm"),
            FieldResult(field="organisation.email_alt", value="a@b.c", confidence=0.8, source="llm"),
        ]
        _absorb_open_duplicates(rows, mapped_start=1)
        assert [r.field for r in rows] == ["organisation.email", "organisation.email_alt"]
        assert rows[0].section == "Letterhead"
        assert rows[1].section == ""
