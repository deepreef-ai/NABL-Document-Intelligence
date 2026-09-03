from types import SimpleNamespace

import pytest

from app.documents.compiler import compile_form
from app.schemas.forms import FORM_MODEL, NablFormType


def _field(field_path, value, accepted=True, source="llm"):
    return SimpleNamespace(field_path=field_path, value=value, accepted=accepted, source=source)


def _doc(doc_type, fields):
    return SimpleNamespace(doc_type=doc_type, fields=fields)


@pytest.mark.parametrize("form_type", list(NablFormType))
def test_every_form_type_is_registered_in_form_model(form_type):
    assert form_type in FORM_MODEL


def test_recognition_form_flat_merges_onto_lab_details_and_appends_equipment():
    docs = [
        _doc(
            "project_reference_document",
            [_field("project_name", "Metro Station Phase 2"), _field("project_reference", "TND-2026-441")],
        ),
        _doc(
            "recognition_equipment_certificate",
            [_field("discipline", "Aggregates"), _field("name", "Sieve Shaker")],
        ),
    ]

    result = compile_form(NablFormType.NABL_159.value, docs)

    assert result["lab_details"]["project_name"] == "Metro Station Phase 2"
    assert result["lab_details"]["project_reference"] == "TND-2026-441"
    assert result["equipment"] == [
        {
            "discipline": "Aggregates",
            "name": "Sieve Shaker",
            "in_house_or_traceability": None,
            "external_lab_name": None,
            "calibration_certificate_uploaded": False,
            "image_uploaded": False,
        }
    ]


def test_regulatory_recognition_certificate_merges_onto_154s_form_root():
    docs = [
        _doc(
            "regulatory_recognition_certificate",
            [_field("last_recognition_certificate_number", "FSSAI-2024-8821")],
        )
    ]

    result = compile_form(NablFormType.NABL_154.value, docs)

    assert result["last_recognition_certificate_number"] == "FSSAI-2024-8821"


def test_a_doc_type_irrelevant_to_the_form_type_is_silently_ignored():
    # legal_proof targets `organisation`, which a recognition-category form doesn't have.
    docs = [_doc("legal_proof", [_field("organisation.gst_number", "27ABCDE1234F1Z5")])]

    result = compile_form(NablFormType.NABL_155.value, docs)

    assert "organisation" not in result


def test_completed_application_form_populates_multiple_sections_from_one_document():
    # A "completed_application_form" upload (an applicant's own filled copy of
    # the NABL form) extracts into several different sections at once — flat
    # org fields AND two different repeating lists — unlike every other
    # doc_type, which only ever targets one.
    docs = [
        _doc(
            "completed_application_form",
            [
                _field("organisation.gst_number", "27ABCDE1234F1Z5"),
                _field("equipment[0].name", "Digital Multimeter"),
                _field("equipment[0].serial_number", "SN-001"),
                _field("equipment[1].name", "Analytical Balance"),
                _field("staff[0].name", "Jane Doe"),
                _field("staff[0].designation", "Quality Manager"),
            ],
        )
    ]

    result = compile_form(NablFormType.NABL_151.value, docs)

    assert result["organisation"]["gst_number"] == "27ABCDE1234F1Z5"
    assert [e["name"] for e in result["equipment"]] == ["Digital Multimeter", "Analytical Balance"]
    assert result["equipment"][0]["serial_number"] == "SN-001"
    assert result["staff"][0]["name"] == "Jane Doe"
    assert result["staff"][0]["designation"] == "Quality Manager"


def test_open_extraction_fields_land_in_extra_fields_not_a_schema_attribute():
    docs = [
        _doc(
            "legal_proof",
            [
                _field("organisation.gst_number", "27ABCDE1234F1Z5"),  # normal schema field
                _field("patient_name", "Gunu", source="open_extraction"),  # no schema slot anywhere
            ],
        )
    ]

    result = compile_form(NablFormType.NABL_151.value, docs)

    assert result["organisation"]["gst_number"] == "27ABCDE1234F1Z5"
    assert result["extra_fields"] == {"patient_name": "Gunu"}
    # never injected as a real attribute anywhere on the compiled form
    assert "patient_name" not in result
    assert "patient_name" not in result["organisation"]


def test_open_extraction_field_cannot_overwrite_a_real_schema_attribute():
    # "gst_number" is the last dotted segment of a real schema field. An
    # open-extraction field with that same bare name must NOT be able to
    # reach _merge_flat_fields' hasattr/setattr path and overwrite it.
    docs = [
        _doc(
            "legal_proof",
            [
                _field("organisation.gst_number", "27ABCDE1234F1Z5"),
                _field("gst_number", "FORGED-VALUE", source="open_extraction"),
            ],
        )
    ]

    result = compile_form(NablFormType.NABL_151.value, docs)

    assert result["organisation"]["gst_number"] == "27ABCDE1234F1Z5"
    assert result["extra_fields"] == {"gst_number": "FORGED-VALUE"}


def test_document_with_only_open_extraction_fields_still_populates_extra_fields():
    # e.g. a doc_type of "other" — no schema-guided fields at all, only
    # open-extraction ones. Must not be silently dropped, and must not
    # break the existing doc_type dispatch for a target-less doc_type.
    docs = [_doc("other", [_field("cane_sugar", "Absent", source="open_extraction")])]

    result = compile_form(NablFormType.NABL_151.value, docs)

    assert result["extra_fields"] == {"cane_sugar": "Absent"}
