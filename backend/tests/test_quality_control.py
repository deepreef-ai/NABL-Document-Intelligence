import json

import pytest

from app.quality_control import pipeline
from app.quality_control.approve import ApprovalRefused, approve_all_passing, approve_document

_MASTER_SCHEMA = {
    "domains": {
        "medical": {
            "keys": ["patient_name", "age", "sample_id"],
            "key_details": [
                {"canonical_key": "patient_name", "field_role": "document_field", "total_frequency": 5},
                {"canonical_key": "age", "field_role": "document_field", "total_frequency": 4},
                {"canonical_key": "sample_id", "field_role": "document_field", "total_frequency": 3},
            ],
        },
    }
}


def _write_master_schema(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "master_schema.json").write_text(json.dumps(_MASTER_SCHEMA), encoding="utf-8")


def _write_normalized_doc(root, document_id, source_path, page_count=1, status="processed", error=None):
    doc_dir = root / "normalized" / document_id
    doc_dir.mkdir(parents=True)
    (doc_dir / "document.json").write_text(json.dumps({
        "document_id": document_id, "original_filename": source_path.name, "source_path": str(source_path),
        "source_format": "pdf", "source_type": "born_digital_pdf", "page_count": page_count, "status": status,
        "error": error,
        "pages": [{"page_number": i + 1, "text": "x", "extraction_method": "pymupdf", "ocr_used": False,
                    "ocr_confidence": None, "elements": []} for i in range(page_count)],
    }), encoding="utf-8")


def _write_label(root, file_stem, **overrides):
    (root / "labels").mkdir(parents=True, exist_ok=True)
    label = {
        "document_id": file_stem,
        "original_filename": f"{file_stem}.pdf",
        "domain": "medical",
        "page_count": 1,
        "source_ocr_used": False,
        "source_ocr_confidence": None,
        "fields": {"patient_name": "John Doe", "age": "45", "sample_id": "S1"},
        "tests": [{"test_name": "Hemoglobin", "result": "13.5", "unit": "g/dL", "reference_range": "13-17"}],
        "annotation_status": "pending",
        "extraction_status": "ok",
        "error": None,
    }
    label.update(overrides)
    (root / "labels" / f"{file_stem}.json").write_text(json.dumps(label), encoding="utf-8")
    return label


def _base_setup(tmp_path):
    labeled_dir = tmp_path / "labeled_dataset"
    normalized_dir = tmp_path / "normalized_dataset"
    master_schema_dir = tmp_path / "master_schema"
    _write_master_schema(master_schema_dir)
    source_file = tmp_path / "source.pdf"
    source_file.write_bytes(b"%PDF-1.4 fake content")
    return labeled_dir, normalized_dir, master_schema_dir, source_file


# --------------------------------------------------------------------------- happy path

def test_valid_label_passes_and_keeps_pending_status(tmp_path):
    labeled_dir, normalized_dir, master_schema_dir, source_file = _base_setup(tmp_path)
    _write_normalized_doc(normalized_dir, "LR_000001", source_file)
    _write_label(labeled_dir, "LR_000001")

    report, results = pipeline.run(labeled_dir, normalized_dir, master_schema_dir)

    assert report.total_documents == 1
    assert report.pending == 1
    assert report.rejected == 0
    assert results[0].passed is True

    updated = json.loads((labeled_dir / "labels" / "LR_000001.json").read_text(encoding="utf-8"))
    assert updated["annotation_status"] == "pending"
    assert updated["qc"]["passed"] is True
    assert updated["fields"] == {"patient_name": "John Doe", "age": "45", "sample_id": "S1"}  # untouched


def test_valid_label_that_is_already_approved_stays_approved(tmp_path):
    labeled_dir, normalized_dir, master_schema_dir, source_file = _base_setup(tmp_path)
    _write_normalized_doc(normalized_dir, "LR_000001", source_file)
    _write_label(labeled_dir, "LR_000001", annotation_status="approved")

    report, _ = pipeline.run(labeled_dir, normalized_dir, master_schema_dir)
    assert report.approved == 1
    updated = json.loads((labeled_dir / "labels" / "LR_000001.json").read_text(encoding="utf-8"))
    assert updated["annotation_status"] == "approved"


# --------------------------------------------------------------------------- individual checks -> rejection

def test_invalid_document_id_is_rejected(tmp_path):
    labeled_dir, normalized_dir, master_schema_dir, source_file = _base_setup(tmp_path)
    _write_normalized_doc(normalized_dir, "LR_000001", source_file)
    _write_label(labeled_dir, "LR_000001", document_id="not-a-real-id")

    _, results = pipeline.run(labeled_dir, normalized_dir, master_schema_dir)
    assert results[0].new_status == "rejected"
    assert any(i.code == "invalid_document_id" for i in results[0].issues)


def test_unreadable_source_file_is_rejected(tmp_path):
    labeled_dir, normalized_dir, master_schema_dir, source_file = _base_setup(tmp_path)
    _write_normalized_doc(normalized_dir, "LR_000001", tmp_path / "does_not_exist.pdf")
    _write_label(labeled_dir, "LR_000001")

    _, results = pipeline.run(labeled_dir, normalized_dir, master_schema_dir)
    assert results[0].new_status == "rejected"
    assert any(i.code == "source_file_unreadable" for i in results[0].issues)


def test_page_count_mismatch_is_rejected(tmp_path):
    labeled_dir, normalized_dir, master_schema_dir, source_file = _base_setup(tmp_path)
    _write_normalized_doc(normalized_dir, "LR_000001", source_file, page_count=3)
    _write_label(labeled_dir, "LR_000001", page_count=1)

    _, results = pipeline.run(labeled_dir, normalized_dir, master_schema_dir)
    assert results[0].new_status == "rejected"
    assert any(i.code == "page_count_mismatch" for i in results[0].issues)


def test_invalid_domain_is_rejected(tmp_path):
    labeled_dir, normalized_dir, master_schema_dir, source_file = _base_setup(tmp_path)
    _write_normalized_doc(normalized_dir, "LR_000001", source_file)
    _write_label(labeled_dir, "LR_000001", domain="astrology")

    _, results = pipeline.run(labeled_dir, normalized_dir, master_schema_dir)
    assert results[0].new_status == "rejected"
    assert any(i.code == "invalid_domain" for i in results[0].issues)


def test_invalid_schema_is_rejected(tmp_path):
    labeled_dir, normalized_dir, master_schema_dir, source_file = _base_setup(tmp_path)
    _write_normalized_doc(normalized_dir, "LR_000001", source_file)
    _write_label(labeled_dir, "LR_000001", tests="not a list")

    _, results = pipeline.run(labeled_dir, normalized_dir, master_schema_dir)
    assert results[0].new_status == "rejected"
    assert any(i.code == "invalid_schema" for i in results[0].issues)


def test_missing_required_fields_is_a_soft_warning_not_a_rejection(tmp_path):
    """Missing fields is a completeness signal, not proof the label is
    wrong — a document that genuinely doesn't mention e.g. a patient's name
    (redacted, or simply absent) shouldn't be auto-rejected for it. It must
    still show up in the label's qc.issues and the report's missing_fields
    count, so a human reviewer sees it."""
    labeled_dir, normalized_dir, master_schema_dir, source_file = _base_setup(tmp_path)
    _write_normalized_doc(normalized_dir, "LR_000001", source_file)
    _write_label(labeled_dir, "LR_000001", fields={"patient_name": None, "age": None, "sample_id": None})

    report, results = pipeline.run(labeled_dir, normalized_dir, master_schema_dir)
    assert results[0].new_status == "pending"
    assert results[0].passed is True
    assert any(i.code == "missing_required_fields" for i in results[0].issues)
    assert report.missing_fields == 1


def test_placeholder_sentinel_value_is_a_soft_warning(tmp_path):
    labeled_dir, normalized_dir, master_schema_dir, source_file = _base_setup(tmp_path)
    _write_normalized_doc(normalized_dir, "LR_000001", source_file)
    _write_label(labeled_dir, "LR_000001", tests=[
        {"test_name": "Hemoglobin", "result": "N/A", "unit": "g/dL", "reference_range": "13-17"},
    ])

    _, results = pipeline.run(labeled_dir, normalized_dir, master_schema_dir)
    assert results[0].new_status == "pending"
    assert any(i.code == "invalid_values" for i in results[0].issues)


def test_swapped_unit_and_reference_range_is_a_soft_warning(tmp_path):
    labeled_dir, normalized_dir, master_schema_dir, source_file = _base_setup(tmp_path)
    _write_normalized_doc(normalized_dir, "LR_000001", source_file)
    _write_label(labeled_dir, "LR_000001", tests=[
        {"test_name": "Hemoglobin", "result": "13.5", "unit": "13-17", "reference_range": None},
    ])

    _, results = pipeline.run(labeled_dir, normalized_dir, master_schema_dir)
    assert results[0].new_status == "pending"
    assert any(i.code == "incorrect_table_mapping" for i in results[0].issues)


def test_extraction_failure_is_rejected_and_counted(tmp_path):
    labeled_dir, normalized_dir, master_schema_dir, source_file = _base_setup(tmp_path)
    _write_normalized_doc(normalized_dir, "LR_000001", source_file)
    _write_label(labeled_dir, "LR_000001", extraction_status="failed", error="provider exploded")

    report, results = pipeline.run(labeled_dir, normalized_dir, master_schema_dir)
    assert results[0].new_status == "rejected"
    assert report.extraction_failures == 1


def test_ocr_failure_from_normalization_is_counted(tmp_path):
    labeled_dir, normalized_dir, master_schema_dir, source_file = _base_setup(tmp_path)
    _write_normalized_doc(normalized_dir, "LR_000001", source_file, status="failed", error="deepreef-ocr call failed: timeout")
    _write_label(labeled_dir, "LR_000001")

    report, results = pipeline.run(labeled_dir, normalized_dir, master_schema_dir)
    assert results[0].new_status == "rejected"
    assert report.ocr_failures == 1
    assert any(i.code == "ocr_failed" for i in results[0].issues)


def test_low_ocr_confidence_is_flagged(tmp_path):
    labeled_dir, normalized_dir, master_schema_dir, source_file = _base_setup(tmp_path)
    _write_normalized_doc(normalized_dir, "LR_000001", source_file)
    _write_label(labeled_dir, "LR_000001", source_ocr_used=True, source_ocr_confidence=0.2)

    report, results = pipeline.run(labeled_dir, normalized_dir, master_schema_dir)
    assert any(i.code == "ocr_low_confidence" for i in results[0].issues)
    assert report.ocr_failures == 1
    assert results[0].new_status == "pending"  # a soft warning, not a hard failure


def test_duplicate_document_content_is_detected_and_both_rejected(tmp_path):
    labeled_dir, normalized_dir, master_schema_dir, source_file = _base_setup(tmp_path)
    _write_normalized_doc(normalized_dir, "LR_000001", source_file)
    _write_normalized_doc(normalized_dir, "LR_000002", source_file)  # same bytes, different id
    _write_label(labeled_dir, "LR_000001")
    _write_label(labeled_dir, "LR_000002")

    report, results = pipeline.run(labeled_dir, normalized_dir, master_schema_dir)
    by_id = {r.document_id: r for r in results}
    assert by_id["LR_000001"].new_status == "rejected"
    assert by_id["LR_000002"].new_status == "rejected"
    assert report.duplicate_documents == 2


def test_duplicate_document_id_field_is_detected(tmp_path):
    labeled_dir, normalized_dir, master_schema_dir, source_file = _base_setup(tmp_path)
    _write_normalized_doc(normalized_dir, "LR_000001", source_file)
    _write_normalized_doc(normalized_dir, "LR_000002", tmp_path / "other.pdf")
    (tmp_path / "other.pdf").write_bytes(b"different content")
    _write_label(labeled_dir, "LR_000001")
    _write_label(labeled_dir, "LR_000002", document_id="LR_000001")  # same id field, different file

    report, _ = pipeline.run(labeled_dir, normalized_dir, master_schema_dir)
    assert report.duplicate_document_ids == 2


# --------------------------------------------------------------------------- the core enforcement

def test_an_approved_label_that_now_fails_qc_is_demoted_to_rejected(tmp_path):
    """This is the central requirement: an invalid annotation must never be
    left "approved", even if a human approved it before something changed."""
    labeled_dir, normalized_dir, master_schema_dir, source_file = _base_setup(tmp_path)
    _write_normalized_doc(normalized_dir, "LR_000001", source_file, page_count=5)  # now disagrees with the label
    _write_label(labeled_dir, "LR_000001", annotation_status="approved", page_count=1)

    report, results = pipeline.run(labeled_dir, normalized_dir, master_schema_dir)
    assert results[0].new_status == "rejected"
    assert report.approved == 0
    assert report.rejected == 1
    updated = json.loads((labeled_dir / "labels" / "LR_000001.json").read_text(encoding="utf-8"))
    assert updated["annotation_status"] == "rejected"


def test_approve_document_succeeds_for_a_valid_label(tmp_path):
    labeled_dir, normalized_dir, master_schema_dir, source_file = _base_setup(tmp_path)
    _write_normalized_doc(normalized_dir, "LR_000001", source_file)
    _write_label(labeled_dir, "LR_000001")

    approve_document("LR_000001", labeled_dir, normalized_dir, master_schema_dir)

    updated = json.loads((labeled_dir / "labels" / "LR_000001.json").read_text(encoding="utf-8"))
    assert updated["annotation_status"] == "approved"


def test_a_previously_rejected_label_recovers_to_pending_once_it_passes(tmp_path):
    """"rejected" must not be a one-way trap: if the underlying issue is
    fixed (or checks are recalibrated) and the label now passes, a re-run
    must reflect that, not leave it stuck rejected forever."""
    labeled_dir, normalized_dir, master_schema_dir, source_file = _base_setup(tmp_path)
    _write_normalized_doc(normalized_dir, "LR_000001", source_file)
    _write_label(labeled_dir, "LR_000001", annotation_status="rejected")

    report, results = pipeline.run(labeled_dir, normalized_dir, master_schema_dir)
    assert results[0].new_status == "pending"
    assert report.rejected == 0
    assert report.pending == 1


def test_approve_document_succeeds_despite_a_soft_warning(tmp_path):
    """Missing a non-critical field is a documented warning, not a hard
    defect — a human approving despite it is the point of this workflow."""
    labeled_dir, normalized_dir, master_schema_dir, source_file = _base_setup(tmp_path)
    _write_normalized_doc(normalized_dir, "LR_000001", source_file)
    _write_label(labeled_dir, "LR_000001", fields={"patient_name": None, "age": None, "sample_id": None})

    label = approve_document("LR_000001", labeled_dir, normalized_dir, master_schema_dir)
    assert label["annotation_status"] == "approved"
    assert any(i["code"] == "missing_required_fields" for i in label["qc"]["issues"])


def test_approve_document_refuses_an_invalid_label_and_leaves_it_unchanged(tmp_path):
    labeled_dir, normalized_dir, master_schema_dir, source_file = _base_setup(tmp_path)
    _write_normalized_doc(normalized_dir, "LR_000001", source_file)
    _write_label(labeled_dir, "LR_000001", domain="astrology")

    with pytest.raises(ApprovalRefused):
        approve_document("LR_000001", labeled_dir, normalized_dir, master_schema_dir)

    unchanged = json.loads((labeled_dir / "labels" / "LR_000001.json").read_text(encoding="utf-8"))
    assert unchanged["annotation_status"] == "pending"  # never touched


def test_approve_all_passing_approves_valid_ones_and_reports_refusals(tmp_path):
    labeled_dir, normalized_dir, master_schema_dir, source_file = _base_setup(tmp_path)
    _write_normalized_doc(normalized_dir, "LR_000001", source_file)
    _write_normalized_doc(normalized_dir, "LR_000002", source_file)
    _write_label(labeled_dir, "LR_000001")  # valid
    _write_label(labeled_dir, "LR_000002", domain="astrology")  # hard-fails

    approved_ids, refused = approve_all_passing(labeled_dir, normalized_dir, master_schema_dir)

    assert approved_ids == ["LR_000001"]
    assert len(refused) == 1
    assert refused[0][0] == "LR_000002"

    assert json.loads((labeled_dir / "labels" / "LR_000001.json").read_text(encoding="utf-8"))["annotation_status"] == "approved"
    assert json.loads((labeled_dir / "labels" / "LR_000002.json").read_text(encoding="utf-8"))["annotation_status"] == "pending"


def test_approve_all_passing_skips_already_approved(tmp_path):
    labeled_dir, normalized_dir, master_schema_dir, source_file = _base_setup(tmp_path)
    _write_normalized_doc(normalized_dir, "LR_000001", source_file)
    _write_label(labeled_dir, "LR_000001", annotation_status="approved")

    approved_ids, refused = approve_all_passing(labeled_dir, normalized_dir, master_schema_dir)
    assert approved_ids == []
    assert refused == []


# --------------------------------------------------------------------------- report shape

def test_report_domain_distribution_and_totals(tmp_path):
    labeled_dir, normalized_dir, master_schema_dir, source_file = _base_setup(tmp_path)
    _write_normalized_doc(normalized_dir, "LR_000001", source_file)
    _write_normalized_doc(normalized_dir, "LR_000002", tmp_path / "b.pdf")
    (tmp_path / "b.pdf").write_bytes(b"other content")
    _write_label(labeled_dir, "LR_000001", domain="medical")
    _write_label(labeled_dir, "LR_000002", domain="milk", fields={}, tests=[])

    report, _ = pipeline.run(labeled_dir, normalized_dir, master_schema_dir)
    assert report.total_documents == 2
    assert report.domain_distribution == {"medical": 1, "milk": 1}

    on_disk = json.loads((labeled_dir / "qc_report.json").read_text(encoding="utf-8"))
    assert on_disk["total_documents"] == 2
