import csv
import json

from app.final_dataset.document_type import derive_document_type
from app.final_dataset.pipeline import build_final_dataset, write_final_dataset
from app.final_dataset.validation import validate_record
from app.final_dataset.models import FinalDocumentRecord


def _write_normalized_doc(root, document_id, source_path, source_format="pdf", page_count=1):
    doc_dir = root / "normalized" / document_id
    doc_dir.mkdir(parents=True)
    (doc_dir / "document.json").write_text(json.dumps({
        "document_id": document_id, "source_path": str(source_path), "source_format": source_format,
        "page_count": page_count, "status": "processed",
        "pages": [{"page_number": i + 1, "text": "x"} for i in range(page_count)],
    }), encoding="utf-8")


def _write_label(root, document_id, **overrides):
    (root / "labels").mkdir(parents=True, exist_ok=True)
    label = {
        "document_id": document_id,
        "original_filename": f"{document_id}.pdf",
        "domain": "medical",
        "page_count": 1,
        "fields": {"patient_name": "John Doe", "age": None},
        "tests": [{"test_name": "Hemoglobin", "result": "13.5", "unit": "g/dL", "reference_range": "13-17"}],
        "annotation_status": "pending",
        "extraction_status": "ok",
    }
    label.update(overrides)
    (root / "labels" / f"{document_id}.json").write_text(json.dumps(label), encoding="utf-8")
    return label


def _setup(tmp_path):
    labeled_dir = tmp_path / "labeled_dataset"
    normalized_dir = tmp_path / "normalized_dataset"
    source_file = tmp_path / "source.pdf"
    source_file.write_bytes(b"%PDF-1.4 fake")
    return labeled_dir, normalized_dir, source_file


# --------------------------------------------------------------------------- document_type

def test_derive_document_type_matches_known_keywords():
    assert derive_document_type("001_Lab-report.png") == "lab_report"
    assert derive_document_type("000_Prescription.png") == "prescription"
    assert derive_document_type("Chahal-Black-Mustard-Oil-FSSAI-Purity-Test.pdf") == "purity_test"


def test_derive_document_type_returns_none_when_unrecognized():
    assert derive_document_type("random_document_123.pdf") is None


# --------------------------------------------------------------------------- validation

def test_validate_record_flags_non_approved_status():
    record = FinalDocumentRecord(
        document_id="LR_000001", original_filename="a.pdf", domain="medical", document_type=None,
        source_format="pdf", source_path="", page_count=1, fields={}, tests=[], annotation_status="pending",
    )
    errors = validate_record(record)
    assert any("annotation_status" in e for e in errors)


def test_validate_record_flags_malformed_test_row():
    record = FinalDocumentRecord(
        document_id="LR_000001", original_filename="a.pdf", domain="medical", document_type=None,
        source_format="pdf", source_path="", page_count=1, fields={}, tests=[{"result": "13.5"}],
        annotation_status="approved",
    )
    errors = validate_record(record)
    assert any("tests[0]" in e for e in errors)


def test_validate_record_passes_a_well_formed_approved_record(tmp_path):
    source = tmp_path / "a.pdf"
    source.write_bytes(b"x")
    record = FinalDocumentRecord(
        document_id="LR_000001", original_filename="a.pdf", domain="medical", document_type="lab_report",
        source_format="pdf", source_path=str(source), page_count=1,
        fields={"patient_name": "X"}, tests=[{"test_name": "Hb", "result": "13", "unit": "g/dL", "reference_range": "13-17"}],
        annotation_status="approved",
    )
    assert validate_record(record) == []


# --------------------------------------------------------------------------- pipeline: only approved included

def test_only_approved_labels_are_included(tmp_path):
    labeled_dir, normalized_dir, source_file = _setup(tmp_path)
    _write_normalized_doc(normalized_dir, "LR_000001", source_file)
    _write_normalized_doc(normalized_dir, "LR_000002", source_file)
    _write_normalized_doc(normalized_dir, "LR_000003", source_file)
    _write_label(labeled_dir, "LR_000001", annotation_status="approved")
    _write_label(labeled_dir, "LR_000002", annotation_status="pending")
    _write_label(labeled_dir, "LR_000003", annotation_status="rejected")

    records, summary = build_final_dataset(labeled_dir, normalized_dir)
    assert [r.document_id for r in records] == ["LR_000001"]
    assert summary.total_documents == 1
    assert summary.approved_annotations == 1


def test_invalid_approved_record_is_excluded_and_logged(tmp_path):
    labeled_dir, normalized_dir, source_file = _setup(tmp_path)
    _write_normalized_doc(normalized_dir, "LR_000001", source_file)
    _write_label(labeled_dir, "LR_000001", annotation_status="approved", domain="astrology")

    records, summary = build_final_dataset(labeled_dir, normalized_dir)
    assert records == []
    assert summary.total_documents == 0
    assert summary.excluded_by_validation == 1
    assert summary.validation_errors[0]["document_id"] == "LR_000001"


# --------------------------------------------------------------------------- summary counts

def test_summary_counts_domains_formats_pages_and_nulls(tmp_path):
    labeled_dir, normalized_dir, source_file = _setup(tmp_path)
    source2 = tmp_path / "source2.jpg"
    source2.write_bytes(b"jpg data")

    _write_normalized_doc(normalized_dir, "LR_000001", source_file, source_format="pdf", page_count=2)
    _write_normalized_doc(normalized_dir, "LR_000002", source2, source_format="jpg", page_count=1)

    _write_label(labeled_dir, "LR_000001", annotation_status="approved", page_count=2,
                 fields={"patient_name": "X", "age": None})
    _write_label(labeled_dir, "LR_000002", annotation_status="approved", domain="milk", page_count=1,
                 fields={"sample_id": "S1"}, tests=[])

    records, summary = build_final_dataset(labeled_dir, normalized_dir)
    assert summary.total_documents == 2
    assert summary.documents_by_domain == {"medical": 1, "milk": 1}
    assert summary.documents_by_format == {"pdf": 1, "jpg": 1}
    assert summary.documents_by_page_count == {"2": 1, "1": 1}
    assert summary.total_fields == 3  # 2 + 1
    assert summary.total_test_records == 1
    assert summary.missing_null_fields == 1  # age=None


# --------------------------------------------------------------------------- output files

def test_write_final_dataset_produces_jsonl_csv_and_summary_with_shared_and_domain_specific_columns(tmp_path):
    labeled_dir, normalized_dir, source_file = _setup(tmp_path)
    source2 = tmp_path / "source2.jpg"
    source2.write_bytes(b"jpg data")

    _write_normalized_doc(normalized_dir, "LR_000001", source_file)
    _write_normalized_doc(normalized_dir, "LR_000002", source2, source_format="jpg")

    _write_label(labeled_dir, "LR_000001", annotation_status="approved",
                 fields={"sample_id": "S1", "patient_name": "X"})
    _write_label(labeled_dir, "LR_000002", annotation_status="approved", domain="milk",
                 fields={"sample_id": "S2", "fat_percent": "4.2"})

    records, summary = build_final_dataset(labeled_dir, normalized_dir)
    output_dir = tmp_path / "final_dataset"
    write_final_dataset(records, summary, output_dir)

    assert (output_dir / "final_labeled_dataset.jsonl").exists()
    assert (output_dir / "final_labeled_dataset.csv").exists()
    assert (output_dir / "dataset_summary.json").exists()
    # No split files of any kind.
    assert not (output_dir / "train.jsonl").exists()
    assert not (output_dir / "test.jsonl").exists()

    jsonl_lines = (output_dir / "final_labeled_dataset.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(jsonl_lines) == 2
    parsed = [json.loads(line) for line in jsonl_lines]
    assert {p["document_id"] for p in parsed} == {"LR_000001", "LR_000002"}

    with (output_dir / "final_labeled_dataset.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    # "sample_id" is common across both domains -> ONE shared column.
    assert "field__sample_id" in rows[0]
    # domain-specific fields both appear as columns, blank where not applicable.
    by_id = {r["document_id"]: r for r in rows}
    assert by_id["LR_000001"]["field__patient_name"] == "X"
    assert by_id["LR_000001"]["field__fat_percent"] == ""
    assert by_id["LR_000002"]["field__fat_percent"] == "4.2"

    on_disk_summary = json.loads((output_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    assert on_disk_summary["total_documents"] == 2
