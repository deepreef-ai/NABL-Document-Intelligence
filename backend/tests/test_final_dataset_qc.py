import json

from app.final_dataset_qc import pipeline
from app.final_dataset_qc.validate import load_jsonl_records


_MASTER_SCHEMA = {
    "domains": {
        "medical": {"keys": ["patient_name", "age", "hemoglobin"]},
        "milk": {"keys": ["sample_id", "fat_percent"]},
    }
}


def _write_master_schema(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "master_schema.json").write_text(json.dumps(_MASTER_SCHEMA), encoding="utf-8")


def _write_normalized_doc(root, document_id, source_path, page_count=1):
    doc_dir = root / "normalized" / document_id
    doc_dir.mkdir(parents=True)
    (doc_dir / "document.json").write_text(json.dumps({
        "document_id": document_id, "source_path": str(source_path), "source_format": "pdf",
        "page_count": page_count, "status": "processed",
        "pages": [{"page_number": i + 1, "text": "x"} for i in range(page_count)],
    }), encoding="utf-8")


def _write_id_registry(root, by_hash):
    root.mkdir(parents=True, exist_ok=True)
    (root / "id_registry.json").write_text(json.dumps({"by_hash": by_hash, "next_seq": len(by_hash) + 1}), encoding="utf-8")


def _record(document_id="LR_000001", **overrides):
    record = {
        "document_id": document_id,
        "original_filename": f"{document_id}.pdf",
        "domain": "medical",
        "document_type": "lab_report",
        "source_format": "pdf",
        "source_path": "",
        "page_count": 1,
        "fields": {"patient_name": "X", "age": None},
        "tests": [{"test_name": "Hemoglobin", "result": "13.5", "unit": "g/dL", "reference_range": "13-17"}],
        "annotation_status": "approved",
    }
    record.update(overrides)
    return record


def _write_jsonl(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _full_setup(tmp_path, record_overrides=None, content_hash_for=None):
    """Builds a normalized doc + id_registry + one matching record, wired
    together so the "happy path" traces correctly end to end; tests then
    override pieces of this to break one thing at a time."""
    import hashlib

    final_dir = tmp_path / "final_dataset"
    normalized_dir = tmp_path / "normalized_dataset"
    master_schema_dir = tmp_path / "master_schema"
    _write_master_schema(master_schema_dir)

    source_file = tmp_path / "source.pdf"
    source_file.write_bytes(b"%PDF-1.4 fake content")
    digest = hashlib.sha256(source_file.read_bytes()).hexdigest()

    document_id = "LR_000001"
    _write_normalized_doc(normalized_dir, document_id, source_file)
    _write_id_registry(normalized_dir, {digest: document_id})

    record = _record(document_id=document_id, source_path=str(source_file))
    if record_overrides:
        record.update(record_overrides)

    final_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(final_dir / "final_labeled_dataset.jsonl", [json.dumps(record)])
    return final_dir, normalized_dir, master_schema_dir


# --------------------------------------------------------------------------- happy path

def test_a_fully_valid_traceable_record_passes(tmp_path):
    final_dir, normalized_dir, master_schema_dir = _full_setup(tmp_path)
    report = pipeline.run(final_dir, normalized_dir, master_schema_dir)

    assert report.total_documents == 1
    assert report.valid_documents == 1
    assert report.invalid_documents == 0
    hard_errors = [e for e in report.errors if e["severity"] == "hard"]
    assert hard_errors == []
    # age=None is a soft "missing value" note, not an error.
    assert report.missing_values == 1


# --------------------------------------------------------------------------- hard checks

def test_duplicate_document_id_is_detected(tmp_path):
    final_dir, normalized_dir, master_schema_dir = _full_setup(tmp_path)
    # Append a second record with the same document_id.
    existing = (final_dir / "final_labeled_dataset.jsonl").read_text(encoding="utf-8")
    with (final_dir / "final_labeled_dataset.jsonl").open("a", encoding="utf-8") as f:
        f.write(existing)

    report = pipeline.run(final_dir, normalized_dir, master_schema_dir)
    assert report.duplicate_documents == 2
    assert report.invalid_documents == 2


def test_invalid_domain_is_detected(tmp_path):
    final_dir, normalized_dir, master_schema_dir = _full_setup(tmp_path, {"domain": "astrology"})
    report = pipeline.run(final_dir, normalized_dir, master_schema_dir)
    assert report.invalid_documents == 1
    assert any(e["check"] == "valid_domain" for e in report.errors)


def test_invalid_source_format_is_detected(tmp_path):
    final_dir, normalized_dir, master_schema_dir = _full_setup(tmp_path, {"source_format": "docx"})
    report = pipeline.run(final_dir, normalized_dir, master_schema_dir)
    assert report.invalid_documents == 1
    assert any(e["check"] == "valid_source_format" for e in report.errors)


def test_non_approved_annotation_is_detected(tmp_path):
    final_dir, normalized_dir, master_schema_dir = _full_setup(tmp_path, {"annotation_status": "pending"})
    report = pipeline.run(final_dir, normalized_dir, master_schema_dir)
    assert report.invalid_annotations == 1
    assert report.invalid_documents == 1
    assert any(e["check"] == "annotation_approved" for e in report.errors)


def test_malformed_test_row_is_detected(tmp_path):
    final_dir, normalized_dir, master_schema_dir = _full_setup(tmp_path, {
        "tests": [{"result": "13.5"}],  # missing test_name/unit/reference_range
    })
    report = pipeline.run(final_dir, normalized_dir, master_schema_dir)
    assert report.invalid_documents == 1
    assert any(e["check"] == "test_row_relationship" for e in report.errors)


def test_page_count_mismatch_breaks_multi_page_check(tmp_path):
    final_dir, normalized_dir, master_schema_dir = _full_setup(tmp_path, {"page_count": 5})
    report = pipeline.run(final_dir, normalized_dir, master_schema_dir)
    assert report.invalid_documents == 1
    assert any(e["check"] == "multi_page_preserved" for e in report.errors)


def test_source_file_not_traceable_to_id_registry_is_detected(tmp_path):
    final_dir, normalized_dir, master_schema_dir = _full_setup(tmp_path)
    # Corrupt the registry so the real file's hash no longer maps back.
    (normalized_dir / "id_registry.json").write_text(json.dumps({"by_hash": {}, "next_seq": 1}), encoding="utf-8")

    report = pipeline.run(final_dir, normalized_dir, master_schema_dir)
    assert report.invalid_documents == 1
    assert any(e["check"] == "source_traceable" for e in report.errors)


def test_duplicate_json_key_in_raw_text_is_detected(tmp_path):
    final_dir = tmp_path / "final_dataset"
    normalized_dir = tmp_path / "normalized_dataset"
    master_schema_dir = tmp_path / "master_schema"
    _write_master_schema(master_schema_dir)
    final_dir.mkdir(parents=True)

    # Hand-crafted raw JSON with a literal duplicate top-level key —
    # json.dumps can never produce this, so it must be written by hand to
    # prove the object_pairs_hook actually catches it.
    raw = (
        '{"document_id": "LR_000001", "document_id": "LR_000001", "domain": "medical", '
        '"source_format": "pdf", "source_path": "", "page_count": 1, '
        '"fields": {}, "tests": [], "annotation_status": "approved"}'
    )
    (final_dir / "final_labeled_dataset.jsonl").write_text(raw + "\n", encoding="utf-8")

    report = pipeline.run(final_dir, normalized_dir, master_schema_dir)
    assert report.invalid_documents == 1
    assert any(e["check"] == "no_duplicate_keys" for e in report.errors)


def test_malformed_jsonl_line_is_reported(tmp_path):
    final_dir = tmp_path / "final_dataset"
    normalized_dir = tmp_path / "normalized_dataset"
    master_schema_dir = tmp_path / "master_schema"
    _write_master_schema(master_schema_dir)
    final_dir.mkdir(parents=True)
    (final_dir / "final_labeled_dataset.jsonl").write_text("{not valid json\n", encoding="utf-8")

    report = pipeline.run(final_dir, normalized_dir, master_schema_dir)
    assert report.total_documents == 0
    assert report.invalid_documents == 1
    assert any(e["check"] == "jsonl_valid" for e in report.errors)


# --------------------------------------------------------------------------- soft checks

def test_unexpected_key_is_counted_but_not_invalidating(tmp_path):
    final_dir, normalized_dir, master_schema_dir = _full_setup(tmp_path, {
        "fields": {"patient_name": "X", "some_new_field_never_seen": "value"},
    })
    report = pipeline.run(final_dir, normalized_dir, master_schema_dir)
    assert report.valid_documents == 1  # unexpected key alone doesn't invalidate
    assert report.unexpected_keys == 1
    assert any(e["check"] == "no_unexpected_keys" and e["severity"] == "soft" for e in report.errors)


def test_placeholder_value_counts_as_a_missing_value(tmp_path):
    final_dir, normalized_dir, master_schema_dir = _full_setup(tmp_path, {
        "fields": {"patient_name": "N/A", "age": "45"},
    })
    report = pipeline.run(final_dir, normalized_dir, master_schema_dir)
    assert report.valid_documents == 1
    assert report.missing_values == 1


# --------------------------------------------------------------------------- helpers

def test_load_jsonl_records_skips_blank_lines(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    records, errors = load_jsonl_records(path)
    assert len(records) == 2
    assert errors == []
