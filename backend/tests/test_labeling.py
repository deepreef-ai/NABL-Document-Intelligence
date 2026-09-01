import json

from app.labeling import extraction, pipeline
from app.labeling.schema_hints import build_domain_hints, format_hints_block


# --------------------------------------------------------------------------- helpers

class FakeChain:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def generate_json(self, system, user_text):
        self.calls += 1
        return self.reply


_SAMPLE_MASTER_SCHEMA = {
    "domains": {
        "medical": {
            "keys": ["patient_name", "age", "hemoglobin"],
            "key_details": [
                {"canonical_key": "patient_name", "field_role": "document_field", "total_frequency": 5},
                {"canonical_key": "age", "field_role": "document_field", "total_frequency": 4},
                {"canonical_key": "hemoglobin", "field_role": "parameter", "total_frequency": 3},
            ],
        },
        "milk": {"keys": [], "key_details": []},
    }
}


def _write_normalized_doc(root, document_id, filename, text, ocr_used=False, ocr_confidence=None, page_count=1):
    doc_dir = root / "normalized" / document_id
    doc_dir.mkdir(parents=True)
    (doc_dir / "document.json").write_text(json.dumps({
        "document_id": document_id, "original_filename": filename, "source_format": "pdf",
        "source_type": "born_digital_pdf", "page_count": page_count, "status": "processed",
        "pages": [{
            "page_number": 1, "text": text,
            "extraction_method": "ocr" if ocr_used else "pymupdf",
            "ocr_used": ocr_used, "ocr_confidence": ocr_confidence, "elements": [],
        }],
    }), encoding="utf-8")


def _write_master_schema(root, data=_SAMPLE_MASTER_SCHEMA):
    root.mkdir(parents=True, exist_ok=True)
    (root / "master_schema.json").write_text(json.dumps(data), encoding="utf-8")


# --------------------------------------------------------------------------- schema_hints

def test_build_domain_hints_splits_by_field_role_and_caps_length():
    hints = build_domain_hints(_SAMPLE_MASTER_SCHEMA)
    assert hints["medical"]["fields"] == ["patient_name", "age"]
    assert hints["medical"]["parameters"] == ["hemoglobin"]
    assert hints["milk"] == {"fields": [], "parameters": []}


def test_format_hints_block_lists_every_domain():
    block = format_hints_block(build_domain_hints(_SAMPLE_MASTER_SCHEMA))
    assert "medical" in block
    assert "patient_name" in block
    assert "milk" in block
    assert "(none known yet)" in block


# --------------------------------------------------------------------------- extraction

def test_extract_label_always_includes_hinted_fields_with_null_when_absent(monkeypatch):
    fake_chain = FakeChain({
        "domain": "medical",
        "fields": {"patient_name": "John Doe"},  # "age" not mentioned by the model
        "tests": [{"test_name": "Hemoglobin", "result": "13.5", "unit": "g/dL", "reference_range": "13-17"}],
    })
    monkeypatch.setattr(extraction, "get_llm_chain", lambda: fake_chain)

    hints = build_domain_hints(_SAMPLE_MASTER_SCHEMA)
    label = extraction.extract_label(
        document_id="LR_000001", original_filename="a.pdf", text="Patient Name: John Doe. Hemoglobin 13.5 g/dL.",
        page_count=1, source_ocr_used=False, source_ocr_confidence=None, domain_hints=hints,
    )

    assert fake_chain.calls == 1
    assert label.domain == "medical"
    assert label.fields == {"patient_name": "John Doe", "age": None}  # hinted field always present, null if absent
    assert label.tests[0].test_name == "Hemoglobin"
    assert label.tests[0].result == "13.5"
    assert label.tests[0].unit == "g/dL"
    assert label.tests[0].reference_range == "13-17"
    assert label.annotation_status == "pending"
    assert label.extraction_status == "ok"


def test_extract_label_keeps_extra_fields_not_on_the_hint_list(monkeypatch):
    fake_chain = FakeChain({
        "domain": "medical",
        "fields": {"patient_name": "Jane", "age": 42, "unexpected_field": "some value"},
        "tests": [],
    })
    monkeypatch.setattr(extraction, "get_llm_chain", lambda: fake_chain)

    hints = build_domain_hints(_SAMPLE_MASTER_SCHEMA)
    label = extraction.extract_label(
        document_id="LR_000002", original_filename="b.pdf", text="text",
        page_count=1, source_ocr_used=False, source_ocr_confidence=None, domain_hints=hints,
    )

    assert label.fields["patient_name"] == "Jane"
    assert label.fields["age"] == 42
    assert label.fields["unexpected_field"] == "some value"


def test_extract_label_defaults_unknown_domain_to_other(monkeypatch):
    fake_chain = FakeChain({"domain": "astrology", "fields": {}, "tests": []})
    monkeypatch.setattr(extraction, "get_llm_chain", lambda: fake_chain)

    label = extraction.extract_label(
        document_id="LR_000003", original_filename="c.pdf", text="text",
        page_count=1, source_ocr_used=False, source_ocr_confidence=None, domain_hints={},
    )
    assert label.domain == "other"


def test_extract_label_ignores_malformed_test_rows(monkeypatch):
    fake_chain = FakeChain({
        "domain": "medical",
        "fields": {},
        "tests": ["not a dict", {"result": "13.5"}, {"test_name": "Hemoglobin", "result": "13.5"}],
    })
    monkeypatch.setattr(extraction, "get_llm_chain", lambda: fake_chain)

    label = extraction.extract_label(
        document_id="LR_000004", original_filename="d.pdf", text="text",
        page_count=1, source_ocr_used=False, source_ocr_confidence=None, domain_hints={},
    )
    assert len(label.tests) == 1
    assert label.tests[0].test_name == "Hemoglobin"
    assert label.tests[0].unit is None
    assert label.tests[0].reference_range is None


# --------------------------------------------------------------------------- pipeline

def test_pipeline_run_labels_every_document_and_carries_ocr_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(extraction, "get_llm_chain", lambda: FakeChain({
        "domain": "medical", "fields": {"patient_name": "X"},
        "tests": [{"test_name": "Hemoglobin", "result": "13.5", "unit": "g/dL", "reference_range": "13-17"}],
    }))

    normalized_dir = tmp_path / "normalized_dataset"
    _write_normalized_doc(normalized_dir, "LR_000001", "a.pdf", "Patient Name: X. Hemoglobin 13.5 g/dL.",
                          ocr_used=True, ocr_confidence=0.93)
    master_schema_dir = tmp_path / "master_schema"
    _write_master_schema(master_schema_dir)
    output_dir = tmp_path / "labeled_dataset"

    stats = pipeline.run(normalized_dir, master_schema_dir, output_dir)

    assert stats.total_documents == 1
    assert stats.processed == 1
    assert stats.failed == 0

    label = json.loads((output_dir / "labels" / "LR_000001.json").read_text(encoding="utf-8"))
    assert label["domain"] == "medical"
    assert label["annotation_status"] == "pending"
    assert label["source_ocr_used"] is True
    assert label["source_ocr_confidence"] == 0.93
    assert label["page_count"] == 1

    index_lines = (output_dir / "label_index.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(index_lines) == 1
    assert json.loads(index_lines[0])["document_id"] == "LR_000001"


def test_pipeline_is_resumable_and_continues_after_a_failure(tmp_path, monkeypatch):
    calls = {"n": 0}

    class FlakyChain:
        def generate_json(self, system, user_text):
            calls["n"] += 1
            if "BROKEN" in user_text:
                raise RuntimeError("provider exploded")
            return {"domain": "medical", "fields": {}, "tests": []}

    monkeypatch.setattr(extraction, "get_llm_chain", lambda: FlakyChain())

    normalized_dir = tmp_path / "normalized_dataset"
    _write_normalized_doc(normalized_dir, "LR_000001", "good.pdf", "Patient Name: X.")
    _write_normalized_doc(normalized_dir, "LR_000002", "bad.pdf", "BROKEN text")
    master_schema_dir = tmp_path / "master_schema"
    _write_master_schema(master_schema_dir)
    output_dir = tmp_path / "labeled_dataset"

    stats = pipeline.run(normalized_dir, master_schema_dir, output_dir)
    assert stats.processed == 1
    assert stats.failed == 1

    failed_label = json.loads((output_dir / "labels" / "LR_000002.json").read_text(encoding="utf-8"))
    assert failed_label["extraction_status"] == "failed"
    assert failed_label["annotation_status"] == "pending"
    assert failed_label["error"]

    # Resuming without --force must not re-call the LLM for either document.
    second = pipeline.run(normalized_dir, master_schema_dir, output_dir)
    assert second.skipped == 2
    assert calls["n"] == 2  # only the first run's two attempts
