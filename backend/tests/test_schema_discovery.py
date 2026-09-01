import json

from app.schema_discovery import llm_discovery, pipeline
from app.schema_discovery.domains import heuristic_domain
from app.schema_discovery.sampling import NormalizedDocRecord, load_normalized_documents, select_representative_sample


# --------------------------------------------------------------------------- helpers

class FakeChain:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def generate_json(self, system, user_text):
        self.calls += 1
        return self.reply


def _write_normalized_doc(root, document_id, filename, text, status="processed"):
    doc_dir = root / "normalized" / document_id
    doc_dir.mkdir(parents=True)
    (doc_dir / "document.json").write_text(json.dumps({
        "document_id": document_id,
        "original_filename": filename,
        "source_format": "pdf",
        "source_type": "born_digital_pdf",
        "page_count": 1,
        "status": status,
        "pages": [{"page_number": 1, "text": text, "extraction_method": "pymupdf", "ocr_used": False, "elements": []}],
    }), encoding="utf-8")


# --------------------------------------------------------------------------- domains.heuristic_domain

def test_heuristic_domain_picks_highest_scoring_bucket():
    assert heuristic_domain("Patient Name: John Doe. Hemoglobin 13.5, WBC count normal.") == "medical"
    assert heuristic_domain("Milk Sample. Fat %: 4.2, SNF: 8.5, Lactometer reading 28.") == "milk"


def test_heuristic_domain_falls_back_to_other_when_no_keywords_match():
    assert heuristic_domain("Lorem ipsum dolor sit amet consectetur") == "other"


# --------------------------------------------------------------------------- sampling

def test_load_normalized_documents_skips_failed_and_concatenates_pages(tmp_path):
    _write_normalized_doc(tmp_path, "LR_000001", "a.pdf", "Hello world")
    _write_normalized_doc(tmp_path, "LR_000002", "b.pdf", "irrelevant", status="failed")

    records = load_normalized_documents(tmp_path)
    assert [r.document_id for r in records] == ["LR_000001"]
    assert records[0].text == "Hello world"


def test_select_representative_sample_caps_per_bucket_and_spans_length_distribution():
    records = [
        NormalizedDocRecord(document_id=f"LR_{i:06d}", original_filename=f"f{i}.pdf",
                             text="patient hemoglobin " * (i + 1), char_count=(i + 1) * 20)
        for i in range(10)
    ]
    sample = select_representative_sample(records, max_per_domain=3)
    assert list(sample.keys()) == ["medical"]
    assert len(sample["medical"]) == 3
    # Shortest and longest documents in the bucket must both be represented.
    ids = {r.document_id for r in sample["medical"]}
    assert "LR_000000" in ids
    assert "LR_000009" in ids


def test_select_representative_sample_does_not_cap_a_bucket_smaller_than_the_limit():
    records = [
        NormalizedDocRecord(document_id="LR_000001", original_filename="a.pdf", text="milk fat snf", char_count=12),
    ]
    sample = select_representative_sample(records, max_per_domain=5)
    assert len(sample["milk"]) == 1


# --------------------------------------------------------------------------- llm_discovery

def test_discover_document_keys_normalizes_domain_and_keys(monkeypatch):
    fake_chain = FakeChain({
        "candidate_domain": "Medical",
        "keys": ["Patient Name", "  Hemoglobin (g/dL)  ", "patient_name", "WBC Count", 42, None],
    })
    monkeypatch.setattr(llm_discovery, "get_llm_chain", lambda: fake_chain)

    result = llm_discovery.discover_document_keys("LR_000001", "some document text")

    assert fake_chain.calls == 1
    assert result.document_id == "LR_000001"
    assert result.candidate_domain == "medical"
    # Case/whitespace variants collapse; duplicates (patient_name appearing
    # twice in different forms) are deduplicated; non-string entries dropped.
    assert result.keys == ["patient_name", "hemoglobin_g_dl", "wbc_count"]


def test_discover_document_keys_defaults_unknown_domain_to_other(monkeypatch):
    fake_chain = FakeChain({"candidate_domain": "astrology", "keys": ["star_sign"]})
    monkeypatch.setattr(llm_discovery, "get_llm_chain", lambda: fake_chain)

    result = llm_discovery.discover_document_keys("LR_000002", "text")
    assert result.candidate_domain == "other"


def test_discover_document_keys_prompt_includes_document_text(monkeypatch):
    captured = {}

    class CapturingChain:
        def generate_json(self, system, user_text):
            captured["user_text"] = user_text
            captured["system"] = system
            return {"candidate_domain": "other", "keys": []}

    monkeypatch.setattr(llm_discovery, "get_llm_chain", lambda: CapturingChain())

    llm_discovery.discover_document_keys("LR_000003", "Sample Report Fat % 4.2")

    assert "Sample Report Fat % 4.2" in captured["user_text"]
    assert "candidate_domain" in captured["user_text"]


# --------------------------------------------------------------------------- pipeline (end-to-end over a tmp dataset)

def test_pipeline_run_samples_and_writes_per_document_and_domain_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(
        llm_discovery, "get_llm_chain",
        lambda: FakeChain({"candidate_domain": "medical", "keys": ["patient_name", "hemoglobin"]}),
    )

    input_dir = tmp_path / "normalized_dataset"
    _write_normalized_doc(input_dir, "LR_000001", "a.pdf", "Patient Name: X. Hemoglobin 13.5.")
    _write_normalized_doc(input_dir, "LR_000002", "b.pdf", "Patient Name: Y. Hemoglobin 14.0.")

    output_dir = tmp_path / "schema_discovery"
    stats = pipeline.run(input_dir, output_dir, max_per_domain=5)

    assert stats.total_documents == 2
    assert stats.sampled == 2
    assert stats.processed == 2
    assert stats.failed == 0

    sample_files = sorted((output_dir / "samples").glob("*.json"))
    assert len(sample_files) == 2
    doc = json.loads(sample_files[0].read_text(encoding="utf-8"))
    assert set(doc.keys()) == {"document_id", "candidate_domain", "keys"}
    assert doc["candidate_domain"] == "medical"
    assert doc["keys"] == ["patient_name", "hemoglobin"]

    domain_keys = json.loads((output_dir / "domain_keys.json").read_text(encoding="utf-8"))
    assert domain_keys["medical"]["sample_count"] == 2
    assert domain_keys["medical"]["keys"] == ["hemoglobin", "patient_name"]
    assert domain_keys["medical"]["key_frequency"] == {"patient_name": 2, "hemoglobin": 2}


def test_pipeline_is_resumable_and_does_not_recall_llm_unless_forced(tmp_path, monkeypatch):
    fake_chain = FakeChain({"candidate_domain": "milk", "keys": ["fat", "snf"]})
    monkeypatch.setattr(llm_discovery, "get_llm_chain", lambda: fake_chain)

    input_dir = tmp_path / "normalized_dataset"
    _write_normalized_doc(input_dir, "LR_000001", "a.pdf", "Milk Sample Fat % 4.2 SNF 8.5")
    output_dir = tmp_path / "schema_discovery"

    first = pipeline.run(input_dir, output_dir, max_per_domain=5)
    assert first.processed == 1
    assert fake_chain.calls == 1

    second = pipeline.run(input_dir, output_dir, max_per_domain=5)
    assert second.skipped == 1
    assert second.processed == 0
    assert fake_chain.calls == 1  # not called again

    third = pipeline.run(input_dir, output_dir, max_per_domain=5, force=True)
    assert third.processed == 1
    assert fake_chain.calls == 2


def test_pipeline_continues_after_one_document_fails(tmp_path, monkeypatch):
    calls = {"n": 0}

    class FlakyChain:
        def generate_json(self, system, user_text):
            calls["n"] += 1
            if "BAD" in user_text:
                raise RuntimeError("provider exploded")
            return {"candidate_domain": "food", "keys": ["moisture"]}

    monkeypatch.setattr(llm_discovery, "get_llm_chain", lambda: FlakyChain())

    input_dir = tmp_path / "normalized_dataset"
    _write_normalized_doc(input_dir, "LR_000001", "good.pdf", "Food product moisture ash content")
    _write_normalized_doc(input_dir, "LR_000002", "bad.pdf", "BAD unrelated text with no domain keywords")
    output_dir = tmp_path / "schema_discovery"

    stats = pipeline.run(input_dir, output_dir, max_per_domain=5)

    assert stats.processed == 1
    assert stats.failed == 1
    assert stats.failures[0][0] == "LR_000002"
    assert len(list((output_dir / "samples").glob("*.json"))) == 1

    index_lines = (output_dir / "discovery_index.jsonl").read_text(encoding="utf-8").strip().splitlines()
    statuses = {json.loads(line)["status"] for line in index_lines}
    assert statuses == {"processed", "failed"}
