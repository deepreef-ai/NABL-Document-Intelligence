import json

from app.benchmark import pipeline, run_pipeline
from app.benchmark.accumulator import MetricAccumulator
from app.benchmark.compare import compare_fields, compare_tests
from app.dataset_normalization.models import NormalizedDocument, NormalizedPage
from app.labeling.models import DocumentLabel, TestRow


# --------------------------------------------------------------------------- compare_fields

def test_compare_fields_counts_exact_matches_and_both_null_as_correct():
    gt = {"patient_name": "John Doe", "age": None}
    pred = {"patient_name": "John Doe", "age": None}
    failures, counters = compare_fields("LR_1", gt, pred)
    assert failures == []
    assert counters["field_correct"] == 2
    assert counters["field_total"] == 2
    assert counters["exact_match"] == 1


def test_compare_fields_detects_wrong_value():
    gt = {"age": "45"}
    pred = {"age": "46"}
    failures, counters = compare_fields("LR_1", gt, pred)
    assert len(failures) == 1
    assert failures[0].error_type == "wrong_value"
    assert counters["field_correct"] == 0


def test_compare_fields_detects_missing_value():
    gt = {"patient_name": "John Doe"}
    pred = {}
    failures, counters = compare_fields("LR_1", gt, pred)
    assert len(failures) == 1
    assert failures[0].error_type == "missing"
    assert counters["missing"] == 1
    assert counters["key_fn"] == 1


def test_compare_fields_detects_wrong_key_when_value_found_elsewhere():
    gt = {"patient_name": "John Doe", "other_field": None}
    pred = {"other_field": "John Doe"}  # value landed under the wrong key
    failures, counters = compare_fields("LR_1", gt, pred)
    wrong_key_failures = [f for f in failures if f.error_type == "wrong_key"]
    assert len(wrong_key_failures) == 1
    assert wrong_key_failures[0].key == "patient_name"


def test_compare_fields_detects_extra_hallucinated_field():
    gt = {"patient_name": "John Doe"}
    pred = {"patient_name": "John Doe", "invented_field": "some value"}
    failures, counters = compare_fields("LR_1", gt, pred)
    extra_failures = [f for f in failures if f.error_type == "extra"]
    assert len(extra_failures) == 1
    assert extra_failures[0].key == "invented_field"
    assert counters["extra"] == 1


def test_compare_fields_flags_hallucination_against_a_null_ground_truth():
    gt = {"patient_name": None}
    pred = {"patient_name": "Someone"}
    failures, _ = compare_fields("LR_1", gt, pred)
    assert failures[0].error_type == "wrong_value"
    assert failures[0].ground_truth is None


def test_compare_fields_treats_placeholder_text_as_null():
    gt = {"age": "N/A"}
    pred = {"age": None}
    failures, counters = compare_fields("LR_1", gt, pred)
    assert failures == []
    assert counters["field_correct"] == 1


# --------------------------------------------------------------------------- compare_tests

def test_compare_tests_matches_by_normalized_test_name_and_scores_result_unit_range():
    gt = [{"test_name": "Hemoglobin", "result": "13.5", "unit": "g/dL", "reference_range": "13-17"}]
    pred = [{"test_name": "hemoglobin", "result": "13.5", "unit": "g/dL", "reference_range": "13-17"}]
    failures, counters = compare_tests("LR_1", gt, pred)
    assert failures == []
    assert counters["result_correct"] == 1
    assert counters["unit_correct"] == 1
    assert counters["reference_range_correct"] == 1


def test_compare_tests_flags_wrong_unit_specifically():
    gt = [{"test_name": "Hemoglobin", "result": "13.5", "unit": "g/dL", "reference_range": "13-17"}]
    pred = [{"test_name": "Hemoglobin", "result": "13.5", "unit": "mg/dL", "reference_range": "13-17"}]
    failures, counters = compare_tests("LR_1", gt, pred)
    unit_failures = [f for f in failures if f.error_type == "wrong_unit"]
    assert len(unit_failures) == 1
    assert counters["unit_correct"] == 0
    assert counters["result_correct"] == 1


def test_compare_tests_detects_missing_test_row():
    gt = [{"test_name": "Hemoglobin", "result": "13.5", "unit": "g/dL", "reference_range": "13-17"}]
    pred = []
    failures, counters = compare_tests("LR_1", gt, pred)
    assert any(f.error_type == "missing" for f in failures)
    assert counters["test_matched"] == 0


def test_compare_tests_detects_extra_test_row():
    gt = []
    pred = [{"test_name": "Glucose", "result": "90", "unit": "mg/dL", "reference_range": "70-100"}]
    failures, counters = compare_tests("LR_1", gt, pred)
    assert any(f.error_type == "extra" for f in failures)


def test_compare_tests_detects_wrong_key_when_result_reassigned_to_different_test_name():
    gt = [{"test_name": "Hemoglobin", "result": "13.5", "unit": "g/dL", "reference_range": "13-17"}]
    pred = [{"test_name": "Haemoglobin Level", "result": "13.5", "unit": "g/dL", "reference_range": "13-17"}]
    failures, _ = compare_tests("LR_1", gt, pred)
    assert any(f.error_type == "wrong_key" for f in failures)


# --------------------------------------------------------------------------- accumulator

def test_metric_accumulator_finalize_computes_precision_recall_f1():
    acc = MetricAccumulator()
    acc.add(
        domain_match=True, extraction_ok=True,
        field_counters={"key_tp": 3, "key_fp": 1, "key_fn": 1, "field_correct": 3, "field_total": 4,
                         "exact_match": 0, "missing": 1, "gt_nonnull_total": 4, "extra": 1,
                         "predicted_nonnull_total": 4, "value_correct": 3, "value_total": 4},
        test_counters={"test_gt_total": 0, "test_matched": 0, "test_pred_total": 0,
                        "result_correct": 0, "unit_correct": 0, "reference_range_correct": 0, "matched_count": 0},
    )
    metrics = acc.finalize()
    assert metrics["key_precision"] == 0.75  # 3/4
    assert metrics["key_recall"] == 0.75     # 3/4
    assert metrics["key_f1"] == 0.75
    assert metrics["missing_field_rate"] == 0.25  # 1/4
    assert metrics["hallucination_rate"] == 0.25  # 1/4


def test_metric_accumulator_returns_none_for_undefined_ratios_with_no_data():
    acc = MetricAccumulator()
    metrics = acc.finalize()
    assert metrics["key_precision"] is None
    assert metrics["key_f1"] is None
    assert metrics["document_count"] == 0


# --------------------------------------------------------------------------- run_pipeline wrapper

def test_run_production_pipeline_wraps_normalize_and_extract(tmp_path, monkeypatch):
    source = tmp_path / "a.pdf"
    source.write_bytes(b"fake pdf bytes")

    fake_normalized = NormalizedDocument(
        document_id="LR_000001", original_filename="a.pdf", source_path=str(source), source_format="pdf",
        source_type="born_digital_pdf", page_count=1, status="processed",
        pages=[NormalizedPage(page_number=1, text="Patient Name: X", extraction_method="pymupdf", ocr_used=False)],
    )
    monkeypatch.setattr(run_pipeline, "normalize_file", lambda *a, **k: fake_normalized)

    fake_label = DocumentLabel(
        document_id="LR_000001", original_filename="a.pdf", domain="medical", page_count=1,
        source_ocr_used=False, source_ocr_confidence=None, fields={"patient_name": "X"},
        tests=[TestRow(test_name="Hemoglobin", result="13.5", unit="g/dL", reference_range="13-17")],
        annotation_status="pending", extraction_status="ok",
    )
    monkeypatch.setattr(run_pipeline, "extract_label", lambda **k: fake_label)

    result = run_pipeline.run_production_pipeline("LR_000001", source, {})
    assert result["pipeline_error"] is None
    assert result["domain"] == "medical"
    assert result["fields"] == {"patient_name": "X"}
    assert result["tests"][0]["test_name"] == "Hemoglobin"


def test_run_production_pipeline_scores_an_exception_as_a_failure(tmp_path, monkeypatch):
    source = tmp_path / "a.pdf"
    source.write_bytes(b"fake")

    def boom(*a, **k):
        raise RuntimeError("OCR exploded")

    monkeypatch.setattr(run_pipeline, "normalize_file", boom)

    result = run_pipeline.run_production_pipeline("LR_000001", source, {})
    assert result["pipeline_error"] == "RuntimeError: OCR exploded"
    assert result["domain"] is None
    assert result["fields"] == {}


# --------------------------------------------------------------------------- predict (phase 1, resumable, mocked production call)

def _write_ground_truth(final_dir, records):
    final_dir.mkdir(parents=True, exist_ok=True)
    with (final_dir / "final_labeled_dataset.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _write_normalized_doc(normalized_dir, document_id, source_type="born_digital_pdf"):
    doc_dir = normalized_dir / "normalized" / document_id
    doc_dir.mkdir(parents=True)
    (doc_dir / "document.json").write_text(json.dumps({
        "document_id": document_id, "source_type": source_type, "page_count": 1, "status": "processed", "pages": [],
    }), encoding="utf-8")


def _write_master_schema(master_schema_dir):
    master_schema_dir.mkdir(parents=True, exist_ok=True)
    (master_schema_dir / "master_schema.json").write_text(json.dumps({"domains": {}}), encoding="utf-8")


def _write_prediction(predictions_dir, document_id, **overrides):
    predictions_dir.mkdir(parents=True, exist_ok=True)
    prediction = {"domain": None, "fields": {}, "tests": [], "page_count": 1, "pipeline_error": None}
    prediction.update(overrides)
    (predictions_dir / f"{document_id}.json").write_text(json.dumps(prediction), encoding="utf-8")


def test_generate_predictions_calls_pipeline_once_per_document_and_caches_to_disk(tmp_path, monkeypatch):
    from app.benchmark import predict

    final_dir = tmp_path / "final_dataset"
    master_schema_dir = tmp_path / "master_schema"
    _write_master_schema(master_schema_dir)
    source = tmp_path / "a.pdf"
    source.write_bytes(b"x")
    _write_ground_truth(final_dir, [
        {"document_id": "LR_000001", "domain": "medical", "source_format": "pdf", "source_path": str(source),
         "page_count": 1, "fields": {}, "tests": [], "annotation_status": "approved"},
    ])

    calls = {"n": 0}

    def fake_run(document_id, source_path, domain_hints):
        calls["n"] += 1
        return {"domain": "medical", "fields": {"patient_name": "X"}, "tests": [], "page_count": 1, "pipeline_error": None}

    monkeypatch.setattr(predict, "run_production_pipeline", fake_run)

    predictions_dir = tmp_path / "predictions"
    stats = predict.generate_predictions(final_dir, predictions_dir, master_schema_dir)
    assert stats.predicted == 1
    assert calls["n"] == 1
    cached = json.loads((predictions_dir / "LR_000001.json").read_text(encoding="utf-8"))
    assert cached["fields"] == {"patient_name": "X"}

    # Resuming without --force must not re-call the pipeline.
    stats2 = predict.generate_predictions(final_dir, predictions_dir, master_schema_dir)
    assert stats2.skipped == 1
    assert calls["n"] == 1


def test_generate_predictions_continues_after_one_document_fails(tmp_path, monkeypatch):
    from app.benchmark import predict

    final_dir = tmp_path / "final_dataset"
    master_schema_dir = tmp_path / "master_schema"
    _write_master_schema(master_schema_dir)
    good_source = tmp_path / "good.pdf"
    good_source.write_bytes(b"x")
    bad_source = tmp_path / "bad.pdf"
    bad_source.write_bytes(b"y")
    _write_ground_truth(final_dir, [
        {"document_id": "LR_000001", "domain": "medical", "source_format": "pdf", "source_path": str(good_source),
         "page_count": 1, "fields": {}, "tests": [], "annotation_status": "approved"},
        {"document_id": "LR_000002", "domain": "medical", "source_format": "pdf", "source_path": str(bad_source),
         "page_count": 1, "fields": {}, "tests": [], "annotation_status": "approved"},
    ])

    def fake_run(document_id, source_path, domain_hints):
        if document_id == "LR_000002":
            return {"domain": None, "fields": {}, "tests": [], "page_count": None, "pipeline_error": "boom"}
        return {"domain": "medical", "fields": {}, "tests": [], "page_count": 1, "pipeline_error": None}

    from app.benchmark import predict as predict_module
    monkeypatch.setattr(predict_module, "run_production_pipeline", fake_run)

    predictions_dir = tmp_path / "predictions"
    stats = predict_module.generate_predictions(final_dir, predictions_dir, master_schema_dir)
    assert stats.predicted == 1
    assert stats.failed == 1
    assert stats.failures[0][0] == "LR_000002"


# --------------------------------------------------------------------------- score (phase 2, cheap, reads only the cache)

def test_score_uses_perfect_and_imperfect_cached_predictions_and_aggregates_by_domain_and_format(tmp_path):
    final_dir = tmp_path / "final_dataset"
    normalized_dir = tmp_path / "normalized_dataset"

    source1 = tmp_path / "doc1.pdf"
    source1.write_bytes(b"x")
    source2 = tmp_path / "doc2.png"
    source2.write_bytes(b"y")

    _write_ground_truth(final_dir, [
        {"document_id": "LR_000001", "domain": "medical", "source_format": "pdf", "source_path": str(source1),
         "page_count": 1, "fields": {"patient_name": "John"}, "tests": [], "annotation_status": "approved"},
        {"document_id": "LR_000002", "domain": "milk", "source_format": "png", "source_path": str(source2),
         "page_count": 1, "fields": {"sample_id": "S1"}, "tests": [], "annotation_status": "approved"},
    ])
    _write_normalized_doc(normalized_dir, "LR_000001", source_type="born_digital_pdf")
    _write_normalized_doc(normalized_dir, "LR_000002", source_type="image")

    predictions_dir = tmp_path / "predictions"
    _write_prediction(predictions_dir, "LR_000001", domain="medical", fields={"patient_name": "John"})
    _write_prediction(predictions_dir, "LR_000002", domain="food", fields={"sample_id": "WRONG"})

    results, failures, rows = pipeline.score(final_dir, normalized_dir, predictions_dir)

    assert results["overall_summary"]["total_documents"] == 2
    assert results["overall_summary"]["successful_documents"] == 2
    assert results["document_level"]["domain_accuracy"] == 0.5  # 1 of 2 correct

    assert results["by_domain"]["medical"]["document_count"] == 1
    assert results["by_domain"]["medical"]["exact_match_rate"] == 1.0
    assert results["by_domain"]["milk"]["document_count"] == 1
    assert results["by_domain"]["milk"]["exact_match_rate"] == 0.0
    assert results["by_domain"]["water"]["document_count"] == 0  # present even with zero documents

    assert results["by_format"]["born_digital_pdf"]["document_count"] == 1
    assert results["by_format"]["image"]["document_count"] == 1
    assert results["by_format"]["pdf"]["document_count"] == 1  # aggregate bucket

    assert any(f.document_id == "LR_000002" and f.key == "sample_id" for f in failures)
    assert len(rows) == 2


def test_score_treats_a_missing_or_failed_prediction_as_fully_missing(tmp_path):
    final_dir = tmp_path / "final_dataset"
    normalized_dir = tmp_path / "normalized_dataset"

    source = tmp_path / "doc.pdf"
    source.write_bytes(b"x")
    _write_ground_truth(final_dir, [
        {"document_id": "LR_000001", "domain": "medical", "source_format": "pdf", "source_path": str(source),
         "page_count": 1, "fields": {"patient_name": "John"},
         "tests": [{"test_name": "Hemoglobin", "result": "13.5", "unit": "g/dL", "reference_range": "13-17"}],
         "annotation_status": "approved"},
    ])
    _write_normalized_doc(normalized_dir, "LR_000001", source_type="scanned_pdf")

    predictions_dir = tmp_path / "predictions"
    _write_prediction(predictions_dir, "LR_000001", pipeline_error="boom")

    results, failures, rows = pipeline.score(final_dir, normalized_dir, predictions_dir)

    assert results["overall_summary"]["failed_documents"] == 1
    assert results["overall_summary"]["successful_documents"] == 0
    assert results["overall_summary"]["overall_field_accuracy"] == 0.0
    assert results["overall_summary"]["missing_field_rate"] == 1.0
    assert any(f.error_type == "missing" and f.key == "patient_name" for f in failures)
    assert any(f.error_type == "missing" and "Hemoglobin" in f.key for f in failures)
    assert rows[0]["extraction_status"] == "failed"


def test_score_handles_a_document_with_no_cached_prediction_at_all(tmp_path):
    """predict.py may not have reached every document yet (interrupted run)
    — score() must still produce a result for it rather than crash."""
    final_dir = tmp_path / "final_dataset"
    normalized_dir = tmp_path / "normalized_dataset"
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"x")
    _write_ground_truth(final_dir, [
        {"document_id": "LR_000001", "domain": "medical", "source_format": "pdf", "source_path": str(source),
         "page_count": 1, "fields": {"patient_name": "John"}, "tests": [], "annotation_status": "approved"},
    ])
    _write_normalized_doc(normalized_dir, "LR_000001")

    results, _, rows = pipeline.score(final_dir, normalized_dir, tmp_path / "predictions_never_created")
    assert results["overall_summary"]["failed_documents"] == 1
    assert rows[0]["extraction_status"] == "failed"


def test_write_results_produces_json_and_csv_only_no_splits(tmp_path):
    results = {"overall_summary": {"total_documents": 1}, "by_domain": {}, "by_format": {}, "failed_fields": []}
    rows = [{"document_id": "LR_000001", "field_accuracy": 1.0}]
    output_dir = tmp_path / "benchmark"

    pipeline.write_results(results, rows, output_dir)

    assert (output_dir / "benchmark_results.json").exists()
    assert (output_dir / "benchmark_results.csv").exists()
    assert not (output_dir / "train.jsonl").exists()
    assert not (output_dir / "test.jsonl").exists()
    assert not (output_dir / "validation.jsonl").exists()
