"""Scores a predictions/ directory (one JSON per document, produced by
generate_predictions_and_score.py's own prediction phase) against
labelled_dataset's ground truth, pooling results overall, by ground-truth
domain, and by source format (accumulator.py). No train/validation/test
split — every document is scored.

Deliberately reads a prediction cache from disk rather than taking
predictions inline: scoring is cheap and safe to re-run any number of times
(e.g. after fixing a bug in compare.py) without repeating a single
expensive LLM call.

A prediction failure (LLM error, unreadable file — recorded as
pipeline_error) is scored as "predicted nothing": every ground-truth
non-null field/test counts as missing rather than the document being
silently excluded, so a real failure lowers the reported accuracy instead
of disappearing from it. A document with no cached prediction at all is
scored the same way.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from app.benchmark.accumulator import MetricAccumulator
from app.benchmark.compare import compare_fields, compare_tests
from app.benchmark.models import FieldFailure

# Kept here rather than a shared "known domains" module: this is the one
# remaining consumer since schema-discovery-based auto-labeling was
# retired in favor of direct (LLM-free) ground-truth authoring.
CANONICAL_DOMAINS = ["medical", "milk", "food", "water", "soil", "chemical", "other"]


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

_FORMAT_LABELS = {
    "image": ("image",),
    "born_digital_pdf": ("pdf", "born_digital_pdf"),
    "scanned_pdf": ("pdf", "scanned_pdf"),
    "mixed_pdf": ("pdf", "scanned_pdf"),
}

_NO_PREDICTION = {"domain": None, "fields": {}, "tests": [], "pipeline_error": "no cached prediction found"}


def _load_predictions(predictions_dir: Path) -> dict[str, dict]:
    predictions = {}
    if predictions_dir.is_dir():
        for path in predictions_dir.glob("*.json"):
            predictions[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return predictions


def _load_normalized_documents(normalized_dir: Path) -> dict[str, dict]:
    docs = {}
    for path in sorted((normalized_dir / "normalized").glob("*/document.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        docs[data.get("document_id", path.parent.name)] = data
    return docs


def _empty_field_counters(ground_truth_fields: dict) -> dict:
    nonnull = sum(1 for v in ground_truth_fields.values() if v is not None)
    return {
        "key_tp": 0, "key_fp": 0, "key_fn": nonnull,
        "field_correct": 0, "field_total": len(ground_truth_fields),
        "missing": nonnull, "gt_nonnull_total": nonnull,
        "extra": 0, "predicted_nonnull_total": 0,
        "value_correct": 0, "value_total": nonnull,
    }


def _empty_test_counters(ground_truth_tests: list[dict]) -> dict:
    return {
        "test_gt_total": len(ground_truth_tests), "test_matched": 0, "test_pred_total": 0,
        "result_correct": 0, "unit_correct": 0, "reference_range_correct": 0, "matched_count": 0, "extra": 0,
    }


def score(final_dataset_dir: Path, normalized_dir: Path, predictions_dir: Path) -> tuple[dict, list[FieldFailure], list[dict]]:
    records = load_jsonl(final_dataset_dir / "final_labeled_dataset.jsonl")
    doc_jsons = _load_normalized_documents(normalized_dir)
    predictions = _load_predictions(predictions_dir)

    overall = MetricAccumulator()
    by_domain: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    by_format: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    all_failures: list[FieldFailure] = []
    per_document_rows: list[dict] = []
    successful = failed = 0

    for record in records:
        document_id = record["document_id"]
        domain = record.get("domain") if record.get("domain") in CANONICAL_DOMAINS else "other"
        ground_truth_fields = record.get("fields", {})
        ground_truth_tests = record.get("tests", [])
        source_type = doc_jsons.get(document_id, {}).get("source_type", "unknown")

        prediction = predictions.get(document_id, _NO_PREDICTION)
        extraction_ok = prediction.get("pipeline_error") is None
        successful += int(extraction_ok)
        failed += int(not extraction_ok)

        if extraction_ok:
            domain_match = prediction["domain"] == domain
            field_failures, field_counters = compare_fields(document_id, ground_truth_fields, prediction["fields"])
            test_failures, test_counters = compare_tests(document_id, ground_truth_tests, prediction["tests"])
        else:
            domain_match = False
            field_failures = [
                FieldFailure(document_id, k, v, None, "missing") for k, v in ground_truth_fields.items() if v is not None
            ]
            field_counters = _empty_field_counters(ground_truth_fields)
            test_failures = [
                FieldFailure(document_id, f"tests.{t.get('test_name')}", t.get("result"), None, "missing") for t in ground_truth_tests
            ]
            test_counters = _empty_test_counters(ground_truth_tests)

        # A document only counts as an exact match if BOTH its fields AND
        # its test/result rows are failure-free — a document can have every
        # metadata field correct while every lab result is wrong, and that
        # is not an exact match.
        exact_match = extraction_ok and not field_failures and not test_failures

        doc_acc = MetricAccumulator()
        doc_acc.add(domain_match=domain_match, extraction_ok=extraction_ok, exact_match=exact_match, field_counters=field_counters, test_counters=test_counters)
        doc_metrics = doc_acc.finalize()

        overall.add(domain_match=domain_match, extraction_ok=extraction_ok, exact_match=exact_match, field_counters=field_counters, test_counters=test_counters)
        by_domain[domain].add(domain_match=domain_match, extraction_ok=extraction_ok, exact_match=exact_match, field_counters=field_counters, test_counters=test_counters)
        for bucket in _FORMAT_LABELS.get(source_type, ("unknown",)):
            by_format[bucket].add(domain_match=domain_match, extraction_ok=extraction_ok, exact_match=exact_match, field_counters=field_counters, test_counters=test_counters)

        all_failures.extend(field_failures)
        all_failures.extend(test_failures)
        per_document_rows.append({
            "document_id": document_id,
            "ground_truth_domain": domain,
            "predicted_domain": prediction["domain"],
            "source_format": record.get("source_format"),
            "source_type": source_type,
            "extraction_status": "ok" if extraction_ok else "failed",
            "pipeline_error": prediction["pipeline_error"],
            **doc_metrics,
        })

    for domain in CANONICAL_DOMAINS:
        by_domain.setdefault(domain, MetricAccumulator())
    for fmt in ("pdf", "born_digital_pdf", "scanned_pdf", "image"):
        by_format.setdefault(fmt, MetricAccumulator())

    overall_metrics = overall.finalize()
    results = {
        "overall_summary": {
            "total_documents": len(records),
            "successful_documents": successful,
            "failed_documents": failed,
            "overall_key_precision": overall_metrics["key_precision"],
            "overall_key_recall": overall_metrics["key_recall"],
            "overall_key_f1": overall_metrics["key_f1"],
            "overall_field_accuracy": overall_metrics["field_accuracy"],
            "overall_exact_match": overall_metrics["exact_match_rate"],
            "missing_field_rate": overall_metrics["missing_field_rate"],
            "hallucination_rate": overall_metrics["hallucination_rate"],
        },
        "document_level": {
            "domain_accuracy": overall_metrics["domain_accuracy"],
            "extraction_success_rate": overall_metrics["extraction_success_rate"],
        },
        "key_level": {
            "precision": overall_metrics["key_precision"],
            "recall": overall_metrics["key_recall"],
            "f1": overall_metrics["key_f1"],
        },
        "field_level": {
            "field_accuracy": overall_metrics["field_accuracy"],
            "exact_match_rate": overall_metrics["exact_match_rate"],
            "missing_field_rate": overall_metrics["missing_field_rate"],
            "hallucination_rate": overall_metrics["hallucination_rate"],
        },
        "value_level": {
            "value_accuracy": overall_metrics["value_accuracy"],
            "unit_accuracy": overall_metrics["unit_accuracy"],
            "reference_range_accuracy": overall_metrics["reference_range_accuracy"],
        },
        "table_level": {
            "test_name_accuracy": overall_metrics["test_name_accuracy"],
            "test_result_accuracy": overall_metrics["test_result_accuracy"],
            "test_unit_mapping_accuracy": overall_metrics["test_unit_mapping_accuracy"],
        },
        "by_domain": {domain: acc.finalize() for domain, acc in sorted(by_domain.items())},
        "by_format": {fmt: acc.finalize() for fmt, acc in sorted(by_format.items())},
        "failed_fields": [f.to_json_dict() for f in all_failures],
    }
    return results, all_failures, per_document_rows


def write_results(results: dict, per_document_rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "benchmark_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if per_document_rows:
        columns = list(per_document_rows[0].keys())
        with (output_dir / "benchmark_results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(per_document_rows)


def print_summary(results: dict) -> None:
    summary = results["overall_summary"]
    print()
    print("=== Benchmark Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print()
    print("--- by_domain ---")
    for domain, metrics in results["by_domain"].items():
        print(f"  {domain}: n={metrics['document_count']} field_accuracy={metrics['field_accuracy']} key_f1={metrics['key_f1']}")
    print("--- by_format ---")
    for fmt, metrics in results["by_format"].items():
        print(f"  {fmt}: n={metrics['document_count']} field_accuracy={metrics['field_accuracy']} key_f1={metrics['key_f1']}")
