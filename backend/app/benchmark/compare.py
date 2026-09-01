"""Scoring logic: compares one FRESH prediction (from run_pipeline.py) to
its ground-truth record from final_labeled_dataset.jsonl. Pure functions,
read-only — they only ever produce FieldFailure entries and raw counters
for pipeline.py's MetricAccumulator to sum; they never modify either side.

A predicted value counts as "wrong_key" (rather than plain "missing") when
the ground-truth value for one field turns up, unchanged, under a
DIFFERENT key/test_name in the prediction — that's a genuinely different
failure mode (the value was found, but mis-attributed) worth distinguishing
from a value that was never extracted at all.
"""
from __future__ import annotations

from typing import Any

from app.benchmark.models import FieldFailure

_PLACEHOLDER_SENTINELS = {"n/a", "na", "null", "none", "-", "--", "tbd", ""}


def _normalize_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return None if text in _PLACEHOLDER_SENTINELS else text


def _normalize_test_name(name: Any) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def compare_fields(document_id: str, ground_truth: dict, predicted: dict) -> tuple[list[FieldFailure], dict]:
    gt_present = {k: v for k, v in ground_truth.items() if _normalize_value(v) is not None}
    pred_present = {k: v for k, v in predicted.items() if _normalize_value(v) is not None}
    gt_keys, pred_keys = set(gt_present), set(pred_present)

    failures: list[FieldFailure] = []
    field_correct = 0
    value_correct = 0
    value_total = 0
    missing = 0
    extra = 0

    for key, gt_value in ground_truth.items():
        gt_norm = _normalize_value(gt_value)
        pred_norm = _normalize_value(predicted.get(key)) if key in predicted else None

        if gt_norm is None:
            if pred_norm is None:
                field_correct += 1
            else:
                failures.append(FieldFailure(document_id, key, None, predicted.get(key), "wrong_value"))
            continue

        value_total += 1
        if pred_norm == gt_norm:
            field_correct += 1
            value_correct += 1
            continue

        if pred_norm is None:
            reassigned_key = next(
                (k for k, v in predicted.items() if k != key and _normalize_value(v) == gt_norm), None,
            )
            if reassigned_key:
                failures.append(FieldFailure(
                    document_id, key, gt_value, f"found under key {reassigned_key!r}: {predicted[reassigned_key]!r}", "wrong_key",
                ))
            else:
                failures.append(FieldFailure(document_id, key, gt_value, None, "missing"))
                missing += 1
        else:
            failures.append(FieldFailure(document_id, key, gt_value, predicted.get(key), "wrong_value"))

    for key in pred_keys - gt_keys:
        failures.append(FieldFailure(document_id, key, None, predicted[key], "extra"))
        extra += 1

    counters = {
        "key_tp": len(gt_keys & pred_keys), "key_fp": len(pred_keys - gt_keys), "key_fn": len(gt_keys - pred_keys),
        "field_correct": field_correct, "field_total": len(ground_truth),
        "exact_match": 1 if not failures else 0,
        "missing": missing, "gt_nonnull_total": len(gt_present),
        "extra": extra, "predicted_nonnull_total": len(pred_present),
        "value_correct": value_correct, "value_total": value_total,
    }
    return failures, counters


def compare_tests(document_id: str, ground_truth_tests: list[dict], predicted_tests: list[dict]) -> tuple[list[FieldFailure], dict]:
    gt_by_name = {_normalize_test_name(row.get("test_name")): row for row in ground_truth_tests if row.get("test_name")}
    pred_by_name = {_normalize_test_name(row.get("test_name")): row for row in predicted_tests if row.get("test_name")}

    matched = set(gt_by_name) & set(pred_by_name)
    failures: list[FieldFailure] = []
    result_correct = unit_correct = reference_range_correct = 0

    for name in matched:
        gt_row, pred_row = gt_by_name[name], pred_by_name[name]
        label = gt_row.get("test_name")

        if _normalize_value(gt_row.get("result")) == _normalize_value(pred_row.get("result")):
            result_correct += 1
        else:
            failures.append(FieldFailure(document_id, f"tests.{label}.result", gt_row.get("result"), pred_row.get("result"), "wrong_value"))

        if _normalize_value(gt_row.get("unit")) == _normalize_value(pred_row.get("unit")):
            unit_correct += 1
        else:
            failures.append(FieldFailure(document_id, f"tests.{label}.unit", gt_row.get("unit"), pred_row.get("unit"), "wrong_unit"))

        if _normalize_value(gt_row.get("reference_range")) == _normalize_value(pred_row.get("reference_range")):
            reference_range_correct += 1
        else:
            failures.append(FieldFailure(document_id, f"tests.{label}.reference_range", gt_row.get("reference_range"), pred_row.get("reference_range"), "wrong_value"))

    unmatched_gt = set(gt_by_name) - set(pred_by_name)
    unmatched_pred = set(pred_by_name) - set(gt_by_name)

    for name in unmatched_gt:
        gt_row = gt_by_name[name]
        gt_result_norm = _normalize_value(gt_row.get("result"))
        reassigned = next(
            (n for n in unmatched_pred if gt_result_norm is not None and _normalize_value(pred_by_name[n].get("result")) == gt_result_norm),
            None,
        )
        if reassigned:
            failures.append(FieldFailure(
                document_id, f"tests.{gt_row.get('test_name')}", gt_row.get("result"),
                f"found under test_name {pred_by_name[reassigned].get('test_name')!r}", "wrong_key",
            ))
            unmatched_pred.discard(reassigned)
        else:
            failures.append(FieldFailure(document_id, f"tests.{gt_row.get('test_name')}", gt_row.get("result"), None, "missing"))

    for name in unmatched_pred:
        failures.append(FieldFailure(document_id, f"tests.{pred_by_name[name].get('test_name')}", None, pred_by_name[name].get("result"), "extra"))

    counters = {
        "test_gt_total": len(gt_by_name), "test_matched": len(matched), "test_pred_total": len(pred_by_name),
        "result_correct": result_correct, "unit_correct": unit_correct, "reference_range_correct": reference_range_correct,
        "matched_count": len(matched),
    }
    return failures, counters
