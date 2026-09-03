"""Scoring logic: compares one FRESH prediction (from run_pipeline.py) to
its ground-truth record from final_labeled_dataset.jsonl. Pure functions,
read-only — they only ever produce FieldFailure entries and raw counters
for pipeline.py's MetricAccumulator to sum; they never modify either side.

A predicted value counts as "wrong_key" (rather than plain "missing") when
the ground-truth value for one field turns up, unchanged, under a
DIFFERENT key/test_name in the prediction — that's a genuinely different
failure mode (the value was found, but mis-attributed) worth distinguishing
from a value that was never extracted at all. Two conditions must hold for
that reassignment claim to be trustworthy, or it produces nonsense pairings:
1. The candidate key must not already be legitimately matched to its OWN
   ground-truth key — otherwise a real match gets "stolen" to explain away
   an unrelated missing field.
2. The matching value must be unique among the remaining candidates —
   common boilerplate results ("NAD", "Absent", "Negative") repeat across
   many unrelated rows in real lab reports, so "some other row happens to
   say the same three-letter value" is not evidence of a shared identity.
   An ambiguous match is scored as plain "missing", not "wrong_key".
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from app.benchmark.models import FieldFailure

_PLACEHOLDER_SENTINELS = {"n/a", "na", "null", "none", "-", "--", "tbd", ""}

# MEASURED 2026-09-03 over the 52-document set: a real chunk of "wrong_value"
# was the same value written two ways — "24.03.2021" vs "24-03-2021", a value
# wrapped across two lines in one source and one line in the other, or a
# non-breaking/full-width character from a PDF text layer. Those are
# transcription cosmetics, not extraction errors, so they are canonicalized
# on BOTH sides before comparison. Deliberately NOT normalized: month names
# ("05-Dec-2024" -> "05-12-2024") and field order (d/m vs m/d), because both
# need an assumption about the source's own convention that a lab report
# does not state — guessing there would silently mark genuinely different
# dates as equal.
_DATE_DMY = re.compile(r"^(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})$")
_DATE_YMD = re.compile(r"^(\d{4})[./-](\d{1,2})[./-](\d{1,2})$")
_WHITESPACE = re.compile(r"\s+")


def _canonical_date(text: str) -> str | None:
    """Same date, different separators/zero-padding -> one string. Returns
    None when `text` is not an all-numeric date, leaving it untouched."""
    dmy = _DATE_DMY.match(text)
    if dmy:
        day, month, year = dmy.groups()
        return f"{int(day):02d}-{int(month):02d}-{year}"
    ymd = _DATE_YMD.match(text)
    if ymd:
        year, month, day = ymd.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    return None


def _normalize_value(value: Any) -> str | None:
    if value is None:
        return None
    # NFKC folds full-width characters, non-breaking spaces and similar
    # PDF-text-layer artifacts onto their plain equivalents.
    text = unicodedata.normalize("NFKC", str(value))
    # A value wrapped across lines in the source must equal the same value
    # read as one line — collapse runs of whitespace rather than only
    # trimming the ends.
    text = _WHITESPACE.sub(" ", text).strip().lower()
    if text in _PLACEHOLDER_SENTINELS:
        return None
    return _canonical_date(text) or text


def _normalize_test_name(name: Any) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def _find_unique_reassignment(target_norm: str | None, candidates: dict[str, str | None]) -> str | None:
    """Among `candidates` (key -> normalized value, already restricted to
    ones not otherwise consumed), return the one key whose value matches
    `target_norm` — but only if there is EXACTLY one such key. Zero or
    multiple matches both mean "not a trustworthy reassignment"."""
    if target_norm is None:
        return None
    matches = [k for k, v in candidates.items() if v == target_norm]
    return matches[0] if len(matches) == 1 else None


def compare_fields(document_id: str, ground_truth: dict, predicted: dict) -> tuple[list[FieldFailure], dict]:
    gt_present = {k: v for k, v in ground_truth.items() if _normalize_value(v) is not None}
    pred_present = {k: v for k, v in predicted.items() if _normalize_value(v) is not None}
    gt_keys, pred_keys = set(gt_present), set(pred_present)

    failures: list[FieldFailure] = []
    field_correct = 0
    value_correct = 0
    value_total = 0
    missing = 0
    # Ground-truth fields whose value WAS extracted, just stored under a
    # different key name. Counted so accumulator.py can report a
    # naming-adjusted precision/recall alongside the strict one — see its
    # finalize(). Kept out of key_tp/key_fp/key_fn, which stay strictly
    # key-identity based so the primary metric never moves.
    wrong_key = 0

    # Only keys that are NOT a legitimate match for their own ground-truth
    # key are candidates for "value found under a different key" — consumed
    # (via .pop) as each reassignment is accepted, so the same extra key can
    # never explain away two different missing fields, and the true "extra"/
    # hallucinated set at the end is exactly whatever's left unconsumed.
    # Iterated in sorted order so the outcome never depends on Python's
    # randomized string-hash seed.
    extra_candidates = {k: _normalize_value(predicted[k]) for k in sorted(pred_keys - gt_keys)}

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
            reassigned_key = _find_unique_reassignment(gt_norm, extra_candidates)
            if reassigned_key:
                failures.append(FieldFailure(
                    document_id, key, gt_value, f"found under key {reassigned_key!r}: {predicted[reassigned_key]!r}", "wrong_key",
                ))
                del extra_candidates[reassigned_key]
                wrong_key += 1
            else:
                failures.append(FieldFailure(document_id, key, gt_value, None, "missing"))
                missing += 1
        else:
            failures.append(FieldFailure(document_id, key, gt_value, predicted.get(key), "wrong_value"))

    for key in extra_candidates:
        failures.append(FieldFailure(document_id, key, None, predicted[key], "extra"))

    counters = {
        "key_tp": len(gt_keys & pred_keys), "key_fp": len(pred_keys - gt_keys), "key_fn": len(gt_keys - pred_keys),
        "key_wrong_key": wrong_key,
        "field_correct": field_correct, "field_total": len(ground_truth),
        "missing": missing, "gt_nonnull_total": len(gt_present),
        "extra": len(extra_candidates), "predicted_nonnull_total": len(pred_present),
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

    unmatched_gt = sorted(set(gt_by_name) - set(pred_by_name))
    # Candidates for reassignment, keyed by normalized test name -> normalized
    # result. Consumed (via .pop) as each unique match is accepted, matching
    # compare_fields' same scoping + uniqueness rules — a boilerplate result
    # like "NAD"/"Absent"/"Negative" repeats across many unrelated rows in
    # real lab reports, so it must not be treated as proof of shared identity
    # unless it is the ONE remaining candidate with that value.
    extra_candidates = {
        n: _normalize_value(pred_by_name[n].get("result")) for n in sorted(set(pred_by_name) - set(gt_by_name))
    }

    for name in unmatched_gt:
        gt_row = gt_by_name[name]
        gt_result_norm = _normalize_value(gt_row.get("result"))
        reassigned = _find_unique_reassignment(gt_result_norm, extra_candidates)
        if reassigned:
            failures.append(FieldFailure(
                document_id, f"tests.{gt_row.get('test_name')}", gt_row.get("result"),
                f"found under test_name {pred_by_name[reassigned].get('test_name')!r}", "wrong_key",
            ))
            del extra_candidates[reassigned]
        else:
            failures.append(FieldFailure(document_id, f"tests.{gt_row.get('test_name')}", gt_row.get("result"), None, "missing"))

    for name in extra_candidates:
        failures.append(FieldFailure(document_id, f"tests.{pred_by_name[name].get('test_name')}", None, pred_by_name[name].get("result"), "extra"))

    counters = {
        "test_gt_total": len(gt_by_name), "test_matched": len(matched), "test_pred_total": len(pred_by_name),
        "result_correct": result_correct, "unit_correct": unit_correct, "reference_range_correct": reference_range_correct,
        "matched_count": len(matched), "extra": len(extra_candidates),
    }
    return failures, counters
