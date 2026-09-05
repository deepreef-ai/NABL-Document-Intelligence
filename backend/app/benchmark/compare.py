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
from typing import Any

from app.benchmark.models import FieldFailure

_PLACEHOLDER_SENTINELS = {"n/a", "na", "null", "none", "-", "--", "tbd", ""}

# Unicode dash variants (en/em dash, minus sign, ...) OCR/LLM output uses
# interchangeably with a plain hyphen — fold to one form before comparing.
_DASH_VARIANTS = re.compile("[‐‑‒–—―−]")
_WHITESPACE = re.compile(r"\s+")
# Separator punctuation, REMOVED before comparing (see _normalize_value):
# values are matched on their characters, words and numbers, not on which
# separator each side happened to use. "." and "-" are deliberately NOT
# here — they carry numeric meaning ("6.5", "40-129").
_SEPARATORS = re.compile(r"[,|/;]")
# Leading "Page"/"Page No"/"Page:" label on a page field (applied to the
# already-normalized text, which has no spaces).
_PAGE_LABEL = re.compile(r"^page(?:no)?[.:]*")


def _normalize_value(value: Any) -> str | None:
    """Two values that differ only in spacing ("40 - 129" vs "40-129",
    "02 Nov 2020" vs "02Nov2020") are the same value — OCR is inconsistent
    about exactly where it inserts a space around a dash or drops one
    between words entirely, and that's not a real extraction error. Ignoring
    ALL internal whitespace here (not just collapsing runs of it) is what it
    takes to treat those as equal; same reasoning already applies to test
    names below, which strip non-alnum chars entirely, more aggressively
    than this."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in _PLACEHOLDER_SENTINELS:
        return None
    text = _DASH_VARIANTS.sub("-", text)
    # Separator punctuation is dropped entirely, so two values are compared
    # on their characters, words and numbers rather than on punctuation. BOTH
    # sides invent separators: the page lays address parts out on separate
    # lines or with "|", the labeler wrote commas, the model wrote "/" or a
    # newline — MEASURED on this corpus, a comma present on one side and
    # absent on the other was the single most common "punctuation only"
    # mismatch (lab_address, lab_website, client_address).
    #
    # "." and "-" are NOT dropped: they carry numeric meaning, and removing
    # them would collapse "6.5" to "65" and "40-129" to "40129".
    text = _SEPARATORS.sub("", text)
    text = _WHITESPACE.sub("", text)
    # Trailing sentence punctuation is not part of the value: a labeler
    # transcribing "EDTA Blood." and a model returning "EDTA Blood" read the
    # same cell. Safe to strip because it can only ever remove terminal
    # punctuation - it cannot make two genuinely different values equal, the
    # way a containment/prefix rule could (that idea was MEASURED and
    # rejected: it wrongly credited a dropped date and a dropped specialty).
    text = text.rstrip(".,;:")
    return text or None


def _normalize_field_value(key: str, value: Any) -> str | None:
    """_normalize_value, plus the page-label rule.

    A page cell prints as "Page 4 Of 15": the label is "Page" and the value
    is "4 Of 15", so a model returning either form has read it correctly.
    MEASURED on this corpus as the most common truncation mismatch. Scoped to
    the page field ON PURPOSE — a general "one value contains the other" rule
    was measured and rejected because it also credited a dropped date
    ("09:55:20" for "03/02/2019 09:55:20") and a dropped specialty."""
    norm = _normalize_value(value)
    if norm and _KEY_ALIASES.get(key, key) == "page":
        norm = _PAGE_LABEL.sub("", norm) or None
    return norm


def _normalize_test_name(name: Any) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


# Field keys that mean the same thing under different names, folded to one
# canonical name before comparison. This exists because BOTH sides disagree
# with themselves: documents/app.py's schema is open-ended ("derive keys
# from the document's own labels"), so the model names a key after however
# the page abbreviates it, AND the hand-written ground truth is itself
# inconsistent across documents — MEASURED 2026-09-03 over the 52 labelled
# documents: lab_name (14 docs) vs laboratory_name (4), remarks (9) vs
# remark (9), client_name (17) vs customer_name (4) vs
# name_of_the_customer (3), date_of_receipt (22) vs sample_received_on (10)
# vs received (6). Without this, one naming disagreement scores as BOTH a
# missing ground-truth field and a hallucinated predicted one.
#
# Deliberately conservative: only names for the genuinely same concept are
# folded. A qualifier that changes meaning is left alone —
# testing_lab_address is not lab_address (a different lab), and
# sample_description_by_customer is not sample_description (different
# provenance, which is the whole point of recording it).
_KEY_ALIASES = {
    "laboratory_name": "lab_name",
    "laboratory_address": "lab_address",
    "laboratory_phone": "lab_phone",
    "report_no": "report_number",
    "ulr_number": "ulr_no",
    "remark": "remarks",
    "sample_type": "sample_description",
    "specimen_type": "sample_description",
    # MEASURED 2026-09-04 by mining model-vs-gold mismatches across the gold
    # set (document counts in comments). Renaming alone never credits a wrong
    # answer: the VALUE still has to match after the fold, so an "address"
    # holding the lab's address still fails against a client_address.
    "address": "client_address",              # 17 docs
    "quantity_condition": "quantity_and_condition",   # 15
    "customer_address": "client_address",     # 6
    "report_number_b": "secondary_report_number",     # 4
    "reported": "report_issue_date",          # 3
    "quantity_received": "sample_quantity",   # 2
    "test_report_no": "report_number",        # 2
    "sample_reference_no": "sample_number",   # 2
    "sample_completed_on": "end_date_of_analysis",    # 2
    "group_2": "secondary_group",             # 2
    "pathologist": "signatory_name",          # 2 - gold used the document's
                                              # own word, prompt says signatory_name
    # From the Gemini run's wrong_key rows. lab_fax->lab_phone is safe
    # despite being different concepts in general: these labs print one
    # "Telefax" line, so the SAME number is the phone and the fax, and the
    # fold only ever credits a matching value anyway.
    "lab_fax": "lab_phone",                   # 7 docs
    "customer_reference": "reference_number", # 6
    # Gold-internal inconsistencies, found by auditing MY OWN labels for the
    # same concept spelled differently across documents (an objective test,
    # independent of whether folding them helps any score):
    #   footer (36 docs) vs disclaimer (5)
    #   testing_lab_address (20) vs tested_in (4)
    #   note (19) vs limits_note (7)
    # NOT folded, deliberately: lab_address vs testing_lab_address (genuinely
    # two different labs), and the "page" convention — gold consistently
    # includes the word "Page" in 43 of 46 documents, so a model dropping it
    # is real variance, not a labelling bug.
    "disclaimer": "footer",
    "tested_in": "testing_lab_address",
    "limits_note": "note",
    "customer_name": "client_name",
    "name_of_the_customer": "client_name",
    "report_date": "report_issue_date",
    "sample_received_on": "date_of_receipt",
    "received": "date_of_receipt",
    "analysis_starting_date": "start_date_of_analysis",
    "analysis_started_on": "start_date_of_analysis",
    "test_started_on": "start_date_of_analysis",
    "analysis_completion_date": "end_date_of_analysis",
    "analysis_completed_on": "end_date_of_analysis",
    "test_completed_on": "end_date_of_analysis",
    "sampled_by": "sample_collected_by",
    "sample_drawn_by": "sample_collected_by",
    "sampling_procedure": "sampling_protocol",
    "sampling_plan_and_procedure": "sampling_protocol",
}


def _canonicalize_keys(source: dict) -> dict:
    """Rewrites known alias keys to their canonical name. A rewrite that
    would land on a key the dict ALREADY has is skipped rather than
    overwriting it — silently dropping one of two real values is exactly
    the bug class _test_row_key was written to avoid, and a document
    carrying both spellings genuinely has two distinct entries."""
    result = {}
    for key, value in source.items():
        canonical = _KEY_ALIASES.get(key, key)
        result[key if canonical in result else canonical] = value
    return result


def _find_unique_reassignment(target_norm: str | None, candidates: dict[Any, str | None]) -> Any | None:
    """Among `candidates` (key -> normalized value, already restricted to
    ones not otherwise consumed), return the one key whose value matches
    `target_norm` — but only if there is EXACTLY one such key. Zero or
    multiple matches both mean "not a trustworthy reassignment". Keys are
    plain field-name strings for compare_fields, (test_name, sample_id)
    tuples for compare_tests — this function only ever moves them around,
    never inspects their shape."""
    if target_norm is None:
        return None
    matches = [k for k, v in candidates.items() if v == target_norm]
    return matches[0] if len(matches) == 1 else None


def compare_fields(document_id: str, ground_truth: dict, predicted: dict) -> tuple[list[FieldFailure], dict]:
    # Both sides folded to canonical names first, so a pure naming
    # disagreement (see _KEY_ALIASES) isn't scored as an extraction error.
    ground_truth = _canonicalize_keys(ground_truth)
    predicted = _canonicalize_keys(predicted)

    gt_present = {k: v for k, v in ground_truth.items() if _normalize_field_value(k, v) is not None}
    pred_present = {k: v for k, v in predicted.items() if _normalize_field_value(k, v) is not None}
    gt_keys, pred_keys = set(gt_present), set(pred_present)

    failures: list[FieldFailure] = []
    field_correct = 0
    value_correct = 0
    value_total = 0
    missing = 0

    # Only keys that are NOT a legitimate match for their own ground-truth
    # key are candidates for "value found under a different key" — consumed
    # (via .pop) as each reassignment is accepted, so the same extra key can
    # never explain away two different missing fields, and the true "extra"/
    # hallucinated set at the end is exactly whatever's left unconsumed.
    # Iterated in sorted order so the outcome never depends on Python's
    # randomized string-hash seed.
    extra_candidates = {k: _normalize_field_value(k, predicted[k]) for k in sorted(pred_keys - gt_keys)}

    for key, gt_value in ground_truth.items():
        gt_norm = _normalize_field_value(key, gt_value)
        pred_norm = _normalize_field_value(key, predicted.get(key)) if key in predicted else None

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
            else:
                failures.append(FieldFailure(document_id, key, gt_value, None, "missing"))
                missing += 1
        else:
            failures.append(FieldFailure(document_id, key, gt_value, predicted.get(key), "wrong_value"))

    for key in extra_candidates:
        failures.append(FieldFailure(document_id, key, None, predicted[key], "extra"))

    counters = {
        "key_tp": len(gt_keys & pred_keys), "key_fp": len(pred_keys - gt_keys), "key_fn": len(gt_keys - pred_keys),
        "field_correct": field_correct, "field_total": len(ground_truth),
        "missing": missing, "gt_nonnull_total": len(gt_present),
        "extra": len(extra_candidates), "predicted_nonnull_total": len(pred_present),
        "value_correct": value_correct, "value_total": value_total,
    }
    return failures, counters


def _test_row_key(row: dict) -> tuple[str, str] | None:
    """(normalized test_name, normalized sample_id). The same analyte name
    routinely repeats across multiple samples in one report (e.g. "pH" once
    per sample point in a water panel) — sample_id is exactly the column
    the schema carries to disambiguate those (see documents/app.py's
    SYSTEM_PROMPT: "If the same label appears multiple times, extract each
    occurrence separately... sample ID if present"). Keying on test_name
    alone would collapse those rows onto each other. A row with no
    sample_id normalizes to "" for that half of the key, so documents that
    never use sample_id (the common case) still match purely by name, same
    as before."""
    name = _normalize_test_name(row.get("test_name")) if row.get("test_name") else ""
    if not name:
        return None
    return (name, _normalize_value(row.get("sample_id")) or "")


def _match_test_rows(ground_truth_tests: list[dict], predicted_tests: list[dict]) -> dict:
    """The matching + reassignment step shared by compare_tests (scoring)
    and build_test_comparison_rows (the human-readable Excel export) — pulled out
    once so the export can never disagree with the scored metrics, the way
    two independent re-implementations of "which row explains which" could
    silently drift apart."""
    gt_by_key = {key: row for row in ground_truth_tests if (key := _test_row_key(row))}
    # A predicted row carrying NO result filled in no value, so it is not a
    # fabricated value and must not count as one - most often it is a section
    # header ("Differential Leucocyte Count") the model mistook for a row.
    # compare_fields already drops null-valued predictions from pred_present;
    # this keeps the two comparators consistent instead of penalising tests
    # for something fields ignore.
    pred_by_key = {
        key: row for row in predicted_tests
        if (key := _test_row_key(row)) and _normalize_value(row.get("result")) is not None
    }
    matched = set(gt_by_key) & set(pred_by_key)

    # Second pass: sample_id is a TIE-BREAKER, not part of a row's identity.
    # Ground truth often carries one (a specimen type, a sub-report number)
    # for a test whose name is already unique in the document, and the model
    # has no reason to reproduce that identifier — so an exact
    # (name, sample_id) match alone would score a perfectly-read row as both
    # missing AND hallucinated. MEASURED 2026-09-03: 55 such rows across two
    # gold-labelled documents. So any still-unmatched pair whose NAME is
    # unambiguous on both sides (exactly one leftover each) is matched on the
    # name alone. Rows whose name genuinely repeats are untouched here and
    # keep needing sample_id to pair correctly, which is the case the key
    # exists for.
    def _leftovers_by_name(keys):
        by_name: dict[str, list[tuple[str, str]]] = {}
        for key in keys:
            by_name.setdefault(key[0], []).append(key)
        return by_name

    gt_left = _leftovers_by_name(set(gt_by_key) - matched)
    pred_left = _leftovers_by_name(set(pred_by_key) - matched)
    for name, gt_keys in gt_left.items():
        pred_keys = pred_left.get(name, [])
        if len(gt_keys) == 1 and len(pred_keys) == 1:
            # Re-file the prediction under the ground truth's key so every
            # downstream consumer (counters, failures, the Excel export) sees
            # one matched pair rather than a name collision.
            pred_by_key[gt_keys[0]] = pred_by_key.pop(pred_keys[0])
            matched.add(gt_keys[0])

    # Candidates for reassignment, keyed by (test_name, sample_id) ->
    # normalized result. Consumed as each unique match is accepted, matching
    # compare_fields' same scoping + uniqueness rules — a boilerplate result
    # like "NAD"/"Absent"/"Negative" repeats across many unrelated rows in
    # real lab reports, so it must not be treated as proof of shared identity
    # unless it is the ONE remaining candidate with that value.
    extra_candidates = {
        k: _normalize_value(pred_by_key[k].get("result")) for k in sorted(set(pred_by_key) - set(gt_by_key))
    }
    reassigned_to: dict[tuple[str, str], tuple[str, str]] = {}
    for key in sorted(set(gt_by_key) - set(pred_by_key)):
        reassigned = _find_unique_reassignment(_normalize_value(gt_by_key[key].get("result")), extra_candidates)
        if reassigned:
            reassigned_to[key] = reassigned
            del extra_candidates[reassigned]

    missing = sorted(set(gt_by_key) - set(pred_by_key) - set(reassigned_to))
    extra = sorted(extra_candidates)  # whatever's left unconsumed
    return {
        "gt_by_key": gt_by_key, "pred_by_key": pred_by_key, "matched": matched,
        "reassigned_to": reassigned_to, "missing": missing, "extra": extra,
    }


def compare_tests(document_id: str, ground_truth_tests: list[dict], predicted_tests: list[dict]) -> tuple[list[FieldFailure], dict]:
    m = _match_test_rows(ground_truth_tests, predicted_tests)
    gt_by_key, pred_by_key = m["gt_by_key"], m["pred_by_key"]
    failures: list[FieldFailure] = []
    result_correct = unit_correct = reference_range_correct = 0

    for key in m["matched"]:
        gt_row, pred_row = gt_by_key[key], pred_by_key[key]
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

    for gt_key, pred_key in m["reassigned_to"].items():
        gt_row = gt_by_key[gt_key]
        failures.append(FieldFailure(
            document_id, f"tests.{gt_row.get('test_name')}", gt_row.get("result"),
            f"found under test_name {pred_by_key[pred_key].get('test_name')!r}", "wrong_key",
        ))

    for key in m["missing"]:
        gt_row = gt_by_key[key]
        failures.append(FieldFailure(document_id, f"tests.{gt_row.get('test_name')}", gt_row.get("result"), None, "missing"))

    for key in m["extra"]:
        failures.append(FieldFailure(document_id, f"tests.{pred_by_key[key].get('test_name')}", None, pred_by_key[key].get("result"), "extra"))

    counters = {
        "test_gt_total": len(gt_by_key), "test_matched": len(m["matched"]), "test_pred_total": len(pred_by_key),
        "result_correct": result_correct, "unit_correct": unit_correct, "reference_range_correct": reference_range_correct,
        "matched_count": len(m["matched"]), "extra": len(m["extra"]),
    }
    return failures, counters


def field_comparison_rows(
    document_id: str, ground_truth: dict, predicted: dict, field_verified: dict | None = None
) -> list[dict]:
    """One row per ground-truth field (correct or not) plus one per
    hallucinated predicted field — a full side-by-side export for human
    review, unlike compare_fields' failures-only list. Built directly on
    compare_fields' own output so it can never disagree with the scored
    metrics; safe to key off FieldFailure.key here because dict keys are
    already unique within one document (unlike test rows, which need
    (test_name, sample_id) — see build_test_comparison_rows).

    `field_verified` is documents/app.py's parallel "did this value appear
    in the OCR text" map. Optional because it postdates the predictions
    already sitting in the cache — a prediction from before that signal
    existed leaves the column blank rather than reading as "unverified"."""
    # Same folding compare_fields applies internally, so these rows line up
    # with the failures it reports (which carry canonical keys).
    ground_truth = _canonicalize_keys(ground_truth)
    predicted = _canonicalize_keys(predicted)
    field_verified = _canonicalize_keys(field_verified or {})
    failures, _ = compare_fields(document_id, ground_truth, predicted)
    failure_by_key = {f.key: f for f in failures if f.error_type != "extra"}

    rows = []
    for key, gt_value in ground_truth.items():
        failure = failure_by_key.get(key)
        rows.append({
            "document_id": document_id, "field": key,
            "ground_truth": gt_value, "predicted": predicted.get(key),
            "status": failure.error_type if failure else "correct",
            "verified": field_verified.get(key, ""),
            "note": failure.prediction if failure and failure.error_type == "wrong_key" else "",
        })
    for f in failures:
        if f.error_type == "extra":
            rows.append({
                "document_id": document_id, "field": f.key,
                "ground_truth": None, "predicted": f.prediction,
                "status": "extra", "verified": field_verified.get(f.key, ""), "note": "",
            })
    return rows


def build_test_comparison_rows(document_id: str, ground_truth_tests: list[dict], predicted_tests: list[dict]) -> list[dict]:
    """One row per ground-truth test (correct or not) plus one per
    hallucinated predicted row — result/unit/reference_range are each
    flagged individually so a partially-wrong row (right result, wrong
    unit) is visible rather than collapsed into one pass/fail. Built on
    _match_test_rows, the same matching compare_tests scores from.

    The "verified" column carries documents/app.py's "did this result
    appear in the OCR text" flag straight off the predicted row, and is
    blank where there's nothing to have verified (a missing row) or where
    the prediction predates that signal."""
    m = _match_test_rows(ground_truth_tests, predicted_tests)
    gt_by_key, pred_by_key = m["gt_by_key"], m["pred_by_key"]
    rows = []

    for key in sorted(m["matched"]):
        gt_row, pred_row = gt_by_key[key], pred_by_key[key]
        mismatches = [
            col for col in ("result", "unit", "reference_range")
            if _normalize_value(gt_row.get(col)) != _normalize_value(pred_row.get(col))
        ]
        rows.append({
            "document_id": document_id, "test_name": gt_row.get("test_name"), "sample_id": gt_row.get("sample_id"),
            "gt_result": gt_row.get("result"), "pred_result": pred_row.get("result"),
            "gt_unit": gt_row.get("unit"), "pred_unit": pred_row.get("unit"),
            "gt_reference_range": gt_row.get("reference_range"), "pred_reference_range": pred_row.get("reference_range"),
            "status": "correct" if not mismatches else "wrong_value",
            "verified": pred_row.get("result_verified", ""),
            "note": ", ".join(mismatches) + " mismatch" if mismatches else "",
        })

    for gt_key, pred_key in m["reassigned_to"].items():
        gt_row, pred_row = gt_by_key[gt_key], pred_by_key[pred_key]
        rows.append({
            "document_id": document_id, "test_name": gt_row.get("test_name"), "sample_id": gt_row.get("sample_id"),
            "gt_result": gt_row.get("result"), "pred_result": None,
            "gt_unit": gt_row.get("unit"), "pred_unit": None,
            "gt_reference_range": gt_row.get("reference_range"), "pred_reference_range": None,
            "status": "wrong_key", "verified": pred_row.get("result_verified", ""),
            "note": f"found under test_name {pred_row.get('test_name')!r}",
        })

    for key in m["missing"]:
        gt_row = gt_by_key[key]
        rows.append({
            "document_id": document_id, "test_name": gt_row.get("test_name"), "sample_id": gt_row.get("sample_id"),
            "gt_result": gt_row.get("result"), "pred_result": None,
            "gt_unit": gt_row.get("unit"), "pred_unit": None,
            "gt_reference_range": gt_row.get("reference_range"), "pred_reference_range": None,
            "status": "missing", "verified": "", "note": "",
        })

    for key in m["extra"]:
        pred_row = pred_by_key[key]
        rows.append({
            "document_id": document_id, "test_name": pred_row.get("test_name"), "sample_id": pred_row.get("sample_id"),
            "gt_result": None, "pred_result": pred_row.get("result"),
            "gt_unit": None, "pred_unit": pred_row.get("unit"),
            "gt_reference_range": None, "pred_reference_range": pred_row.get("reference_range"),
            "status": "extra", "verified": pred_row.get("result_verified", ""), "note": "",
        })

    return rows


# Human-readable failure taxonomy for the reviewable export. Kept as data so
# the Excel sheet, any summary count, and the docs all name a failure the same
# way — MEASURED shares across the gold set are in the benchmark notes.
FAILURE_CATEGORIES = {
    "never_extracted": "Value is on the page but the model returned nothing for it",
    "wrong_key": "Correct value, but returned under a different field name",
    "wrong_value_truncated": "Right field, but the value is cut short or has extra text appended",
    "wrong_value_punctuation": "Right field, value differs only in punctuation/symbols",
    "wrong_value_different": "Right field, but genuinely different text (a misread)",
    "hallucinated": "Returned a field that has no counterpart in the ground truth",
    "test_never_extracted": "Test row is on the page but was not returned",
    "test_wrong_name": "Test row found, but under a different test name",
    "test_wrong_result": "Test row matched, but its result value is wrong",
    "test_wrong_unit": "Test row matched, but its unit is wrong",
    "test_wrong_reference_range": "Test row matched, but its reference range is wrong",
    "test_hallucinated": "Returned a test row with no counterpart in the ground truth",
}


def _value_difference_kind(gold: Any, predicted: Any) -> str:
    """Why two values at the SAME key disagree. Distinguishing these matters:
    a truncation is usually a labelling-convention clash (the model returning
    "4 Of 15" where the label is "Page" and the value "4 Of 15"), whereas
    'different text' is a genuine misread worth fixing."""
    g, p = str(gold), str(predicted)
    if p in g or g in p:
        return "wrong_value_truncated"
    if _normalize_test_name(g) == _normalize_test_name(p):
        return "wrong_value_punctuation"
    return "wrong_value_different"


def failure_rows(
    document_id: str, ground_truth: dict, predicted: dict, field_verified: dict | None = None
) -> list[dict]:
    """One row per MISMATCH, showing BOTH sides' key and value plus a
    categorised reason — the sheet you work from when deciding what to fix.

    Distinct from field_comparison_rows, which emits every field including the
    correct ones (good for auditing, noisy for triage). Built on the same
    _canonicalize_keys / _match_test_rows the metrics use, so a failure listed
    here is exactly a failure the scores counted."""
    field_verified = _canonicalize_keys(field_verified or {})
    gold_f = _canonicalize_keys(ground_truth.get("fields", {}))
    pred_f = _canonicalize_keys(predicted.get("fields", {}))
    gold_n = {k: _normalize_value(v) for k, v in gold_f.items()}
    pred_n = {k: _normalize_value(v) for k, v in pred_f.items()}

    rows: list[dict] = []

    def add(category, gt_key=None, gt_value=None, pred_key=None, pred_value=None, detail=""):
        rows.append({
            "document_id": document_id,
            "category": category,
            "reason": FAILURE_CATEGORIES.get(category, category) + (f" — {detail}" if detail else ""),
            "gt_key": gt_key, "gt_value": gt_value,
            "pred_key": pred_key, "pred_value": pred_value,
            "in_ocr_text": field_verified.get(pred_key, "") if pred_key else "",
        })

    matched_pred_keys = set()
    for key, gold_value in gold_f.items():
        gv = gold_n[key]
        if gv is None:
            continue
        pv = pred_n.get(key)
        if pv == gv:
            matched_pred_keys.add(key)
            continue
        if pv is not None:
            matched_pred_keys.add(key)
            add(_value_difference_kind(gold_value, pred_f[key]), key, gold_value, key, pred_f[key])
            continue
        # not at its own key — is the value sitting under a different one?
        elsewhere = next((pk for pk, pn in pred_n.items() if pn == gv and pk not in gold_n), None)
        if elsewhere:
            matched_pred_keys.add(elsewhere)
            add("wrong_key", key, gold_value, elsewhere, pred_f[elsewhere],
                detail=f"model called it {elsewhere!r}")
        else:
            add("never_extracted", key, gold_value)

    for key, value in pred_f.items():
        if key in matched_pred_keys or pred_n[key] is None or key in gold_n:
            continue
        add("hallucinated", pred_key=key, pred_value=value)

    # ---- test rows ----
    m = _match_test_rows(ground_truth.get("tests", []), predicted.get("tests", []))
    gt_by_key, pred_by_key = m["gt_by_key"], m["pred_by_key"]
    for key in sorted(m["matched"]):
        g_row, p_row = gt_by_key[key], pred_by_key[key]
        name = g_row.get("test_name")
        for column, category in (
            ("result", "test_wrong_result"),
            ("unit", "test_wrong_unit"),
            ("reference_range", "test_wrong_reference_range"),
        ):
            if _normalize_value(g_row.get(column)) != _normalize_value(p_row.get(column)):
                add(category, f"tests.{name}.{column}", g_row.get(column),
                    f"tests.{p_row.get('test_name')}.{column}", p_row.get(column))
    for gt_key, pred_key in m["reassigned_to"].items():
        g_row = gt_by_key[gt_key]
        add("test_wrong_name", f"tests.{g_row.get('test_name')}", g_row.get("result"),
            f"tests.{pred_by_key[pred_key].get('test_name')}", pred_by_key[pred_key].get("result"))
    for key in m["missing"]:
        g_row = gt_by_key[key]
        add("test_never_extracted", f"tests.{g_row.get('test_name')}", g_row.get("result"))
    for key in m["extra"]:
        p_row = pred_by_key[key]
        add("test_hallucinated", pred_key=f"tests.{p_row.get('test_name')}", pred_value=p_row.get("result"))

    return rows



def gt_vs_predicted_rows(
    document_id: str, ground_truth: dict, predicted: dict
) -> list[dict]:
    """Flat side-by-side of EVERY field and test row — hits included — as four
    columns: ground-truth field/value against predicted field/value.

    Distinct from failure_rows (mismatches only, for triage) and from
    field_comparison_rows (gold-keyed, carries the scorer's status/notes).
    This one is the "show me everything, colour the hits green" view, so a
    reviewer can page through a document and see what was got and what was
    missed without cross-referencing sheets.

    `hit` drives the row colour. A row appears for every ground-truth value
    AND every predicted value, so a miss shows an empty predicted side and a
    hallucination shows an empty ground-truth side."""
    gold_f = _canonicalize_keys(ground_truth.get("fields", {}))
    pred_f = _canonicalize_keys(predicted.get("fields", {}))
    gold_n = {k: _normalize_value(v) for k, v in gold_f.items()}
    pred_n = {k: _normalize_value(v) for k, v in pred_f.items()}

    rows: list[dict] = []

    def add(gt_field, gt_value, pred_field, pred_value, hit):
        rows.append({
            "document_id": document_id,
            "ground_truth_field": gt_field, "ground_truth_value": gt_value,
            "predicted_field": pred_field, "predicted_value": pred_value,
            "hit": "HIT" if hit else "MISS",
        })

    consumed = set()
    for key, gold_value in gold_f.items():
        gv = gold_n[key]
        if gv is None:
            continue
        if pred_n.get(key) == gv:
            consumed.add(key)
            add(key, gold_value, key, pred_f[key], True)
        elif pred_n.get(key) is not None:
            consumed.add(key)
            add(key, gold_value, key, pred_f[key], False)
        else:
            # value may be sitting under a different predicted key
            elsewhere = next((pk for pk, pn in pred_n.items() if pn == gv and pk not in gold_n), None)
            if elsewhere:
                consumed.add(elsewhere)
                add(key, gold_value, elsewhere, pred_f[elsewhere], False)
            else:
                add(key, gold_value, None, None, False)

    for key, value in pred_f.items():
        if key in consumed or pred_n[key] is None or key in gold_n:
            continue
        add(None, None, key, value, False)

    m = _match_test_rows(ground_truth.get("tests", []), predicted.get("tests", []))
    gt_by_key, pred_by_key = m["gt_by_key"], m["pred_by_key"]
    for key in sorted(m["matched"]):
        g_row, p_row = gt_by_key[key], pred_by_key[key]
        hit = _normalize_value(g_row.get("result")) == _normalize_value(p_row.get("result"))
        add(f"tests.{g_row.get('test_name')}", g_row.get("result"),
            f"tests.{p_row.get('test_name')}", p_row.get("result"), hit)
    for gt_key, pred_key in m["reassigned_to"].items():
        g_row = gt_by_key[gt_key]
        add(f"tests.{g_row.get('test_name')}", g_row.get("result"),
            f"tests.{pred_by_key[pred_key].get('test_name')}", pred_by_key[pred_key].get("result"), False)
    for key in m["missing"]:
        g_row = gt_by_key[key]
        add(f"tests.{g_row.get('test_name')}", g_row.get("result"), None, None, False)
    for key in m["extra"]:
        p_row = pred_by_key[key]
        add(None, None, f"tests.{p_row.get('test_name')}", p_row.get("result"), False)

    return rows
