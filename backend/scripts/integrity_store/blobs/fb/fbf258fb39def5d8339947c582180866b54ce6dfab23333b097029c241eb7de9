"""Pools raw counts (never per-document averages) into the ratio metrics
the benchmark reports — the same accumulator class is used for the overall
totals, each domain bucket, each format bucket, AND each single document's
own CSV row, so every reported number comes from one consistent
computation, never a duplicated formula.
"""
from __future__ import annotations

from dataclasses import dataclass


def _f1(precision: float | None, recall: float | None) -> float | None:
    """Harmonic mean, None when either side is undefined (no data)."""
    if precision is None or recall is None:
        return None
    return round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0


@dataclass
class MetricAccumulator:
    document_count: int = 0
    domain_correct: int = 0
    extraction_success: int = 0
    key_tp: int = 0
    key_fp: int = 0
    key_fn: int = 0
    # Ground-truth values that WERE extracted but under a different key name
    # (compare.py's "wrong_key"). Reported separately, never folded into
    # key_tp/fp/fn — see finalize()'s naming-adjusted block.
    key_wrong_key: int = 0
    field_correct: int = 0
    field_total: int = 0
    exact_match: int = 0
    missing: int = 0
    gt_nonnull_total: int = 0
    extra: int = 0
    predicted_nonnull_total: int = 0
    value_correct: int = 0
    value_total: int = 0
    # Numerator/denominator for *_fill_precision below: unlike value_correct/
    # predicted_nonnull_total (which pool fields and test rows together),
    # these stay split so a form auto-fill decision can be made per surface
    # — fields and the test/result table have very different reliability in
    # practice (a wrong header field vs. a wrong lab result row).
    field_value_correct: int = 0
    field_predicted_nonnull_total: int = 0
    test_predicted_total: int = 0
    unit_correct: int = 0
    reference_range_correct: int = 0
    test_name_matched: int = 0
    test_name_total: int = 0
    test_result_correct: int = 0
    test_result_total: int = 0  # == matched test rows

    def add(self, *, domain_match: bool, extraction_ok: bool, exact_match: bool, field_counters: dict, test_counters: dict) -> None:
        self.document_count += 1
        self.domain_correct += int(domain_match)
        self.extraction_success += int(extraction_ok)
        self.exact_match += int(exact_match)

        self.key_tp += field_counters["key_tp"]
        self.key_fp += field_counters["key_fp"]
        self.key_fn += field_counters["key_fn"]
        self.key_wrong_key += field_counters.get("key_wrong_key", 0)
        self.field_correct += field_counters["field_correct"]
        self.field_total += field_counters["field_total"]
        self.missing += field_counters["missing"]
        self.gt_nonnull_total += field_counters["gt_nonnull_total"]
        self.extra += field_counters["extra"]
        self.predicted_nonnull_total += field_counters["predicted_nonnull_total"]
        self.value_correct += field_counters["value_correct"]
        self.value_total += field_counters["value_total"]
        self.field_value_correct += field_counters["value_correct"]
        self.field_predicted_nonnull_total += field_counters["predicted_nonnull_total"]

        matched = test_counters["matched_count"]
        # Table-row values fold into the same combined value_accuracy as
        # field values (VALUE LEVEL is "every extracted value", fields and
        # test results together) while also being tracked on their own for
        # TABLE LEVEL's test_result_accuracy.
        self.value_correct += test_counters["result_correct"]
        self.value_total += matched
        self.test_result_correct += test_counters["result_correct"]
        self.test_result_total += matched
        self.test_predicted_total += test_counters["test_pred_total"]
        self.unit_correct += test_counters["unit_correct"]
        self.reference_range_correct += test_counters["reference_range_correct"]
        self.test_name_matched += test_counters["test_matched"]
        self.test_name_total += test_counters["test_gt_total"]
        # Hallucinated rows count on both sides regardless of whether the
        # hallucination showed up as a document-level field or a fabricated
        # test/table row — same "extra" concept, same denominator family as
        # field-level hallucinations (HALLUCINATION_RATE is over every
        # predicted, non-null item: fields and test rows together).
        self.extra += test_counters["extra"]
        self.predicted_nonnull_total += test_counters["test_pred_total"]

    def finalize(self) -> dict:
        def ratio(numerator: int, denominator: int) -> float | None:
            return round(numerator / denominator, 4) if denominator else None

        def prf(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
            p = ratio(tp, tp + fp)
            r = ratio(tp, tp + fn)
            if p is None or r is None:
                return p, r, None
            return p, r, (round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0)

        precision, recall, f1 = prf(self.key_tp, self.key_fp, self.key_fn)

        # Naming-adjusted: credits a ground-truth value that WAS extracted but
        # under a different key name (compare.py's "wrong_key" — e.g. the model
        # said "fax" where the label says "lab_fax"). MEASURED 2026-09-03 on
        # the 52-document set: 88 of 177 apparent false positives are this,
        # worth ~+0.09 F1. Reported ALONGSIDE the strict figures, never
        # replacing them: strict is the schema-exact lower bound (the key you
        # need to fill a specific form slot), adjusted is the value-found
        # upper bound. Quoting either one alone is misleading.
        adj_p, adj_r, adj_f1 = prf(
            self.key_tp + self.key_wrong_key,
            self.key_fp - self.key_wrong_key,
            self.key_fn - self.key_wrong_key,
        )

        return {
            "document_count": self.document_count,
            "domain_accuracy": ratio(self.domain_correct, self.document_count),
            "extraction_success_rate": ratio(self.extraction_success, self.document_count),
            "key_precision": precision,
            "key_recall": recall,
            "key_f1": f1,
            "renamed_key_count": self.key_wrong_key,
            "key_precision_naming_adjusted": adj_p,
            "key_recall_naming_adjusted": adj_r,
            "key_f1_naming_adjusted": adj_f1,
            "field_accuracy": ratio(self.field_correct, self.field_total),
            "exact_match_rate": ratio(self.exact_match, self.document_count),
            "missing_field_rate": ratio(self.missing, self.gt_nonnull_total),
            "hallucination_rate": ratio(self.extra, self.predicted_nonnull_total),
            "value_accuracy": ratio(self.value_correct, self.value_total),
            # Precision of what actually gets filled in, NOT recall: the
            # denominator is every non-null value/row the system output
            # (right, wrong, or invented), not every ground-truth value —
            # so a missing field doesn't drag this down (it's a visibly
            # empty box, low risk), but a wrong or fabricated one does (it
            # LOOKS filled-in and correct, high risk if auto-filled
            # unattended). A value that landed under the wrong key/test
            # name does NOT count as correct here even if the raw text was
            # right — for a form auto-fill, that's still the wrong box.
            # This is the number to gate an auto-fill decision on, not
            # value_accuracy (which credits nothing for a value the system
            # never attempted) or hallucination_rate (which only flags
            # fully invented keys, not a wrong value at a real one).
            "field_fill_precision": ratio(self.field_value_correct, self.field_predicted_nonnull_total),
            "test_row_fill_precision": ratio(self.test_result_correct, self.test_predicted_total),
            "unit_accuracy": ratio(self.unit_correct, self.test_result_total),
            "reference_range_accuracy": ratio(self.reference_range_correct, self.test_result_total),
            "test_name_accuracy": ratio(self.test_name_matched, self.test_name_total),
            "test_result_accuracy": ratio(self.test_result_correct, self.test_result_total),
            "test_unit_mapping_accuracy": ratio(self.unit_correct, self.test_result_total),
        }
