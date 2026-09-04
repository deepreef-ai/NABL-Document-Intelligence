"""Pools raw counts (never per-document averages) into the ratio metrics
the benchmark reports — the same accumulator class is used for the overall
totals, each domain bucket, each format bucket, AND each single document's
own CSV row, so every reported number comes from one consistent
computation, never a duplicated formula.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _percentile(values: list[int], pct: float) -> int | None:
    """Nearest-rank percentile. No numpy dependency for one number, and
    nearest-rank is the honest choice for small samples: it returns a value
    that a document actually cost, not an interpolation between two."""
    if not values:
        return None
    ordered = sorted(values)
    import math

    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


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
    unit_correct: int = 0
    reference_range_correct: int = 0
    test_name_matched: int = 0
    test_name_total: int = 0
    test_result_correct: int = 0
    test_result_total: int = 0  # == matched test rows
    # --- LLM call metrics (spec section 16) -----------------------------
    # Additive and completely separate from the quality counters above: a
    # call reduction that costs accuracy must stay visible, so these are
    # reported ALONGSIDE precision/recall/F1, never folded into them.
    classification_calls: int = 0
    extraction_calls: int = 0
    recovery_calls: int = 0
    vision_calls: int = 0
    total_llm_calls: int = 0
    calls_per_document: list[int] = field(default_factory=list)

    def add(self, *, domain_match: bool, extraction_ok: bool, exact_match: bool, field_counters: dict, test_counters: dict, call_log: dict | None = None) -> None:
        self.document_count += 1
        self.domain_correct += int(domain_match)
        self.extraction_success += int(extraction_ok)
        self.exact_match += int(exact_match)

        if call_log:
            self.classification_calls += int(call_log.get("classification_calls") or 0)
            self.extraction_calls += int(call_log.get("extraction_calls") or 0)
            self.recovery_calls += int(call_log.get("recovery_calls") or 0)
            self.vision_calls += int(call_log.get("vision_calls") or 0)
            total = int(call_log.get("total_llm_calls") or 0)
            self.total_llm_calls += total
            self.calls_per_document.append(total)

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

        matched = test_counters["matched_count"]
        # Table-row values fold into the same combined value_accuracy as
        # field values (VALUE LEVEL is "every extracted value", fields and
        # test results together) while also being tracked on their own for
        # TABLE LEVEL's test_result_accuracy.
        self.value_correct += test_counters["result_correct"]
        self.value_total += matched
        self.test_result_correct += test_counters["result_correct"]
        self.test_result_total += matched
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
            "unit_accuracy": ratio(self.unit_correct, self.test_result_total),
            "reference_range_accuracy": ratio(self.reference_range_correct, self.test_result_total),
            "test_name_accuracy": ratio(self.test_name_matched, self.test_name_total),
            "test_result_accuracy": ratio(self.test_result_correct, self.test_result_total),
            "test_unit_mapping_accuracy": ratio(self.unit_correct, self.test_result_total),
            # --- call metrics (see the counters' comment above) ---------
            "classification_calls": self.classification_calls,
            "extraction_calls": self.extraction_calls,
            "recovery_calls": self.recovery_calls,
            "vision_calls": self.vision_calls,
            "total_llm_calls": self.total_llm_calls,
            "average_calls": (
                round(self.total_llm_calls / len(self.calls_per_document), 4)
                if self.calls_per_document else None
            ),
            # p95 by nearest-rank on the sorted per-document counts. On a
            # 52-document run that is the 50th value, i.e. "the worst
            # document short of the outliers" - the number that says whether
            # the budget actually holds in practice, which a mean hides.
            "p95_calls": _percentile(self.calls_per_document, 95),
            "max_calls": max(self.calls_per_document) if self.calls_per_document else None,
        }
