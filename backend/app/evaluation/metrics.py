"""Pure-Python, dependency-free evaluation metrics.

These only MEASURE; nothing here encodes or assumes an accuracy level.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

VERIFIED_STATUSES = frozenset({"verified", "supported"})
REVIEW_STATUSES = frozenset({"needs_review", "review", "flagged"})


def levenshtein(a: Sequence[Any], b: Sequence[Any]) -> int:
    """Edit distance (insert/delete/substitute, unit cost) over two sequences."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """edits / len(reference). Empty reference: 0.0 if hypothesis empty else 1.0."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis) / len(reference)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Whitespace-token Levenshtein / number of reference words."""
    ref, hyp = reference.split(), hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def _norm(value: Any) -> str:
    # TODO: app.documents.value_matching.strict_value_key does not exist on this
    # base (feat/demo22). Replace with it (strict numeric rules) once available.
    return " ".join(str(value).split()).casefold()


def values_match(gold: Any, pred: Any) -> bool:
    return _norm(gold) == _norm(pred)


def _pred_value(entry: Any) -> Any:
    return entry.get("value") if isinstance(entry, Mapping) else entry


def _pred_status(entry: Any) -> str:
    if isinstance(entry, Mapping):
        return str(entry.get("status") or "").strip().lower()
    return ""


def field_exact_match(gold: Mapping[str, Any], pred: Mapping[str, Any]) -> dict:
    """Compare predicted to gold fields. Missing predicted fields count as wrong.
    Extra predicted fields (not in gold) are ignored. Returns counts and rate."""
    matched = sum(
        1 for k, g in gold.items()
        if k in pred and _pred_value(pred[k]) is not None
        and values_match(g, _pred_value(pred[k]))
    )
    total = len(gold)
    return {"matched": matched, "total": total,
            "rate": matched / total if total else None}


def false_verification_rate(gold: Mapping[str, Any], pred: Mapping[str, Any]) -> dict:
    """Among predicted fields with a verified/supported status that also exist in
    gold, the fraction whose value is wrong vs gold."""
    verified = wrong = 0
    for k, entry in pred.items():
        if k not in gold or _pred_status(entry) not in VERIFIED_STATUSES:
            continue
        verified += 1
        if not values_match(gold[k], _pred_value(entry)):
            wrong += 1
    return {"wrong_verified": wrong, "verified": verified,
            "rate": wrong / verified if verified else None}


def review_rate(pred: Mapping[str, Any]) -> dict:
    """Fraction of predicted fields flagged for human review."""
    flagged = sum(1 for e in pred.values() if _pred_status(e) in REVIEW_STATUSES)
    total = len(pred)
    return {"flagged": flagged, "total": total,
            "rate": flagged / total if total else None}


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile, q in (0, 100]."""
    s = sorted(values)
    return s[max(1, math.ceil(q / 100 * len(s))) - 1]


def latency_summary(samples: Mapping[str, Sequence[float]]) -> dict:
    """Per-stage {count, p50, p95, mean} from {stage: [ms, ...]}."""
    out = {}
    for stage, vals in samples.items():
        if not vals:
            continue
        out[stage] = {"count": len(vals), "p50": percentile(vals, 50),
                      "p95": percentile(vals, 95), "mean": sum(vals) / len(vals)}
    return out
