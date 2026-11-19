# app.evaluation

Measurement-only tool (no new dependencies). It reports numbers for whatever
you feed it; it contains no accuracy claims.

    cd backend
    python -m app.evaluation.run --gold <file|dir> --pred <file|dir> --out report.json

Gold and pred are JSON files (or directories of `*.json`). Each file is one
document or `{"documents": [...]}`. Documents are joined on `id`.

Gold document:

    {"id": "doc1", "text": "reference text", "fields": {"name": "Value"}}

Prediction document:

    {"id": "doc1", "text": "ocr text",
     "fields": {"name": {"value": "Value", "status": "verified"}},
     "latency_ms": {"ocr": 120, "extract": 900}}

A predicted field may also be a bare value (no status).
`status` of `verified`/`supported` counts as verified; `needs_review`/`review`/`flagged` counts as review.

Metrics (corpus-level, pooled over documents):
- CER / WER: total Levenshtein edits / total reference chars / words (whitespace tokens).
- field exact match: matched gold fields / gold fields. Missing predictions count as wrong.
  Comparison is case- and whitespace-insensitive string equality (TODO: switch to
  `strict_value_key` from `app.documents.value_matching`, which does not exist on this base).
- false verification rate: verified fields (present in gold) whose value is wrong / verified fields.
- review rate: flagged fields / predicted fields.
- latency: per stage count, p50, p95 (nearest-rank), mean, in the units given.

Rates are `null` when the denominator is 0. Documents without both sides are skipped
(gold-only ids are listed under `documents_missing_pred`).
