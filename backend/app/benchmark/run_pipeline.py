"""The ONLY extraction path this benchmark uses: the exact same production
functions Steps 2 and 5 already run — app.dataset_normalization.normalize's
normalize_file (PDF/OCR extraction via documents/pdf_utils.py and
documents/local_ocr.py, unmodified) feeding into
app.labeling.extraction.extract_label (domain classification + field/key
extraction, unmodified). Nothing here reimplements either step — this
module only calls them fresh against the original raw file and reshapes
the result into a plain dict for compare.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.dataset_normalization.normalize import normalize_file
from app.dataset_normalization.text_quality import TextQualityThresholds
from app.labeling.extraction import extract_label


def run_production_pipeline(document_id: str, source_path: Path, domain_hints: dict) -> dict[str, Any]:
    """Returns {"domain", "fields", "tests", "page_count", "pipeline_error"}.
    pipeline_error is None on success; on any exception from the real
    pipeline (OCR failure, LLM failure, unreadable file, ...) it holds a
    message and domain/fields/tests come back empty — the benchmark scores
    that as "predicted nothing" rather than crashing the whole run."""
    try:
        data = source_path.read_bytes()
        normalized = normalize_file(document_id, source_path, data, TextQualityThresholds())
        text = "\n".join(page.text for page in normalized.pages)
        label = extract_label(
            document_id=document_id,
            original_filename=source_path.name,
            text=text,
            page_count=normalized.page_count,
            source_ocr_used=any(page.ocr_used for page in normalized.pages),
            source_ocr_confidence=None,
            domain_hints=domain_hints,
        )
        return {
            "domain": label.domain,
            "fields": label.fields,
            "tests": [
                {"test_name": t.test_name, "result": t.result, "unit": t.unit, "reference_range": t.reference_range}
                for t in label.tests
            ],
            "page_count": normalized.page_count,
            "pipeline_error": None,
        }
    except Exception as exc:  # noqa: BLE001 — a production-pipeline failure is a benchmark result, not a crash
        return {
            "domain": None, "fields": {}, "tests": [], "page_count": None,
            "pipeline_error": f"{type(exc).__name__}: {exc}",
        }
