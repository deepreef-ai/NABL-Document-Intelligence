"""Output structures for the labeling stage: the per-document label shape
requested for Step 5 (document_id/domain/fields/tests) plus the metadata
Step 6's quality control needs to check it — page_count and OCR provenance
to cross-check against the source normalized document, extraction_status to
tell "the LLM call itself failed" apart from "it succeeded but found little",
and annotation_status as the pending/reviewed/approved/rejected state Step 6
enforces transitions on.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

ANNOTATION_STATUSES = ("pending", "reviewed", "approved", "rejected")


@dataclass
class TestRow:
    test_name: str
    result: str | int | float | bool | None
    unit: str | int | float | bool | None
    reference_range: str | int | float | bool | None


@dataclass
class DocumentLabel:
    document_id: str
    original_filename: str
    domain: str
    page_count: int
    source_ocr_used: bool
    source_ocr_confidence: float | None
    fields: dict[str, Any]
    tests: list[TestRow]
    annotation_status: str  # "pending" | "reviewed" | "approved" | "rejected"
    extraction_status: str  # "ok" | "failed"
    error: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LabelIndexEntry:
    document_id: str
    original_filename: str
    domain: str
    annotation_status: str
    extraction_status: str
    field_count: int
    test_count: int
    error: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)
