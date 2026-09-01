from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FinalDocumentRecord:
    document_id: str
    original_filename: str
    domain: str
    document_type: str | None
    source_format: str
    source_path: str
    page_count: int
    fields: dict[str, Any]
    tests: list[dict[str, Any]]
    annotation_status: str

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetSummary:
    total_documents: int = 0
    documents_by_domain: dict[str, int] = field(default_factory=dict)
    documents_by_format: dict[str, int] = field(default_factory=dict)
    documents_by_page_count: dict[str, int] = field(default_factory=dict)
    total_fields: int = 0
    total_test_records: int = 0
    approved_annotations: int = 0
    missing_null_fields: int = 0
    excluded_by_validation: int = 0
    validation_errors: list[dict[str, Any]] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)
