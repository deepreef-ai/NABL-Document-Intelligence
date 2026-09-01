from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class QCIssue:
    code: str
    message: str

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QCContext:
    """Everything a single check function needs beyond the label itself —
    built once per document by pipeline.py so checks stay pure functions."""
    doc_json: dict | None
    required_fields: list[str]
    ocr_confidence_threshold: float
    duplicate_document_id: bool = False
    duplicate_content_with: list[str] = field(default_factory=list)


@dataclass
class QCResult:
    document_id: str
    passed: bool
    issues: list[QCIssue]
    previous_status: str
    new_status: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "passed": self.passed,
            "issues": [i.to_json_dict() for i in self.issues],
            "previous_status": self.previous_status,
            "new_status": self.new_status,
        }
