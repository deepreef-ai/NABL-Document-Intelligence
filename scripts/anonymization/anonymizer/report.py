"""Per-document and per-run reporting.

A deliberate constraint runs through this module: **the report must not become
the leak.** A log line reading `redacted AADHAAR "4321 8765 1234"` recreates
exactly the disclosure the tool exists to prevent, in a file that is far more
likely to be emailed around than the scan itself.

So reports record kind, page, box and rule — never the value — unless
`include_pii_samples` is explicitly switched on, and even then only a
two-character head with a length. When it is on, the report inherits the
sensitivity of the source document, and the header says so.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

STATUS_OK = "OK"
STATUS_FAILED_REVIEW = "FAILED_REVIEW"
STATUS_ERROR = "ERROR"
STATUS_SKIPPED = "SKIPPED"


@dataclass
class PageSummary:
    page: int
    source: str                  # text_layer | ocr | empty
    detections: int = 0
    redactions_applied: int = 0
    ocr_mean_confidence: float | None = None
    low_confidence: bool = False
    note: str = ""


@dataclass
class DetectionRecord:
    kind: str
    page: int
    rule: str
    source: str
    confidence: float
    bbox: tuple[float, float, float, float]
    sample: str = ""


@dataclass
class DocumentReport:
    source_path: str
    output_path: str
    relative_path: str
    file_type: str                       # pdf | image
    status: str = STATUS_OK
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0
    page_count: int = 0
    total_detections: int = 0
    total_redactions: int = 0
    detections_by_kind: dict[str, int] = field(default_factory=dict)
    pages: list[PageSummary] = field(default_factory=list)
    detections: list[DetectionRecord] = field(default_factory=list)
    verification_passed: bool | None = None
    verification_summary: str = ""
    verification_findings: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str = ""
    config_fingerprint: str = ""
    contains_pii_samples: bool = False

    def to_dict(self) -> dict:
        data = asdict(self)
        data["_note"] = (
            "This report contains truncated PII samples and must be handled as "
            "sensitive." if self.contains_pii_samples else
            "This report records detection kinds and positions only — no PII values."
        )
        return data

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                        encoding="utf-8")


@dataclass
class RunReport:
    input_root: str
    output_root: str
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str = ""
    documents: list[DocumentReport] = field(default_factory=list)
    config: dict = field(default_factory=dict)

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for d in self.documents:
            out[d.status] = out.get(d.status, 0) + 1
        return out

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_root": self.input_root,
            "output_root": self.output_root,
            "started_at": self.started_at,
            "finished_at": self.finished_at or datetime.now(timezone.utc).isoformat(),
            "totals": {
                "documents": len(self.documents),
                **self.counts,
                "detections": sum(d.total_detections for d in self.documents),
                "redactions": sum(d.total_redactions for d in self.documents),
            },
            "config": self.config,
            "documents": [
                {
                    "relative_path": d.relative_path,
                    "status": d.status,
                    "file_type": d.file_type,
                    "pages": d.page_count,
                    "detections": d.total_detections,
                    "redactions": d.total_redactions,
                    "detections_by_kind": d.detections_by_kind,
                    "verification_passed": d.verification_passed,
                    "verification_summary": d.verification_summary,
                    "error": d.error,
                    "warnings": d.warnings,
                }
                for d in self.documents
            ],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def write_csv(self, path: Path) -> None:
        import csv

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "relative_path", "status", "file_type", "pages", "detections",
                "redactions", "verification_passed", "verification_summary", "error",
            ])
            for d in self.documents:
                writer.writerow([
                    d.relative_path, d.status, d.file_type, d.page_count,
                    d.total_detections, d.total_redactions,
                    "" if d.verification_passed is None else d.verification_passed,
                    d.verification_summary, d.error,
                ])
