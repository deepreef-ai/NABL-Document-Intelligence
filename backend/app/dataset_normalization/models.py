"""The one common representation every input format (PDF, scanned PDF, JPG,
JPEG, PNG, TIFF) is normalized into. Deliberately independent of the live
app's DB models (models.py's Application/Document/ExtractedField) — this is
an offline batch artifact (a JSON file per source document), not a database
row, and has no notion of an Application to belong to.

`OcrElement.bbox` is `[x1, y1, x2, y2]` (corner points) to match the target
schema given for this stage — the existing `Rect` type used everywhere else
in this project is `{x, y, w, h}`; the conversion happens once, in
normalize.py, at the point a NormalizedPage is built. Nothing about the
existing Rect/OcrResult types changes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# "born_digital_pdf" | "scanned_pdf" | "mixed_pdf" | "image" — mixed_pdf means
# at least one page used pymupdf text AND at least one page needed OCR.
SourceType = Literal["born_digital_pdf", "scanned_pdf", "mixed_pdf", "image"]
ExtractionMethod = Literal["pymupdf", "ocr"]
DocumentStatus = Literal["processed", "failed", "skipped"]


@dataclass
class OcrElement:
    text: str
    bbox: list[float]  # [x1, y1, x2, y2]
    confidence: float


@dataclass
class NormalizedPage:
    page_number: int  # 1-based, matching the target schema's examples
    text: str
    extraction_method: ExtractionMethod
    ocr_used: bool
    elements: list[OcrElement] = field(default_factory=list)
    ocr_confidence: float | None = None  # page-level mean confidence, only when ocr_used


@dataclass
class NormalizedDocument:
    document_id: str
    original_filename: str
    source_path: str
    source_format: str  # "pdf" | "jpg" | "jpeg" | "png" | "tif" | "tiff"
    source_type: SourceType
    page_count: int
    status: DocumentStatus
    pages: list[NormalizedPage] = field(default_factory=list)
    error: str | None = None
    processing_duration_seconds: float | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IndexEntry:
    document_id: str
    original_filename: str
    source_format: str
    source_type: str | None
    page_count: int
    status: str
    extraction_method: str | None  # summary: "pymupdf" | "ocr" | "mixed" | None (failed before any page)
    ocr_used: bool
    error: str | None
    processing_duration_seconds: float | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)
