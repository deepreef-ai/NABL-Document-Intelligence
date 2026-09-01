"""Per-file normalization: PDF or image bytes in, a NormalizedDocument out.
Reuses documents/pdf_utils.py (PyMuPDF) and documents/local_ocr.py (the
existing deepreef-ocr-backed OCR) exactly as they already are — no changes
to either, no second OCR/PDF implementation.

Confidence caveat (documented, not hidden): OcrResult only exposes ONE
confidence value per OCR call — the mean across every line RapidOCR found
on that page/image (see local_ocr.py / ocr_client.py's OcrResult; the
underlying deepreef-ocr engine computes a per-line value internally but
collapses it to a mean before returning). Rather than modify that existing,
shared contract just for this stage, every OcrElement on a given
OCR'd page is stamped with that same page-level mean — a real measured
value, not an invented one, just not literally per-element. If true
per-element confidence is ever needed, it would mean adding one field to
deepreef-ocr's own engine.py return dict (small, additive, but a change to
the shared OCR component, so deliberately not done here).
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import fitz  # PyMuPDF — already a project dependency (documents/pdf_utils.py)
from PIL import Image

from app.documents import local_ocr, pdf_utils
from app.documents.geometry import Rect
from app.dataset_normalization.models import NormalizedDocument, NormalizedPage, OcrElement
from app.dataset_normalization.text_quality import TextQualityThresholds, is_meaningful_page_text

IMAGE_SOURCE_FORMATS = {".jpg": "jpg", ".jpeg": "jpeg", ".png": "png", ".tif": "tif", ".tiff": "tiff"}


class UnsupportedOrInvalidFile(RuntimeError):
    pass


def _rect_to_bbox(rect: Rect) -> list[float]:
    """The target schema for this stage wants [x1, y1, x2, y2] corner
    points; the existing Rect type everywhere else in this project is
    {x, y, w, h}. Converting only here, at the output boundary — Rect
    itself is unchanged."""
    return [rect.x, rect.y, rect.x + rect.w, rect.y + rect.h]


def normalize_pdf(
    document_id: str,
    original_filename: str,
    source_path: str,
    data: bytes,
    thresholds: TextQualityThresholds,
) -> NormalizedDocument:
    t0 = time.perf_counter()

    # Password-protected PDFs open "successfully" via fitz but every later
    # text/page call on them returns empty/garbage rather than raising —
    # catching it explicitly here gives a clear, honest failure reason
    # instead of a silently-empty "processed" document.
    probe = fitz.open(stream=data, filetype="pdf")
    try:
        if probe.needs_pass:
            raise UnsupportedOrInvalidFile("password-protected PDF")
        n_pages = probe.page_count
    finally:
        probe.close()

    if n_pages == 0:
        raise UnsupportedOrInvalidFile("PDF has zero pages")

    text_pages = pdf_utils.extract_text_and_boxes(data)
    text_by_page_index = {p.page_number: p for p in text_pages}  # 0-indexed, matches pdf_utils

    pages: list[NormalizedPage] = []
    any_pymupdf = False
    any_ocr = False

    for page_index in range(n_pages):
        page_text_obj = text_by_page_index.get(page_index)
        page_text = page_text_obj.text if page_text_obj else ""

        if is_meaningful_page_text(page_text, thresholds):
            any_pymupdf = True
            pages.append(
                NormalizedPage(
                    page_number=page_index + 1,
                    text=page_text,
                    extraction_method="pymupdf",
                    ocr_used=False,
                    elements=[],  # PyMuPDF spans aren't OCR detections; nothing to score a confidence against
                )
            )
        else:
            any_ocr = True
            png_bytes = pdf_utils.rasterize_page(data, page_index)
            ocr_result = local_ocr.extract_english(png_bytes)
            elements = [
                OcrElement(text=line, bbox=_rect_to_bbox(box), confidence=ocr_result.confidence)
                for line, box in zip(ocr_result.lines, ocr_result.boxes)
            ]
            pages.append(
                NormalizedPage(
                    page_number=page_index + 1,
                    text=ocr_result.text,
                    extraction_method="ocr",
                    ocr_used=True,
                    elements=elements,
                    ocr_confidence=ocr_result.confidence,
                )
            )

    if any_pymupdf and any_ocr:
        source_type = "mixed_pdf"
    elif any_ocr:
        source_type = "scanned_pdf"
    else:
        source_type = "born_digital_pdf"

    return NormalizedDocument(
        document_id=document_id,
        original_filename=original_filename,
        source_path=source_path,
        source_format="pdf",
        source_type=source_type,
        page_count=n_pages,
        status="processed",
        pages=pages,
        processing_duration_seconds=round(time.perf_counter() - t0, 3),
    )


def normalize_image(
    document_id: str,
    original_filename: str,
    source_path: str,
    source_format: str,
    data: bytes,
) -> NormalizedDocument:
    t0 = time.perf_counter()

    # Validate the file can actually be opened as an image before spending
    # an OCR call on it — a clearer failure reason than whatever exception
    # decode_and_resize/RapidOCR would raise on garbage bytes.
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
    except Exception as exc:  # noqa: BLE001 — PIL raises several distinct error types for a bad image
        raise UnsupportedOrInvalidFile(f"not a valid image: {exc}") from exc

    ocr_result = local_ocr.extract_english(data)
    elements = [
        OcrElement(text=line, bbox=_rect_to_bbox(box), confidence=ocr_result.confidence)
        for line, box in zip(ocr_result.lines, ocr_result.boxes)
    ]
    page = NormalizedPage(
        page_number=1,
        text=ocr_result.text,
        extraction_method="ocr",
        ocr_used=True,
        elements=elements,
        ocr_confidence=ocr_result.confidence,
    )

    return NormalizedDocument(
        document_id=document_id,
        original_filename=original_filename,
        source_path=source_path,
        source_format=source_format,
        source_type="image",
        page_count=1,
        status="processed",
        pages=[page],
        processing_duration_seconds=round(time.perf_counter() - t0, 3),
    )


def normalize_file(document_id: str, path: Path, data: bytes, thresholds: TextQualityThresholds) -> NormalizedDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return normalize_pdf(document_id, path.name, str(path), data, thresholds)
    if suffix in IMAGE_SOURCE_FORMATS:
        return normalize_image(document_id, path.name, str(path), IMAGE_SOURCE_FORMATS[suffix], data)
    raise UnsupportedOrInvalidFile(f"unsupported extension: {suffix}")
