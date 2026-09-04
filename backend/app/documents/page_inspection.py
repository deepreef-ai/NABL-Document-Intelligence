"""Page-level document inspection: what each page contains and how its text
was obtained, before any LLM is involved.

Reuses the OFFLINE package's representation rather than inventing a second
one: `dataset_normalization.models.NormalizedDocument`/`NormalizedPage`
already carry page_number / text / extraction_method / ocr_confidence, and
`dataset_normalization.text_quality.is_meaningful_page_text` already decides
"does this page have usable native text" with configurable, multi-signal
thresholds (char count AND alphanumeric ratio AND word count) — strictly
better than the ad-hoc 20-character check the live pipeline used. One
vocabulary for both the live and offline paths.

Classification is per PAGE, never per document. A 20-page PDF is routinely
pages 1-2 native text, 3-4 scanned, 5 native again; `pdf_utils.has_text_layer`
is document-level and would call that whole file born-digital.

Table detection is new — nothing in the repo did it. PyMuPDF's own
`page.find_tables()` is used where available (it ships with the version
already installed and was simply never called), with a ruling-line/aligned-
whitespace heuristic as the fallback so an OCR'd page, which has no PyMuPDF
table structure at all, can still be flagged.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import fitz

from app.config import get_settings
from app.dataset_normalization.models import NormalizedDocument, NormalizedPage
from app.dataset_normalization.text_quality import TextQualityThresholds, is_meaningful_page_text
from app.documents import local_ocr, pdf_utils
from app.documents.geometry import Rect
from app.documents.ocr_client import SUPPORTED_SCRIPTS, OcrClient

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# A page needs at least this many rows that look like "label ... value" or
# "name value unit range" before the whitespace heuristic calls it a table.
_MIN_TABLE_ROWS = 3
# Three or more runs of 2+ spaces on one line reads as columnar layout.
_COLUMNAR = re.compile(r"\S(?:\s{2,}\S+){2,}")
# A lab result row: a name, then a number, optionally unit and a range.
_RESULT_ROW = re.compile(r"[A-Za-z][A-Za-z ()/.-]{2,}\s+[<>]?\d+(?:\.\d+)?", re.MULTILINE)


@dataclass
class InspectedPage:
    """A NormalizedPage plus the live pipeline's extra needs (spans for
    grounding, the rendered image for vision escalation)."""

    page: NormalizedPage
    spans: list[tuple[str, Rect]] = field(default_factory=list)
    image: bytes | None = None

    @property
    def page_number(self) -> int:
        return self.page.page_number

    @property
    def text(self) -> str:
        return self.page.text

    @property
    def has_table(self) -> bool:
        return bool(self.page.has_table)

    @property
    def needs_image(self) -> bool:
        """Whether this page's IMAGE should accompany its text to the model.

        A born-digital page's PyMuPDF text is exact — the image adds tokens
        and no information. A page whose text came from OCR is different: the
        transcription is a guess, and the image is the actual evidence.

        This is what makes the fallback chain usable. MEASURED 2026-09-03:
        attaching an image to every page meant a 9-page chunk carried 9
        images, and EVERY non-Nova provider rejects multi-image requests
        ("gemini takes one image per request, got 9 pages"), so when Nova
        was unavailable the document extracted nothing at all. It also
        matches the stated policy: OCR first, vision only when necessary.
        """
        return self.page.extraction_method == "ocr" or self.needs_visual_check

    @property
    def needs_visual_check(self) -> bool:
        """Low-confidence OCR marks a page as POSSIBLY needing a look. It
        does not by itself justify a vision call — see recovery.py."""
        if self.page.extraction_method != "ocr" or self.page.ocr_confidence is None:
            return False
        return self.page.ocr_confidence < get_settings().ocr_low_confidence_threshold


@dataclass
class InspectedDocument:
    document: NormalizedDocument
    pages: list[InspectedPage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return self.document.page_count

    @property
    def source_format(self) -> str:
        return self.document.source_format

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)

    def to_dict(self) -> dict:
        return self.document.to_json_dict()


def _table_from_pymupdf(page: fitz.Page) -> bool:
    try:
        return len(list(page.find_tables().tables)) > 0
    except Exception:  # noqa: BLE001 — older/newer PyMuPDF, or a page it can't parse
        return False


def detect_table(text: str, page: fitz.Page | None = None) -> bool:
    """True when the page looks like it contains a results table.

    Tries PyMuPDF's real table finder first (it uses ruling lines and text
    alignment), then falls back to text shape — necessary because an OCR'd
    page has no PyMuPDF table structure to find, and scanned lab reports are
    exactly where tables matter most.
    """
    if page is not None and _table_from_pymupdf(page):
        return True
    if not text:
        return False
    columnar_rows = len(_COLUMNAR.findall(text))
    result_rows = len(_RESULT_ROW.findall(text))
    return columnar_rows >= _MIN_TABLE_ROWS or result_rows >= _MIN_TABLE_ROWS


def _ocr_page(png: bytes, script: str, ocr_client: OcrClient | None):
    if script in SUPPORTED_SCRIPTS:
        return (ocr_client or OcrClient()).extract(png, script)
    return local_ocr.extract_english(png)


def inspect_pdf(
    data: bytes,
    document_id: str = "unknown",
    filename: str = "",
    script: str = "english",
    ocr_client: OcrClient | None = None,
    thresholds: TextQualityThresholds | None = None,
) -> InspectedDocument:
    """Per-page: native text or OCR, confidence, table flag, spans, image.

    The document is opened ONCE here. pdf_utils opens it fresh on every call,
    so the old path re-opened a 17-page PDF 17+ times just to rasterize it.
    """
    thresholds = thresholds or TextQualityThresholds()
    dpi = get_settings().page_image_dpi
    native = {p.page_number: p for p in pdf_utils.extract_text_and_boxes(data)}

    pages: list[InspectedPage] = []
    warnings: list[str] = []
    used_native = used_ocr = False

    doc = fitz.open(stream=data, filetype="pdf")
    try:
        if doc.needs_pass:
            raise ValueError("password-protected PDF")
        for i in range(doc.page_count):
            fitz_page = doc[i]
            native_page = native.get(i)
            native_text = native_page.text if native_page else ""
            spans = list(native_page.spans) if native_page else []

            image: bytes | None = None
            try:
                image = fitz_page.get_pixmap(dpi=dpi).tobytes("png")
            except Exception as exc:  # noqa: BLE001 — text alone is still usable
                warnings.append(f"page {i + 1} could not be rendered: {exc}")

            if is_meaningful_page_text(native_text, thresholds):
                used_native = True
                page = NormalizedPage(
                    page_number=i + 1,
                    text=native_text,
                    extraction_method="pymupdf",
                    ocr_used=False,
                    has_table=detect_table(native_text, fitz_page),
                )
            else:
                used_ocr = True
                text, confidence = "", None
                if image is not None:
                    try:
                        ocr = _ocr_page(image, script, ocr_client)
                        text, confidence = ocr.text, ocr.confidence
                        spans = list(zip(ocr.lines, ocr.boxes))
                    except Exception as exc:  # noqa: BLE001 — one page must not lose the rest
                        log.warning("page_inspection: OCR failed on page %d: %s", i, exc)
                        warnings.append(f"page {i + 1} OCR failed: {exc}")
                page = NormalizedPage(
                    page_number=i + 1,
                    text=text,
                    extraction_method="ocr",
                    ocr_used=True,
                    ocr_confidence=confidence,
                    # No PyMuPDF table structure on a scanned page — text shape only.
                    has_table=detect_table(text),
                )
            pages.append(InspectedPage(page=page, spans=spans, image=image))
    finally:
        doc.close()

    if used_native and used_ocr:
        source_type = "mixed_pdf"
    elif used_ocr:
        source_type = "scanned_pdf"
    else:
        source_type = "born_digital_pdf"

    document = NormalizedDocument(
        document_id=document_id,
        original_filename=filename,
        source_path=filename,
        source_format="pdf",
        source_type=source_type,
        page_count=len(pages),
        status="processed",
        pages=[p.page for p in pages],
    )
    return InspectedDocument(document=document, pages=pages, warnings=warnings)


def inspect_image(
    data: bytes,
    document_id: str = "unknown",
    filename: str = "",
    suffix: str = ".png",
    script: str = "english",
    ocr_client: OcrClient | None = None,
) -> InspectedDocument:
    """A standalone image is one page: OCR it, keep the image for vision."""
    warnings: list[str] = []
    text, confidence, spans = "", None, []
    try:
        ocr = _ocr_page(data, script, ocr_client)
        text, confidence = ocr.text, ocr.confidence
        spans = list(zip(ocr.lines, ocr.boxes))
    except Exception as exc:  # noqa: BLE001 — the image itself still goes to the model
        log.warning("page_inspection: OCR failed for image file: %s", exc)
        warnings.append(f"OCR failed, page usable as image only: {exc}")

    page = NormalizedPage(
        page_number=1,
        text=text,
        extraction_method="ocr",
        ocr_used=True,
        ocr_confidence=confidence,
        has_table=detect_table(text),
    )
    document = NormalizedDocument(
        document_id=document_id,
        original_filename=filename,
        source_path=filename,
        source_format=suffix.lstrip(".").lower(),
        source_type="image",
        page_count=1,
        status="processed",
        pages=[page],
    )
    return InspectedDocument(
        document=document,
        pages=[InspectedPage(page=page, spans=spans, image=data)],
        warnings=warnings,
    )


def inspect(
    data: bytes,
    suffix: str,
    document_id: str = "unknown",
    filename: str = "",
    script: str = "english",
    ocr_client: OcrClient | None = None,
) -> InspectedDocument:
    suffix = suffix.lower()
    if suffix == ".pdf":
        return inspect_pdf(data, document_id, filename, script, ocr_client)
    if suffix in IMAGE_EXTENSIONS:
        return inspect_image(data, document_id, filename, suffix, script, ocr_client)
    raise ValueError(f"unsupported file type: {suffix}")
