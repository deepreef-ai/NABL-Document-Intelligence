"""Builds page-level Chunk objects for the completed_application_form (whole
NABL form) pipeline — one chunk per PDF page, filling in local OCR text for
any individual page whose PyMuPDF text layer is too thin to be useful.

pdf_utils.has_text_layer() is a document-level check; a mostly-digital PDF
can still have a handful of scanned/photographed pages mixed in (a signed
declaration page, a scanned annexure). This catches those individually
instead of OCR'ing the whole document."""
from dataclasses import dataclass, field

from app.documents import local_ocr, pdf_utils
from app.documents.geometry import Rect

# Below this many characters, a page's own PyMuPDF text is treated as
# "nothing useful" and local RapidOCR is tried on that one page instead.
MIN_PAGE_TEXT_CHARS = 20


@dataclass
class Chunk:
    page_number: int
    text: str
    spans: list[tuple[str, Rect]] = field(default_factory=list)


def build_chunks(data: bytes, pages: list[pdf_utils.PageText]) -> list[Chunk]:
    """`pages` is pdf_utils.extract_text_and_boxes()'s output. Best-effort:
    if the per-page OCR fallback itself fails, the page's original (thin)
    text is kept rather than losing the page entirely."""
    chunks = []
    for page in pages:
        text, spans = page.text, list(page.spans)
        if len(text.strip()) < MIN_PAGE_TEXT_CHARS:
            try:
                png = pdf_utils.rasterize_page(data, page.page_number)
                ocr_result = local_ocr.extract_english(png)
            except local_ocr.LocalOcrError:
                ocr_result = None
            if ocr_result and len(ocr_result.text.strip()) > len(text.strip()):
                text = ocr_result.text
                spans = list(zip(ocr_result.lines, ocr_result.boxes))
        chunks.append(Chunk(page_number=page.page_number, text=text, spans=spans))
    return chunks
