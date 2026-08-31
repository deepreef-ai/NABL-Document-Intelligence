from dataclasses import dataclass, field

import fitz  # PyMuPDF

from app.documents.geometry import Rect

# Below this many extracted characters, treat the PDF as scanned/image-only —
# a handful of stray characters (form field labels burned into a scan, etc.)
# shouldn't be mistaken for a real text layer.
TEXT_LAYER_MIN_CHARS = 20


@dataclass
class PageText:
    page_number: int
    text: str
    spans: list[tuple[str, Rect]] = field(default_factory=list)


def has_text_layer(pdf_bytes: bytes) -> bool:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        total_chars = sum(len(page.get_text("text").strip()) for page in doc)
        return total_chars >= TEXT_LAYER_MIN_CHARS
    finally:
        doc.close()


def extract_text_and_boxes(pdf_bytes: bytes) -> list[PageText]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pages = []
        for i, page in enumerate(doc):
            spans: list[tuple[str, Rect]] = []
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        x0, y0, x1, y1 = span["bbox"]
                        spans.append((text, Rect(x=x0, y=y0, w=x1 - x0, h=y1 - y0)))
            pages.append(PageText(page_number=i, text=page.get_text("text"), spans=spans))
        return pages
    finally:
        doc.close()


def rasterize_page(pdf_bytes: bytes, page_number: int, dpi: int = 200) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pix = doc[page_number].get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    finally:
        doc.close()


def page_count(pdf_bytes: bytes) -> int:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return doc.page_count
    finally:
        doc.close()
