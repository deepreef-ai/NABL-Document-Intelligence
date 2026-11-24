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


def extract_tables(pdf_bytes: bytes, page_number: int, max_tables: int = 4) -> list[list[list[str]]]:
    """The page's tables as row/cell grids, via PyMuPDF's own table finder.

    WHY this exists: a PDF's text layer flattens a table into reading order,
    which routinely separates a cell from the row it belongs to. MEASURED on
    Sample-Smart-Report-Clinics.pdf, whose haemoglobin table has a real units
    column: the text layer emits "g/dL" as its own standalone line, with
    nothing tying it to any particular analyte. A model reading that stream
    cannot attribute the unit to a row, which is why 19 documents came back
    with a unit on zero of their rows while 37 came back with one on every
    row — the difference is whether the layout happened to keep cells inline.

    find_tables() reconstructs the grid from ruling lines and alignment, so
    the UNITS column lands back beside its own results. Verified on
    1601353705-Cow-Milk: returns ['S. NO','PARAMETERS','METHOD','UNITS',
    'RESULTS','LIMITS*'] as a header with each row's cells aligned under it.

    Returns [] on any failure — a page whose tables cannot be read is still
    worth extracting from its text.
    """
    # fitz.open is INSIDE the try: bytes that are not a real PDF raise
    # FzErrorFormat right here, and this function is called opportunistically
    # on whatever the caller happens to hold. Structure is a bonus, never a
    # requirement — a page whose tables cannot be read is still worth
    # extracting from its text, so every failure returns [].
    doc = None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        found = doc[page_number].find_tables()
        grids = []
        for table in list(getattr(found, "tables", []))[:max_tables]:
            rows = [[("" if cell is None else str(cell).replace("\n", " ").strip())
                     for cell in row]
                    for row in table.extract()]
            # A single-column "table" is a layout artefact, not tabular data.
            if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
                grids.append(rows)
        return grids
    except Exception:  # noqa: BLE001 — see above
        return []
    finally:
        if doc is not None:
            doc.close()


def page_count(pdf_bytes: bytes) -> int:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return doc.page_count
    finally:
        doc.close()
