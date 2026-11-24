"""Applying redactions — the part that must genuinely remove, not merely cover.

The classic anonymiser failure is drawing a black rectangle over text and
shipping the file with the text still selectable underneath. Two rules here
prevent that:

1. **Text-layer pages** are redacted with PyMuPDF's redaction annotations and
   `apply_redactions()`, which deletes the glyphs from the content stream.
   A drawn rectangle is never used on its own.

2. **Scanned pages** are rebuilt. The page is rasterised, the boxes are painted
   onto the pixels, and a *new* page is created holding only that image. The
   original scan is never copied into the output, so the untouched original
   cannot survive as an unreferenced XObject — which is exactly what happens
   if you blank a page's content stream and insert a new image over the top.

The output document is assembled fresh rather than edited in place, and saved
with garbage collection, so nothing unreferenced rides along.
"""
from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass, field

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageDraw

from .config import RedactionConfig
from .detect import Detection

log = logging.getLogger(__name__)

BBox = tuple[float, float, float, float]

# Pseudonym pools. Small and obviously synthetic on purpose: a pseudonym that
# looks like a real Indian name risks being mistaken for genuine data, and the
# point of the output is that it is visibly not a real person.
_FIRST = ("ALFA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT", "GOLF",
          "HOTEL", "INDIA", "JULIET", "KILO", "LIMA", "MIKE", "NOVEMBER")
_LAST = ("ANDERSON", "BAKER", "CARTER", "DIXON", "EVANS", "FLETCHER",
         "GORDON", "HARPER", "IRVING", "JENKINS", "KELLER", "LOWRY")


@dataclass
class PageRedactions:
    """What to redact on one page, and how that page must be handled."""

    page_number: int
    kind: str                       # "text_layer" | "ocr"
    boxes: list[tuple[BBox, str]] = field(default_factory=list)  # (bbox, detection kind)
    # The unit the boxes are expressed in, as dots per inch. 72 means PDF
    # points (what the text layer uses); an OCR page's boxes arrive in pixels
    # at whatever DPI the page was rasterised for recognition.
    #
    # This used to be a bare "raster_scale" that the rebuild step multiplied by
    # the output zoom whenever it equalled 1.0 — which double-scaled every OCR
    # box by output_dpi/72 and placed every redaction on a scanned page roughly
    # 2.8x away from the text it was meant to cover. Boxes were painted, the
    # count looked right, and the identifier was still legible. Naming the unit
    # makes the conversion impossible to get wrong by accident.
    source_dpi: float = 72.0


def _stable_digest(salt: str, kind: str, value: str) -> str:
    key = f"{salt}|{kind}|{value.strip().lower()}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()


def pseudonym_for(kind: str, value: str, salt: str) -> str:
    """A stable synthetic stand-in for one real value.

    Stable means the same person maps to the same pseudonym everywhere the
    salt is held constant, so longitudinal structure survives anonymisation.
    It also means the mapping is reversible by anyone holding the salt and a
    guess at the value — so the salt is a secret, and the default (empty,
    regenerated per run) deliberately breaks linkage instead.
    """
    digest = _stable_digest(salt, kind, value)
    n = int(digest[:12], 16)
    if kind in ("PATIENT_NAME", "DOCTOR_NAME"):
        first = _FIRST[n % len(_FIRST)]
        last = _LAST[(n // len(_FIRST)) % len(_LAST)]
        prefix = "DR " if kind == "DOCTOR_NAME" else ""
        return f"{prefix}{first} {last}"
    if kind in ("AADHAAR",):
        return f"XXXX XXXX {n % 10000:04d}"
    if kind in ("PHONE",):
        return f"+91 9{n % 1000000000:09d}"
    if kind in ("EMAIL",):
        return f"patient{n % 100000:05d}@example.invalid"
    if kind in ("PATIENT_ID", "PAN", "PASSPORT", "VOTER_ID", "DRIVING_LICENCE", "IFSC"):
        return f"{kind[:3]}{n % 10_000_000:07d}"
    if kind == "DOB":
        return "01-JAN-1900"
    if kind == "ADDRESS":
        return "[ADDRESS REMOVED]"
    return f"[{kind}]"


def replacement_text(kind: str, value: str, cfg: RedactionConfig) -> str:
    if cfg.mode == "black_box":
        return ""
    if cfg.mode == "pseudonym":
        return pseudonym_for(kind, value, cfg.pseudonym_salt)
    return f"[{kind}]"


def _pad(bbox: BBox, padding: float) -> BBox:
    x0, y0, x1, y1 = bbox
    return (x0 - padding, y0 - padding, x1 + padding, y1 + padding)


def merge_adjacent(boxes: list[BBox], gap: float = 3.0) -> list[BBox]:
    """Merge boxes that sit on the same line and nearly touch.

    A three-word name arrives as three boxes; redacting them separately leaves
    thin unredacted slivers between the words where descenders and the tails of
    letters can still be read.
    """
    if not boxes:
        return []
    ordered = sorted(boxes, key=lambda b: (round((b[1] + b[3]) / 2, 1), b[0]))
    merged = [list(ordered[0])]
    for b in ordered[1:]:
        cur = merged[-1]
        same_line = abs(((b[1] + b[3]) / 2) - ((cur[1] + cur[3]) / 2)) <= max(
            (cur[3] - cur[1]) * 0.6, 2.0)
        if same_line and b[0] - cur[2] <= gap:
            cur[2] = max(cur[2], b[2])
            cur[1] = min(cur[1], b[1])
            cur[3] = max(cur[3], b[3])
        else:
            merged.append(list(b))
    return [tuple(m) for m in merged]  # type: ignore[misc]


def plan_from_detections(
    detections: list[Detection], page_number: int, kind: str, source_dpi: float = 72.0,
) -> PageRedactions:
    plan = PageRedactions(page_number=page_number, kind=kind, source_dpi=source_dpi)
    for d in detections:
        for bbox in d.bboxes:
            plan.boxes.append((bbox, d.kind))
    return plan


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------
def _redact_text_page(dst_page, plan: PageRedactions, cfg: RedactionConfig) -> int:
    """True removal on a page that has a real text layer."""
    by_kind: dict[str, list[BBox]] = {}
    for bbox, kind in plan.boxes:
        by_kind.setdefault(kind, []).append(bbox)

    applied = 0
    for kind, boxes in by_kind.items():
        for bbox in merge_adjacent(boxes):
            rect = fitz.Rect(*_pad(bbox, cfg.padding_px))
            if rect.is_empty or rect.is_infinite:
                continue
            label = replacement_text(kind, "", cfg)
            if label:
                # Size the stamp to the box so a long label cannot overflow
                # into neighbouring content.
                size = max(4.0, min(9.0, (rect.height * 0.62)))
                dst_page.add_redact_annot(
                    rect, text=label, fontsize=size, fontname="helv",
                    align=fitz.TEXT_ALIGN_LEFT,
                    fill=cfg.box_colour, text_color=cfg.label_colour,
                )
            else:
                dst_page.add_redact_annot(rect, fill=cfg.box_colour)
            applied += 1

    if applied:
        # PDF_REDACT_IMAGE_PIXELS blanks only the pixels of an image that fall
        # inside a redaction rectangle, rather than deleting whole images that
        # merely intersect one — a logo overlapping a header box survives.
        dst_page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS)
    return applied


def paint_boxes_on_image(
    image: Image.Image, boxes: list[tuple[BBox, str]], cfg: RedactionConfig,
) -> tuple[Image.Image, int]:
    """Burn redactions into pixels. Used for scanned pages and image files."""
    if not boxes:
        return image, 0
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    fill = tuple(int(c * 255) for c in cfg.box_colour)
    text_fill = tuple(int(c * 255) for c in cfg.label_colour)

    by_kind: dict[str, list[BBox]] = {}
    for bbox, kind in boxes:
        by_kind.setdefault(kind, []).append(bbox)

    painted = 0
    for kind, kind_boxes in by_kind.items():
        for bbox in merge_adjacent(kind_boxes):
            x0, y0, x1, y1 = _pad(bbox, cfg.padding_px)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(out.width, x1), min(out.height, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            draw.rectangle([x0, y0, x1, y1], fill=fill)
            label = replacement_text(kind, "", cfg)
            if label:
                # Pillow's default bitmap font is small but always present;
                # loading a TTF would add a dependency and a path assumption.
                try:
                    draw.text((x0 + 2, y0 + 1), label, fill=text_fill)
                except Exception:  # noqa: BLE001 — a missing glyph must not fail the page
                    pass
            painted += 1
    return out, painted


def _rebuild_scanned_page(dst_doc, src_page, plan: PageRedactions, cfg: RedactionConfig) -> int:
    """Rasterise, paint, and emit a brand-new page holding only that image.

    The source page is not copied. That is the point: a scanned page's original
    image must not reach the output in any form, referenced or otherwise.
    """
    zoom = cfg.raster_dpi / 72.0
    pix = src_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    # One conversion, from the boxes' own unit into output pixels.
    factor = cfg.raster_dpi / (plan.source_dpi or 72.0)
    scaled = [((b[0] * factor, b[1] * factor, b[2] * factor, b[3] * factor), k)
              for b, k in plan.boxes]
    redacted, painted = paint_boxes_on_image(image, scaled, cfg)

    buf = io.BytesIO()
    redacted.save(buf, format="PNG", optimize=True)
    rect = src_page.rect
    new_page = dst_doc.new_page(width=rect.width, height=rect.height)
    new_page.insert_image(fitz.Rect(0, 0, rect.width, rect.height), stream=buf.getvalue())
    return painted


def build_redacted_pdf(
    src_path, plans: dict[int, PageRedactions], cfg: RedactionConfig,
) -> tuple[bytes, int]:
    """Assemble the anonymised PDF. Returns (pdf_bytes, redactions_applied).

    Built page by page into a new document so that nothing from a scanned
    page's original content survives, and saved with full garbage collection
    so unreferenced objects are dropped rather than left in the file.
    """
    src = fitz.open(src_path)
    dst = fitz.open()
    applied = 0
    try:
        for index in range(src.page_count):
            plan = plans.get(index)
            src_page = src[index]
            if plan is not None and plan.kind == "ocr":
                applied += _rebuild_scanned_page(dst, src_page, plan, cfg)
                continue
            dst.insert_pdf(src, from_page=index, to_page=index)
            if plan is not None and plan.boxes:
                applied += _redact_text_page(dst[dst.page_count - 1], plan, cfg)
        out = dst.tobytes(garbage=4, deflate=True, clean=True)
    finally:
        dst.close()
        src.close()
    return out, applied


# --------------------------------------------------------------------------
# Standalone images
# --------------------------------------------------------------------------
def load_image(path) -> Image.Image:
    image = Image.open(path)
    image.load()
    return image


def image_to_array(image: Image.Image) -> "np.ndarray":
    return np.array(image.convert("RGB"))
