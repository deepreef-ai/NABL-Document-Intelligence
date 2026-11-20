"""Turning a page into positioned words, from either source.

Detection and redaction both need the same thing: a list of words, each with a
box on the page. A born-digital PDF gives that directly; a scan gives OCR
lines, which have to be split back into words with estimated boxes.

Keeping both behind one `Word` type is what lets the detector be written once.
The `source` field survives onto every detection so the report can say whether
a finding came from exact text or from a transcription.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

log = logging.getLogger(__name__)

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class Word:
    text: str
    bbox: BBox           # x0, y0, x1, y1 — PDF points for a text layer, pixels for a raster
    page: int
    line: int            # words sharing a line share this id, within a page
    order: int           # position within the line
    source: str          # "text_layer" | "ocr"
    confidence: float = 1.0

    @property
    def cleaned(self) -> str:
        """Text without surrounding punctuation, for label matching."""
        return self.text.strip().strip(":;,.–—-|")


@dataclass
class PageWords:
    page_number: int          # 0-based
    words: list[Word]
    source: str               # "text_layer" | "ocr" | "empty"
    width: float = 0.0
    height: float = 0.0
    ocr_mean_confidence: float = 1.0

    @property
    def text(self) -> str:
        out, current = [], -1
        buf: list[str] = []
        for w in self.words:
            if w.line != current and buf:
                out.append(" ".join(buf))
                buf = []
            current = w.line
            buf.append(w.text)
        if buf:
            out.append(" ".join(buf))
        return "\n".join(out)


def words_from_pdf_page(page, page_number: int) -> PageWords:
    """Words and boxes straight from the PDF's own text layer.

    PyMuPDF's "words" mode returns one tuple per whitespace-delimited token
    with its block and line indices, which is exactly the grouping the
    label-and-value detectors need.
    """
    raw = page.get_text("words")  # x0, y0, x1, y1, word, block, line, word_no
    words: list[Word] = []
    for x0, y0, x1, y1, text, block, line_no, word_no in sorted(
        raw, key=lambda t: (t[5], t[6], t[7])
    ):
        if not text.strip():
            continue
        words.append(Word(
            text=text, bbox=(float(x0), float(y0), float(x1), float(y1)),
            page=page_number, line=block * 1000 + line_no, order=word_no,
            source="text_layer", confidence=1.0,
        ))
    rect = page.rect
    return PageWords(page_number, words, "text_layer" if words else "empty",
                     rect.width, rect.height)


def split_line_into_words(
    text: str, bbox: BBox, page: int, line_id: int, confidence: float,
) -> list[Word]:
    """Split one OCR line into words, estimating each word's box.

    OCR engines return a box per LINE, but redaction needs to cover a single
    value inside that line — blacking out the whole line would take the label
    and often the neighbouring column with it. Character width is assumed
    uniform across the line, which is wrong for proportional fonts but close
    enough at the scale of a redaction box, and the configured padding absorbs
    the error.
    """
    x0, y0, x1, y1 = bbox
    stripped = text.rstrip()
    if not stripped.strip():
        return []
    total = len(stripped)
    span = (x1 - x0) / total if total else 0.0

    words: list[Word] = []
    order = 0
    for m in re.finditer(r"\S+", stripped):
        wx0 = x0 + span * m.start()
        wx1 = x0 + span * m.end()
        words.append(Word(
            text=m.group(), bbox=(wx0, y0, wx1, y1), page=page,
            line=line_id, order=order, source="ocr", confidence=confidence,
        ))
        order += 1
    return words


def words_from_ocr(
    ocr_lines: Sequence[tuple[Sequence[Sequence[float]], str, float]],
    page_number: int,
    width: float,
    height: float,
) -> PageWords:
    """Convert an engine's (quad, text, score) triples into positioned words.

    Lines are sorted top-to-bottom then left-to-right before being numbered, so
    a detector scanning "the rest of this line" sees reading order rather than
    whatever order the engine happened to emit.
    """
    entries = []
    for quad, text, score in ocr_lines:
        if not text or not text.strip():
            continue
        xs = [float(p[0]) for p in quad]
        ys = [float(p[1]) for p in quad]
        entries.append(((min(xs), min(ys), max(xs), max(ys)), text, float(score)))

    # Group into visual rows: a line belongs to the current row while its
    # vertical centre stays within half a line-height of the row's centre.
    entries.sort(key=lambda e: (e[0][1], e[0][0]))
    words: list[Word] = []
    confidences: list[float] = []
    line_id = 0
    row_centre = None
    row_tolerance = 0.0

    for bbox, text, score in entries:
        centre = (bbox[1] + bbox[3]) / 2
        height_of_line = max(bbox[3] - bbox[1], 1.0)
        if row_centre is None or abs(centre - row_centre) > row_tolerance:
            line_id += 1
            row_centre = centre
            row_tolerance = height_of_line * 0.6
        words.extend(split_line_into_words(text, bbox, page_number, line_id, score))
        confidences.append(score)

    mean_conf = sum(confidences) / len(confidences) if confidences else 1.0
    return PageWords(page_number, words, "ocr" if words else "empty",
                     width, height, mean_conf)


def group_by_line(words: Iterable[Word]) -> list[list[Word]]:
    """Words bucketed into lines, each line in reading order."""
    lines: dict[int, list[Word]] = {}
    for w in words:
        lines.setdefault(w.line, []).append(w)
    ordered = []
    for key in sorted(lines):
        ordered.append(sorted(lines[key], key=lambda w: (w.order, w.bbox[0])))
    return ordered
