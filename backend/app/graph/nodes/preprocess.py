"""C. Document Preprocessing Agent — deterministic, no LLM.

Turns a file into a page-indexed text corpus with honest per-page metadata.

The governing rule is the business rule "never silently skip a page". Every
page in the document gets a PageRecord, including the ones that yielded
nothing — a page with no text is recorded as EMPTY or UNREADABLE, not omitted.
Downstream, completeness checking counts pages against `total_pages`, so an
omitted page would look like a page that never existed rather than one we
failed on.

Page classification is per page, never per document. A twenty-page PDF is
routinely native text for two pages, scanned for the next two, then native
again; a document-level "does this have a text layer" check calls the whole
file born-digital and silently mis-reads a third of it.
"""
from __future__ import annotations

import logging

from app.graph.config import get_graph_settings
from app.graph.errors import ErrorType
from app.graph.nodes.helpers import audit, clean_text, err, normalise_for_match
from app.graph.schemas import PageRecord, PageStatus
from app.graph.state import GraphState, WorkflowStatus

log = logging.getLogger(__name__)

NODE = "preprocessing"

# Below this many characters a page's own text layer is treated as absent and
# OCR is attempted. Matched to the legacy pipeline's threshold so a page's
# classification does not depend on which entry point read it.
_MIN_CHARS = 20

# A page whose text is mostly punctuation and stray marks is OCR noise, not
# text, however many characters it has.
_MIN_ALNUM_RATIO = 0.35


def _quality_ok(text: str, min_chars: int) -> bool:
    s = clean_text(text)
    if len(s) < min_chars:
        return False
    alnum = sum(1 for c in s if c.isalnum())
    return (alnum / max(len(s), 1)) >= _MIN_ALNUM_RATIO


def _layout_text(page) -> str:
    """Rebuild a page's reading order from word geometry.

    PyMuPDF's default `get_text()` follows the PDF's internal content order,
    which on a form-style layout emits every LABEL as one block and every
    VALUE somewhere else. MEASURED on a real cytology report: page 1 came out
    as "Requesting Physician / Patient Name / Referring Physician /
    Accession No. / ... / Collected / Received / Reported", followed later by a
    bare run of three dates. Nothing in that text says which date is which, so
    the model guessed — differently on each page — and the workflow reported
    four "conflicts" the document does not actually contain. It also read the
    Chart No. as the accession number.

    `get_text(sort=True)` does not help: it reorders blocks, and the labels are
    a block of their own.

    Grouping words into visual lines by y-position, then ordering by x, puts
    "Accession No." back beside "UH0000000". Wide horizontal gaps become a run
    of spaces so column structure survives as something a reader — and a
    model — can still see.
    """
    words = page.get_text("words")   # (x0, y0, x1, y1, word, block, line, word_no)
    if not words:
        return ""

    heights = [w[3] - w[1] for w in words if w[3] > w[1]]
    # Tolerance from the median glyph height: a fixed pixel value is wrong for
    # both a dense report and a large-print certificate.
    line_tol = (sorted(heights)[len(heights) // 2] * 0.6) if heights else 3.0

    rows: list[tuple[float, list]] = []
    for w in sorted(words, key=lambda w: (w[1], w[0])):
        for row in rows:
            if abs(row[0] - w[1]) <= line_tol:
                row[1].append(w)
                break
        else:
            rows.append((w[1], [w]))

    lines = []
    for _, row in rows:
        row.sort(key=lambda w: w[0])
        widths = [(w[2] - w[0]) / max(len(w[4]), 1) for w in row if w[4]]
        char_w = (sorted(widths)[len(widths) // 2] if widths else 4.0) or 4.0
        parts, prev_x1 = [], None
        for w in row:
            if prev_x1 is not None:
                parts.append("   " if (w[0] - prev_x1) > char_w * 3.2 else " ")
            parts.append(w[4])
            prev_x1 = w[2]
        lines.append("".join(parts).rstrip())
    return "\n".join(lines)


def _page_native_text(page) -> str:
    """Layout-aware text, falling back to the raw order if it loses content.

    The rebuild is almost always better, but a page whose words carry no usable
    geometry would come back short — and losing text to gain ordering is a bad
    trade, so raw extraction wins whenever it is materially longer.
    """
    raw = clean_text(page.get_text() or "")
    try:
        rebuilt = clean_text(_layout_text(page))
    except Exception:  # noqa: BLE001 — a layout heuristic must never lose a page
        return raw
    if len(rebuilt) < len(raw) * 0.92:
        return raw
    return rebuilt or raw


def _image_area_ratio(page) -> float:
    """How much of the page is raster image, 0..1.

    Overlapping images are counted twice, so this over-estimates. That is the
    right direction to be wrong in: the number only decides whether a second
    read is worth attempting, and reading a page needlessly costs a little
    time while skipping one loses the letterhead entirely.
    """
    try:
        infos = page.get_image_info()
    except Exception:  # noqa: BLE001 — older PyMuPDF, or a malformed page
        return 0.0
    page_area = page.rect.width * page.rect.height
    if page_area <= 0:
        return 0.0
    covered = 0.0
    for info in infos:
        bbox = info.get("bbox")
        if not bbox:
            continue
        x0, y0, x1, y1 = bbox
        covered += max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return min(covered / page_area, 1.0)


def _merge_ocr_lines(native: str, ocr_text: str) -> str:
    """Append the OCR lines the text layer does not already have.

    The text layer stays authoritative: it has real reading order and no
    recognition errors, and OCR of the same content would only introduce
    disagreement about the same fact. What OCR is here for is the part of the
    page that has NO text layer — a letterhead rendered as a picture. So every
    OCR line that already appears in the native text is dropped, and only what
    is genuinely new is appended.

    Short lines are dropped too. A one- or two-character OCR line is almost
    always a mark on the scan rather than a word, and once appended it becomes
    something the extractor has to explain.
    """
    if not ocr_text:
        return native
    native_norm = normalise_for_match(native)
    additions: list[str] = []
    seen: set[str] = set()
    for line in ocr_text.splitlines():
        candidate = clean_text(line)
        norm = normalise_for_match(candidate)
        if len(norm) < 4 or norm in native_norm or norm in seen:
            continue
        seen.add(norm)
        additions.append(candidate)
    if not additions:
        return native
    # Labelled, so a reader of the page text — and anyone auditing an evidence
    # quote that came from here — can tell which half of the page it is from.
    return native + "\n[image region]\n" + "\n".join(additions)


def _ocr_page(pdf_bytes: bytes, page_number: int, dpi: int) -> tuple[str, float | None, str]:
    """Returns (text, confidence, error). Never raises."""
    try:
        from app.documents import local_ocr, pdf_utils

        # rasterize_page indexes the document directly (doc[n]), so it is
        # ZERO-based, while every page number in this package is one-based
        # because that is what a reader and an evidence citation mean by
        # "page 2". Passing the one-based number here rasterised the NEXT
        # page for every page, and threw on the last one.
        image = pdf_utils.rasterize_page(pdf_bytes, page_number - 1, dpi=dpi)
        result = local_ocr.extract_english(image)
        # Line-per-line, not the joined blob. `.text` comes back as one long
        # string, which is fine for a page being read wholly by OCR and wrong
        # for one being merged into an existing text layer: with no line
        # boundaries the whole page reads as a single unseen "line" and gets
        # appended, duplicating the results table the text layer already had.
        lines = [clean_text(ln) for ln in (getattr(result, "lines", None) or [])]
        text = "\n".join(ln for ln in lines if ln) or clean_text(getattr(result, "text", "") or "")
        return text, getattr(result, "confidence", None), ""
    except Exception as exc:  # noqa: BLE001 — OCR failure is a recorded page state
        return "", None, str(exc)[:300]


def _detect_table(page) -> bool:
    """PyMuPDF's own table finder where available, ruled-line heuristic where not."""
    try:
        finder = page.find_tables()
        if getattr(finder, "tables", None):
            return True
    except Exception:  # noqa: BLE001 — older PyMuPDF, fall through to the heuristic
        pass
    try:
        drawings = page.get_drawings()
        horizontals = sum(
            1 for d in drawings for item in d.get("items", [])
            if item[0] == "l" and abs(item[1].y - item[2].y) < 1.5
        )
        return horizontals >= 4
    except Exception:  # noqa: BLE001
        return False


def preprocess_document(state: GraphState) -> dict:
    settings = get_graph_settings()
    path = state.file_path
    page_text: dict[int, str] = {}
    page_meta: dict[int, PageRecord] = {}
    processed: list[int] = []
    unreadable: list[int] = []
    ocr_pages: list[int] = []
    entries = []
    errors = []

    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        return {
            "current_node": NODE,
            "current_status": WorkflowStatus.FAILED,
            "error_details": f"could not read file: {exc}",
            "errors": err(ErrorType.INVALID_FILE, str(exc), NODE),
            "audit_log": audit(NODE, "read_failed", str(exc), level="error"),
        }

    # ---------------------------------------------------------------- PDF
    if state.file_type == "pdf":
        import fitz

        with fitz.open(path) as doc:
            total = doc.page_count
            for index in range(total):
                page_no = index + 1
                text, status, ocr_conf, note = "", PageStatus.EMPTY, None, ""
                has_table = False
                width = height = None
                try:
                    page = doc[index]
                    width, height = page.rect.width, page.rect.height
                    native = _page_native_text(page)
                    has_table = _detect_table(page)
                except Exception as exc:  # noqa: BLE001
                    native = ""
                    note = f"page read error: {exc}"[:300]
                    errors += err(
                        ErrorType.PAGE_EXTRACTION_FAILED, note, NODE, page_number=page_no,
                        recovery_action="attempted OCR",
                    )

                if _quality_ok(native, _MIN_CHARS):
                    text, status = native, PageStatus.NATIVE_TEXT
                    # A good text layer can still be missing whole regions of
                    # the page: a letterhead, a stamp, a scanned annexure
                    # pasted in as a picture. Those carry the lab's name and
                    # address on most reports, so a page that is partly image
                    # gets read twice and the halves merged.
                    if (
                        settings.ocr_enabled
                        and settings.ocr_augments_text_layer
                        and _image_area_ratio(page) >= settings.ocr_image_area_threshold
                    ):
                        img_text, img_conf, img_error = _ocr_page(data, page_no, settings.ocr_dpi)
                        merged = _merge_ocr_lines(text, img_text)
                        if merged != text:
                            text = merged
                            ocr_conf = img_conf
                            note = "text layer augmented with OCR of image regions"
                            entries += audit(NODE, "ocr_augmented_text_layer", note,
                                             page_number=page_no)
                        elif img_error:
                            # Not a page failure: the text layer is fine and
                            # this was an attempt at extra, so it is recorded
                            # and the page carries on as NATIVE_TEXT.
                            entries += audit(NODE, "image_ocr_failed", img_error[:200],
                                             page_number=page_no, level="warning")
                elif settings.ocr_enabled:
                    ocr_text, ocr_conf, ocr_error = _ocr_page(data, page_no, settings.ocr_dpi)
                    if ocr_text and _quality_ok(ocr_text, _MIN_CHARS):
                        text, status = ocr_text, PageStatus.OCR
                        ocr_pages.append(page_no)
                        if ocr_conf is not None and ocr_conf < settings.ocr_low_confidence_threshold:
                            note = f"low OCR confidence {ocr_conf:.2f}"
                            entries += audit(NODE, "low_ocr_confidence", note, page_number=page_no,
                                             level="warning")
                    elif ocr_error:
                        status = PageStatus.OCR_FAILED
                        note = f"OCR failed: {ocr_error}"
                        errors += err(ErrorType.OCR_FAILURE, note, NODE, page_number=page_no,
                                      recovery_action="page marked unreadable")
                    else:
                        # OCR ran and genuinely found nothing. A blank page is a
                        # legitimate document feature, not a failure — but it is
                        # recorded so completeness can tell it apart from a gap.
                        status = PageStatus.EMPTY
                        note = note or "no text found by text layer or OCR"
                else:
                    status = PageStatus.UNREADABLE
                    note = "no usable text layer and OCR is disabled"

                if status in (PageStatus.OCR_FAILED, PageStatus.UNREADABLE):
                    unreadable.append(page_no)
                else:
                    processed.append(page_no)

                page_text[page_no] = text
                page_meta[page_no] = PageRecord(
                    page_number=page_no, text=text, char_count=len(text), status=status,
                    ocr_applied=(status == PageStatus.OCR), ocr_confidence=ocr_conf,
                    has_table=has_table, width=width, height=height, notes=note,
                )

    # --------------------------------------------------------------- DOCX
    elif state.file_type == "docx":
        from app.documents.docx_utils import extract_text as docx_text

        try:
            body = clean_text(docx_text(data))
        except Exception as exc:  # noqa: BLE001
            body = ""
            errors += err(ErrorType.PAGE_EXTRACTION_FAILED, str(exc), NODE, page_number=1)

        # A DOCX has no page boundaries. Paginate at a fixed character count so
        # every downstream page-level guarantee (evidence cites a page,
        # completeness counts pages) still holds rather than being special-cased.
        per_page = 3000
        pieces = [body[i:i + per_page] for i in range(0, len(body), per_page)] or [""]
        for i, piece in enumerate(pieces, start=1):
            ok = _quality_ok(piece, _MIN_CHARS)
            page_text[i] = piece
            page_meta[i] = PageRecord(
                page_number=i, text=piece, char_count=len(piece),
                status=PageStatus.NATIVE_TEXT if ok else PageStatus.EMPTY,
                notes="" if ok else "synthetic page has little text",
            )
            (processed if ok or piece == "" else unreadable).append(i)
        entries += audit(NODE, "docx_paginated",
                         f"{len(pieces)} synthetic page(s) at {per_page} chars each")

    # -------------------------------------------------------------- image
    else:
        ocr_text, ocr_conf, ocr_error = "", None, ""
        if settings.ocr_enabled:
            try:
                from app.documents import local_ocr

                result = local_ocr.extract_english(data)
                ocr_text = clean_text(getattr(result, "text", "") or "")
                ocr_conf = getattr(result, "confidence", None)
            except Exception as exc:  # noqa: BLE001
                ocr_error = str(exc)[:300]

        if ocr_text:
            status = PageStatus.OCR
            ocr_pages.append(1)
            processed.append(1)
        elif ocr_error:
            status = PageStatus.OCR_FAILED
            unreadable.append(1)
            errors += err(ErrorType.OCR_FAILURE, ocr_error, NODE, page_number=1)
        else:
            status = PageStatus.EMPTY
            processed.append(1)

        page_text[1] = ocr_text
        page_meta[1] = PageRecord(
            page_number=1, text=ocr_text, char_count=len(ocr_text), status=status,
            ocr_applied=bool(ocr_text), ocr_confidence=ocr_conf,
            notes=ocr_error or "",
        )

    total_pages = len(page_meta)
    readable = [p for p, m in page_meta.items() if m.is_readable]

    entries += audit(
        NODE, "preprocessed",
        f"{total_pages} page(s): {len(readable)} readable, {len(ocr_pages)} via OCR, "
        f"{len(unreadable)} unreadable",
    )

    # Every page empty is not "an empty document" in the file sense — the file
    # opened fine — but it is nothing to extract from, and saying so here is
    # far clearer than letting extraction return zero fields for a mystery reason.
    if not readable:
        errors += err(
            ErrorType.EMPTY_DOCUMENT,
            "No page yielded usable text (text layer and OCR both produced nothing).",
            NODE, recovery_action="workflow will report REJECTED",
        )
        entries += audit(NODE, "no_readable_pages", "document has no extractable text", level="error")

    return {
        "current_node": NODE,
        "total_pages": total_pages,
        "page_text": page_text,
        "page_metadata": page_meta,
        "processed_pages": sorted(processed),
        "unreadable_pages": sorted(unreadable),
        "ocr_pages": sorted(ocr_pages),
        "audit_log": entries,
        "errors": errors,
    }
