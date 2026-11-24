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

import io
import logging
import os

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

# Mid-grey. Chart bands and table shading fall to white, printed text to
# black; a scan faint enough to lose text at this level loses it to the
# detector anyway, and the un-thresholded pass is still read alongside.
_BINARY_THRESHOLD = 128


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


def _image_regions(page, min_fraction: float = 0.005, pad: float = 4.0, limit: int = 8) -> list:
    """Bounding boxes of the page's raster images, overlapping ones merged.

    Tiny images (below min_fraction of the page) are dropped: a bullet or a
    rule rendered as a picture carries no text worth a recogniser pass.
    """
    import fitz

    try:
        infos = page.get_image_info()
    except Exception:  # noqa: BLE001
        return []
    area = page.rect.width * page.rect.height or 1.0
    rects = []
    for info in infos:
        bbox = info.get("bbox")
        if not bbox:
            continue
        r = fitz.Rect(bbox) + (-pad, -pad, pad, pad)
        r &= page.rect
        if r.is_empty or (r.width * r.height) / area < min_fraction:
            continue
        rects.append(r)
    merged: list = []
    for r in sorted(rects, key=lambda x: (x.y0, x.x0)):
        for i, m in enumerate(merged):
            if m.intersects(r):
                merged[i] = fitz.Rect(m) | r     # store the union back
                break
        else:
            merged.append(fitz.Rect(r))
    return merged[:limit]


def _ocr_image_regions(page, settings) -> tuple[str, float | None, str]:
    """OCR only the picture parts of a digital page (letterhead, logo, stamp).

    The text layer already holds everything else, so reading the whole page
    again is wasted work. MEASURED 2026-11-24 on the 54-report dataset:
    whole-page augmentation ran the old RapidOCR route at ~10 s/page on any
    page whose images covered >= 3% (a logo is enough) — 50 of
    hp-lab-report.pdf's 60 s, and the reason digital PDFs (median 22.7 s) were
    slower than scans (9.8 s). Cropping to the image regions keeps the
    letterhead text this feature exists for. Returns (text, confidence, error).
    """
    regions = _image_regions(page)
    if not regions:
        return "", None, ""
    lines: list[str] = []
    confs: list[float] = []
    try:
        for rect in regions:
            png = page.get_pixmap(clip=rect, dpi=settings.ocr_dpi).tobytes("png")
            if settings.ocr_engine == "deepreef_local":
                from app.documents import deepreef_local

                result = deepreef_local.extract_page(png, rendered_pdf_page=True)
                got = result.lines
            else:
                from app.documents import local_ocr

                result = local_ocr.extract_english(png)
                got = getattr(result, "lines", None) or []
            lines += [clean_text(ln) for ln in got if clean_text(ln)]
            if getattr(result, "confidence", None) is not None:
                confs.append(result.confidence)
    except Exception as exc:  # noqa: BLE001 — augmentation is optional extra text
        return "", None, str(exc)[:300]
    return "\n".join(lines), (sum(confs) / len(confs) if confs else None), ""


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


def _binarised(image_bytes: bytes) -> bytes | None:
    """Black text on white, with the colour thrown away.

    A trend chart's points sit on coloured bands, and colour is noise to a text
    detector: MEASURED on `001_Lab-report.png`, reading the page as-is finds
    three of the five plotted values and reading a threshold of it finds all
    five, plus the top axis tick and the two dates OCR otherwise fuses.
    """
    try:
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(image_bytes)) as im:
            grey = ImageOps.autocontrast(im.convert("L"))
            bw = grey.point(lambda v: 0 if v < _BINARY_THRESHOLD else 255, mode="L").convert("RGB")
            out = io.BytesIO()
            bw.save(out, format="PNG")
            return out.getvalue()
    except Exception:  # noqa: BLE001 — a second rendering is a bonus, never required
        return None


def _with_chart(text: str, ocr_result, image_bytes: bytes | None = None) -> str:
    """Append a chart's recovered structure to a page's OCR text.

    OCR flattens a plot into loose tokens, and the extractor then reads the
    chart's TITLE next to the first number it finds. MEASURED on a scanned
    HbA1c trend chart: it emitted "GLYCOSYLATED HEMOGLOBIN (HBA1C) = 7.52 %",
    where 7.52 is the second tick on the y-axis and the patient's readings are
    5.6, 5.4, 5.5, 4.8 and 5.0.

    The page is read a SECOND time in black and white when a chart is present,
    because the coloured bands behind the points defeat the text detector. The
    extra pass costs one OCR call and only happens on pages that have a chart.

    Never fatal: a chart is a bonus on top of the text, so a failure here
    leaves the page exactly as OCR read it.
    """
    try:
        from app.documents import local_ocr
        from app.graph.charts import annotate, best_chart, detect_chart

        lines = getattr(ocr_result, "lines", None) or []
        boxes = getattr(ocr_result, "boxes", None) or []
        if not lines or len(boxes) != len(lines):
            return text

        if not detect_chart(lines, boxes).is_chart():
            return text        # no chart, so no second pass to pay for

        renderings = [(lines, boxes)]
        bw = _binarised(image_bytes) if image_bytes else None
        if bw:
            second = local_ocr.extract_english(bw)
            renderings.append((second.lines or [], second.boxes or []))

        return annotate(text, best_chart(*renderings))
    except Exception:  # noqa: BLE001 — chart detection must never lose a page
        log.debug("chart detection failed", exc_info=True)
        return text


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
        return _with_chart(text, result, image), getattr(result, "confidence", None), ""
    except Exception as exc:  # noqa: BLE001 — OCR failure is a recorded page state
        return "", None, str(exc)[:300]


def _ocr_scanned_page(
    pdf_bytes: bytes, page_number: int, dpi: int, script: str
) -> tuple[str, float | None, str, str]:
    """Read a scanned page with the engine that has a model for its script.

    Returns (text, confidence, error, engine).

    deepreef-ocr ("Aakash") and RapidOCR are NOT a primary and a spare — they
    are different models covering different alphabets. deepreef-ocr has
    Devanagari, Arabic, Tamil, Telugu and Kannada; RapidOCR has Latin, and
    raises for anything else. So the script chooses the engine, exactly as
    documents/pipeline.py has always done.

    RapidOCR is still tried when deepreef-ocr fails, because a Lambda can be
    unreachable, unauthorised or timing out for reasons that have nothing to
    do with the page — and a Latin-alphabet reading of a Devanagari page is
    poor, but it beats an unreadable page.
    """
    from app.documents.ocr_client import SUPPORTED_SCRIPTS

    if script in SUPPORTED_SCRIPTS:
        try:
            from app.documents import pdf_utils
            from app.documents.ocr_client import OcrClient

            image = pdf_utils.rasterize_page(pdf_bytes, page_number - 1, dpi=dpi)
            result = OcrClient().extract(image, script)
            text = clean_text(getattr(result, "text", "") or "")
            if text:
                return text, getattr(result, "confidence", None), "", f"deepreef:{script}"
            note = "deepreef-ocr returned no text"
        except Exception as exc:  # noqa: BLE001 — a remote OCR failure is a fallback
            note = f"deepreef-ocr failed: {str(exc)[:200]}"
        log.info("ocr: deepreef fell back to rapidocr page=%d script=%s reason=%s",
                 page_number, script, note)
        text, conf, error = _ocr_page(pdf_bytes, page_number, dpi)
        return text, conf, (error or note if not text else ""), "rapidocr(fallback)"

    if get_graph_settings().ocr_engine == "deepreef_local":
        try:
            from app.documents import deepreef_local, ocr_page_cache

            image = deepreef_local.render_for_ocr(pdf_bytes, page_number, dpi)
            # Identical pixels + model + settings => identical reading, so a
            # re-uploaded report skips OCR (~4.3 s/page, MEASURED 2026-11-24).
            # The cache never raises: any problem is a miss and OCR runs.
            key = ocr_page_cache.page_key(image)
            result = ocr_page_cache.get(key) if key else None
            source = " (cache)" if result is not None else ""
            if result is None:
                result = deepreef_local.extract_page(image, rendered_pdf_page=True)
                if key and result.lines:
                    ocr_page_cache.put(key, result)
            text = "\n".join(clean_text(ln) for ln in result.lines if clean_text(ln))
            if text:
                return text, result.confidence, "", f"deepreef-local:en [{result.model_used}]{source}"
            note = "deepreef-local returned no text"
        except Exception as exc:  # noqa: BLE001 — fall back to RapidOCR, recorded
            note = f"deepreef-local failed: {str(exc)[:200]}"
        log.info("ocr: deepreef-local fell back to rapidocr page=%d reason=%s", page_number, note)
        text, conf, error = _ocr_page(pdf_bytes, page_number, dpi)
        return text, conf, (error or note if not text else ""), "rapidocr(fallback)"

    text, conf, error = _ocr_page(pdf_bytes, page_number, dpi)
    return text, conf, error, "rapidocr"


def _prefetch_scanned_pages(path: str, data: bytes, settings, script: str) -> dict:
    """OCR every scanned page up front, several at a time.

    Only for the deepreef_local engine, whose engines are one per thread with
    pinned ONNX threads; the RapidOCR path shares one engine and stays
    sequential. MEASURED 2026-11-23: sequential OCR was 125s of a 148s run on a
    30-page scan (~4.3-5.3s per page). Returns {page_no: _ocr_scanned_page(...)};
    any page missing from it is read inline by the normal loop.
    settings.ocr_worker_mode picks threads (default) or spawned processes.
    """
    if settings.ocr_engine in _REMOTE_OCR_ENGINES:
        return _prefetch_remote(path, data, settings, script)
    workers = max(1, int(settings.ocr_page_workers))
    if (settings.ocr_engine != "deepreef_local" or workers == 1 or not settings.ocr_enabled
            or (script or "english") not in ("english", "en", "latin")):
        return {}
    import fitz
    from concurrent.futures import ThreadPoolExecutor

    scanned = []
    with fitz.open(path) as doc:
        for index in range(doc.page_count):
            try:
                if not _quality_ok(_page_native_text(doc[index]), _MIN_CHARS):
                    scanned.append(index + 1)
            except Exception:  # noqa: BLE001 — the loop records page read errors itself
                continue
    if len(scanned) < 2:
        return {}
    mode = str(getattr(settings, "ocr_worker_mode", "thread") or "thread").lower()
    if mode != "process":
        with ThreadPoolExecutor(max_workers=min(workers, len(scanned)), thread_name_prefix="ocr") as pool:
            futures = {p: pool.submit(_ocr_scanned_page, data, p, settings.ocr_dpi, script) for p in scanned}
            return {p: f.result() for p, f in futures.items()}
    # Process mode: a dead worker (BrokenProcessPool, OOM kill, spawn/pickle
    # error) must never fail the document, so any pool failure returns {} and
    # the normal sequential loop OCRs every page inline.
    try:
        with _make_ocr_process_pool(min(workers, len(scanned))) as pool:
            futures = {p: pool.submit(_ocr_page_in_process, data, p, settings.ocr_dpi, script)
                       for p in scanned}
            return {p: f.result() for p, f in futures.items()}
    except Exception as exc:  # noqa: BLE001 — fall back to sequential OCR
        log.warning("ocr: process pool failed, falling back to sequential OCR: %s", str(exc)[:200])
        return {}


#: Engines that read pages somewhere else, all pages at once (E1, E2):
#:   deepreef_service — the team's engine in the OCR worker service
#:                      (ocr_service/app.py) on a stronger host, OCR_SERVICE_URL
#:   deepreef_lambda  — the deepreef-ocr Lambda, one invoke per page in parallel
#:                      (needs lambda:InvokeFunction; images leave this host)
_REMOTE_OCR_ENGINES = ("deepreef_service", "deepreef_lambda")


def _scanned_page_numbers(path: str) -> list[int]:
    import fitz

    scanned = []
    with fitz.open(path) as doc:
        for index in range(doc.page_count):
            try:
                if not _quality_ok(_page_native_text(doc[index]), _MIN_CHARS):
                    scanned.append(index + 1)
            except Exception:  # noqa: BLE001 — the page loop records read errors
                continue
    return scanned


def _prefetch_remote(path: str, data: bytes, settings, script: str) -> dict:
    """Read every scanned page with a remote engine in one parallel batch.

    Only successful pages are returned. A page that failed remotely (service
    down, AccessDenied, timeout) is simply absent, so the page loop reads it
    inline — _ocr_scanned_page sends any non-deepreef_local engine to RapidOCR
    — and a remote outage costs speed, never a page.
    """
    if not settings.ocr_enabled or (script or "english") not in ("english", "en", "latin"):
        return {}
    scanned = _scanned_page_numbers(path)
    if not scanned:
        return {}
    from app.documents import deepreef_local

    images = {p: deepreef_local.render_for_ocr(data, p, settings.ocr_dpi) for p in scanned}
    out: dict = {}
    try:
        if settings.ocr_engine == "deepreef_service":
            from app.documents.ocr_service_client import extract_pages_remote

            results = extract_pages_remote(images, base_url=settings.ocr_service_url,
                                           workers=max(1, settings.ocr_page_workers),
                                           rendered_pdf_page=True)
            for p, r in results.items():
                if isinstance(r, Exception):
                    log.info("ocr: service failed page=%d: %s", p, str(r)[:160])
                    continue
                text = "\n".join(clean_text(ln) for ln in r.lines if clean_text(ln))
                if text:
                    out[p] = (text, r.confidence, "", f"deepreef-service:en [{r.model_used}]")
        else:
            from app.config import get_settings as get_app_settings
            from app.documents.deepreef_lambda import extract_pages_lambda

            app_settings = get_app_settings()
            results = extract_pages_lambda(images, function_name=app_settings.ocr_lambda_function_name,
                                           region=app_settings.ocr_lambda_region,
                                           max_workers=max(1, settings.ocr_page_workers) * 4,
                                           timeout=max(30.0, app_settings.ocr_timeout_seconds))
            for p, (text, conf, error) in results.items():
                if error:
                    log.info("ocr: lambda failed page=%d: %s", p, error[:160])
                    continue
                text = clean_text(text) and "\n".join(clean_text(ln) for ln in text.splitlines() if clean_text(ln))
                if text:
                    out[p] = (text, conf, "", "deepreef-lambda:en")
    except Exception as exc:  # noqa: BLE001 — remote engine unavailable: read locally
        log.warning("ocr: %s unavailable, reading pages locally: %s", settings.ocr_engine, str(exc)[:200])
        return {}
    return out


def _ocr_page_in_process(pdf_bytes: bytes, page_no: int, dpi: int, script: str):
    """Process-pool worker: module-level so "spawn" can pickle it by name.
    Returns _ocr_scanned_page's picklable (text, conf, error, engine_label)."""
    return _ocr_scanned_page(pdf_bytes, page_no, dpi, script)


def _make_ocr_process_pool(max_workers: int):
    """ProcessPoolExecutor with the "spawn" start method: fork is unsafe on
    macOS (ObjC runtime) and with ONNX/BLAS threads already running. Each
    process loads its own OCR models (~200-300 MB). Separate so tests can
    replace it without spawning."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    return ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context("spawn"))


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


def _apply_docling(state, path, settings, page_text, page_meta, errors,
                   *, digital_pages=None, scanned_pages=None) -> list:
    """Enrich already-extracted pages with Docling's structure. Never raises.

    Additive by construction. `page_text` gains only table rows the text layer
    did not already contain, `has_table` becomes an answer from a table model
    rather than a ruled-line heuristic, and `layout` carries the blocks and
    grids for anything downstream that wants them. Every other field the
    RapidOCR + PyMuPDF pass produced is left untouched.

    Returns audit entries. Logs are structural only — counts, timings and the
    fallback reason — never document text, cell values or field content.
    """
    if not getattr(settings, "use_docling", False):
        return []

    # Docling reads a page's LAYOUT, not its pixels: do_ocr is off, so on a
    # scanned page it has no text to attach structure to and returns nothing
    # after a minute of model time. MEASURED on a 2-page report: 78s for a
    # structure pass. A document with no digital page is not worth that.
    digital = list(digital_pages or [])
    if digital_pages is not None and not digital:
        reason = "every page is scanned; Docling has no text layer to read"
        log.info("docling: skipped document=%s reason=%s", os.path.basename(path), reason)
        return audit(NODE, "docling_skipped", reason)

    from app.graph.docling_adapter import (
        merge_ocr_and_docling_output,
        parse_document_with_docling,
    )

    name = os.path.basename(path)
    log.info("docling: start document=%s pages=%d digital=%d scanned=%d",
             name, len(page_meta), len(digital), len(scanned_pages or []))

    result = parse_document_with_docling(
        path,
        timeout_seconds=float(getattr(settings, "docling_timeout_seconds", 120.0)),
        max_pages=int(getattr(settings, "docling_max_pages", 0) or 0),
        expected_pages=len(page_meta),
    )

    if not result.ok:
        log.info(
            "docling: fallback document=%s reason=%s seconds=%.2f",
            name, result.fallback_reason, result.seconds,
        )
        return audit(
            NODE, "docling_fallback",
            f"structure pass skipped: {result.fallback_reason}",
            level="warning",
        )

    merged_pages = 0
    allowed = set(digital) if digital_pages is not None else set(page_meta)
    for page_no, record in page_meta.items():
        if page_no not in allowed:
            continue        # a scanned page's text is the OCR engine's, not Docling's
        layout = result.pages.get(page_no)
        if layout is None:
            continue
        record.layout = layout
        if layout.tables:
            record.has_table = True
        before = page_text.get(page_no, "")
        after = merge_ocr_and_docling_output(before, layout)
        if after != before:
            page_text[page_no] = after
            record.text = after
            record.char_count = len(after)
            merged_pages += 1

    log.info(
        "docling: done document=%s seconds=%.2f blocks=%d tables=%d rows=%d cells=%d "
        "pages_enriched=%d",
        name, result.seconds, result.block_count, result.table_count,
        result.row_count, result.cell_count, merged_pages,
    )
    return audit(
        NODE, "docling_applied",
        f"{result.block_count} block(s), {result.table_count} table(s), "
        f"{result.row_count} row(s) in {result.seconds:.2f}s; "
        f"{merged_pages} page(s) gained table text",
    )


def preprocess_document(state: GraphState) -> dict:
    settings = get_graph_settings()
    path = state.file_path
    page_text: dict[int, str] = {}
    page_meta: dict[int, PageRecord] = {}
    processed: list[int] = []
    unreadable: list[int] = []
    ocr_pages: list[int] = []
    # PyMuPDF's verdict per page, and the fork the whole node turns on: a page
    # with a usable text layer is DIGITAL and goes to Docling for structure; a
    # page without one is SCANNED and goes to a script-routed OCR engine.
    digital_pages: list[int] = []
    scanned_pages: list[int] = []
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

        prefetched = _prefetch_scanned_pages(path, data, settings, state.script or "english")
        if prefetched:
            entries += audit(NODE, "ocr_prefetched",
                             f"{len(prefetched)} scanned page(s) read in parallel "
                             f"({settings.ocr_page_workers} at a time, {settings.ocr_engine})")

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
                    # Table detection only where there is a text layer. On a
                    # scanned page PyMuPDF's finder reads vector rules, not
                    # text. MEASURED 2026-11-24 on a 30-page scan: 289 ms/page
                    # (8.4s per document) and "table" on all 29 scanned pages,
                    # notes-only pages included. Scanned pages get the flag
                    # from their OCR text further down instead.
                    if _quality_ok(native, _MIN_CHARS):
                        has_table = _detect_table(page)
                except Exception as exc:  # noqa: BLE001
                    native = ""
                    note = f"page read error: {exc}"[:300]
                    errors += err(
                        ErrorType.PAGE_EXTRACTION_FAILED, note, NODE, page_number=page_no,
                        recovery_action="attempted OCR",
                    )

                if _quality_ok(native, _MIN_CHARS):
                    # DIGITAL page: it has its own text layer, so no OCR
                    # engine reads it and Docling can recover its structure.
                    text, status = native, PageStatus.NATIVE_TEXT
                    digital_pages.append(page_no)
                    # A good text layer can still be missing whole regions of
                    # the page: a letterhead, a stamp, a scanned annexure
                    # pasted in as a picture. Those carry the lab's name and
                    # address on most reports, so a page that is partly image
                    # gets read twice and the halves merged.
                    # test_rows (e-Attestation) reads image regions on the first
                    # page only: a letterhead repeats on every page, and the
                    # attesting lab is identified by its login, not by the
                    # report — MEASURED 2026-11-24, image-region OCR is still
                    # ~2.6 s per page.
                    from app.graph.nodes.helpers import is_test_rows_profile

                    if (
                        settings.ocr_enabled
                        and settings.ocr_augments_text_layer
                        and not (is_test_rows_profile(state) and page_no > 1)
                        and _image_area_ratio(page) >= settings.ocr_image_area_threshold
                    ):
                        if settings.ocr_augment_regions_only:
                            img_text, img_conf, img_error = _ocr_image_regions(page, settings)
                        else:
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
                    # SCANNED page: no usable text layer, so an OCR engine
                    # reads it and Docling is skipped for it entirely —
                    # with do_ocr=False Docling has nothing to read here and
                    # would spend a minute per document returning nothing.
                    ocr_text, ocr_conf, ocr_error, engine = prefetched.get(page_no) or _ocr_scanned_page(
                        data, page_no, settings.ocr_dpi, state.script or "english")
                    scanned_pages.append(page_no)
                    if ocr_text and _quality_ok(ocr_text, _MIN_CHARS):
                        text, status = ocr_text, PageStatus.OCR
                        ocr_pages.append(page_no)
                        from app.graph.nodes.chunking import _last_table_header

                        has_table = _last_table_header(ocr_text) is not None
                        entries += audit(NODE, "page_ocr", f"read by {engine}",
                                         page_number=page_no)
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

        # ---- document structure (optional, additive) --------------------
        # Runs AFTER the text layer and RapidOCR, never instead of them: by
        # this point every page already has its text and its confidence, and
        # what Docling adds is the structure reading order destroyed — heading
        # levels, and which cells belong to which row. A failure here leaves
        # the pages exactly as they are.
        entries += _apply_docling(
            state, path, settings, page_text, page_meta, errors,
            digital_pages=digital_pages, scanned_pages=scanned_pages,
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
        # Image uploads (PNG/JPG photos and scans) read the same way as scanned
        # PDF pages: deepreef-ocr with row-grouped lines. MEASURED 2026-11-24 on
        # 006_Lab-report.png: the old path returned the whole page as ONE line
        # (RapidOCR `.text`), so the compact extractor's line citations could
        # not be verified and all 20 rows were refused (0 extracted, old
        # pipeline 33). A photo keeps the full clean-up (illumination,
        # contrast, sharpen): rendered_pdf_page=False.
        if settings.ocr_enabled and settings.ocr_engine == "deepreef_local" and \
                (state.script or "english") in ("english", "en", "latin"):
            try:
                from app.documents import deepreef_local

                page = deepreef_local.extract_page(data, rendered_pdf_page=False)
                ocr_text = "\n".join(clean_text(ln) for ln in page.lines if clean_text(ln))
                ocr_conf = page.confidence
                if ocr_text:
                    entries += audit(NODE, "page_ocr", f"read by deepreef-local:en [{page.model_used}]",
                                     page_number=1)
            except Exception as exc:  # noqa: BLE001 — fall back to RapidOCR below
                log.info("ocr: deepreef-local failed on image, using rapidocr: %s", str(exc)[:200])
        if settings.ocr_enabled and not ocr_text:
            try:
                from app.documents import local_ocr

                result = local_ocr.extract_english(data)
                # One OCR box per line, not the joined blob (see above).
                lines = [clean_text(ln) for ln in (getattr(result, "lines", None) or [])]
                text = "\n".join(ln for ln in lines if ln) or clean_text(getattr(result, "text", "") or "")
                ocr_text = _with_chart(text, result, data)
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
