"""Docling: document structure, and nothing else.

Why this module exists
----------------------
`preprocess_document` turns a file into text, and everything downstream reads
only `page_text` and `page_metadata`. That is enough to extract scalar fields
and, with effort, to guess at tables — but the structure of the page is gone by
the time an extractor sees it. `has_table` is a BOOLEAN from PyMuPDF's
`find_tables()`; no row, no cell, no heading level and no reading order exists
anywhere in the system. Results tables are flattened to text and the model
rebuilds the rows from prose.

Docling recovers that structure. This module is the ONLY place that knows it
exists: it takes a path, returns the project's own normalized types, and no
`DoclingDocument` ever escapes. Deleting this file and its one call site
removes the feature completely.

What it is NOT allowed to do
----------------------------
**Docling's internal OCR stays off.** `do_ocr=False`, asserted in
`_pipeline_options` and pinned by a test. RapidOCR is the project's OCR engine
and running a second one over the same page would produce two readings of the
same text, differing slightly, with nothing to say which is right. Docling is
here for layout; the words are RapidOCR's.

It also never replaces text. `merge_ocr_and_docling_output` treats the OCR/
text-layer content as authoritative and adds only what is genuinely absent —
table rows the flattened text does not already contain. Anything Docling reads
that OCR already read is dropped rather than appended, because a duplicated
value is worse than a missing one: it invents a second reading of one fact.

Failure is normal, not exceptional
----------------------------------
Docling is an optional dependency (45 packages, including torch), it is slow
enough to need a timeout, and it can return a document that parses but makes no
sense. Every one of those is a FALLBACK, not an error: the caller keeps the
RapidOCR + PyMuPDF result it already had, the reason is recorded, and the
document still processes. A structure pass that cannot run must never cost a
reader their extraction.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

log = logging.getLogger(__name__)

#: Marks the block appended to a page's text. The extraction prompt keys off
#: this, and `merge_ocr_and_docling_output` uses it to avoid appending twice.
LAYOUT_MARKER = "[document structure]"


# --------------------------------------------------------------------------
# the project's own normalized types — no Docling object crosses this boundary
# --------------------------------------------------------------------------


@dataclass
class LayoutBlock:
    """One structural element: a heading, a paragraph, a list, a caption."""

    kind: str = "text"          # heading | text | list | caption | table | other
    text: str = ""
    level: int | None = None    # heading depth, 1 = outermost; None when not a heading
    page_number: int = 0
    bbox: tuple[float, float, float, float] | None = None
    provenance: str = "docling"


@dataclass
class LayoutTable:
    """One table, as a grid.

    `rows` is row-major and rectangular — short rows are padded — because a
    ragged grid cannot be rendered or compared. `merged_cells` records spans
    where Docling reports them, so a header cell covering three columns is not
    silently flattened into one.
    """

    rows: list[list[str]] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0
    merged_cells: list[dict[str, int]] = field(default_factory=list)
    page_number: int = 0
    bbox: tuple[float, float, float, float] | None = None
    provenance: str = "docling"

    @property
    def cell_count(self) -> int:
        return sum(len(r) for r in self.rows)

    def is_valid(self) -> bool:
        """A table needs at least two rows and two columns to BE a table.

        One row is a heading that happened to have gaps in it; one column is a
        list. Admitting either produces "tables" that carry no relationship
        between values, which is the only thing a table is for.
        """
        return self.n_rows >= 2 and self.n_cols >= 2 and any(any(c for c in r) for r in self.rows)


@dataclass
class PageLayout:
    """Everything Docling recovered for one page."""

    page_number: int
    blocks: list[LayoutBlock] = field(default_factory=list)
    tables: list[LayoutTable] = field(default_factory=list)

    @property
    def headings(self) -> list[LayoutBlock]:
        return [b for b in self.blocks if b.kind == "heading"]


@dataclass
class DoclingResult:
    """The outcome of a structure pass — including the ways it can not happen."""

    ok: bool = False
    pages: dict[int, PageLayout] = field(default_factory=dict)
    fallback_reason: str = ""
    seconds: float = 0.0

    @property
    def block_count(self) -> int:
        return sum(len(p.blocks) for p in self.pages.values())

    @property
    def table_count(self) -> int:
        return sum(len(p.tables) for p in self.pages.values())

    @property
    def row_count(self) -> int:
        return sum(t.n_rows for p in self.pages.values() for t in p.tables)

    @property
    def cell_count(self) -> int:
        return sum(t.cell_count for p in self.pages.values() for t in p.tables)


class _Converter(Protocol):
    """The one Docling call we make, narrow enough to substitute in tests."""

    def convert(self, source: str) -> Any: ...


# --------------------------------------------------------------------------
# availability
# --------------------------------------------------------------------------


def docling_available() -> bool:
    """Is Docling importable?

    Checked without importing it: `find_spec` answers the question at the cost
    of a path lookup, where a real import costs seconds and several hundred MB
    of torch. The disabled path must not pay for a feature it is not using.
    """
    try:
        from importlib.util import find_spec

        return find_spec("docling") is not None
    except (ImportError, ValueError):       # ValueError: package present but broken
        return False


def _pipeline_options() -> Any:
    """Docling configured for layout only.

    `do_ocr=False` is the load-bearing line. Docling ships its own OCR and
    would happily run it over every scanned page, producing a second reading of
    text RapidOCR has already read — two values for one fact, differing in the
    margins, with nothing to arbitrate. Table structure stays ON: it is the
    reason this integration exists.
    """
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    options = PdfPipelineOptions()
    options.do_ocr = False                  # RapidOCR is the only OCR engine
    options.do_table_structure = True
    if hasattr(options, "table_structure_options"):
        options.table_structure_options.do_cell_matching = True
    return options


def _build_converter() -> _Converter:
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=_pipeline_options())
        }
    )


# --------------------------------------------------------------------------
# parse
# --------------------------------------------------------------------------


def parse_document_with_docling(
    file_path: str,
    *,
    timeout_seconds: float = 120.0,
    max_pages: int = 0,
    expected_pages: int = 0,
    converter_factory: Callable[[], _Converter] | None = None,
) -> DoclingResult:
    """Run a structure pass over one PDF.

    Never raises. Every failure — missing dependency, timeout, crash, a
    document that comes back empty or with the wrong number of pages — returns
    `ok=False` and a `fallback_reason`, because the caller already holds a
    usable RapidOCR + PyMuPDF result and must keep it.

    `expected_pages` is PyMuPDF's page count. When Docling disagrees the whole
    result is discarded: a layout keyed to the wrong pages would attach a
    table to a page it is not on, and a confidently misplaced table is worse
    than no table.

    `converter_factory` exists so the adapter is testable without Docling
    installed — the tests inject a stub and exercise every path above.
    """
    started = time.monotonic()

    if not os.path.exists(file_path):
        return DoclingResult(fallback_reason=f"file not found: {os.path.basename(file_path)}")

    factory = converter_factory or (_build_converter if docling_available() else None)
    if factory is None:
        return DoclingResult(fallback_reason="docling is not installed")

    if max_pages and expected_pages and expected_pages > max_pages:
        return DoclingResult(
            fallback_reason=f"document has {expected_pages} pages, over the {max_pages}-page limit"
        )

    try:
        document = _convert_with_timeout(factory, file_path, timeout_seconds)
    except TimeoutError:
        elapsed = time.monotonic() - started
        return DoclingResult(
            fallback_reason=f"docling timed out after {timeout_seconds:.0f}s",
            seconds=round(elapsed, 2),
        )
    except ImportError as exc:
        return DoclingResult(fallback_reason=f"docling import failed: {exc}")
    except Exception as exc:  # noqa: BLE001 — any parser failure is a fallback
        log.warning("docling failed on %s: %s", os.path.basename(file_path), exc)
        return DoclingResult(
            fallback_reason=f"docling raised {type(exc).__name__}: {str(exc)[:200]}",
            seconds=round(time.monotonic() - started, 2),
        )

    result = normalize_docling_output(document, expected_pages=expected_pages)
    result.seconds = round(time.monotonic() - started, 2)
    return result


def _convert_with_timeout(factory: Callable[[], _Converter], path: str, timeout: float) -> Any:
    """Convert on a worker thread so a hung parse cannot stop the pipeline.

    A thread, not a process: Docling holds model weights that would have to be
    reloaded per document, and the cost of that dwarfs the parse. The thread is
    abandoned rather than killed on timeout — Python cannot safely interrupt
    one — so it finishes into nothing and is collected. That leaks a worker for
    the length of one stuck parse, which is the accepted price of not blocking
    the request forever.
    """
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FuturesTimeout

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="docling")
    try:
        future = executor.submit(lambda: factory().convert(path))
        try:
            return future.result(timeout=timeout)
        except FuturesTimeout as exc:
            future.cancel()
            raise TimeoutError(str(exc)) from exc
    finally:
        # Do not wait: the point of the timeout is not to block on the thread
        # we just gave up on.
        executor.shutdown(wait=False)


# --------------------------------------------------------------------------
# normalize
# --------------------------------------------------------------------------

_HEADING_KINDS = {"section_header", "title", "subtitle", "page_header"}
_SKIP_KINDS = {"page_footer", "picture", "formula"}


def normalize_docling_output(document: Any, *, expected_pages: int = 0) -> DoclingResult:
    """Convert a Docling document into this project's own layout types.

    Defensive throughout: Docling's object model varies across versions, and a
    structure pass that raises on an unexpected attribute would take down a
    document the old pipeline could have processed. Anything unreadable is
    skipped and counted rather than propagated.
    """
    conv = getattr(document, "document", document)
    if conv is None:
        return DoclingResult(fallback_reason="docling returned no document")

    pages: dict[int, PageLayout] = {}

    def page_for(number: int) -> PageLayout:
        if number not in pages:
            pages[number] = PageLayout(page_number=number)
        return pages[number]

    skipped = 0
    for item in _iter_items(conv):
        try:
            page_no, bbox = _provenance(item)
            if page_no <= 0:
                skipped += 1
                continue
            kind = str(getattr(item, "label", "") or "text").lower()
            if kind in _SKIP_KINDS:
                continue
            text = str(getattr(item, "text", "") or "").strip()
            if not text:
                continue
            page_for(page_no).blocks.append(LayoutBlock(
                kind="heading" if kind in _HEADING_KINDS else _block_kind(kind),
                text=text,
                level=_heading_level(item) if kind in _HEADING_KINDS else None,
                page_number=page_no,
                bbox=bbox,
            ))
        except Exception:  # noqa: BLE001 — one odd item must not lose the page
            skipped += 1

    for table in getattr(conv, "tables", None) or []:
        try:
            normalized = _normalize_table(table)
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        if normalized and normalized.is_valid():
            page_for(normalized.page_number).tables.append(normalized)
        elif normalized:
            skipped += 1

    if not pages:
        return DoclingResult(fallback_reason="docling produced no usable blocks or tables")

    # PyMuPDF is the authority on how many pages the document has. A layout
    # that disagrees is keyed to pages that do not exist, and attaching a table
    # to the wrong page is worse than attaching none.
    if expected_pages:
        stray = [p for p in pages if p > expected_pages]
        if stray:
            return DoclingResult(
                fallback_reason=(
                    f"docling reported page(s) {sorted(stray)[:5]} beyond the "
                    f"{expected_pages} PyMuPDF found"
                )
            )

    if skipped:
        log.debug("docling: skipped %d unreadable item(s)", skipped)
    return DoclingResult(ok=True, pages=pages)


def _iter_items(conv: Any):
    """Docling's items, whichever traversal this version offers."""
    if hasattr(conv, "iterate_items"):
        for entry in conv.iterate_items():
            yield entry[0] if isinstance(entry, tuple) else entry
        return
    yield from (getattr(conv, "texts", None) or [])


def _block_kind(label: str) -> str:
    if "list" in label:
        return "list"
    if "caption" in label:
        return "caption"
    if "table" in label:
        return "table"
    return "text"


def _heading_level(item: Any) -> int:
    level = getattr(item, "level", None)
    try:
        return max(1, int(level))
    except (TypeError, ValueError):
        return 1


def _provenance(item: Any) -> tuple[int, tuple[float, float, float, float] | None]:
    """(page number, bbox) from an item's provenance, when it has one."""
    prov = getattr(item, "prov", None) or []
    if not prov:
        return 0, None
    first = prov[0]
    page_no = int(getattr(first, "page_no", 0) or 0)
    bbox = getattr(first, "bbox", None)
    if bbox is None:
        return page_no, None
    try:
        return page_no, (float(bbox.l), float(bbox.t), float(bbox.r), float(bbox.b))
    except (AttributeError, TypeError, ValueError):
        return page_no, None


def _normalize_table(table: Any) -> LayoutTable | None:
    """A Docling table as a rectangular grid, with its spans recorded."""
    page_no, bbox = _provenance(table)
    data = getattr(table, "data", None)
    if data is None:
        return None

    n_rows = int(getattr(data, "num_rows", 0) or 0)
    n_cols = int(getattr(data, "num_cols", 0) or 0)
    grid: list[list[str]] = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    merged: list[dict[str, int]] = []

    for cell in getattr(data, "table_cells", None) or []:
        r = int(getattr(cell, "start_row_offset_idx", 0) or 0)
        c = int(getattr(cell, "start_col_offset_idx", 0) or 0)
        if not (0 <= r < n_rows and 0 <= c < n_cols):
            continue
        grid[r][c] = str(getattr(cell, "text", "") or "").strip()

        row_span = int(getattr(cell, "row_span", 1) or 1)
        col_span = int(getattr(cell, "col_span", 1) or 1)
        if row_span > 1 or col_span > 1:
            # Recorded rather than expanded. A merged header covering three
            # columns is one fact about the table's shape; writing its text
            # into three cells would make it look like three values.
            merged.append({"row": r, "col": c, "row_span": row_span, "col_span": col_span})

    return LayoutTable(
        rows=grid, n_rows=n_rows, n_cols=n_cols, merged_cells=merged,
        page_number=page_no, bbox=bbox,
    )


# --------------------------------------------------------------------------
# merge
# --------------------------------------------------------------------------


def _tables_of(layout: Any) -> list[LayoutTable]:
    """The tables on a page layout, whether it is a PageLayout or a dict.

    A layout stored on a PageRecord and then CHECKPOINTED comes back as a
    plain dict: LangGraph's serializer round-trips the dataclass structurally,
    not by type. MEASURED — a checkpoint round-trip of a PageRecord returns
    `layout` as dict, so `layout.tables` raises AttributeError on any resumed
    run. Nothing downstream reads `.layout` today, which makes this latent
    rather than broken; accepting both shapes here keeps it that way.
    """
    if layout is None:
        return []
    if isinstance(layout, dict):
        raw = layout.get("tables") or []
        out: list[LayoutTable] = []
        for entry in raw:
            if isinstance(entry, LayoutTable):
                out.append(entry)
            elif isinstance(entry, dict):
                try:
                    out.append(LayoutTable(**entry))
                except TypeError:
                    continue      # a shape this version does not know
        return out
    return list(getattr(layout, "tables", None) or [])


def merge_ocr_and_docling_output(
    page_text: str,
    layout: PageLayout | None,
    *,
    already_present: Callable[[str, str], bool] | None = None,
) -> str:
    """Add Docling's tables to a page's text, without repeating it.

    The OCR/text-layer content is authoritative and is never rewritten: it is
    what the evidence gate matches quotes against, and replacing it would
    invalidate every citation. What gets appended is the part reading order
    destroyed — which cells belong to which row — and only for rows the text
    does not already contain.

    That last condition is the whole safeguard. A born-digital results table is
    usually read perfectly well by the text layer, so appending Docling's copy
    would give the extractor the same twelve analytes twice and invite twelve
    duplicate rows. Row-level, not table-level: a table half of which the text
    layer missed contributes only its missing half.
    """
    tables = _tables_of(layout)
    if not tables:
        return page_text
    if LAYOUT_MARKER in page_text:
        return page_text        # already merged; never append twice

    contains = already_present or _text_contains_row
    lines: list[str] = []

    for index, table in enumerate(tables, start=1):
        rows = [r for r in table.rows if any(c.strip() for c in r)]
        fresh = [r for r in rows if not contains(page_text, " ".join(r))]
        if not fresh:
            continue        # the text layer already has this table
        header = f"table {index} ({table.n_rows} rows x {table.n_cols} columns"
        if table.merged_cells:
            header += f", {len(table.merged_cells)} merged cell(s)"
        lines.append(header + "):")
        lines += ["  " + " | ".join(c.strip() for c in row) for row in fresh]

    if not lines:
        return page_text
    return (page_text + "\n" + LAYOUT_MARKER + "\n" + "\n".join(lines)).strip()


def _text_contains_row(page_text: str, row: str) -> bool:
    """Is this row already in the page's text, ignoring how it is spaced?

    Compared on alphanumerics alone: the text layer separates cells with runs
    of spaces and Docling with single ones, and that difference is not a
    difference in content.
    """
    cells = [c for c in row.split() if c]
    if not cells:
        return True
    haystack = "".join(ch for ch in page_text.lower() if ch.isalnum())
    needle = "".join(ch for ch in row.lower() if ch.isalnum())
    if not needle:
        return True
    if needle in haystack:
        return True
    # A row counts as present when every one of its non-trivial cells is —
    # the cells may be interleaved with other columns in reading order.
    meaningful = [c for c in cells if len("".join(ch for ch in c if ch.isalnum())) >= 2]
    if not meaningful:
        return True
    return all(
        "".join(ch for ch in c.lower() if ch.isalnum()) in haystack
        for c in meaningful
    )
