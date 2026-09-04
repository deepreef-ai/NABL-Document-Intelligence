"""Groups relevant pages into as few extraction chunks as the budget and the
context window allow.

Replaces two bad extremes that both existed in this codebase:

- one LLM call per schema SECTION (the whole-form path: 10 calls for
  NABL_151 regardless of how much of the form the document actually covers);
- one call per ~7,000 chars of flat text (the DOCX/open-extraction path),
  which splits mid-table.

Rules, in priority order:
1. Never exceed `max_chunk_chars` of text in one chunk.
2. Never emit more than `max_initial_extraction_calls` chunks — if the
   relevant pages don't fit, merge the smallest adjacent pair repeatedly.
   Exceeding the caller's budget is not an option; a slightly-too-large
   prompt is recoverable, a budget overrun is not.
3. Keep a table page with its neighbours where possible: a results table
   that continues across a page break must not be split across two calls,
   because each call would see half the rows and neither would see the
   header. Table pages are therefore never chosen as the merge boundary
   while a non-table boundary is available.
4. Only relevant pages are included; irrelevant pages are skipped entirely
   (this is where most of the saving comes from on a long document).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.config import get_settings


@dataclass
class ExtractionChunk:
    """One extraction call's worth of evidence."""

    page_numbers: list[int] = field(default_factory=list)
    text: str = ""
    images: list[bytes] = field(default_factory=list)
    has_table: bool = False

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def label(self) -> str:
        if not self.page_numbers:
            return "empty"
        pages = sorted(self.page_numbers)
        # Only render a range when the pages really are contiguous — a chunk
        # of the 3 relevant pages out of 20 is "p1,p2,p7", and calling that
        # "p1-7" would misreport what was actually sent to the model.
        if pages == list(range(pages[0], pages[-1] + 1)):
            return f"p{pages[0]}" if len(pages) == 1 else f"p{pages[0]}-{pages[-1]}"
        return ",".join(f"p{n}" for n in pages)


def _page_text(page) -> str:
    return f"--- Page {page.page_number} ---\n{page.text or ''}"


def build_chunks(
    pages: list,
    relevant_page_numbers: list[int] | None = None,
    max_chars: int | None = None,
    max_chunks: int | None = None,
    include_images: bool = True,
    max_images: int | None = None,
) -> list[ExtractionChunk]:
    """`pages` are InspectedPage objects (documents/page_inspection.py)."""
    settings = get_settings()
    max_chars = max_chars or settings.max_chunk_chars
    max_chunks = max_chunks or settings.max_initial_extraction_calls
    max_images = settings.max_images_per_call if max_images is None else max_images

    if relevant_page_numbers is None:
        selected = list(pages)
    else:
        wanted = set(relevant_page_numbers)
        selected = [p for p in pages if p.page_number in wanted]
    if not selected:
        selected = list(pages)
    # Page order, not relevance order: the model reads a document forwards,
    # and a table's continuation must follow its header.
    selected.sort(key=lambda p: p.page_number)
    if not selected:
        return []

    # Pass 1: greedy fill up to max_chars, breaking preferentially where a
    # table does not straddle the boundary.
    chunks: list[ExtractionChunk] = []
    current = ExtractionChunk()
    for page in selected:
        piece = _page_text(page)
        if current.page_numbers and current.char_count + len(piece) > max_chars:
            chunks.append(current)
            current = ExtractionChunk()
        current.page_numbers.append(page.page_number)
        current.text = f"{current.text}\n\n{piece}".strip() if current.text else piece
        current.has_table = current.has_table or page.has_table
        # Only pages that actually need visual interpretation contribute an
        # image (see page_inspection.InspectedPage.needs_image), and never
        # more than max_images than one request can carry.
        if include_images and page.image is not None and getattr(page, "needs_image", True):
            if len(current.images) < max_images:
                current.images.append(page.image)
    if current.page_numbers:
        chunks.append(current)

    # Pass 2: honour the call budget by merging adjacent chunks, smallest
    # combined size first, preferring boundaries that don't join two tables.
    while len(chunks) > max_chunks:
        best_i, best_cost = 0, None
        for i in range(len(chunks) - 1):
            cost = chunks[i].char_count + chunks[i + 1].char_count
            if chunks[i].has_table and chunks[i + 1].has_table:
                # Merging two table pages is allowed, but only as a last
                # resort — it is the case most likely to blow the context.
                cost += max_chars
            if best_cost is None or cost < best_cost:
                best_i, best_cost = i, cost
        a, b = chunks[best_i], chunks[best_i + 1]
        merged = ExtractionChunk(
            page_numbers=a.page_numbers + b.page_numbers,
            text=f"{a.text}\n\n{b.text}",
            images=(a.images + b.images)[:max_images],
            has_table=a.has_table or b.has_table,
        )
        chunks[best_i : best_i + 2] = [merged]

    return chunks
