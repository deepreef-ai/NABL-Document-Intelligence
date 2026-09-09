"""E. Chunk Management Node — deterministic, no LLM.

Divides the document into token-safe units of work, then audits its own output.

Two rules drive the design, and they conflict on large documents:

- **Every page must appear in at least one chunk.** This is the guarantee the
  completeness node checks and the reason the whole system can claim not to
  skip pages. It is non-negotiable.
- **A chunk must be small enough to be *answered*, not merely read.** The
  binding constraint on real documents is output tokens, not input context: a
  chunk containing fifty pages of tables fits comfortably in the context
  window and then produces a reply that truncates mid-JSON. Truncated JSON is
  worse than a smaller chunk, because the whole chunk's fields are lost, not
  just the overflow.

So chunking here never merges past the size limit to satisfy a call budget.
If a document needs eighty chunks, it gets eighty chunks. Cost is controlled
by the rate limiter and the workflow timeout, not by silently sending prompts
that cannot be answered.

Pages larger than the limit on their own are split rather than dropped, with
the split points recorded, because a single dense page is still a page we
promised not to skip.
"""
from __future__ import annotations

import logging

from app.graph.config import get_graph_settings
from app.graph.errors import ErrorType
from app.graph.nodes.helpers import audit, content_hash, err
from app.graph.schemas import ChunkRecord, ChunkStatus
from app.graph.state import GraphState

log = logging.getLogger(__name__)

NODE = "chunk_management"


def _page_block(page_no: int, text: str) -> str:
    # The page marker is load-bearing: the extraction prompt tells the model to
    # cite one of these numbers, and evidence validation checks the citation.
    return f"--- Page {page_no} ---\n{text}"


def _split_oversized(page_no: int, text: str, limit: int) -> list[str]:
    """Split one very large page on paragraph boundaries where possible."""
    body_limit = max(1000, limit - 200)
    if len(text) <= body_limit:
        return [text]
    parts, current = [], ""
    for para in text.split("\n\n"):
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) > body_limit and current:
            parts.append(current)
            current = para
        else:
            current = candidate
    if current:
        parts.append(current)
    # A single paragraph over the limit still has to be cut somewhere.
    final: list[str] = []
    for part in parts:
        while len(part) > body_limit:
            final.append(part[:body_limit])
            part = part[body_limit:]
        if part:
            final.append(part)
    return final


def create_chunks(state: GraphState) -> dict:
    settings = get_graph_settings()
    limit = settings.max_chunk_chars
    overlap = max(0, settings.chunk_overlap_pages)

    pages = sorted(state.page_text)
    chunks: list[ChunkRecord] = []
    entries = []
    errors = []
    seq = 0

    current_pages: list[int] = []
    current_text = ""
    current_table = False

    def flush() -> None:
        nonlocal seq, current_pages, current_text, current_table
        if not current_pages:
            return
        seq += 1
        chunk_id = f"{state.document_id}::c{seq:04d}"
        chunks.append(ChunkRecord(
            chunk_id=chunk_id,
            page_numbers=list(current_pages),
            text=current_text,
            char_count=len(current_text),
            status=ChunkStatus.PENDING,
            has_table=current_table,
            content_hash=content_hash(current_text),
        ))
        current_pages, current_text, current_table = [], "", False

    for page_no in pages:
        text = state.page_text.get(page_no, "")
        meta = state.page_metadata.get(page_no)
        has_table = bool(meta and meta.has_table)

        if not text.strip():
            # An empty page still has to be represented, or completeness will
            # report it as never chunked. It gets its own zero-content chunk
            # marked SKIPPED_EMPTY: accounted for, but no LLM call spent on it.
            seq += 1
            chunks.append(ChunkRecord(
                chunk_id=f"{state.document_id}::c{seq:04d}",
                page_numbers=[page_no], text="", char_count=0,
                status=ChunkStatus.SKIPPED_EMPTY, has_table=False,
                content_hash=content_hash(f"empty-{page_no}"),
            ))
            entries += audit(NODE, "empty_page_chunk", "page has no text", page_number=page_no)
            continue

        pieces = _split_oversized(page_no, text, limit)
        if len(pieces) > 1:
            flush()
            for i, piece in enumerate(pieces):
                seq += 1
                chunks.append(ChunkRecord(
                    chunk_id=f"{state.document_id}::c{seq:04d}",
                    page_numbers=[page_no],
                    text=_page_block(page_no, piece),
                    char_count=len(piece),
                    status=ChunkStatus.PENDING,
                    has_table=has_table,
                    content_hash=content_hash(piece),
                ))
            entries += audit(
                NODE, "page_split", f"page split into {len(pieces)} chunks (dense page)",
                page_number=page_no, level="warning",
            )
            continue

        block = _page_block(page_no, text)
        if current_pages and len(current_text) + len(block) + 2 > limit:
            flush()
            # Carry the previous page(s) in as read-only context so a value
            # whose label sits on the page before is still resolvable. Context
            # pages are NOT claimed as covered by this chunk.
            if overlap and chunks:
                prev = chunks[-1].page_numbers[-overlap:]
                ctx = "\n\n".join(
                    _page_block(p, state.page_text.get(p, "")) for p in prev
                    if state.page_text.get(p, "").strip()
                )
                if ctx:
                    current_text = f"[context from preceding page(s), do not extract as new]\n{ctx}\n\n"

        current_pages.append(page_no)
        current_text = f"{current_text}\n\n{block}".strip() if current_text else block
        current_table = current_table or has_table

    flush()

    # ---------------------------------------------------------------- audit
    covered: set[int] = set()
    for c in chunks:
        covered.update(c.page_numbers)
    missing = sorted(set(pages) - covered)
    if missing:
        errors += err(
            ErrorType.PAGE_NOT_CHUNKED,
            f"pages missing from every chunk: {missing}",
            NODE, recovery_action="none — this is a chunking bug, workflow will fail completeness",
        )
        entries += audit(NODE, "pages_not_chunked", str(missing), level="error")

    seen_hashes: dict[str, str] = {}
    for c in chunks:
        if c.status == ChunkStatus.SKIPPED_EMPTY:
            continue
        prior = seen_hashes.get(c.content_hash)
        if prior:
            # Identical content in two chunks wastes a call and doubles every
            # field it contains. Report it; do not silently drop the chunk,
            # because a genuinely repeated page is real document content.
            entries += audit(
                NODE, "duplicate_chunk_content",
                f"{c.chunk_id} has the same content as {prior}",
                chunk_id=c.chunk_id, level="warning",
            )
            errors += err(
                ErrorType.DUPLICATE_CHUNK,
                f"{c.chunk_id} duplicates {prior}", NODE, chunk_id=c.chunk_id,
                recovery_action="both retained; duplicate values merged downstream",
                resolution_status="RECOVERED",
            )
        else:
            seen_hashes[c.content_hash] = c.chunk_id

        if c.char_count > limit * 1.05:
            errors += err(
                ErrorType.OVERSIZED_CHUNK,
                f"{c.chunk_id} is {c.char_count} chars, over the {limit} limit",
                NODE, chunk_id=c.chunk_id,
                recovery_action="recovery agent will split it if extraction fails",
            )

    if not chunks:
        errors += err(
            ErrorType.MISSING_CHUNK, "no chunks were produced", NODE,
            recovery_action="none",
        )

    work = [c for c in chunks if c.status == ChunkStatus.PENDING]
    entries += audit(
        NODE, "chunked",
        f"{len(chunks)} chunk(s) covering {len(covered)}/{len(pages)} page(s); "
        f"{len(work)} need extraction, {len(chunks) - len(work)} empty",
    )

    return {
        "current_node": NODE,
        "chunks": chunks,
        "audit_log": entries,
        "errors": errors,
    }
