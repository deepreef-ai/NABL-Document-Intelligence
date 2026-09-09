"""H. Retry and Recovery Agent — deterministic control, no LLM of its own.

Sits between a failed extraction round and the next one, and decides *how* the
next attempt should differ. Re-sending an identical prompt to a provider that
just failed on it is the least likely thing to work, so this node changes
something concrete before the retry:

- **Truncated or oversized** → split the chunk in half and requeue both parts.
  Two smaller answers fit where one large one did not.
- **Invalid JSON / schema mismatch** → leave the chunk alone; the LLM layer
  already escalates to a stricter instruction on its own retries.
- **Rate limited or timed out** → leave it alone but let the backoff run; the
  chunk is fine, the provider was busy.
- **Poor text quality** → re-run OCR on the chunk's pages at a higher DPI and
  rebuild the chunk text, then requeue.
- **Attempts exhausted** → stop. Mark the chunk permanently failed, record it,
  and let completeness and the final decision account for it. A chunk that has
  failed three times with a clear error is not going to succeed on the fourth,
  and looping is how a workflow burns a budget with nothing to show.

The retry ceiling is enforced here and nowhere else, so there is one place to
look when asking "why did it stop trying".
"""
from __future__ import annotations

import logging

from app.graph.config import get_graph_settings
from app.graph.errors import ErrorType
from app.graph.nodes.helpers import audit, clean_text, content_hash, err
from app.graph.schemas import ChunkRecord, ChunkStatus, PageStatus
from app.graph.state import GraphState

log = logging.getLogger(__name__)

NODE = "retry_recovery"

_SPLIT_TRIGGERS = {
    ErrorType.LLM_TRUNCATED.value,
    ErrorType.OVERSIZED_CHUNK.value,
    ErrorType.LLM_INCOMPLETE_RESPONSE.value,
}
_OCR_TRIGGERS = {
    ErrorType.POOR_QUALITY_TEXT.value,
    ErrorType.OCR_FAILURE.value,
}


def _last_error_type(state: GraphState, chunk_id: str) -> str:
    for record in reversed(state.errors):
        if record.chunk_id == chunk_id:
            return record.error_type
    return ""


def _split_chunk(chunk: ChunkRecord, suffix_seq: int) -> list[ChunkRecord]:
    """Halve a chunk on a paragraph boundary, preserving its page claims.

    Both halves claim the SAME page numbers. That is deliberate: the pages are
    still covered, and completeness must not see a page lose its coverage
    because we split the chunk that carried it.
    """
    text = chunk.text
    midpoint = len(text) // 2
    split_at = text.find("\n\n", midpoint)
    if split_at == -1:
        split_at = midpoint
    halves = [text[:split_at].strip(), text[split_at:].strip()]
    out = []
    for i, part in enumerate(h for h in halves if h):
        out.append(ChunkRecord(
            chunk_id=f"{chunk.chunk_id}-s{suffix_seq}{i + 1}",
            page_numbers=list(chunk.page_numbers),
            text=part,
            char_count=len(part),
            status=ChunkStatus.PENDING,
            has_table=chunk.has_table,
            context_pages=list(chunk.context_pages),
            attempt_count=chunk.attempt_count,   # inherited: splitting is not a free reset
            content_hash=content_hash(part),
        ))
    return out


def _reocr_pages(state: GraphState, pages: list[int]) -> tuple[dict[int, str], list[int]]:
    """Re-run OCR at higher DPI. Returns (new text by page, pages improved)."""
    settings = get_graph_settings()
    improved: list[int] = []
    updated: dict[int, str] = {}
    if state.file_type != "pdf":
        return updated, improved
    try:
        with open(state.file_path, "rb") as fh:
            data = fh.read()
        from app.documents import local_ocr, pdf_utils
    except Exception:  # noqa: BLE001
        return updated, improved

    for page in pages:
        try:
            # Zero-based, like everywhere rasterize_page is called — see the
            # note in preprocess.py. Page numbers in this package are one-based.
            image = pdf_utils.rasterize_page(data, page - 1, dpi=min(400, settings.ocr_dpi * 2))
            result = local_ocr.extract_english(image)
            text = clean_text(getattr(result, "text", "") or "")
        except Exception:  # noqa: BLE001 — a failed re-OCR just means no improvement
            continue
        if len(text) > len(state.page_text.get(page, "")):
            updated[page] = text
            improved.append(page)
    return updated, improved


def recover(state: GraphState) -> dict:
    settings = get_graph_settings()
    chunks = list(state.chunks)
    index = {c.chunk_id: c for c in chunks}
    entries = []
    errors = []
    retry_reasons: list[str] = []

    failing = [c for c in chunks if c.status == ChunkStatus.FAILED]
    if not failing:
        return {
            "current_node": NODE,
            "audit_log": audit(NODE, "nothing_to_recover", "no failed chunks"),
        }

    page_text = dict(state.page_text)
    page_meta = dict(state.page_metadata)
    new_chunks: list[ChunkRecord] = []
    abandoned: list[str] = []
    requeued: list[str] = []
    split_seq = 0

    for chunk in failing:
        cause = _last_error_type(state, chunk.chunk_id)

        # --- ceiling ------------------------------------------------------
        if chunk.attempt_count >= settings.max_attempts_per_chunk:
            abandoned.append(chunk.chunk_id)
            errors += err(
                ErrorType.RETRY_LIMIT_EXCEEDED,
                f"{chunk.chunk_id} failed {chunk.attempt_count} time(s); "
                f"last error: {chunk.last_error[:200]}",
                NODE, chunk_id=chunk.chunk_id, retry_count=chunk.attempt_count,
                recovery_action="abandoned; reported as a failed chunk",
                resolution_status="ABANDONED",
            )
            entries += audit(
                NODE, "chunk_abandoned",
                f"retry limit reached after {chunk.attempt_count} attempt(s)",
                chunk_id=chunk.chunk_id, level="error",
            )
            continue

        # --- split --------------------------------------------------------
        if cause in _SPLIT_TRIGGERS and chunk.char_count > 2000:
            split_seq += 1
            parts = _split_chunk(chunk, split_seq)
            if len(parts) > 1:
                new_chunks.extend(parts)
                chunk.status = ChunkStatus.SKIPPED_EMPTY  # superseded by its parts
                chunk.last_error = f"split into {len(parts)} smaller chunks"
                requeued.extend(p.chunk_id for p in parts)
                retry_reasons.append(f"{chunk.chunk_id}: split after {cause}")
                entries += audit(
                    NODE, "chunk_split",
                    f"split into {len(parts)} parts after {cause}",
                    chunk_id=chunk.chunk_id,
                )
                continue

        # --- re-OCR -------------------------------------------------------
        if cause in _OCR_TRIGGERS:
            updated, improved = _reocr_pages(state, chunk.page_numbers)
            if improved:
                page_text.update(updated)
                for page in improved:
                    meta = page_meta.get(page)
                    if meta is not None:
                        page_meta[page] = meta.model_copy(update={
                            "text": updated[page],
                            "char_count": len(updated[page]),
                            "status": PageStatus.OCR,
                            "ocr_applied": True,
                            "notes": (meta.notes + " | re-OCR at higher DPI").strip(" |"),
                        })
                chunk.text = "\n\n".join(
                    f"--- Page {p} ---\n{page_text.get(p, '')}" for p in chunk.page_numbers
                )
                chunk.char_count = len(chunk.text)
                chunk.content_hash = content_hash(chunk.text)
                retry_reasons.append(f"{chunk.chunk_id}: re-OCR improved pages {improved}")
                entries += audit(
                    NODE, "reocr_applied", f"pages {improved} re-read at higher DPI",
                    chunk_id=chunk.chunk_id,
                )

        # --- plain retry ---------------------------------------------------
        chunk.status = ChunkStatus.PENDING
        requeued.append(chunk.chunk_id)
        retry_reasons.append(f"{chunk.chunk_id}: retry after {cause or 'unknown error'}")
        entries += audit(
            NODE, "chunk_requeued",
            f"attempt {chunk.attempt_count + 1} of {settings.max_attempts_per_chunk} "
            f"after {cause or 'unknown error'}",
            chunk_id=chunk.chunk_id,
        )

    # Abandoned chunks keep FAILED status so completeness and the final
    # decision both see them. They are never quietly dropped.
    for cid in abandoned:
        index[cid].status = ChunkStatus.FAILED

    chunks.extend(new_chunks)

    entries += audit(
        NODE, "recovery_complete",
        f"{len(requeued)} chunk(s) requeued, {len(new_chunks)} new from splits, "
        f"{len(abandoned)} abandoned",
        level="warning" if abandoned else "info",
    )

    return {
        "current_node": NODE,
        "chunks": chunks,
        "page_text": page_text,
        "page_metadata": page_meta,
        "failed_chunks": sorted(abandoned),
        "retry_count": state.retry_count + 1,
        "retry_reasons": retry_reasons,
        "audit_log": entries,
        "errors": errors,
    }
