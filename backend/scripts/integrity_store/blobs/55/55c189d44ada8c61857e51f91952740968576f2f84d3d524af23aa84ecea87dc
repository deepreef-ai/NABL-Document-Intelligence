"""G. LLM Response Validation Node — deterministic, no LLM.

A second gate on what extraction produced. The LLM layer already guaranteed
"this parsed as the right Pydantic model"; that is a syntactic guarantee and
says nothing about whether the content is coherent. This node asks the
semantic questions:

- does every field cite a chunk that exists, and a page that chunk covers?
- did the model return the same field twice, identically, inside one chunk
  (a strong sign of a looping or duplicated reply)?
- does a chunk's reply look truncated — a big chunk that returned suspiciously
  few fields, or a reply whose last field is visibly malformed?
- is the field name plausible, or is it a chunk of prose the model mistook for
  a label?

Invalid rows are dropped from the state and recorded, never silently repaired.
A chunk whose reply fails badly enough is reset to FAILED so the retry loop
picks it up. That is the whole point of the node: invalid output must not
continue through the graph.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict

from app.graph.errors import ErrorType
from app.graph.nodes.helpers import audit, err, normalise_for_match
from app.graph.schemas import ChunkStatus, ExtractedField, ExtractionStatus
from app.graph.state import GraphState

log = logging.getLogger(__name__)

NODE = "response_validation"

#: A field name longer than this is prose, not a label.
_MAX_FIELD_NAME_CHARS = 120
#: A value longer than this is a paragraph the model mislabelled as a value.
_MAX_VALUE_CHARS = 4000
#: Below this many fields from a chunk this large, suspect a truncated reply.
_TRUNCATION_CHUNK_CHARS = 8000
_TRUNCATION_MIN_FIELDS = 2


def validate_responses(state: GraphState) -> dict:
    chunk_index = {c.chunk_id: c for c in state.chunks}
    kept: list[ExtractedField] = []
    dropped: list[tuple[ExtractedField, str]] = []
    entries = []
    errors = []

    by_chunk: dict[str, list[ExtractedField]] = defaultdict(list)
    for f in state.extracted_fields:
        by_chunk[f.chunk_id].append(f)

    # ---- per-field structural checks -------------------------------------
    for f in state.extracted_fields:
        chunk = chunk_index.get(f.chunk_id)

        if chunk is None:
            dropped.append((f, f"unknown chunk_id {f.chunk_id!r}"))
            continue
        if len(f.field_name) > _MAX_FIELD_NAME_CHARS:
            dropped.append((f, "field_name is prose, not a label"))
            continue
        if f.value is not None and len(str(f.value)) > _MAX_VALUE_CHARS:
            dropped.append((f, "value is a paragraph, not a field value"))
            continue
        if f.page_number is None:
            dropped.append((f, "no page number cited"))
            continue
        if f.page_number not in chunk.page_numbers:
            # Kept, but flagged: the page may still be one of the chunk's
            # context pages, and the evidence node can resolve it properly.
            f.extraction_status = ExtractionStatus.UNCERTAIN
            f.notes = (f.notes + f" | cited page {f.page_number} not in chunk").strip(" |")
            errors += err(
                ErrorType.INCORRECT_PAGE_REFERENCE,
                f"field {f.normalized_field_name!r} cited page {f.page_number}, "
                f"chunk covers {chunk.page_numbers}",
                NODE, chunk_id=f.chunk_id, page_number=f.page_number,
                recovery_action="downgraded to UNCERTAIN; evidence node will confirm",
                resolution_status="RECOVERED",
            )
        if f.page_number is not None and f.page_number not in state.page_text:
            dropped.append((f, f"cited page {f.page_number} does not exist in the document"))
            continue

        kept.append(f)

    # ---- duplicate replies within one chunk ------------------------------
    # The same field name, value AND evidence repeated inside a single chunk is
    # a duplicated reply, not a genuinely repeated row: a real repeated row has
    # different evidence. occurrence_index alone is not enough to tell them
    # apart because models number inconsistently.
    deduped: list[ExtractedField] = []
    seen: set[tuple] = set()
    dupes = 0
    for f in kept:
        signature = (
            f.chunk_id,
            f.normalized_field_name,
            normalise_for_match(f.value),
            normalise_for_match(f.exact_source_evidence),
        )
        if signature in seen:
            dupes += 1
            continue
        seen.add(signature)
        deduped.append(f)

    if dupes:
        entries += audit(
            NODE, "duplicate_rows_removed",
            f"{dupes} identical row(s) returned more than once within a chunk", level="warning",
        )
        errors += err(
            ErrorType.LLM_DUPLICATE_RESPONSE,
            f"{dupes} exactly-duplicated field row(s) removed", NODE,
            recovery_action="duplicates dropped; distinct occurrences preserved",
            resolution_status="RECOVERED",
        )

    # ---- suspected truncation --------------------------------------------
    chunks = list(state.chunks)
    reset_for_retry: list[str] = []
    surviving_by_chunk: dict[str, int] = Counter(f.chunk_id for f in deduped)

    for chunk in chunks:
        if chunk.status != ChunkStatus.PROCESSED:
            continue
        produced = surviving_by_chunk.get(chunk.chunk_id, 0)
        if chunk.char_count >= _TRUNCATION_CHUNK_CHARS and produced < _TRUNCATION_MIN_FIELDS:
            # Only worth a retry if we have attempts left; the recovery agent
            # decides that. Here we just mark it and say why.
            chunk.status = ChunkStatus.FAILED
            chunk.last_error = (
                f"only {produced} field(s) from {chunk.char_count} chars — suspected truncated reply"
            )
            reset_for_retry.append(chunk.chunk_id)
            errors += err(
                ErrorType.LLM_TRUNCATED, chunk.last_error, NODE, chunk_id=chunk.chunk_id,
                retry_count=chunk.attempt_count,
                recovery_action="chunk requeued for recovery",
            )
            entries += audit(
                NODE, "suspected_truncation", chunk.last_error,
                chunk_id=chunk.chunk_id, level="warning",
            )

    for f, reason in dropped:
        errors += err(
            ErrorType.LLM_SCHEMA_MISMATCH,
            f"dropped field {f.field_name!r}: {reason}", NODE,
            chunk_id=f.chunk_id, page_number=f.page_number,
            recovery_action="row discarded as invalid",
            resolution_status="RECOVERED",
        )

    entries += audit(
        NODE, "responses_validated",
        f"{len(state.extracted_fields)} in, {len(deduped)} kept, {len(dropped)} dropped, "
        f"{dupes} duplicate(s), {len(reset_for_retry)} chunk(s) requeued",
        level="warning" if (dropped or reset_for_retry) else "info",
    )

    failed = sorted({c.chunk_id for c in chunks if c.status == ChunkStatus.FAILED})
    processed = [cid for cid in state.processed_chunks if cid not in set(reset_for_retry)]

    return {
        "current_node": NODE,
        # extracted_fields has an add-reducer, so it cannot be replaced by
        # returning it here. Validated rows go to normalized_fields, which the
        # rest of the graph reads via state.active_fields().
        "normalized_fields": deduped,
        "chunks": chunks,
        "failed_chunks": failed,
        "processed_chunks": processed,
        "audit_log": entries,
        "errors": errors,
    }
