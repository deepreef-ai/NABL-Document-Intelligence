"""F. Dynamic Extraction Agent — LLM-backed, one call per chunk.

Processes every chunk still marked PENDING or FAILED. Chunks already PROCESSED
are left alone, which is what makes targeted re-analysis and partial retry
possible: the node is re-entrant, and re-entering it after a failure re-runs
only what failed rather than the whole document.

Concurrency is bounded by `max_concurrent_llm_calls` and further shaped by the
rate limiter inside the LLM layer. Both are needed: the semaphore stops us
opening three hundred sockets, the limiter stops us exceeding the provider's
per-minute allowance.

Everything this node returns is *claimed*, not verified. It does not check
whether the evidence is real — that is the evidence node's job, deliberately
separated so that a model which is good at extraction but casual about quoting
cannot mark its own homework.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from app.graph.config import get_graph_settings
from app.graph.errors import ErrorType
from app.graph.llm import call_structured
from app.graph.nodes.analysis import structure_hint
from app.graph.nodes.helpers import (
    audit,
    clean_text,
    err,
    field_uid,
    infer_data_type,
    is_placeholder,
    snake_case,
)
from app.graph.schemas import (
    ChunkExtraction,
    ChunkRecord,
    ChunkStatus,
    ExtractedField,
    ExtractionStatus,
    LlmCallMetric,
    TestRow,
)
from app.graph.state import GraphState

log = logging.getLogger(__name__)

NODE = "dynamic_extraction"


def _section_for_page(state: GraphState, page: int | None) -> str:
    """The analysed section covering `page`, or a page label as a last resort.

    Never returns "" — a field with no heading is invisible in a grouped view,
    and "Page 7" at least tells the reader where to look.
    """
    if page is None:
        return "Unplaced"
    structure = state.document_structure
    if structure is not None:
        for section in structure.sections:
            start = section.start_page
            end = section.end_page or start
            if start is not None and end is not None and start <= page <= end:
                return section.name.strip() or f"Page {page}"
    return f"Page {page}"


def _sanitise(field: ExtractedField, chunk: ChunkRecord, state: GraphState) -> ExtractedField | None:
    """Repair what is safely repairable; return None for what is not.

    Rejecting here rather than downstream keeps obviously-broken rows out of
    the state entirely. The judgement calls (is the evidence real? is the page
    right?) belong to the evidence node and are NOT made here.
    """
    field.field_name = clean_text(field.field_name)
    if not field.field_name:
        return None

    field.chunk_id = chunk.chunk_id
    field.normalized_field_name = snake_case(field.normalized_field_name or field.field_name)
    field.exact_source_evidence = clean_text(field.exact_source_evidence)

    # A page number the chunk does not cover is a citation error. Where the
    # chunk covers exactly one page we can repair it; otherwise it is left
    # wrong and the evidence node will catch it, because guessing which of
    # five pages the model meant would be inventing provenance.
    if field.page_number not in chunk.page_numbers:
        if len(chunk.page_numbers) == 1:
            field.page_number = chunk.page_numbers[0]
        # else: leave as-is for evidence validation to flag as WRONG_PAGE

    # The review UI groups fields under their heading, so an absent
    # section_name collapses the whole document into one anonymous bucket and
    # throws away the structure the analysis pass already worked out. Fill it
    # from the section covering this page when the model omitted it — a
    # derived heading is honest (it comes from the document's own structure)
    # and far more useful than nothing.
    if not clean_text(field.section_name):
        field.section_name = _section_for_page(state, field.page_number)

    if is_placeholder(field.value):
        field.value = None
        field.normalized_value = None
        if field.extraction_status == ExtractionStatus.VERIFIED:
            field.extraction_status = ExtractionStatus.NOT_FOUND

    if field.value is not None and not field.data_type:
        field.data_type = infer_data_type(field.value)
    if field.normalized_value is None and field.value is not None:
        field.normalized_value = clean_text(field.value)

    # A VERIFIED claim with no quote is downgraded, not trusted. The prompt
    # asks for a verbatim quote; a model that skipped it has not shown its work.
    if field.extraction_status == ExtractionStatus.VERIFIED and not field.exact_source_evidence:
        field.extraction_status = ExtractionStatus.UNCERTAIN
        field.notes = (field.notes + " | no evidence quote returned").strip(" |")

    field.field_uid = field_uid(field)
    return field


def _extract_one(chunk: ChunkRecord, state: GraphState, hint: str, max_fields: int):
    """Runs in a worker thread. Returns (chunk_id, fields, error, metrics)."""
    from app.graph.prompts import EXTRACTION_SYSTEM, extraction_user

    collected: list[LlmCallMetric] = []

    outcome = call_structured(
        node=NODE,
        system=EXTRACTION_SYSTEM,
        user_text=extraction_user(
            chunk_id=chunk.chunk_id,
            page_numbers=chunk.page_numbers,
            text=chunk.text,
            structure_hint=hint,
            max_fields=max_fields,
        ),
        output_model=ChunkExtraction,
        on_metric=collected.append,
    )

    if not outcome.ok:
        return chunk.chunk_id, [], [], (outcome.error_type, outcome.error_message, outcome.attempts), collected

    parsed: ChunkExtraction = outcome.parsed  # type: ignore[assignment]

    # A reply carrying somebody else's chunk_id means the provider returned a
    # stale or crossed response. Treating it as this chunk's data would
    # attribute fields to the wrong pages, so it is rejected outright.
    if parsed.chunk_id and parsed.chunk_id != chunk.chunk_id:
        return chunk.chunk_id, [], [], (
            ErrorType.LLM_UNRELATED_RESPONSE,
            f"reply cited chunk_id {parsed.chunk_id!r}, expected {chunk.chunk_id!r}",
            outcome.attempts,
        ), collected

    fields = []
    for f in parsed.fields:
        cleaned = _sanitise(f, chunk, state)
        if cleaned is not None:
            fields.append(cleaned)

    # A page the chunk does not cover is a citation error; where the chunk
    # is a single page we can repair it, otherwise leave it for review.
    tests = []
    for t in parsed.tests:
        if not (t.test_name or t.result):
            continue
        if t.page_number not in chunk.page_numbers and len(chunk.page_numbers) == 1:
            t.page_number = chunk.page_numbers[0]
        tests.append(t)
    return chunk.chunk_id, fields, tests, None, collected


def extract_fields(state: GraphState) -> dict:
    settings = get_graph_settings()
    metrics = state.metrics.model_copy(deep=True)
    lock = Lock()

    todo = [c for c in state.chunks if c.status in (ChunkStatus.PENDING, ChunkStatus.FAILED)]
    if not todo:
        return {
            "current_node": NODE,
            "audit_log": audit(NODE, "nothing_to_do", "no pending or failed chunks"),
        }

    chunk_index = {c.chunk_id: c for c in state.chunks}
    all_fields: list[ExtractedField] = []
    all_tests: list[TestRow] = []
    processed: list[str] = list(state.processed_chunks)
    failed: list[str] = []
    entries = []
    errors = []

    workers = max(1, min(settings.max_concurrent_llm_calls, len(todo)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _extract_one, chunk, state,
                structure_hint(state, chunk.page_numbers),
                settings.max_fields_per_chunk,
            ): chunk
            for chunk in todo
        }
        for future in as_completed(futures):
            chunk = futures[future]
            try:
                chunk_id, fields, chunk_tests, failure, call_metrics = future.result()
            except Exception as exc:  # noqa: BLE001 — a worker crash is a chunk failure
                chunk_id, fields, chunk_tests, failure, call_metrics = (
                    chunk.chunk_id, [], [], (ErrorType.NODE_EXCEPTION, str(exc)[:500], 1), []
                )

            with lock:
                for m in call_metrics:
                    metrics.record(m)

            record = chunk_index[chunk_id]
            record.attempt_count += 1

            if failure is not None:
                error_type, message, attempts = failure
                record.status = ChunkStatus.FAILED
                record.last_error = message[:500]
                failed.append(chunk_id)
                errors += err(
                    error_type or ErrorType.LLM_UNAVAILABLE, message, NODE,
                    chunk_id=chunk_id, retry_count=record.attempt_count,
                    recovery_action="queued for the recovery agent",
                )
                entries += audit(
                    NODE, "chunk_failed", message[:300], chunk_id=chunk_id, level="warning",
                )
                continue

            record.status = ChunkStatus.PROCESSED
            record.last_error = ""
            if chunk_id not in processed:
                processed.append(chunk_id)
            all_fields.extend(fields)
            all_tests.extend(chunk_tests)
            entries += audit(
                NODE, "chunk_extracted", f"{len(fields)} field(s)", chunk_id=chunk_id,
            )

    # Empty chunks were never queued; mark them accounted-for so completeness
    # does not read "never processed" as "silently skipped".
    for c in state.chunks:
        if c.status == ChunkStatus.SKIPPED_EMPTY and c.chunk_id not in processed:
            processed.append(c.chunk_id)

    entries += audit(
        NODE, "extraction_round_complete",
        f"{len(todo)} chunk(s) attempted, {len(failed)} failed, "
        f"{len(all_fields)} field(s) and {len(all_tests)} test row(s) extracted",
    )

    return {
        "current_node": NODE,
        "chunks": list(state.chunks),          # attempt_count / status mutated in place
        "extracted_fields": all_fields,        # reducer appends
        "tests": all_tests,                    # reducer appends
        "processed_chunks": processed,
        "failed_chunks": failed,
        "extraction_attempts": state.extraction_attempts + 1,
        "metrics": metrics,
        "audit_log": entries,
        "errors": errors,
    }
