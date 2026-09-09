"""L. Completeness Agent — deterministic, no LLM.

Answers one question: is there anything we were supposed to have looked at and
did not? It is the node that turns "we processed the document" from an
assumption into a checked claim.

The distinction it exists to preserve: a page with no fields is not the same as
a page we never read. The first is a cover sheet; the second is information
loss. Only the page-level bookkeeping from preprocessing and chunking can tell
them apart, and this is where those two records are reconciled.

It also nominates targets for re-analysis rather than triggering a blanket
re-run. Re-reading a whole 300-page document because four pages looked thin is
exactly the waste the business rule about targeted re-analysis forbids.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from app.graph.config import get_graph_settings
from app.graph.errors import ErrorType
from app.graph.nodes.helpers import audit, err
from app.graph.schemas import (
    ChunkStatus,
    CompletenessReport,
    EvidenceStatus,
    PageStatus,
)
from app.graph.state import GraphState

log = logging.getLogger(__name__)

NODE = "completeness_check"

#: A readable page this long that produced nothing is worth a second look.
_SUSPICIOUS_PAGE_CHARS = 400


def check_completeness(state: GraphState) -> dict:
    settings = get_graph_settings()
    entries = []
    errors = []
    reasons: list[str] = []

    all_pages = set(state.page_text)
    report = CompletenessReport(pages_total=len(all_pages), chunks_total=len(state.chunks))

    # ---- page coverage ----------------------------------------------------
    covered: set[int] = set()
    for c in state.chunks:
        covered.update(c.page_numbers)
    not_chunked = sorted(all_pages - covered)
    report.pages_not_in_any_chunk = not_chunked
    if not_chunked:
        reasons.append(f"{len(not_chunked)} page(s) appear in no chunk: {not_chunked[:20]}")
        errors += err(
            ErrorType.PAGE_NOT_CHUNKED,
            f"pages never chunked: {not_chunked}", NODE,
            recovery_action="none available — blocks a successful verdict",
        )

    readable = [p for p, m in state.page_metadata.items() if m.is_readable]
    report.pages_readable = len(readable)
    report.pages_unreadable = sorted(
        p for p, m in state.page_metadata.items()
        if m.status in (PageStatus.UNREADABLE, PageStatus.OCR_FAILED)
    )
    if report.pages_unreadable:
        reasons.append(f"{len(report.pages_unreadable)} page(s) unreadable: {report.pages_unreadable[:20]}")

    # A page counts as processed if it is in a chunk that reached a terminal
    # state — including SKIPPED_EMPTY, which is a decision, not an omission.
    terminal = {ChunkStatus.PROCESSED, ChunkStatus.SKIPPED_EMPTY}
    processed_pages: set[int] = set()
    for c in state.chunks:
        if c.status in terminal:
            processed_pages.update(c.page_numbers)
    report.all_pages_processed = not not_chunked and processed_pages >= (all_pages - set(report.pages_unreadable))

    # ---- chunk coverage ---------------------------------------------------
    failed = [c.chunk_id for c in state.chunks if c.status == ChunkStatus.FAILED]
    pending = [c.chunk_id for c in state.chunks if c.status in (ChunkStatus.PENDING, ChunkStatus.IN_PROGRESS)]
    report.chunks_failed = sorted(failed)
    report.all_chunks_processed = not failed and not pending

    if failed:
        reasons.append(f"{len(failed)} chunk(s) failed permanently")
        errors += err(
            ErrorType.INCOMPLETE_PROCESSING,
            f"chunks failed after all retries: {failed[:20]}", NODE,
            recovery_action="reported as failed chunks; blocks a VALIDATED verdict",
        )
    if pending:
        reasons.append(f"{len(pending)} chunk(s) never ran")
        errors += err(
            ErrorType.MISSING_CHUNK, f"chunks never processed: {pending[:20]}", NODE,
            recovery_action="none — indicates the graph exited early",
        )

    # ---- evidence coverage ------------------------------------------------
    fields = state.active_fields()
    evidence_by_uid = {r.field_uid: r for r in state.evidence_validation_results}
    missing_evidence = [
        f.field_uid for f in fields
        if evidence_by_uid.get(f.field_uid) is None
        or evidence_by_uid[f.field_uid].status in (EvidenceStatus.NO_EVIDENCE, EvidenceStatus.UNSUPPORTED)
    ]
    report.fields_without_evidence = missing_evidence
    if missing_evidence:
        reasons.append(f"{len(missing_evidence)} field(s) lack usable evidence")

    # ---- pages that produced nothing --------------------------------------
    fields_by_page: dict[int, int] = defaultdict(int)
    for f in fields:
        if f.page_number is not None:
            fields_by_page[f.page_number] += 1

    barren = []
    for page in sorted(readable):
        if fields_by_page.get(page, 0) == 0:
            meta = state.page_metadata.get(page)
            if meta and meta.char_count >= _SUSPICIOUS_PAGE_CHARS:
                barren.append(page)
    report.pages_with_no_fields = barren
    if barren:
        reasons.append(
            f"{len(barren)} substantive page(s) produced no fields: {barren[:20]}"
        )

    # ---- targets for re-analysis ------------------------------------------
    # Only the barren pages, and only if we have rounds left. Failed chunks are
    # the recovery agent's business, not re-analysis.
    targets: list[str] = []
    if barren and state.reanalysis_count < settings.max_reanalysis_rounds:
        for c in state.chunks:
            if c.status == ChunkStatus.PROCESSED and any(p in barren for p in c.page_numbers):
                targets.append(c.chunk_id)
    report.reanalysis_targets = sorted(set(targets))

    # ---- information loss --------------------------------------------------
    report.information_loss_detected = bool(
        not_chunked or pending or (failed and not state.failed_chunks)
    )
    if report.information_loss_detected:
        errors += err(
            ErrorType.INFORMATION_LOSS,
            "pages or chunks are unaccounted for", NODE,
            recovery_action="blocks a successful verdict",
        )

    report.reasons = reasons

    entries += audit(
        NODE, "completeness_checked",
        f"pages {report.pages_readable}/{report.pages_total} readable, "
        f"{len(report.chunks_failed)} chunk(s) failed, "
        f"{len(missing_evidence)} field(s) without evidence, "
        f"{len(report.reanalysis_targets)} chunk(s) queued for re-analysis",
        level="warning" if reasons else "info",
    )

    return {
        "current_node": NODE,
        "completeness_report": report,
        "audit_log": entries,
        "errors": errors,
    }


def mark_reanalysis_targets(state: GraphState) -> dict:
    """Requeue exactly the chunks completeness nominated, and nothing else."""
    report = state.completeness_report
    if report is None or not report.reanalysis_targets:
        return {
            "current_node": "targeted_reanalysis",
            "audit_log": audit("targeted_reanalysis", "nothing_to_reanalyse", ""),
        }

    targets = set(report.reanalysis_targets)
    chunks = list(state.chunks)
    requeued = []
    for c in chunks:
        if c.chunk_id in targets:
            c.status = ChunkStatus.PENDING
            # Do NOT reset attempt_count: re-analysis shares the retry ceiling,
            # otherwise a chunk could alternate between the two loops forever.
            requeued.append(c.chunk_id)

    processed = [cid for cid in state.processed_chunks if cid not in targets]

    return {
        "current_node": "targeted_reanalysis",
        "chunks": chunks,
        "processed_chunks": processed,
        "reanalysis_count": state.reanalysis_count + 1,
        "retry_reasons": [f"re-analysis round {state.reanalysis_count + 1}: {len(requeued)} chunk(s)"],
        "audit_log": audit(
            "targeted_reanalysis", "chunks_requeued",
            f"{len(requeued)} chunk(s) covering pages that produced nothing",
        ),
    }
