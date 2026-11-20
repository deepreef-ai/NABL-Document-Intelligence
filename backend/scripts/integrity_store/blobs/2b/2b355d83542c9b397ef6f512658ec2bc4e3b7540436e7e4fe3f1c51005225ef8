"""P. Final Decision Agent — deterministic, no LLM.

Returns exactly one status, and the reasoning behind it. Deterministic on
purpose: the verdict is the one thing in this system that must be reproducible
from the same state, and a model asked to summarise a run will sometimes call
a document VALIDATED because the summary reads well.

Precedence, highest first. The order matters more than the individual rules,
because several conditions are usually true at once and the reader needs to
know which one dominates:

1. REJECTED                — the document could not be processed at all.
2. INCOMPLETE              — something is unaccounted for. Silent gaps outrank
                             every other consideration; a partially-read
                             document with confident fields is more dangerous
                             than an obviously failed one.
3. CONFLICTED              — the document contradicts itself and we did not
                             resolve it. Distinct from MANUAL_REVIEW_REQUIRED
                             because the cause is the document, not our
                             uncertainty about it.
4. MANUAL_REVIEW_REQUIRED  — we produced a usable result but something needs a
                             person: an ambiguous mapping, an unsupported value.
5. LOW_CONFIDENCE          — complete and consistent, but too little of it is
                             firmly verified to hand on unchecked.
6. VALIDATED               — everything above is clear.
"""
from __future__ import annotations

import logging
import time

from app.graph.config import get_graph_settings
from app.graph.nodes.helpers import audit
from app.graph.schemas import (
    ConflictResolution,
    ExtractionStatus,
    FinalStatus,
    MappingStatus,
)
from app.graph.state import GraphState, WorkflowStatus

log = logging.getLogger(__name__)

NODE = "final_decision"


def decide(state: GraphState) -> dict:
    settings = get_graph_settings()
    qc = state.quality_control_report
    completeness = state.completeness_report
    fields = state.active_fields()

    reasons: list[str] = []
    status: FinalStatus

    fatal = [e for e in state.errors if e.resolution_status == "ABANDONED" and e.node == "document_validation"]

    # ---- 1. rejected -------------------------------------------------------
    if state.final_status == FinalStatus.REJECTED or fatal:
        status = FinalStatus.REJECTED
        reasons.append(state.error_details or "the document could not be processed")

    # ---- 2. incomplete -----------------------------------------------------
    elif qc is not None and qc.critical_issues:
        status = FinalStatus.INCOMPLETE
        reasons.extend(qc.critical_issues[:6])

    elif completeness is not None and completeness.critical_failure:
        status = FinalStatus.INCOMPLETE
        reasons.append("completeness check failed: pages or chunks unaccounted for")
        reasons.extend(completeness.reasons[:4])

    elif not fields:
        status = FinalStatus.INCOMPLETE
        reasons.append("no field-value pairs were extracted from the document")

    else:
        unresolved = [c for c in state.conflicts if c.resolution != ConflictResolution.RESOLVED]
        needs_person = [
            i for i in state.manual_review_items
            if i.kind in ("ambiguous_mapping", "unsupported", "failed_chunk")
        ]
        verified = [f for f in fields if f.extraction_status == ExtractionStatus.VERIFIED]
        verified_ratio = len(verified) / len(fields)
        failed_chunks = list(state.failed_chunks)

        # ---- 3. conflicted -------------------------------------------------
        if unresolved:
            status = FinalStatus.CONFLICTED
            reasons.append(
                f"{len(unresolved)} field(s) hold contradictory values that could not be "
                f"resolved from the document; every candidate is preserved for review"
            )
            reasons.extend(
                f"{c.normalized_field_name}: {c.reason[:120]}" for c in unresolved[:4]
            )

        # ---- 4. manual review ----------------------------------------------
        elif needs_person or failed_chunks:
            status = FinalStatus.MANUAL_REVIEW_REQUIRED
            if failed_chunks:
                reasons.append(
                    f"{len(failed_chunks)} chunk(s) could not be processed after all retries; "
                    f"the pages they cover may hold data that is not in this result"
                )
            if needs_person:
                reasons.append(f"{len(needs_person)} item(s) need a person to decide")
                reasons.extend(f"{i.kind}: {i.reference}" for i in needs_person[:4])

        # ---- 5. low confidence ----------------------------------------------
        elif verified_ratio < (1.0 - settings.low_confidence_ratio_limit):
            status = FinalStatus.LOW_CONFIDENCE
            reasons.append(
                f"only {verified_ratio:.0%} of {len(fields)} extracted field(s) are fully verified "
                f"against the source; the rest are uncertain and should be checked"
            )

        # ---- 6. validated -----------------------------------------------------
        else:
            status = FinalStatus.VALIDATED
            filled = sum(1 for m in state.form_mappings if m.mapping_status == MappingStatus.MAPPED)
            reasons.append(
                f"all {len(state.page_text)} page(s) processed, all chunks accounted for, "
                f"{len(verified)} of {len(fields)} field(s) verified against source evidence"
            )
            if state.form_mappings:
                reasons.append(f"{filled} target form field(s) filled from verified values")
            if qc is not None and qc.issues:
                reasons.append(f"{len(qc.issues)} advisory issue(s) recorded, none blocking")

    elapsed = (time.monotonic() - state.started_at) if state.started_at else 0.0
    metrics = state.metrics.model_copy(deep=True)
    metrics.processing_time_seconds = round(elapsed, 2)
    metrics.retry_count = state.retry_count

    reasoning = " | ".join(reasons)

    return {
        "current_node": NODE,
        "final_status": status,
        "final_reasoning": reasoning,
        "current_status": (
            WorkflowStatus.FAILED if status == FinalStatus.REJECTED else WorkflowStatus.COMPLETED
        ),
        "metrics": metrics,
        "audit_log": audit(
            NODE, "decided", f"{status.value}: {reasoning[:400]}",
            level="error" if status in (FinalStatus.REJECTED, FinalStatus.INCOMPLETE) else "info",
        ),
    }


def build_result(state: GraphState):
    """Assemble the spec-section-8 payload from the final state."""
    from app.graph.schemas import (
        DocumentSummary,
        FinalResult,
        QualityControlOutput,
    )

    from app.graph.structured import build_structured

    qc = state.quality_control_report
    structured = build_structured(
        state.active_fields(), state.tests,
        filename=state.display_name or state.file_name,
        document_type=state.document_type,
    )
    return FinalResult(
        overall_status=state.final_status or FinalStatus.INCOMPLETE,
        document_summary=DocumentSummary(
            document_type=state.document_type,
            page_count=state.total_pages,
            processed_pages=len(state.processed_pages),
            processed_chunks=len(state.processed_chunks),
            failed_chunks=len(state.failed_chunks),
            ocr_pages=len(state.ocr_pages),
        ),
        extracted_fields=state.active_fields(),
        tests=state.tests,
        structured_document=structured,
        form_mappings=state.form_mappings,
        filled_form=state.filled_form,
        missing_fields=state.missing_fields,
        conflicts=state.conflicts,
        failed_chunks=list(state.failed_chunks),
        manual_review_items=state.manual_review_items,
        quality_control=QualityControlOutput(
            all_pages_processed=bool(qc and qc.all_pages_processed),
            all_chunks_processed=bool(qc and qc.all_chunks_processed),
            all_values_have_evidence=bool(qc and qc.all_values_have_evidence),
            conflicts_resolved=bool(qc and qc.conflicts_resolved),
            form_mapping_verified=bool(qc and qc.form_mapping_verified),
            information_loss_detected=bool(qc and qc.information_loss_detected),
        ),
        final_reasoning=state.final_reasoning,
        audit_log=state.audit_log,
        metrics=state.metrics,
    )
