"""K. Conflict Detection and Resolution Agent — deterministic detection, LLM adjudication.

Detection is mechanical and happens first: group the fields by canonical name,
and flag any group holding more than one materially different value. "Materially
different" uses the same normalisation as everywhere else, so "02 Nov 2020" and
"2020-11-02" do not register as a conflict, and neither do two rows that differ
only in spacing.

Two things detection must NOT treat as conflicts, because they are normal
document features rather than disagreements:

- **Repeating entities.** Twelve equipment rows each with a different
  `serial_number` is a table, not twelve conflicting claims about one serial
  number. A group whose members carry distinct `table_context` values is a
  repeating entity and is left alone.
- **Null against a value.** A field found blank on one page and filled on
  another is not a contradiction; the filled one is simply where the data is.

What survives is adjudicated by the LLM against explicit criteria — clarity,
specificity, recency, context, and only then confidence as a tie-break. The
business rule that confidence must never be the sole criterion is enforced
here in code as well as in the prompt: a resolution that cites confidence
alone is downgraded to MANUAL_REVIEW_REQUIRED regardless of what the model said.

Nothing is ever discarded. Resolution picks a preferred candidate; every other
candidate stays in the conflict record and in `normalized_fields`.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from pydantic import BaseModel, Field

from app.graph.errors import ErrorType
from app.graph.llm import call_structured
from app.graph.nodes.helpers import audit, err, normalise_for_match, truncate
from app.graph.prompts import CONFLICT_SYSTEM, conflict_user
from app.graph.schemas import (
    ConflictCandidate,
    ConflictResolution,
    ExtractedField,
    ExtractionStatus,
    FieldConflict,
    ManualReviewItem,
)
from app.graph.state import GraphState

log = logging.getLogger(__name__)

NODE = "conflict_resolution"

#: Beyond this many conflicts in one document the prompt stops being useful;
#: the remainder go straight to manual review rather than being guessed at.
_MAX_ADJUDICATED = 40


class _Resolution(BaseModel):
    normalized_field_name: str
    chosen_field_uid: str | None = None
    resolution: str = "UNRESOLVED"
    reason: str = ""
    criteria_used: list[str] = Field(default_factory=list)


class _ResolutionSet(BaseModel):
    resolutions: list[_Resolution] = Field(default_factory=list)


def _is_repeating_entity(group: list[ExtractedField]) -> bool:
    """Distinct table_context on most members means these are table rows."""
    contexts = [normalise_for_match(f.table_context) for f in group if f.table_context]
    if len(contexts) < 2:
        return False
    return len(set(contexts)) >= max(2, int(len(group) * 0.6))


def _is_per_page_attribute(group: list[ExtractedField], content_pages: set[int]) -> bool:
    """One occurrence on every page — a repeating page header, not a clash.

    A multi-page report restates its header on each continuation page, and
    those restatements legitimately differ: page "1 of 2" then "2 of 2", an
    accession number that increments per page, a reported date amended on the
    later page. MEASURED on a real cytology report: nine header fields each
    appeared once per page and four of them differed, producing four
    "conflicts" in a document that contradicts nothing.

    The distinguishing signal is coverage. A per-page attribute appears exactly
    once on EVERY page. A genuine disagreement — a report number stated on
    page 1 and restated differently in an amendment on page 3 of 3 — appears on
    some pages and not others, and still registers as a conflict.
    """
    if len(content_pages) < 2:
        return False
    pages = [f.page_number for f in group if f.page_number is not None]
    if len(pages) != len(set(pages)):
        return False                       # twice on one page: not a header
    return set(pages) == content_pages


def detect_and_resolve_conflicts(state: GraphState) -> dict:
    metrics = state.metrics.model_copy(deep=True)
    fields = [f.model_copy(deep=True) for f in state.active_fields()]
    entries = []
    errors = []

    by_name: dict[str, list[ExtractedField]] = defaultdict(list)
    for f in fields:
        by_name[f.normalized_field_name].append(f)

    # Pages that actually produced fields — the denominator for deciding
    # whether something appears on "every page".
    content_pages = {f.page_number for f in fields if f.page_number is not None}

    conflicts: list[FieldConflict] = []
    for name, group in by_name.items():
        with_values = [f for f in group if f.value is not None]
        if len(with_values) < 2:
            continue
        distinct = {normalise_for_match(f.normalized_value or f.value) for f in with_values}
        if len(distinct) < 2:
            continue
        if _is_repeating_entity(with_values):
            entries += audit(
                NODE, "repeating_entity",
                f"{name!r} has {len(with_values)} row-scoped values — treated as a table, not a conflict",
            )
            continue

        if _is_per_page_attribute(with_values, content_pages):
            # Tag each occurrence with the page it belongs to, so the review UI
            # groups them as one-per-page records instead of showing a bare
            # list of values that look like they disagree.
            for f in with_values:
                if not f.table_context:
                    f.table_context = f"Page {f.page_number}"
            entries += audit(
                NODE, "per_page_attribute",
                f"{name!r} appears once on every page with differing values — "
                f"treated as a per-page header field, not a conflict",
            )
            continue

        conflicts.append(FieldConflict(
            normalized_field_name=name,
            candidates=[
                ConflictCandidate(
                    field_uid=f.field_uid,
                    value=f.normalized_value if f.normalized_value is not None else f.value,
                    page_number=f.page_number,
                    chunk_id=f.chunk_id,
                    section_name=f.section_name,
                    confidence_score=f.confidence_score,
                    evidence=truncate(f.exact_source_evidence, 240),
                )
                for f in with_values
            ],
        ))

    if not conflicts:
        entries += audit(NODE, "no_conflicts", f"{len(by_name)} distinct field name(s) checked")
        return {
            "current_node": NODE,
            "conflicts": [],
            "normalized_fields": fields,
            "audit_log": entries,
        }

    # ---- adjudication -----------------------------------------------------
    adjudicate = conflicts[:_MAX_ADJUDICATED]
    overflow = conflicts[_MAX_ADJUDICATED:]

    payload = [
        {
            "normalized_field_name": c.normalized_field_name,
            "candidates": [
                {
                    "field_uid": cand.field_uid,
                    "value": cand.value,
                    "page": cand.page_number,
                    "section": cand.section_name,
                    "confidence": cand.confidence_score,
                    "evidence": cand.evidence,
                }
                for cand in c.candidates
            ],
        }
        for c in adjudicate
    ]

    outcome = call_structured(
        node=NODE, system=CONFLICT_SYSTEM, user_text=conflict_user(payload),
        output_model=_ResolutionSet, max_attempts=2, on_metric=metrics.record,
    )

    resolutions: dict[str, _Resolution] = {}
    if outcome.ok:
        parsed: _ResolutionSet = outcome.parsed  # type: ignore[assignment]
        resolutions = {r.normalized_field_name: r for r in parsed.resolutions}
    else:
        errors += err(
            outcome.error_type or ErrorType.LLM_UNAVAILABLE,
            f"conflict adjudication failed: {outcome.error_message}", NODE,
            retry_count=outcome.attempts,
            recovery_action="all conflicts escalated to manual review",
            resolution_status="RECOVERED",
        )
        entries += audit(NODE, "adjudication_unavailable",
                         "every conflict escalated", level="warning")

    review_items: list[ManualReviewItem] = []
    resolved = escalated = 0

    for conflict in conflicts:
        r = resolutions.get(conflict.normalized_field_name)
        valid_uids = {c.field_uid for c in conflict.candidates}

        if r is None:
            conflict.resolution = ConflictResolution.MANUAL_REVIEW_REQUIRED
            conflict.reason = "no adjudication returned for this conflict"
        elif r.chosen_field_uid and r.chosen_field_uid not in valid_uids:
            # A chosen uid that is not one of the candidates is a fabricated
            # answer; it must not silently become the winner.
            conflict.resolution = ConflictResolution.MANUAL_REVIEW_REQUIRED
            conflict.reason = f"adjudicator returned an unknown candidate id {r.chosen_field_uid!r}"
            errors += err(
                ErrorType.CONFLICTING_FIELD, conflict.reason, NODE,
                recovery_action="escalated to manual review", resolution_status="OPEN",
            )
        elif r.resolution == "RESOLVED" and r.chosen_field_uid:
            criteria = [c for c in r.criteria_used if c]
            if criteria and set(criteria) <= {"confidence"}:
                # Business rule: confidence alone is never sufficient.
                conflict.resolution = ConflictResolution.MANUAL_REVIEW_REQUIRED
                conflict.reason = (
                    "resolution rested on confidence alone, which is not a sufficient criterion"
                )
            else:
                conflict.resolution = ConflictResolution.RESOLVED
                conflict.chosen_field_uid = r.chosen_field_uid
                conflict.reason = r.reason
                conflict.criteria_used = criteria
                resolved += 1
        else:
            conflict.resolution = ConflictResolution.MANUAL_REVIEW_REQUIRED
            conflict.reason = r.reason or "no candidate is better supported"

        if conflict.resolution != ConflictResolution.RESOLVED:
            escalated += 1
            review_items.append(ManualReviewItem(
                kind="conflict",
                reference=conflict.normalized_field_name,
                summary=f"{len(conflict.candidates)} differing values: {conflict.reason}",
                page_number=conflict.candidates[0].page_number if conflict.candidates else None,
                candidates=[
                    {"field_uid": c.field_uid, "value": c.value, "page": c.page_number,
                     "evidence": c.evidence}
                    for c in conflict.candidates
                ],
                suggested_action="choose the correct value, or mark the field not applicable",
            ))

    for conflict in overflow:
        conflict.resolution = ConflictResolution.MANUAL_REVIEW_REQUIRED
        conflict.reason = f"beyond the {_MAX_ADJUDICATED}-conflict adjudication limit"

    # ---- stamp the fields --------------------------------------------------
    chosen = {c.chosen_field_uid for c in conflicts if c.chosen_field_uid}
    conflicted_names = {
        c.normalized_field_name for c in conflicts
        if c.resolution != ConflictResolution.RESOLVED
    }
    for f in fields:
        if f.normalized_field_name in conflicted_names:
            f.extraction_status = ExtractionStatus.CONFLICTED
        elif f.field_uid in chosen:
            f.notes = (f.notes + " | preferred value for a resolved conflict").strip(" |")

    entries += audit(
        NODE, "conflicts_processed",
        f"{len(conflicts)} conflict(s): {resolved} resolved, {escalated} escalated",
        level="warning" if escalated else "info",
    )

    return {
        "current_node": NODE,
        "conflicts": conflicts,
        "normalized_fields": fields,
        "manual_review_items": list(state.manual_review_items) + review_items,
        "metrics": metrics,
        "audit_log": entries,
        "errors": errors,
    }
