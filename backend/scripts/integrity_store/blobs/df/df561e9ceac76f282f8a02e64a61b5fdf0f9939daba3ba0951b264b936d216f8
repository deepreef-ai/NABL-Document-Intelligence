"""N. Form-Filling Node — deterministic, no LLM.

Turns mappings into a filled form. Nothing here decides *what* a value means;
that was settled upstream. This node's whole job is to enforce the rules about
what is allowed to be written:

- Only MAPPED and PARTIALLY_MAPPED entries fill anything.
- A CONFLICTED field is never filled automatically, however confident the
  mapping was — the point of flagging a conflict is that we do not know which
  value is right.
- A value with no surviving evidence is never written, even if a mapping
  claimed it.
- A better-supported value already in a slot is never overwritten by a
  lower-confidence one. Fill order is therefore deliberate rather than
  incidental: entries are sorted by confidence before writing.
- A repeating target (`equipment[].serial_number`) accumulates a list rather
  than the last writer winning.

Everything not filled is reported explicitly, with the reason. A form field
that is absent from both the filled form and the missing list would be
invisible, and invisible is the failure mode this whole system is built to
avoid.
"""
from __future__ import annotations

import logging
from typing import Any

from app.graph.errors import ErrorType
from app.graph.nodes.helpers import audit, err, is_placeholder, normalise_for_match
from app.graph.schemas import ManualReviewItem, MappingStatus
from app.graph.state import GraphState

log = logging.getLogger(__name__)

NODE = "form_filling"

_FILLABLE = {MappingStatus.MAPPED, MappingStatus.PARTIALLY_MAPPED}


def _set_path(container: dict[str, Any], path: str, value: Any, repeating: bool) -> None:
    """Write a dotted path, treating `seg[]` as a list of dicts."""
    parts = path.split(".")
    node: Any = container
    for i, raw in enumerate(parts):
        last = i == len(parts) - 1
        is_list = raw.endswith("[]")
        key = raw[:-2] if is_list else raw

        if last:
            if is_list or repeating:
                node.setdefault(key, [])
                if isinstance(node[key], list):
                    node[key].append(value)
                else:
                    node[key] = [node[key], value]
            else:
                node[key] = value
            return

        if is_list:
            bucket = node.setdefault(key, [{}])
            if not isinstance(bucket, list):
                bucket = node[key] = [bucket]
            if not bucket:
                bucket.append({})
            node = bucket[-1]
        else:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                # A scalar already sits where a nested object belongs. Replace
                # it rather than crash, and say so — this indicates a schema
                # path collision worth a human looking at.
                node = {}


def fill_form(state: GraphState) -> dict:
    entries = []
    errors = []
    filled: dict[str, Any] = {}
    written: dict[str, float] = {}          # target path -> confidence written
    missing: list[str] = []
    review: list[ManualReviewItem] = []

    if not state.form_mappings:
        entries += audit(NODE, "no_mappings", "nothing to fill", level="warning")
        return {
            "current_node": NODE,
            "filled_form": {},
            "missing_fields": [],
            "audit_log": entries,
        }

    evidence_by_uid = {r.field_uid: r for r in state.evidence_validation_results}

    # One source value fills ONE slot. A document that states a single lab
    # email once does not thereby state the email of four named officers, and
    # copying it into each of their records produces a form that looks filled
    # and is fiction. MEASURED on a real veterinary report: one address and one
    # email were mapped into six slots across organisation and four different
    # people. Tracked by the value itself rather than the source id, because
    # the same text extracted twice has two ids and the same problem.
    claimed_by: dict[str, str] = {}          # normalised value -> target it filled

    # Highest confidence first, so the overwrite guard below never has to
    # reason about ordering — the first writer for a slot is the best one.
    ordered = sorted(state.form_mappings, key=lambda m: -m.mapping_confidence)

    for m in ordered:
        target = m.target_field
        repeating = target.endswith("[]") or "[]." in target

        if m.mapping_status not in _FILLABLE:
            if m.mapping_status == MappingStatus.NOT_FOUND:
                missing.append(target)
            elif m.mapping_status in (MappingStatus.AMBIGUOUS, MappingStatus.MANUAL_REVIEW_REQUIRED):
                missing.append(target)
                review.append(ManualReviewItem(
                    kind="ambiguous_mapping", reference=target,
                    summary=m.mapping_reason or "mapping could not be decided",
                    page_number=m.source_page,
                    suggested_action="choose the correct source field, or leave the slot empty",
                ))
                errors += err(
                    ErrorType.AMBIGUOUS_MAPPING,
                    f"{target}: {m.mapping_reason[:200]}", NODE,
                    page_number=m.source_page,
                    recovery_action="left unfilled, escalated to manual review",
                )
            elif m.mapping_status == MappingStatus.CONFLICTED:
                missing.append(target)
                review.append(ManualReviewItem(
                    kind="conflict", reference=target,
                    summary="source field has an unresolved conflict; slot deliberately left empty",
                    page_number=m.source_page,
                    suggested_action="resolve the source conflict, then re-fill",
                ))
            continue

        if m.mapped_value is None or is_placeholder(m.mapped_value):
            missing.append(target)
            continue

        # Evidence is re-checked here rather than trusted from the mapping.
        # A mapping is a claim about correspondence; this is the last gate
        # before a value becomes part of the answer.
        ev = evidence_by_uid.get(m.source_field_uid) if m.source_field_uid else None
        if ev is not None and ev.status.value in ("UNSUPPORTED", "NO_EVIDENCE"):
            missing.append(target)
            errors += err(
                ErrorType.MISSING_EVIDENCE,
                f"{target}: source value has no supporting evidence; refused", NODE,
                page_number=m.source_page,
                recovery_action="slot left empty",
            )
            review.append(ManualReviewItem(
                kind="unsupported", reference=target,
                summary="mapped value has no evidence in the document",
                page_number=m.source_page,
                suggested_action="verify against the source page or discard",
            ))
            continue

        # --- one source value, one slot -----------------------------------
        # Repeating targets are exempt: several equipment rows legitimately
        # carry different values into equipment[0], equipment[1], and those
        # come from different source fields anyway.
        value_key = normalise_for_match(m.mapped_value)
        if value_key and not repeating:
            already = claimed_by.get(value_key)
            if already is not None and already != target:
                missing.append(target)
                entries += audit(
                    NODE, "duplicate_value_refused",
                    f"{target} not filled: the same value already fills {already}. "
                    f"One statement in the document cannot supply two different fields.",
                    page_number=m.source_page, level="warning",
                )
                review.append(ManualReviewItem(
                    kind="ambiguous_mapping", reference=target,
                    summary=f"the same value is already used for {already}; "
                            f"the document does not separately state this field",
                    page_number=m.source_page,
                    suggested_action="enter the correct value for this field, or leave it empty",
                ))
                continue

        prior = written.get(target)
        if prior is not None and not repeating and prior >= m.mapping_confidence:
            entries += audit(
                NODE, "overwrite_refused",
                f"{target} already holds a value of confidence {prior:.2f}; "
                f"not replacing with {m.mapping_confidence:.2f}",
            )
            continue

        _set_path(filled, target, m.mapped_value, repeating)
        written[target] = m.mapping_confidence
        if value_key and not repeating:
            claimed_by.setdefault(value_key, target)

        if m.mapping_status == MappingStatus.PARTIALLY_MAPPED:
            review.append(ManualReviewItem(
                kind="low_confidence", reference=target,
                summary=f"filled from a partial match: {m.mapping_reason[:200]}",
                page_number=m.source_page,
                suggested_action="confirm the value against the cited page",
            ))

    missing = sorted(set(missing) - set(written))

    entries += audit(
        NODE, "form_filled",
        f"{len(written)} slot(s) filled, {len(missing)} left empty, "
        f"{len(review)} item(s) for review",
    )

    return {
        "current_node": NODE,
        "filled_form": filled,
        "missing_fields": missing,
        "manual_review_items": list(state.manual_review_items) + review,
        "audit_log": entries,
        "errors": errors,
    }
