"""M. Dynamic Form-Mapping Agent — LLM-backed, with a deterministic pre-pass.

Discovers the target form's fields and works out which extracted field supplies
each one. The form schema is genuinely dynamic: it can be supplied inline by the
caller, or introspected from a Pydantic model (which is how the NABL forms
arrive), and neither the field list nor the document's vocabulary is known in
advance.

Two passes, cheapest first:

1. **Exact and near-exact name matches** are settled deterministically. If the
   document has `gst_number` and the form wants `organisation.gst_number`, no
   model needs to be consulted, and spending a call on it would add latency and
   a chance of being talked out of a correct answer.
2. **Everything else** goes to the LLM, which sees only the still-unmapped
   target fields and the candidate source fields. Its job is semantic
   correspondence, and it is told explicitly that string similarity is not
   correspondence.

Only fields that survived evidence validation are offered as candidates. A
value the document does not support cannot map to anything, and filtering here
rather than at fill time means the model is never tempted by it.
"""
from __future__ import annotations

import logging
from typing import Any, get_args, get_origin

from pydantic import BaseModel, Field

from app.graph.config import get_graph_settings
from app.graph.errors import ErrorType
from app.graph.llm import call_structured
from app.graph.nodes.helpers import audit, err, normalise_for_match, similarity, truncate
from app.graph.prompts import MAPPING_SYSTEM, mapping_user
from app.graph.schemas import (
    ExtractedField,
    ExtractionStatus,
    FormMapping,
    MappingStatus,
    TargetFormField,
    TargetFormSchema,
)
from app.graph.state import GraphState

log = logging.getLogger(__name__)

NODE = "form_mapping"

#: Above this, a name match is taken as correspondence without asking.
_EXACT_MATCH_THRESHOLD = 0.97


class _RawMapping(BaseModel):
    source_field: str = ""
    target_field: str = ""
    source_field_uid: str = ""
    mapping_confidence: float = 0.0
    mapping_reason: str = ""
    mapping_status: str = "NOT_FOUND"


class _MappingSet(BaseModel):
    mappings: list[_RawMapping] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Schema discovery
# --------------------------------------------------------------------------


def _flatten_model(model: type[BaseModel], prefix: str = "", depth: int = 0) -> list[TargetFormField]:
    """Introspect a Pydantic model into a flat list of dotted field paths.

    Repeating sections become `equipment[].serial_number` so the mapping agent
    can see that many values are expected, rather than treating the list as one
    scalar slot and mapping a single row into it.
    """
    if depth > 4:
        return []
    out: list[TargetFormField] = []
    for name, info in model.model_fields.items():
        path = f"{prefix}{name}"
        annotation = info.annotation
        origin = get_origin(annotation)
        args = get_args(annotation)

        inner = None
        repeating = False
        if origin in (list, set, tuple) and args:
            inner, repeating = args[0], True
        elif origin is not None and args:
            # Optional[X] / Union — take the first non-None member.
            inner = next((a for a in args if a is not type(None)), None)  # noqa: E721
        else:
            inner = annotation

        if isinstance(inner, type) and issubclass(inner, BaseModel):
            child_prefix = f"{path}[]." if repeating else f"{path}."
            out.extend(_flatten_model(inner, child_prefix, depth + 1))
            continue

        out.append(TargetFormField(
            name=f"{path}[]" if repeating else path,
            data_type=getattr(inner, "__name__", "string"),
            required=info.is_required(),
            description=(info.description or "")[:200],
            repeating=repeating,
            parent=prefix.rstrip(".[]") or None,
        ))
    return out


def discover_form_schema(state: GraphState) -> TargetFormSchema | None:
    """Use whatever the caller gave us; introspect a named NABL form otherwise."""
    if state.target_form_schema and state.target_form_schema.fields:
        return state.target_form_schema

    form_id = (state.target_form_schema.form_id if state.target_form_schema else "") or ""
    if not form_id:
        return None
    try:
        from app.schemas.forms import FORM_MODEL, NablFormType

        model = FORM_MODEL[NablFormType(form_id)]
    except Exception:  # noqa: BLE001 — an unknown form id is not an error here
        return None
    return TargetFormSchema(form_id=form_id, fields=_flatten_model(model), source="pydantic")


# --------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------


def _candidates(state: GraphState) -> list[ExtractedField]:
    """Fields good enough to be offered to a form."""
    out = []
    for f in state.active_fields():
        if f.value is None:
            continue
        if f.extraction_status == ExtractionStatus.CONFLICTED:
            continue  # offered separately, as a CONFLICTED mapping
        if f.evidence_status is not None and f.evidence_status.value in ("UNSUPPORTED", "NO_EVIDENCE"):
            continue
        out.append(f)
    return out


def map_to_form(state: GraphState) -> dict:
    settings = get_graph_settings()
    metrics = state.metrics.model_copy(deep=True)
    entries = []
    errors = []

    schema = discover_form_schema(state)
    if schema is None or not schema.fields:
        entries += audit(
            NODE, "no_target_form",
            "no target form schema supplied or discoverable; extraction results returned unmapped",
            level="warning",
        )
        return {
            "current_node": NODE,
            "form_mappings": [],
            "audit_log": entries,
        }

    sources = _candidates(state)
    by_uid = {f.field_uid: f for f in sources}
    mappings: list[FormMapping] = []
    unmapped_targets: list[TargetFormField] = []

    # ---- pass 1: deterministic name correspondence ------------------------
    source_by_norm: dict[str, list[ExtractedField]] = {}
    for f in sources:
        source_by_norm.setdefault(normalise_for_match(f.normalized_field_name), []).append(f)

    for target in schema.fields:
        leaf = target.name.rsplit(".", 1)[-1].replace("[]", "")
        key = normalise_for_match(leaf)
        exact = source_by_norm.get(key)
        if exact and len({normalise_for_match(f.value) for f in exact}) == 1:
            best = max(exact, key=lambda f: f.confidence_score)
            mappings.append(FormMapping(
                source_field=best.normalized_field_name,
                target_field=target.name,
                mapped_value=best.normalized_value if best.normalized_value is not None else best.value,
                mapping_confidence=1.0,
                source_page=best.page_number,
                source_evidence=truncate(best.exact_source_evidence, 300),
                mapping_reason="exact field-name correspondence",
                mapping_status=MappingStatus.MAPPED,
                source_field_uid=best.field_uid,
            ))
        else:
            unmapped_targets.append(target)

    entries += audit(
        NODE, "deterministic_pass",
        f"{len(mappings)} of {len(schema.fields)} target field(s) matched by name",
    )

    # ---- pass 2: semantic mapping ----------------------------------------
    if unmapped_targets and sources:
        target_payload = [
            {"name": t.name, "type": t.data_type, "required": t.required,
             "description": t.description, "repeating": t.repeating}
            for t in unmapped_targets[:200]
        ]
        source_payload = [
            {"field_uid": f.field_uid, "name": f.normalized_field_name,
             "original_name": f.field_name,
             "value": truncate(str(f.value), 120), "type": f.data_type,
             "section": f.section_name, "page": f.page_number}
            for f in sources[:300]
        ]

        outcome = call_structured(
            node=NODE, system=MAPPING_SYSTEM,
            user_text=mapping_user(target_payload, source_payload),
            output_model=_MappingSet, max_attempts=2, on_metric=metrics.record,
        )

        if not outcome.ok:
            errors += err(
                outcome.error_type or ErrorType.LLM_UNAVAILABLE,
                f"semantic form mapping failed: {outcome.error_message}", NODE,
                retry_count=outcome.attempts,
                recovery_action="unmatched target fields reported as NOT_FOUND",
                resolution_status="RECOVERED",
            )
            entries += audit(NODE, "semantic_mapping_unavailable",
                             f"{len(unmapped_targets)} target field(s) left unmapped", level="warning")
            for t in unmapped_targets:
                mappings.append(FormMapping(
                    source_field="", target_field=t.name,
                    mapping_reason="semantic mapping unavailable",
                    mapping_status=MappingStatus.NOT_FOUND,
                ))
        else:
            parsed: _MappingSet = outcome.parsed  # type: ignore[assignment]
            returned = {m.target_field for m in parsed.mappings}
            known_targets = {t.name for t in unmapped_targets}

            for raw in parsed.mappings:
                if raw.target_field not in known_targets:
                    continue  # a target we did not ask about; ignore rather than invent a slot
                try:
                    status = MappingStatus(raw.mapping_status)
                except ValueError:
                    status = MappingStatus.MANUAL_REVIEW_REQUIRED

                src = by_uid.get(raw.source_field_uid)
                if src is None and raw.source_field:
                    matches = [
                        f for f in sources
                        if similarity(f.normalized_field_name, raw.source_field) >= _EXACT_MATCH_THRESHOLD
                    ]
                    src = matches[0] if len(matches) == 1 else None

                if status in (MappingStatus.MAPPED, MappingStatus.PARTIALLY_MAPPED) and src is None:
                    # Claimed a mapping but named a source we do not have. That
                    # is a fabricated value in the making, so it is refused.
                    status = MappingStatus.MANUAL_REVIEW_REQUIRED
                    raw.mapping_reason = (
                        f"{raw.mapping_reason} | source field {raw.source_field!r} not found in "
                        f"the verified set; mapping refused"
                    ).strip(" |")
                    errors += err(
                        ErrorType.UNSUPPORTED_MAPPING_VALUE,
                        f"mapping to {raw.target_field!r} named an unknown source field",
                        NODE, recovery_action="escalated to manual review",
                    )

                if status == MappingStatus.MAPPED and raw.mapping_confidence < settings.mapping_confidence_threshold:
                    status = MappingStatus.PARTIALLY_MAPPED
                    raw.mapping_reason = (
                        f"{raw.mapping_reason} | confidence below "
                        f"{settings.mapping_confidence_threshold}"
                    ).strip(" |")

                mappings.append(FormMapping(
                    source_field=src.normalized_field_name if src else raw.source_field,
                    target_field=raw.target_field,
                    mapped_value=(
                        (src.normalized_value if src.normalized_value is not None else src.value)
                        if src and status in (MappingStatus.MAPPED, MappingStatus.PARTIALLY_MAPPED)
                        else None
                    ),
                    mapping_confidence=raw.mapping_confidence,
                    source_page=src.page_number if src else None,
                    source_evidence=truncate(src.exact_source_evidence, 300) if src else "",
                    mapping_reason=raw.mapping_reason,
                    mapping_status=status,
                    source_field_uid=src.field_uid if src else "",
                ))

            # Every target field must appear in the output, including the ones
            # nothing maps to — a silently absent target is indistinguishable
            # from one nobody looked at.
            for t in unmapped_targets:
                if t.name not in returned:
                    mappings.append(FormMapping(
                        source_field="", target_field=t.name,
                        mapping_reason="no corresponding field found in the document",
                        mapping_status=MappingStatus.NOT_FOUND,
                    ))

    # ---- conflicted fields surface as conflicted mappings ------------------
    conflicted_names = {
        c.normalized_field_name for c in state.conflicts
        if c.resolution.value != "RESOLVED"
    }
    for m in mappings:
        if m.source_field and normalise_for_match(m.source_field) in {
            normalise_for_match(n) for n in conflicted_names
        }:
            m.mapping_status = MappingStatus.CONFLICTED
            m.mapped_value = None
            m.mapping_reason = (m.mapping_reason + " | source field has an unresolved conflict").strip(" |")

    counts: dict[str, int] = {}
    for m in mappings:
        counts[m.mapping_status.value] = counts.get(m.mapping_status.value, 0) + 1

    entries += audit(
        NODE, "mapped",
        f"{len(mappings)} mapping(s) for {len(schema.fields)} target field(s): "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
    )

    return {
        "current_node": NODE,
        "target_form_schema": schema,
        "form_mappings": mappings,
        "metrics": metrics,
        "audit_log": entries,
        "errors": errors,
    }
