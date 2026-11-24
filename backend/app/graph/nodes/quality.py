"""O. Final Quality-Control Agent — deterministic, no LLM.

The last gate before a verdict. It re-derives every claim the result is about
to make, from the state, rather than trusting the flags earlier nodes set —
the point of a final check is to catch the case where an upstream node got its
own bookkeeping wrong.

Issues are sorted into two buckets, and the distinction is the whole design:

- **critical_issues** block a successful verdict outright. Unaccounted pages,
  chunks that never ran, values in the form with no evidence behind them.
  These mean the result is not trustworthy, not merely imperfect.
- **issues** are things a reader must be told but which do not invalidate the
  result: a page that legitimately held nothing, a field left uncertain, an
  unresolved conflict that has been correctly escalated rather than guessed.

A document can be perfectly processed and still have plenty in the second
bucket. Conflating the two would make the system either uselessly strict or
quietly dishonest.
"""
from __future__ import annotations

import logging

from app.graph.config import get_graph_settings
from app.graph.errors import ErrorType, is_critical
from app.graph.nodes.helpers import audit, err, is_placeholder
from app.graph.schemas import (
    ChunkStatus,
    ConflictResolution,
    EvidenceStatus,
    ExtractionStatus,
    MappingStatus,
    QualityControlReport,
)
from app.graph.state import GraphState

log = logging.getLogger(__name__)

NODE = "quality_control"


def _walk(value, path: str = ""):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _walk(v, f"{path}.{k}" if path else k)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _walk(v, f"{path}[{i}]")
    else:
        yield path, value


def run_quality_control(state: GraphState) -> dict:
    settings = get_graph_settings()
    report = QualityControlReport()
    entries = []
    errors = []
    issues: list[str] = []
    critical: list[str] = []

    fields = state.active_fields()
    completeness = state.completeness_report

    # ---- 1. page and chunk coverage, re-derived --------------------------
    all_pages = set(state.page_text)
    covered: set[int] = set()
    for c in state.chunks:
        covered.update(c.page_numbers)
    uncovered = sorted(all_pages - covered)

    unfinished = [
        c.chunk_id for c in state.chunks
        if c.status in (ChunkStatus.PENDING, ChunkStatus.IN_PROGRESS)
    ]
    failed = [c.chunk_id for c in state.chunks if c.status == ChunkStatus.FAILED]

    report.all_pages_processed = not uncovered
    report.all_chunks_processed = not unfinished and not failed

    if uncovered:
        critical.append(f"{len(uncovered)} page(s) were never included in any chunk: {uncovered[:15]}")
    if unfinished:
        critical.append(f"{len(unfinished)} chunk(s) never ran: {unfinished[:15]}")
    if failed:
        issues.append(f"{len(failed)} chunk(s) failed permanently and are reported as such")

    # ---- 2. evidence ------------------------------------------------------
    evidence_by_uid = {r.field_uid: r for r in state.evidence_validation_results}
    unsupported = [
        f for f in fields
        if (ev := evidence_by_uid.get(f.field_uid)) is None
        or ev.status in (EvidenceStatus.UNSUPPORTED, EvidenceStatus.NO_EVIDENCE)
    ]
    report.all_values_have_evidence = not unsupported
    if unsupported:
        issues.append(f"{len(unsupported)} extracted field(s) lack usable evidence and are marked uncertain")

    # ---- 3. schema validity ------------------------------------------------
    schema_ok = True
    for f in fields:
        if not f.field_name or not f.normalized_field_name:
            schema_ok = False
            critical.append(f"a field is missing its name (uid {f.field_uid})")
            break
        if f.page_number is not None and f.page_number not in all_pages:
            schema_ok = False
            critical.append(f"field {f.normalized_field_name!r} cites page {f.page_number}, which does not exist")
            break
    report.schema_valid = schema_ok

    # ---- 4. conflicts ------------------------------------------------------
    unresolved = [c for c in state.conflicts if c.resolution != ConflictResolution.RESOLVED]
    report.conflicts_resolved = not unresolved
    if unresolved:
        issues.append(
            f"{len(unresolved)} conflict(s) unresolved and escalated for review "
            f"(every candidate preserved)"
        )
    # Losing a candidate is a real defect: it means information was discarded.
    for c in state.conflicts:
        if len(c.candidates) < 2:
            critical.append(f"conflict on {c.normalized_field_name!r} has fewer than two candidates recorded")

    # ---- 5. the filled form -------------------------------------------------
    mapping_by_target = {m.target_field: m for m in state.form_mappings}
    supported_values = {
        str(f.normalized_value if f.normalized_value is not None else f.value)
        for f in fields if f.value is not None
    }

    unsupported_in_form: list[str] = []
    for path, value in _walk(state.filled_form):
        if value is None or is_placeholder(value):
            continue
        if str(value) not in supported_values:
            unsupported_in_form.append(path)

    if unsupported_in_form:
        # A value in the form that no extracted field supplies is either a
        # transformation bug or an invented value. Either way it must not ship.
        critical.append(
            f"{len(unsupported_in_form)} value(s) in the filled form do not correspond to any "
            f"extracted field: {unsupported_in_form[:10]}"
        )
        errors += err(
            ErrorType.UNSUPPORTED_MAPPING_VALUE,
            f"unsupported values present in the filled form: {unsupported_in_form[:10]}",
            NODE, recovery_action="blocks a successful verdict",
        )

    conflicted_filled = [
        t for t, m in mapping_by_target.items()
        if m.mapping_status == MappingStatus.CONFLICTED and any(
            p == t or p.startswith(t.replace("[]", "")) for p, _ in _walk(state.filled_form)
        )
    ]
    if conflicted_filled:
        critical.append(f"conflicted field(s) were filled automatically: {conflicted_filled[:10]}")

    mapped = [m for m in state.form_mappings if m.mapping_status == MappingStatus.MAPPED]
    report.form_mapping_verified = bool(state.form_mappings) and not unsupported_in_form and not conflicted_filled
    if state.target_form_schema and state.target_form_schema.fields:
        expected = {f.name for f in state.target_form_schema.fields}
        reported = {m.target_field for m in state.form_mappings}
        absent = sorted(expected - reported)
        if absent:
            issues.append(f"{len(absent)} target field(s) have no mapping entry at all: {absent[:10]}")
            errors += err(
                ErrorType.MISSING_TARGET_FORM_FIELD,
                f"target fields with no mapping entry: {absent[:20]}", NODE,
                recovery_action="reported as missing",
                resolution_status="RECOVERED",
            )
        required_missing = [
            f.name for f in state.target_form_schema.fields
            if f.required and mapping_by_target.get(f.name, None) is not None
            and mapping_by_target[f.name].mapping_status != MappingStatus.MAPPED
        ]
        report.required_fields_present = not required_missing
        if required_missing:
            issues.append(f"{len(required_missing)} required target field(s) unfilled: {required_missing[:10]}")
    else:
        report.required_fields_present = True

    # ---- 6. information loss -----------------------------------------------
    loss = bool(uncovered or unfinished)
    if completeness is not None and completeness.information_loss_detected:
        loss = True
    report.information_loss_detected = loss
    if loss:
        critical.append("information loss detected: pages or chunks are unaccounted for")

    # ---- 7. upstream critical errors ----------------------------------------
    upstream = [e for e in state.errors if is_critical(e) and e.resolution_status == "OPEN"]
    for e in upstream:
        critical.append(f"unresolved {e.error_type} from {e.node}: {e.error_message[:160]}")

    # ---- 8. confidence profile ----------------------------------------------
    if fields:
        verified = sum(1 for f in fields if f.extraction_status == ExtractionStatus.VERIFIED)
        ratio_uncertain = 1.0 - (verified / len(fields))
        if ratio_uncertain > settings.low_confidence_ratio_limit:
            issues.append(
                f"{ratio_uncertain:.0%} of fields are not fully verified "
                f"(limit {settings.low_confidence_ratio_limit:.0%})"
            )
    else:
        issues.append("no fields were extracted from this document")

    report.issues = issues
    report.critical_issues = critical

    entries += audit(
        NODE, "quality_controlled",
        f"{len(critical)} critical issue(s), {len(issues)} advisory issue(s)",
        level="error" if critical else ("warning" if issues else "info"),
    )

    return {
        "current_node": NODE,
        "quality_control_report": report,
        "audit_log": entries,
        "errors": errors,
    }
