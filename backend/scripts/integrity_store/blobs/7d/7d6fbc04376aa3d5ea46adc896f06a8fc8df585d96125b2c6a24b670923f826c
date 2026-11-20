"""J. Field Normalisation and Merging Agent — deterministic first, LLM second.

Two distinct jobs that are easy to conflate:

- **Normalising a value** is safe and mechanical: fold whitespace, unify dash
  characters, canonicalise an obvious date. Done deterministically, and the
  original is always retained alongside.
- **Merging two field NAMES** is a semantic judgement about whether two labels
  denote the same attribute of the same entity, and it is the single most
  destructive operation in this pipeline if it is wrong. It is done cautiously,
  with the LLM proposing and deterministic rules vetoing.

The guard rails on merging:

- Exact normalised-name matches merge without asking anyone.
- Everything else is *proposed* by the LLM and then vetoed if the members
  disagree on data type, or come from different sections while holding
  different values — the classic "equipment serial_number vs reference-material
  serial_number" trap, where the names are identical and the fields are not.
- No occurrence is ever deleted. Merging rewrites `normalized_field_name` to a
  canonical value and records a MergeRecord; every original name, page and
  quote survives on the field itself.

That last point is what makes the merge reversible, which matters because a
reviewer disagreeing with a merge is a normal outcome, not an exception.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import date

from app.graph.errors import ErrorType
from app.graph.llm import call_structured
from app.graph.nodes.helpers import audit, clean_text, err, normalise_for_match, snake_case
from app.graph.prompts import MERGE_SYSTEM, merge_user
from app.graph.schemas import ExtractedField, MergeRecord
from app.graph.state import GraphState
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

NODE = "field_normalization"

#: Below this many distinct names, semantic grouping is not worth a call.
_MIN_NAMES_FOR_LLM = 4
#: Above this, the prompt gets unwieldy; the exact-match pass still applies.
_MAX_NAMES_FOR_LLM = 250


class _MergeGroup(BaseModel):
    canonical_name: str
    original_names: list[str] = Field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0


class _MergeProposal(BaseModel):
    groups: list[_MergeGroup] = Field(default_factory=list)
    notes: str = ""


# --------------------------------------------------------------------------
# Deterministic value normalisation
# --------------------------------------------------------------------------

_DATE_PATTERNS = [
    (re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$"), lambda m: (int(m[1]), int(m[2]), int(m[3]))),
    (re.compile(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})$"), lambda m: (int(m[3]), int(m[2]), int(m[1]))),
]
_NUMBER = re.compile(r"^[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")


def normalise_value(value, data_type: str):
    """Only unambiguous transformations. Anything else returns cleaned text.

    Deliberately does NOT try to parse ambiguous dates: 03/04/2024 is either
    3 April or 4 March depending on locale, and guessing would silently
    corrupt a value the reviewer believes was verified.
    """
    if value is None:
        return None
    s = clean_text(value)
    if not s:
        return None

    if data_type == "date":
        for pattern, parts in _DATE_PATTERNS:
            m = pattern.match(s)
            if m:
                y, mo, d = parts(m)
                # Day > 12 disambiguates D/M from M/D; otherwise leave it alone.
                if pattern.pattern.startswith("^(\\d{1,2})") and d <= 12 and mo <= 12:
                    return s
                try:
                    return date(y, mo, d).isoformat()
                except ValueError:
                    return s
        return s

    if data_type in ("number", "integer") and _NUMBER.match(s):
        return s.replace(",", "")

    return s


# --------------------------------------------------------------------------
# Merge vetoes
# --------------------------------------------------------------------------


def _veto(group_fields: list[ExtractedField]) -> str:
    """Return a reason to refuse the merge, or "" to allow it."""
    types = {f.data_type for f in group_fields if f.value is not None and f.data_type}
    if len(types) > 1 and not types <= {"string", "null"}:
        return f"members disagree on data type ({sorted(types)})"

    sections = {normalise_for_match(f.section_name) for f in group_fields if f.section_name}
    values = {normalise_for_match(f.value) for f in group_fields if f.value is not None}
    if len(sections) > 1 and len(values) > 1:
        return (
            f"members come from {len(sections)} different sections and hold "
            f"{len(values)} different values — likely different entities"
        )
    return ""


def normalize_and_merge(state: GraphState) -> dict:
    metrics = state.metrics.model_copy(deep=True)
    fields = [f.model_copy(deep=True) for f in state.active_fields()]
    entries = []
    errors = []

    if not fields:
        return {
            "current_node": NODE,
            "normalized_fields": [],
            "duplicate_fields": [],
            "audit_log": audit(NODE, "skipped", "no fields to normalise"),
        }

    # ---- 1. deterministic value normalisation ----------------------------
    for f in fields:
        f.normalized_field_name = snake_case(f.normalized_field_name or f.field_name)
        f.normalized_value = normalise_value(f.value, f.data_type)

    # ---- 2. exact-name grouping (free, no judgement needed) --------------
    by_name: dict[str, list[ExtractedField]] = defaultdict(list)
    for f in fields:
        by_name[f.normalized_field_name].append(f)

    merges: list[MergeRecord] = []
    for name, group in by_name.items():
        originals = {f.field_name for f in group}
        if len(group) > 1 and len(originals) > 1:
            merges.append(MergeRecord(
                canonical_name=name,
                original_names=sorted(originals),
                field_uids=[f.field_uid for f in group],
                reason="identical normalised field name",
                confidence=1.0,
            ))

    # ---- 3. semantic grouping across different names ---------------------
    distinct = sorted(by_name)
    if _MIN_NAMES_FOR_LLM <= len(distinct) <= _MAX_NAMES_FOR_LLM:
        summaries = []
        for name in distinct:
            sample = by_name[name][0]
            summaries.append({
                "normalized_field_name": name,
                "original_name": sample.field_name,
                "example_value": str(sample.value)[:120] if sample.value is not None else None,
                "data_type": sample.data_type,
                "section": sample.section_name,
                "occurrences": len(by_name[name]),
            })

        outcome = call_structured(
            node=NODE, system=MERGE_SYSTEM, user_text=merge_user(summaries),
            output_model=_MergeProposal, max_attempts=2, on_metric=metrics.record,
        )

        if not outcome.ok:
            errors += err(
                outcome.error_type or ErrorType.LLM_UNAVAILABLE,
                f"semantic merge proposal failed: {outcome.error_message}", NODE,
                retry_count=outcome.attempts,
                recovery_action="exact-name merges retained; no semantic merging applied",
                resolution_status="RECOVERED",
            )
            entries += audit(NODE, "semantic_merge_unavailable",
                             "exact-name merges only", level="warning")
        else:
            proposal: _MergeProposal = outcome.parsed  # type: ignore[assignment]
            applied = vetoed = 0
            for group in proposal.groups:
                members = [snake_case(n) for n in group.original_names]
                present = [n for n in members if n in by_name]
                if len(present) < 2:
                    continue
                canonical = snake_case(group.canonical_name) or present[0]
                group_fields = [f for n in present for f in by_name[n]]

                reason_to_refuse = _veto(group_fields)
                if reason_to_refuse:
                    vetoed += 1
                    entries += audit(
                        NODE, "merge_vetoed",
                        f"refused to merge {present} into {canonical!r}: {reason_to_refuse}",
                        level="warning",
                    )
                    errors += err(
                        ErrorType.DUPLICATE_FIELD,
                        f"proposed merge of {present} rejected: {reason_to_refuse}", NODE,
                        recovery_action="fields kept separate",
                        resolution_status="RECOVERED",
                    )
                    continue

                for f in group_fields:
                    if f.normalized_field_name != canonical:
                        f.merged_from = sorted(set(f.merged_from + [f.normalized_field_name]))
                        f.normalized_field_name = canonical
                merges.append(MergeRecord(
                    canonical_name=canonical,
                    original_names=sorted({f.field_name for f in group_fields}),
                    field_uids=[f.field_uid for f in group_fields],
                    reason=group.reason or "semantically equivalent field names",
                    confidence=group.confidence,
                ))
                applied += 1

            entries += audit(
                NODE, "semantic_merge",
                f"{applied} group(s) merged, {vetoed} vetoed, from {len(distinct)} distinct name(s)",
            )
    elif len(distinct) > _MAX_NAMES_FOR_LLM:
        entries += audit(
            NODE, "semantic_merge_skipped",
            f"{len(distinct)} distinct names exceeds the {_MAX_NAMES_FOR_LLM} limit; "
            f"exact-name merging only", level="warning",
        )

    entries += audit(
        NODE, "normalised",
        f"{len(fields)} field(s), {len({f.normalized_field_name for f in fields})} distinct "
        f"canonical name(s), {len(merges)} merge record(s)",
    )

    return {
        "current_node": NODE,
        "normalized_fields": fields,
        "duplicate_fields": merges,
        "metrics": metrics,
        "audit_log": entries,
        "errors": errors,
    }
