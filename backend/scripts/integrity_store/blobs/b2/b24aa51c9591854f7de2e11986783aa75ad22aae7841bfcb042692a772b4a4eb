"""Adapter: run the LangGraph workflow, return the legacy `PipelineResult`.

This is what makes the graph the extraction path without a rewrite of the
review UI, the compiler, or the upload router. Those all speak `PipelineResult`
and `FieldResult`, and none of them needs to know that the thing producing
those objects changed underneath.

The mapping is not quite one-to-one, and the differences matter:

- The graph produces far richer per-field provenance (evidence status, merge
  history, occurrence index) than `FieldResult` can hold. What does not fit is
  folded into the field's `source` and the run's warnings rather than dropped
  — and the full record is always in the graph's own database and result JSON.
- The graph has no bounding boxes. It works from page text, so a field carries
  a page number but no rectangle. The review UI already handles a page-level
  highlight (it has to, for the vision-OCR path), so this degrades to that
  rather than breaking.
- A CONFLICTED or unsupported field is deliberately NOT emitted as a normal
  field. Letting it through would put a value the graph refused to trust into
  a form slot, which is precisely what the workflow exists to prevent; it is
  surfaced as a warning instead.
"""
from __future__ import annotations

import logging
import re

from app.documents.grounding import FieldResult, PipelineResult
from app.graph.nodes.helpers import normalise_for_match
from app.graph.runner import run_document
from app.graph.structured import route_field
from app.graph.schemas import (
    EvidenceStatus,
    ExtractedField,
    ExtractionStatus,
    FinalStatus,
    MappingStatus,
)

log = logging.getLogger(__name__)

#: Open-ended graph fields land just under the review UI's 0.85 threshold on
#: purpose — a reviewer should confirm them rather than have them silently
#: accepted. Same reasoning the legacy pipeline used for its open extraction.
_UNVERIFIED_CONFIDENCE = 0.80


def _extraction_source(state_type: str, ocr_pages: int, total_pages: int) -> str:
    if state_type == "docx":
        return "docx"
    if ocr_pages == 0:
        return "born_digital_pdf"
    if ocr_pages >= total_pages:
        return "ocr_pdf"
    return "mixed_pdf"


def _absorb_open_duplicates(fields: list[FieldResult], *, mapped_start: int) -> None:
    """Merge each named form slot with the open extraction it duplicates.

    Mutates `fields` in place: the open row's section and group move onto the
    mapped row, and the open row goes. Names count as related when one's words
    are a subset of the other's — "laboratory_name" and
    "organisation.laboratory_name" are the same fact under two spellings;
    "sex" and "blood_group" both reading "M" are not.
    """
    mapped = fields[mapped_start:]
    if not mapped:
        return

    open_rows = fields[:mapped_start]
    by_value: dict[str, list[int]] = {}
    for i, row in enumerate(open_rows):
        by_value.setdefault(normalise_for_match(row.value), []).append(i)

    absorbed: set[int] = set()
    for slot in mapped:
        words = set(re.split(r"[^a-z0-9]+", slot.field.lower())) - {""}
        for i in by_value.get(normalise_for_match(slot.value), []):
            if i in absorbed:
                continue
            other = set(re.split(r"[^a-z0-9]+", open_rows[i].field.lower())) - {""}
            if not (words <= other or other <= words):
                continue
            slot.section = slot.section or open_rows[i].section
            slot.group = slot.group or open_rows[i].group
            slot.source_page = slot.source_page if slot.source_page is not None else open_rows[i].source_page
            absorbed.add(i)
            break

    fields[:] = [r for i, r in enumerate(open_rows) if i not in absorbed] + mapped


def _summarise_reasoning(reasoning: str, limit: int = 400) -> str:
    """Collapse a " | "-joined reason list that repeats the same kind of item.

    MEASURED on a filled application form: the decision node produced five
    `ambiguous_mapping:` segments differing only by which slot was ambiguous,
    and the reviewer got a wall of text saying one thing five times. Segments
    are grouped by the text before their first ":", the first of each kind is
    kept verbatim, and the rest become a count.
    """
    segments = [seg.strip() for seg in (reasoning or "").split("|") if seg.strip()]
    if not segments:
        return ""

    kept: list[str] = []
    seen: dict[str, int] = {}
    for seg in segments:
        kind = seg.split(":", 1)[0].strip().lower()
        seen[kind] = seen.get(kind, 0) + 1
        if seen[kind] == 1:
            kept.append(seg)

    for kind, count in seen.items():
        if count > 1:
            for i, seg in enumerate(kept):
                if seg.split(":", 1)[0].strip().lower() == kind:
                    kept[i] = f"{seg} (and {count - 1} more like it)"
                    break

    return " | ".join(kept)[:limit]


def _to_api_page(page_number: int | None) -> int | None:
    """Graph page numbers are 1-based; the API's `source_page` is 0-based.

    Two conventions, both deliberate, meeting at this boundary. Inside the
    graph a page number is what a reader means by "page 2", because it ends up
    in an evidence citation a person has to check. Everything the API already
    speaks — pdf_utils.extract_pages, the /render endpoint, the review
    viewer's page tabs — indexes from zero.

    Passing the graph's number straight through made the review screen label a
    page-1 field "p2" and jump the viewer one page past the value, and on the
    last page it jumped somewhere that does not exist.
    """
    if page_number is None:
        return None
    return max(page_number - 1, 0)


def _collapse_repeats(extracted: list[ExtractedField]) -> list[FieldResult]:
    """Turn the graph's raw field list into one row per fact.

    A multi-page report restates its header on every page, and the same value
    gets picked up under more than one name, so the raw list says the same
    thing three ways. Three distinct repeats show up and each needs its own
    answer:

    1. IDENTICAL name and value — accession_no on page 1 and again on page 2.
       One fact; collapse. Values that genuinely DIFFER (chart_no 1400000 vs
       1400001) are kept, because dropping one invents agreement the document
       does not have.
    2. SAME NAME, one value swallowing the other — clinical_data as "No
       clinical data specified" and as "No clinical data specified No therapy",
       where "No therapy" is the neighbouring `therapy` field. The long reading
       crossed a field boundary, so the short one wins.
    3. SAME VALUE under related names — isomorphic_erythrocytes and
       erythrocytes_per_10_hpfs_isomorphic, both "760". One reading of one
       number. Related means one name's words are a subset of the other's;
       without that test sex "M" would swallow any other "M" on the page.

    Unrelated fields that merely share a value are NOT collapsed, and neither
    are two genuinely different values under one name — a reviewer needs to see
    a disagreement the document actually contains.
    """
    usable = [
        f for f in extracted
        if f.value is not None
        and f.extraction_status != ExtractionStatus.CONFLICTED
        and f.evidence_status not in (EvidenceStatus.UNSUPPORTED, EvidenceStatus.NO_EVIDENCE)
    ]
    # Best-supported first, so the row that survives a collapse is the one the
    # document states most clearly rather than whichever chunk finished first.
    # The original position rides along and is restored at the end: confidence
    # order would scramble the form, where a reviewer expects the fields in the
    # order the page printed them.
    order = {id(f): i for i, f in enumerate(usable)}
    usable.sort(key=lambda f: -f.confidence_score)

    kept: list[tuple[int, FieldResult]] = []
    seen: set[tuple[str, str]] = set()
    by_value: dict[str, set[str]] = {}      # value -> the name words holding it
    by_name: dict[str, list[str]] = {}      # name  -> the values kept under it

    for f in usable:
        name = f.normalized_field_name
        norm = normalise_for_match(f.value)
        words = set(re.split(r"[^a-z0-9]+", name.lower())) - {""}

        if (name, norm) in seen:
            continue  # rule 1

        # rule 2
        overlapping = [v for v in by_name.get(name, []) if norm and (norm in v or v in norm)]
        if any(len(v) <= len(norm) for v in overlapping):
            continue
        for longer in overlapping:
            kept = [(i, r) for i, r in kept
                    if not (r.field == name and normalise_for_match(r.value) == longer)]
            by_name[name] = [v for v in by_name[name] if v != longer]
            seen.discard((name, longer))

        # rule 3
        holder = by_value.get(norm)
        if holder is not None and norm and (holder <= words or words <= holder):
            continue

        seen.add((name, norm))
        by_value.setdefault(norm, words)
        by_name.setdefault(name, []).append(norm)

        verified = (
            f.extraction_status == ExtractionStatus.VERIFIED
            and f.evidence_status == EvidenceStatus.SUPPORTED
        )
        kept.append((order[id(f)], FieldResult(
            field=name,
            value=str(f.normalized_value if f.normalized_value is not None else f.value),
            confidence=f.confidence_score if verified else min(f.confidence_score, _UNVERIFIED_CONFIDENCE),
            source_page=_to_api_page(f.page_number),
            source_bbox=None,  # graph works from text; no rectangle available
            source="open_extraction",
            section=f.section_name or "",
            group=route_field(name),
        )))

    return [r for _, r in sorted(kept, key=lambda pair: pair[0])]


def run_graph_pipeline(
    file_path: str,
    *,
    document_id: str,
    form_type: str = "",
    display_name: str = "",
) -> PipelineResult:
    """Process one document through the graph and adapt the result."""
    result = run_document(
        file_path, document_id=document_id, form_id=form_type,
        display_name=display_name,
    )

    # A multi-page report restates its header on every page, and the same value
    # gets picked up under more than one name, so the raw field list says the
    # same thing three ways. Three separate repeats show up, and each needs its
    # own answer:
    #
    #  1. IDENTICAL name and value — accession_no on page 1 and page 2. One
    #     fact; collapse. Values that genuinely DIFFER (chart_no 1400000 vs
    #     1400001) are kept, because dropping one invents agreement.
    #  2. SAME NAME, one value swallowing the other — clinical_data as
    #     "No clinical data specified" and as "No clinical data specified No
    #     therapy", where "No therapy" is the neighbouring `therapy` field.
    #     The long one crossed a field boundary, so the short one wins.
    #  3. SAME VALUE under related names — isomorphic_erythrocytes and
    #     erythrocytes_per_10_hpfs_isomorphic, both "760". One reading of one
    #     number. Related means one name's words are a subset of the other's;
    #     without that test, sex "M" would collapse into any other "M".
    #
    # Best-supported first, so the row that survives is the one the document
    # states most clearly rather than whichever chunk finished first.
    fields = _collapse_repeats(result.extracted_fields)

    warnings: list[str] = []

    # --- named form slots -------------------------------------------------
    # The graph's form_mappings are the only thing that can fill a NAMED NABL
    # slot. Without this the compiler saw nothing but open_extraction fields,
    # routed every one of them into extra_fields, and the compiled form came
    # out empty however well the document was read.
    #
    # `source="llm"` (not open_extraction) is what makes compiler.py treat
    # these as schema fields, and the paths are already the shape
    # _merge_full_form_fields expects — "organisation.gst_number", or
    # "equipment[0].name" once the repeating marker carries an index.
    repeat_counter: dict[str, int] = {}
    mapped_count = 0
    # One source value fills ONE slot. The graph's own form_filling node
    # enforces this, but the compiled-form path does not go through it, so the
    # rule has to hold here too: MEASURED on a real filled application, a
    # single lab email was written into organisation.email AND all four
    # senior-management roles, none of whom the document names. Highest
    # confidence first, so the slot that keeps the value is the best-supported.
    claimed: dict[str, str] = {}
    # winning slot -> the slots that wanted the same value
    displaced: dict[str, list[str]] = {}
    for m in sorted(result.form_mappings, key=lambda x: -x.mapping_confidence):
        if m.mapping_status not in (MappingStatus.MAPPED, MappingStatus.PARTIALLY_MAPPED):
            continue
        if m.mapped_value is None or str(m.mapped_value).strip() == "":
            continue

        repeating = "[]" in m.target_field
        key = normalise_for_match(m.mapped_value)
        if key and not repeating:
            already = claimed.get(key)
            if already is not None and already != m.target_field:
                # Collected, not appended one line at a time. A single lab
                # email is a candidate for organisation.email AND all four
                # senior-management roles, so this fired five times and the
                # notes panel became five near-identical sentences about one
                # fact. Summarised once below.
                displaced.setdefault(already, []).append(m.target_field)
                continue
            claimed[key] = m.target_field

        path = m.target_field
        if "[]" in path:
            # "equipment[].name" -> "equipment[0].name", then [1], [2]...
            # so several values for one repeating target build several records
            # instead of overwriting each other.
            stem = path.split("[]", 1)[0]
            index = repeat_counter.get(stem, 0)
            repeat_counter[stem] = index + 1
            path = path.replace("[]", f"[{index}]", 1)

        fields.append(FieldResult(
            field=path,
            value=str(m.mapped_value),
            confidence=m.mapping_confidence,
            source_page=_to_api_page(m.source_page),
            source_bbox=None,
            source="llm",
            group=route_field(path),
        ))
        mapped_count += 1

    # A mapped slot and the open field it came from are ONE reading. The
    # letterhead's lab name arrives twice: once as the open `laboratory_name`
    # and once as `organisation.laboratory_name` filling the NABL slot, and the
    # review screen showed both with two Confirm buttons for the same fact.
    #
    # The mapped row is the one that survives, because dropping it would empty
    # the compiled form — but it inherits the open row's sub-heading and
    # section first, so the surviving row still knows where on the page it came
    # from. Only related names collapse: an unrelated field that happens to
    # share a value is a different fact and stays.
    _absorb_open_duplicates(fields, mapped_start=len(fields) - mapped_count)

    if result.overall_status != FinalStatus.VALIDATED:
        # INCOMPLETE beside a healthy field count confused reviewers into
        # thinking the run had broken. It means "some of what was asked for is
        # not here", which is normal for a document that simply does not carry
        # every field, so say that instead of shouting the enum.
        label = {
            FinalStatus.INCOMPLETE: "Partial read",
            FinalStatus.LOW_CONFIDENCE: "Read with low confidence",
            FinalStatus.CONFLICTED: "Conflicting values found",
            FinalStatus.MANUAL_REVIEW_REQUIRED: "Needs a closer look",
            FinalStatus.REJECTED: "Extraction rejected",
        }.get(result.overall_status, result.overall_status.value)
        # final_reasoning is a " | "-joined list from the decision node and
        # repeats itself: five ambiguous_mapping segments differing only by
        # slot name. Keep the first of each kind, count the rest.
        warnings.append(f"{label}: {_summarise_reasoning(result.final_reasoning)}")
    if result.failed_chunks:
        warnings.append(
            f"{len(result.failed_chunks)} chunk(s) failed; pages they cover may be incomplete"
        )
    for conflict in result.conflicts:
        if conflict.resolution.value != "RESOLVED":
            warnings.append(
                f"unresolved conflict on {conflict.normalized_field_name!r} "
                f"({len(conflict.candidates)} candidate values kept for review)"
            )
    unsupported = sum(
        1 for f in result.extracted_fields
        if f.evidence_status in (EvidenceStatus.UNSUPPORTED, EvidenceStatus.NO_EVIDENCE)
    )
    if unsupported:
        # Deliberately NOT phrased as a failure. These are values the model
        # proposed and then could not quote from the page — on a real report
        # one of them was "SAMPLE" for a referring_physician line the document
        # left blank. Dropping them is the guard working, but "N extraction(s)
        # failed" read as a defect and sent a reviewer looking for a bug.
        warnings.append(
            f"{unsupported} proposed value(s) not shown: they could not be quoted "
            f"from the page, so they were dropped rather than presented as read"
        )

    # compiler.py dispatches on doc_type. Arbitrary dotted schema paths are
    # only handled by the completed_application_form branch
    # (_merge_full_form_fields) — every other branch targets exactly one list
    # or one flat object and would silently drop these. So when the graph has
    # produced named form slots, that is the branch we need, whatever the
    # document itself happens to be. The real document type is still reported
    # in the warnings and in the graph's own record.
    compiled_doc_type = (
        "completed_application_form" if mapped_count
        else (result.document_summary.document_type or "other")
    )
    if displaced:
        total = sum(len(v) for v in displaced.values())
        holders = ", ".join(sorted(displaced))
        warnings.append(
            f"{total} form slot(s) left empty because their value already fills "
            f"{holders} — one reading of one value fills one slot"
        )

    # NOT added to warnings: "N value(s) mapped into named form fields" is good
    # news, and the review screen prints this list under a red "Extraction
    # notes" heading. A success reported as a warning teaches a reviewer to
    # distrust the whole panel. The count is visible in the fields themselves.

    return PipelineResult(
        doc_type=compiled_doc_type,
        doc_confidence=1.0 if result.overall_status == FinalStatus.VALIDATED else 0.5,
        extraction_source=_extraction_source(
            "docx" if file_path.lower().endswith(".docx") else "pdf",
            result.document_summary.ocr_pages,
            result.document_summary.page_count,
        ),
        fields=fields,
        extraction_warnings=warnings[:20],
        page_count=result.document_summary.page_count,
        # The results table and the gold-shaped document. Both were built by
        # the graph already and then thrown away here, so the review screen had
        # to reconstruct a taxonomy of its own from field names and every
        # analyte arrived as a scalar with its unit and range fused in.
        tests=[t for t in result.structured_document.get("tests", []) if t],
        structured=result.structured_document,
    )
