"""Assemble the gold-dataset-shaped document JSON.

The graph's own output is a flat list of `ExtractedField` occurrences, which is
the right internal shape — it keeps every occurrence, its page, and its
evidence. It is the wrong shape to hand a reader, for two reasons the gold
dataset gets right and a flat list gets wrong:

1. **A test is a row, not a scalar.** A urine panel has twelve analytes each
   with a result, a unit and a reference range. Flattened, that becomes
   `ph = "6.5 5-9"` — value and range fused into one string, and nothing to
   sort, compare or validate. The gold records put them in `tests[]` with
   named columns, so each analyte keeps its parts separate.

2. **Scalar facts belong once.** A two-page report restates its header on the
   continuation page, so `accession_no` appears twice. A reader wants one
   accession number, not two identical rows — while genuinely differing
   restatements must still be visible rather than silently collapsed.

Section names and the tests[] columns are taken from the 53 hand-authored
records in labelled_dataset/, so the output can be diffed directly against
ground truth instead of needing a translation layer.
"""
from __future__ import annotations

import re
from typing import Any

from app.graph.nodes.helpers import clean_text, normalise_for_match
from app.graph.schemas import EvidenceStatus, ExtractedField, ExtractionStatus, TestRow

#: Section order as it appears in the gold records. MEASURED across the 53
#: hand-authored files in labelled_dataset/: document_info in all 53, lab_info
#: in 50, report_info 35, sample_info 32, notes 31, signatories 28,
#: client_info 24, patient_info 13, findings 4. The long tail below that is
#: one-offs (stone_types, karyotypic_trend_graph) and is not worth a bucket —
#: those land in report_info, which is where a reader looks for them anyway.
SECTIONS = (
    "document_info",
    "lab_info",
    "client_info",
    "patient_info",
    "sample_info",
    "report_info",
    "findings",
    "signatories",
    "notes",
)

#: What each section is called on screen. The JSON keeps the gold key so the
#: output can be diffed against ground truth directly; a reviewer reading a
#: form should not have to.
SECTION_LABELS: dict[str, str] = {
    "document_info": "Document",
    "lab_info": "Laboratory",
    "client_info": "Client",
    "patient_info": "Patient",
    "sample_info": "Sample",
    "report_info": "Report",
    "findings": "Findings",
    "signatories": "Signatories",
    "notes": "Notes",
    "tests": "Test Results",
}

#: Where a field name lands when the model did not say. Checked in order, so
#: the first match wins and overlaps resolve to the more specific reading.
_ROUTING: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("patient_info", (
        "patient", "accession", "chart_no", "lifetime_id", "mrn", "uhid",
        "sex", "gender", "dob", "date_of_birth", "age", "referring", "requesting",
    )),
    ("lab_info", (
        "lab_", "laboratory", "organisation", "organization", "clia", "cap_",
        "license", "licence", "accreditation", "accredited", "address",
        "phone", "telephone", "mobile", "fax", "email", "website", "url",
        "gst", "pan_number", "cin", "scope", "certification", "certified",
        "iso_",
    )),
    ("client_info", ("client", "customer", "consignee", "recipient", "billed")),
    ("sample_info", (
        "sample", "specimen", "collected", "drawn", "received", "quantity",
        "condition", "container", "volume", "appearance", "animal", "breed",
    )),
    ("signatories", (
        "signator", "signed", "authorised_by", "authorized_by", "approved_by",
        "reviewed_by", "verified_by", "pathologist", "cytotechnologist",
        "technical_manager", "quality_manager", "analyst", "chemist",
    )),
    ("notes", (
        "disclaimer", "end_of_report", "note", "remark", "terms",
        "conditions", "footer", "caution", "advice", "recommendation",
    )),
    ("findings", (
        "finding", "diagnosis", "impression", "conclusion", "interpretation",
        "microscopic", "macroscopic", "background", "cellularity",
        "epithelial", "concretion", "erythrocyte",
    )),
    ("report_info", (
        "report", "issued", "released", "dated", "status", "page", "revision",
        "version", "comment", "clinical",
    )),
)


def route_field(field_name: str) -> str:
    """Which gold section a field name belongs to.

    Public because it is the ONE place that decides. The review form and the
    structured JSON used to classify independently — the backend routing a
    field to `lab_info` while the frontend filed the same field under "Other
    Details" — so the form and the exported JSON disagreed about the same
    document. There is one taxonomy, it is the gold dataset's, and this is it.
    """
    name = field_name.lower()
    for section, needles in _ROUTING:
        if any(n in name for n in needles):
            return section
    return "report_info"


#: Kept as the old private name so existing callers in this module read the
#: same; `route_field` is what everything outside it uses.
_route = route_field


def _usable(f: ExtractedField) -> bool:
    """Only values the workflow stands behind reach the structured output."""
    if f.value is None or not clean_text(f.value):
        return False
    if f.extraction_status == ExtractionStatus.CONFLICTED:
        return False
    return f.evidence_status not in (EvidenceStatus.UNSUPPORTED, EvidenceStatus.NO_EVIDENCE)


def build_structured(
    fields: list[ExtractedField],
    tests: list[TestRow],
    *,
    filename: str = "",
    document_type: str = "",
) -> dict[str, Any]:
    """Group fields into the gold sections and attach the tests array.

    Duplicate scalars collapse to one entry. A field restated identically on
    every page is one fact, so it is written once; a field whose restatements
    genuinely differ keeps them all, suffixed by page, because dropping one
    would be inventing agreement the document does not have.
    """
    grouped: dict[str, dict[str, Any]] = {s: {} for s in SECTIONS}
    # canonical name -> {normalised value -> (value, [pages])}
    seen: dict[str, dict[str, tuple[Any, list[int]]]] = {}

    for f in fields:
        if not _usable(f):
            continue
        name = f.normalized_field_name
        # VERBATIM, not the normalised form. The gold records keep what the
        # page printed — "17.12.2025", not "2025-12-17" — because this file is
        # a record of the document, and a reformatted date is already an
        # interpretation of it. The normalised value is still what the review
        # form and the compiled application form use; it just is not what
        # "the extraction" means when you diff it against ground truth.
        value = f.value if f.value is not None else f.normalized_value
        key = normalise_for_match(value)
        bucket = seen.setdefault(name, {})
        if key in bucket:
            # Same fact restated — record the extra page, do not duplicate.
            if f.page_number is not None and f.page_number not in bucket[key][1]:
                bucket[key][1].append(f.page_number)
        else:
            bucket[key] = (value, [f.page_number] if f.page_number is not None else [])

    # One value read under two names is one fact. MEASURED on a food-testing
    # report: the letterhead address came back as both `address` and
    # `laboratory_address`, differing only by a space, and the record carried
    # the lab's address twice. Related means one name's words are a subset of
    # the other's — without that test, two analytes that happen to read "25"
    # would collapse into one.
    claimed: dict[str, set[str]] = {}
    for name in sorted(seen, key=len):
        words = set(re.split(r"[^a-z0-9]+", name.lower())) - {""}
        for key in list(seen[name]):
            holder = claimed.get(key)
            if holder is not None and (holder <= words or words <= holder):
                del seen[name][key]
            else:
                claimed.setdefault(key, words)
    seen = {name: variants for name, variants in seen.items() if variants}

    for name, variants in seen.items():
        section = _route(name)
        if len(variants) == 1:
            grouped[section][name] = next(iter(variants.values()))[0]
            continue
        # Genuinely different restatements: keep each, labelled by page, so the
        # difference is visible instead of one silently winning.
        for value, pages in variants.values():
            suffix = f"_page_{pages[0]}" if pages else ""
            grouped[section][f"{name}{suffix}"] = value

    grouped["document_info"] = {
        "original_filename": filename,
        "document_title": document_type or None,
        **grouped.get("document_info", {}),
    }

    out: dict[str, Any] = {s: grouped[s] for s in SECTIONS if grouped[s]}
    out["tests"] = [t.model_dump(exclude_none=True) for t in _dedupe_tests(tests)]
    return out


def _dedupe_tests(tests: list[TestRow]) -> list[TestRow]:
    """Drop rows repeated verbatim, keep genuinely repeated measurements.

    A continuation page that reprints the same results table produces exact
    duplicates and those are noise. The same analyte measured twice on
    different dates is real data, so the sample date is part of the identity.
    """
    out: list[TestRow] = []
    seen: set[tuple] = set()
    for t in tests:
        key = (
            normalise_for_match(t.test_name),
            normalise_for_match(t.result),
            normalise_for_match(t.unit),
            normalise_for_match(t.sample_date),
            normalise_for_match(t.panel_name),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out
