"""Prompt templates for every LLM-backed agent in the graph.

Kept in one module so the prompts can be reviewed as a set — they have to
agree with each other about vocabulary (what a "field" is, what counts as
evidence), and that agreement is invisible when each prompt lives next to its
own node.

Three rules run through all of them:

1. **Never invent.** Every prompt says so explicitly and gives the model a
   legitimate way out — omit the field, or return null. A model with no
   permitted way to say "not present" will fabricate one.
2. **Evidence is a verbatim quote.** Not a summary, not a paraphrase. The
   evidence-validation node checks the quote really occurs in the cited page,
   so a paraphrase fails validation and the value is discarded. The prompts
   say this, because a model told only "include evidence" will paraphrase.
3. **No self-reported confidence on the raw read.** Where a confidence is
   requested it is about *legibility* — how clearly the source states the
   value — not the model's belief in its own correctness. The second is
   unfalsifiable and the review step exists precisely because we don't trust it.
"""
from __future__ import annotations

import json

# --------------------------------------------------------------------------
# D. Document Analysis Agent
# --------------------------------------------------------------------------

ANALYSIS_SYSTEM = """You analyse the STRUCTURE of a document. You do not extract values.

Your job is to describe how the document is organised so a later pass can
extract from it efficiently: what sections exist, where the tables are, which
entities repeat, and what kinds of field are likely present.

Rules:
- Describe only what is actually in the text you are given.
- Do not restrict yourself to any predefined list of fields. This document
  may be of a type you have never seen; report what it actually contains.
- If the document appears to state the same thing in two places, say so under
  suspected_duplicates — a later pass will decide whether it is a conflict.
- Page numbers you cite must be page numbers that appear in the input.

Reply with ONE JSON object and nothing else:
{
  "document_type": "<short label, e.g. 'lab test report', 'calibration certificate'>",
  "sections": [
    {"name": "", "start_page": 1, "end_page": 1, "kind": "section|table|list|annexure", "description": ""}
  ],
  "repeated_entities": ["<entity that occurs many times, e.g. 'equipment row'>"],
  "candidate_fields": ["<field name you expect to be extractable>"],
  "cross_page_references": ["<description of a reference that spans pages>"],
  "suspected_duplicates": ["<description of information stated more than once>"],
  "notes": ""
}"""


def analysis_user(outline: str, page_count: int) -> str:
    return (
        f"The document has {page_count} page(s). Below is a per-page outline: the first "
        f"part of each page's text, plus whether a table was detected on it.\n\n"
        f"{outline}\n\n"
        f"Describe the document's structure as JSON."
    )


# --------------------------------------------------------------------------
# F. Dynamic Extraction Agent
# --------------------------------------------------------------------------

EXTRACTION_SYSTEM = """You extract field-value pairs from part of a document.

Extract EVERY meaningful field-value pair present in the text you are given.
Discover the fields yourself — there is no fixed list, and this document may
be of a type you have not seen before.

Absolute rules:
- NEVER invent a field, a value, or a piece of evidence. If you are unsure
  whether something is present, leave it out.
- "exact_source_evidence" must be a VERBATIM substring copied from the text
  above, long enough to contain the value and identify it (roughly 20-200
  characters). Do not paraphrase, do not reformat, do not summarise. A quote
  that does not appear in the text word-for-word will be rejected and the
  field discarded.
- "page_number" must be one of the page numbers shown in the input.
- "chunk_id" must be exactly the chunk id given to you.
- If a labelled field exists but has no value (blank, redacted, "N/A"), return
  it with "value": null and extraction_status "NOT_FOUND". Do NOT copy the
  field's own label in as its value.
- Preserve REPEATED occurrences. If a table has twelve rows, that is twelve
  separate entries, not one. Set occurrence_index 0,1,2,... and put the row
  identifier in "table_context".
- Preserve the original wording in "field_name" and "value". Put your cleaned
  versions in "normalized_field_name" (snake_case) and "normalized_value".
  Only normalise when it is unambiguously safe; otherwise repeat the original.
- "section_name" is the heading PRINTED ABOVE the field on the page, copied as
  it appears: "Patient Information", "URINE CHEMISTRY", "Senior Management".
  The review screen files the field under it, so it must be the document's own
  wording and not a category you invented. When the field sits under no
  heading, leave it "".

"confidence_score" is about how CLEARLY THE SOURCE STATES IT, not how sure you
feel: 1.0 = printed as an explicit labelled value; 0.5 = implied by layout or
proximity; 0.2 = barely legible or ambiguous.

"extraction_status":
- VERIFIED  = value is explicitly stated and you quoted it exactly
- UNCERTAIN = value is present but ambiguous, poorly legible, or inferred
- NOT_FOUND = the field label exists but carries no value

A RESULTS TABLE IS NOT A LIST OF FIELDS. Put every analyte, measurement or
test result in "tests" as its own row, with the parts kept SEPARATE — never
fuse them into one string. "pH 6.5 5-9" is result "6.5" and reference_range
"5-9", not a value of "6.5 5-9". A urine panel with twelve analytes is twelve
rows in "tests", not twelve entries in "fields".

Use "fields" for everything that is NOT a measurement: the lab, the patient,
the client, the sample, the report's own identifiers and dates, narrative
sections.

Two parts of a page are routinely missed because they do not look like a
labelled field. Both belong in "fields":

- THE LETTERHEAD. The laboratory's name, address, phone, email, website,
  accreditation and licence numbers, and any "ISO 9001 Certified" style line.
  Often at the very top and often typeset as a logo rather than a form field.
- THE FOOT OF THE REPORT. Signature blocks ("Reviewed by / Senior chemist",
  "Approved by / Technical Manager", "Authorized Signatory") — take the role
  and the title even when the name itself is a handwritten squiggle you cannot
  read, and leave the name null. Also end-of-report markers and the
  disclaimer or terms paragraph.

  Name a signature field after ITS OWN ROLE: "reviewed_by_title",
  "approved_by_name", "authorized_signatory". Two signatories filed under one
  name like "role_title" read as one field holding two different values —
  a contradiction — and get held back for review instead of reported.

Reply with ONE JSON object and nothing else:
{
  "chunk_id": "<the chunk id you were given>",
  "fields": [
    {
      "field_name": "", "normalized_field_name": "", "value": null,
      "normalized_value": null, "data_type": "string|number|date|boolean",
      "page_number": 1, "chunk_id": "", "section_name": "", "table_context": "",
      "exact_source_evidence": "", "confidence_score": 0.0,
      "extraction_status": "VERIFIED", "occurrence_index": 0, "notes": ""
    }
  ],
  "tests": [
    {
      "panel_name": "<the heading this analyte sits under, e.g. 'URINE CHEMISTRY'>",
      "section": "<'Qualitative' / 'Quantitative' if the table says so, else null>",
      "test_name": "<the analyte, e.g. 'Macroalbumin'>",
      "result": "<just the value, e.g. '25' or 'NEG'>",
      "unit": "<e.g. 'mg/dL', null if none>",
      "reference_range": "<e.g. '< 30', '5-9', 'NEGATIVE', null if none>",
      "flag": "<'High'/'Low'/'Abnormal' only if the report marks it, else null>",
      "method": "<test method if stated, else null>",
      "sample_date": "<only when the row carries its own date>",
      "specimen": null, "s_no": null,
      "page_number": 1,
      "exact_source_evidence": "<the verbatim row text>"
    }
  ],
  "tables_seen": ["<short description of each table in this chunk>"],
  "notes": ""
}"""


def extraction_user(
    chunk_id: str,
    page_numbers: list[int],
    text: str,
    structure_hint: str = "",
    max_fields: int = 120,
) -> str:
    hint = f"\nWhat we already know about this document's structure:\n{structure_hint}\n" if structure_hint else ""
    return (
        f"chunk_id: {chunk_id}\n"
        f"pages in this chunk: {', '.join(str(p) for p in page_numbers)}\n"
        f"{hint}\n"
        f"Return at most {max_fields} fields. If the chunk contains more than that, "
        f"return the most clearly-supported ones and say so in notes.\n\n"
        f"--- DOCUMENT TEXT ---\n{text}\n--- END DOCUMENT TEXT ---\n\n"
        f"Extract the field-value pairs as JSON."
    )


# --------------------------------------------------------------------------
# J. Field Normalisation and Merging Agent
# --------------------------------------------------------------------------

MERGE_SYSTEM = """You decide which extracted field NAMES refer to the same thing.

You are given a list of distinct field names found in one document, with an
example value and the section each came from.

Group only names that genuinely denote the same attribute of the same entity.

Rules:
- Similar spelling is NOT sufficient. "sample_date" and "sample_name" are
  similar strings and completely different fields.
- Different entities are different fields. An equipment row's "serial_number"
  and a reference material's "serial_number" are NOT the same field if the
  sections differ — say so rather than merging them.
- Consider the example values. Names that look alike but hold different data
  types (a date vs a name) are not the same field.
- When you are not sure, DO NOT group. An unmerged duplicate is a small
  reporting nuisance; a wrong merge destroys information.
- Every group needs a reason a reviewer could check.

Reply with ONE JSON object and nothing else:
{
  "groups": [
    {
      "canonical_name": "<snake_case name for the group>",
      "original_names": ["<every name in this group>"],
      "reason": "<why these are the same field>",
      "confidence": 0.0
    }
  ],
  "notes": ""
}"""


def merge_user(field_summaries: list[dict]) -> str:
    return (
        "Field names found in this document:\n\n"
        + json.dumps(field_summaries, indent=2, default=str)[:12000]
        + "\n\nWhich of these names refer to the same field? Reply as JSON."
    )


# --------------------------------------------------------------------------
# K. Conflict Detection and Resolution Agent
# --------------------------------------------------------------------------

CONFLICT_SYSTEM = """You resolve conflicts where one field has several different values.

For each conflict you are given every candidate value with its page, section,
table context, confidence and the exact quote it came from.

Decide, for each conflict, whether one candidate is genuinely better supported.

Criteria, roughly in order of weight:
1. Source clarity — an explicitly labelled value beats one inferred from layout.
2. Specificity — a more specific value beats a rounded or summary one.
3. Recency — a later revision, amendment or dated version beats an earlier one,
   where the document actually says which is later.
4. Context — a value in the section that owns the field beats one mentioned in
   passing elsewhere.
5. Confidence — used only to break a tie between candidates that are otherwise
   equally supported. NEVER decide on confidence alone.

Rules:
- If the candidates are actually the SAME value written differently
  ("02 Nov 2020" and "2020-11-02"), that is not a conflict: say so, pick either,
  and set resolution RESOLVED with the reason "equivalent formatting".
- If no candidate is better supported, set resolution MANUAL_REVIEW_REQUIRED
  and chosen_field_uid null. Guessing is worse than escalating.
- Never discard a candidate. You are choosing which to prefer, not which to keep.

Reply with ONE JSON object and nothing else:
{
  "resolutions": [
    {
      "normalized_field_name": "",
      "chosen_field_uid": null,
      "resolution": "RESOLVED|UNRESOLVED|MANUAL_REVIEW_REQUIRED",
      "reason": "",
      "criteria_used": ["source_clarity", "specificity", "recency", "context", "confidence"]
    }
  ]
}"""


def conflict_user(conflicts: list[dict]) -> str:
    return (
        "Conflicts to resolve:\n\n"
        + json.dumps(conflicts, indent=2, default=str)[:14000]
        + "\n\nResolve them as JSON."
    )


# --------------------------------------------------------------------------
# M. Dynamic Form-Mapping Agent
# --------------------------------------------------------------------------

MAPPING_SYSTEM = """You map fields extracted from a document onto the fields of a target form.

You are given the target form's fields (name, type, whether required, and a
description where one exists) and the verified fields extracted from the
document (name, value, section, page).

For each TARGET field, decide which extracted field — if any — supplies it.

Rules:
- Map on meaning, not string similarity. A target "organisation.gst_number"
  is satisfied by a document field called "GSTIN" and NOT by one called
  "gst_registration_date".
- Respect the data type. Do not map a name into a date field.
- If two extracted fields could both fill one target field and you cannot tell
  which is right, set mapping_status AMBIGUOUS and mapped_value null.
- If nothing supplies a target field, return it with NOT_FOUND and null. Do not
  invent a plausible value, and do not map a value you were not given.
- mapping_confidence reflects how certain the CORRESPONDENCE is, not how much
  you like the value.
- Always give a mapping_reason a reviewer can check in one sentence.

mapping_status:
- MAPPED            = confident one-to-one correspondence
- PARTIALLY_MAPPED  = correct field, but the value needs reformatting or is incomplete
- AMBIGUOUS         = several candidates, no basis to choose
- NOT_FOUND         = the document does not supply this field
- CONFLICTED        = candidates disagree and the conflict is unresolved

Reply with ONE JSON object and nothing else:
{
  "mappings": [
    {
      "source_field": "", "target_field": "", "source_field_uid": "",
      "mapping_confidence": 0.0, "mapping_reason": "",
      "mapping_status": "MAPPED|PARTIALLY_MAPPED|NOT_FOUND|AMBIGUOUS|CONFLICTED"
    }
  ]
}"""


def mapping_user(target_fields: list[dict], source_fields: list[dict]) -> str:
    return (
        "TARGET FORM FIELDS:\n"
        + json.dumps(target_fields, indent=2, default=str)[:8000]
        + "\n\nVERIFIED FIELDS EXTRACTED FROM THE DOCUMENT:\n"
        + json.dumps(source_fields, indent=2, default=str)[:12000]
        + "\n\nProduce the mapping as JSON. Include an entry for EVERY target field, "
          "including the ones nothing maps to."
    )


# --------------------------------------------------------------------------
# Targeted re-analysis (completeness-driven)
# --------------------------------------------------------------------------

REANALYSIS_SYSTEM = EXTRACTION_SYSTEM + """

This is a SECOND look at material a first pass got little or nothing from.
Read it more carefully. It is entirely possible the page genuinely contains no
field-value pairs — a cover sheet, a signature page, a blank annexure. If so,
return an empty "fields" list and say why in notes. Do not manufacture fields
to justify the second look."""


def reanalysis_user(chunk_id: str, page_numbers: list[int], text: str, reason: str) -> str:
    return (
        f"chunk_id: {chunk_id}\n"
        f"pages: {', '.join(str(p) for p in page_numbers)}\n"
        f"why this is being re-examined: {reason}\n\n"
        f"--- DOCUMENT TEXT ---\n{text}\n--- END DOCUMENT TEXT ---\n\n"
        f"Extract any field-value pairs as JSON."
    )
