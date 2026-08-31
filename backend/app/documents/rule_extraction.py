"""Deterministic, regex-based extraction for organisation-level identifiers
that have an unambiguous, checkable format — GST/PAN/TAN numbers. Runs before
any LLM call for the completed_application_form pipeline.

Deliberately NOT extended to phone/email/dates: those can legitimately appear
several times in a multi-person document (one per staff member), and which
schema field a given one belongs to is exactly the kind of ambiguity that
needs an LLM's context, not a regex — a rule that's "reliable" only for a
single-instance, checksummed identifier stays in scope; a rule that would
have to guess whose contact info it found does not.
"""
import re

from app.documents.chunking import Chunk
from app.documents.grounding import FieldResult, ground

# GSTIN: 2-digit state code + PAN (5 letters, 4 digits, 1 letter) + 1
# alphanumeric entity code + literal 'Z' + 1 alphanumeric checksum = 15 chars.
GST_PATTERN = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]\b")
# PAN: 5 letters, 4 digits, 1 letter.
PAN_PATTERN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
# TAN: 4 letters, 5 digits, 1 letter.
TAN_PATTERN = re.compile(r"\b[A-Z]{4}\d{5}[A-Z]\b")

_PATTERNS = (
    ("organisation.gst_number", GST_PATTERN),
    ("organisation.pan_number", PAN_PATTERN),
    ("organisation.tan_number", TAN_PATTERN),
)


def extract_identifiers(chunks: list[Chunk]) -> list[FieldResult]:
    """Scans every chunk in page order; returns at most one of each
    identifier — a real application states its own GST/PAN/TAN once, not
    once per page, so the first match wins rather than the last."""
    found: dict[str, FieldResult] = {}
    for chunk in chunks:
        for field_path, pattern in _PATTERNS:
            if field_path in found:
                continue
            match = pattern.search(chunk.text)
            if not match:
                continue
            value = match.group(0)
            rect = ground(value, chunk.spans)
            found[field_path] = FieldResult(
                field=field_path,
                value=value,
                confidence=1.0,
                source_page=chunk.page_number if rect else None,
                source_bbox=rect,
                source="rule_based",
            )
    return list(found.values())
