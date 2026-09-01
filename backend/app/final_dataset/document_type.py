"""Deterministic, filename-keyword classification of a document's TYPE —
distinct from its DOMAIN (e.g. two "medical" documents might be a
prescription vs. a lab report). No LLM call, no guessing: a fixed rule
table matched against the original filename, defaulting to None when
nothing matches — the same "don't invent it" rule every other stage in
this pipeline follows, applied to a filename-derived classification instead
of an extracted value.
"""
from __future__ import annotations

_PATTERNS: list[tuple[str, str]] = [
    ("prescription", "prescription"),
    ("lab-report", "lab_report"),
    ("lab_report", "lab_report"),
    ("labreport", "lab_report"),
    ("certificate", "certificate"),
    ("coa", "certificate_of_analysis"),
    ("nutritional", "nutritional_label"),
    ("nutrition", "nutritional_label"),
    ("purity", "purity_test"),
    ("test-result", "test_result"),
    ("test_result", "test_result"),
    ("cln", "clinical_report"),
]


def derive_document_type(original_filename: str) -> str | None:
    name = (original_filename or "").lower()
    for keyword, doc_type in _PATTERNS:
        if keyword in name:
            return doc_type
    return None
