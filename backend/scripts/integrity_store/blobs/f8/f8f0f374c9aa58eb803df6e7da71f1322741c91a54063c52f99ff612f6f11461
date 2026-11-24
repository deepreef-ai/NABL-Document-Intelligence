"""Small shared utilities for graph nodes.

Two things worth knowing before using these:

- `audit()` and `err()` return *lists*, because the corresponding state fields
  carry an `operator.add` reducer. A node returns `{"audit_log": audit(...)}`
  and LangGraph appends. Returning a bare entry would replace the whole log.

- `normalise_for_match()` is the single definition of "these two strings are
  the same value" used by evidence checking, duplicate detection and conflict
  comparison. Having three slightly different definitions of equality in three
  nodes was the most likely way for this system to silently disagree with
  itself, so there is exactly one.
"""
from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from difflib import SequenceMatcher
from typing import Any

from app.graph.errors import ErrorType, make_error
from app.graph.schemas import AuditEntry, ExtractedField, ProcessingErrorRecord

# Unicode dashes/quotes that OCR and LLMs use interchangeably with ASCII.
_DASHES = re.compile("[‐‑‒–—―−]")
_QUOTES = re.compile("[‘’‚‛′]")
_DQUOTES = re.compile("[“”„‟″]")
# Horizontal whitespace only. Newlines are NOT collapsed: a page's line
# structure is load-bearing here — it is how a table reads as rows, how the
# extraction model sees which label belongs to which value, and how a quoted
# piece of evidence stays recognisable to a human checking the page. Folding
# it away turns every page into one long line and silently costs all three.
_HSPACE = re.compile(r"[^\S\n]+")
_BLANKS = re.compile(r"\n{3,}")
_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def audit(
    node: str,
    event: str,
    detail: str = "",
    *,
    page_number: int | None = None,
    chunk_id: str | None = None,
    level: str = "info",
) -> list[AuditEntry]:
    return [AuditEntry(
        node=node, event=event, detail=detail[:2000],
        page_number=page_number, chunk_id=chunk_id, level=level,
    )]


def err(
    error_type: ErrorType,
    message: str,
    node: str,
    **kw: Any,
) -> list[ProcessingErrorRecord]:
    return [make_error(error_type, message, node, **kw)]


def now() -> float:
    return time.monotonic()


# --------------------------------------------------------------------------
# Text normalisation — one definition, used everywhere
# --------------------------------------------------------------------------


def clean_text(value: Any) -> str:
    """Fold the cosmetic differences OCR and LLMs introduce, keep the rest."""
    if value is None:
        return ""
    s = unicodedata.normalize("NFKC", str(value))
    s = _DASHES.sub("-", s)
    s = _QUOTES.sub("'", s)
    s = _DQUOTES.sub('"', s)
    s = _HSPACE.sub(" ", s)             # runs of spaces/tabs -> one space
    s = _BLANKS.sub("\n\n", s)          # 3+ blank lines -> one blank line
    return "\n".join(line.rstrip() for line in s.split("\n")).strip()


def normalise_for_match(value: Any) -> str:
    """Aggressive form for equality tests: lowercase, alphanumerics only.

    Drops ALL internal whitespace rather than collapsing it, because OCR is
    inconsistent about spaces around dashes and inside numbers ("40 - 129" vs
    "40-129", "IN 123" vs "IN123") and that is not a real difference in value.
    """
    return _NON_ALNUM.sub("", clean_text(value).lower())


def similarity(a: Any, b: Any) -> float:
    na, nb = normalise_for_match(a), normalise_for_match(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def snake_case(name: str) -> str:
    s = clean_text(name).lower()
    s = re.sub(r"[^\w\s-]", " ", s)
    s = re.sub(r"[\s\-]+", "_", s).strip("_")
    return re.sub(r"_+", "_", s) or "unnamed_field"


_PLACEHOLDERS = {
    "", "n/a", "na", "null", "none", "nil", "-", "--", "---", "tbd",
    "not available", "not applicable", "not found", "unknown", "no data",
}


def is_placeholder(value: Any) -> bool:
    """True for the sentinel strings a model returns instead of a real null.

    They are all truthy in Python, so without this they get written into a
    form as visible garbage.
    """
    if value is None:
        return True
    return clean_text(value).lower() in _PLACEHOLDERS


# --------------------------------------------------------------------------
# Field identity
# --------------------------------------------------------------------------


def field_uid(f: ExtractedField) -> str:
    """Stable identity for one extracted occurrence.

    Includes the occurrence index and the evidence, so two genuinely distinct
    rows of the same table do not collide — collapsing them is exactly the
    information loss this system exists to avoid.
    """
    basis = "|".join([
        snake_case(f.normalized_field_name or f.field_name),
        str(f.page_number),
        f.chunk_id or "",
        str(f.occurrence_index),
        normalise_for_match(f.value)[:80],
        normalise_for_match(f.exact_source_evidence)[:60],
    ])
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def content_hash(text: str) -> str:
    return hashlib.sha1(normalise_for_match(text).encode("utf-8")).hexdigest()[:16]


def infer_data_type(value: Any) -> str:
    s = clean_text(value)
    if not s:
        return "null"
    if re.fullmatch(r"[-+]?\d+", s):
        return "integer"
    if re.fullmatch(r"[-+]?\d*[.,]\d+", s):
        return "number"
    if re.search(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", s) or re.search(r"\b\d{4}-\d{2}-\d{2}\b", s):
        return "date"
    if s.lower() in {"true", "false", "yes", "no"}:
        return "boolean"
    return "string"


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def page_outline(page_text: dict[int, str], page_meta: dict, per_page_chars: int = 350) -> str:
    """Compact per-page digest for the structure-analysis prompt."""
    parts = []
    for page in sorted(page_text):
        meta = page_meta.get(page)
        flags = []
        if meta is not None:
            if getattr(meta, "has_table", False):
                flags.append("table")
            if getattr(meta, "ocr_applied", False):
                flags.append("ocr")
            status = getattr(meta, "status", None)
            if status is not None:
                flags.append(getattr(status, "value", str(status)))
        head = truncate(clean_text(page_text[page]), per_page_chars)
        parts.append(f"[page {page}{' | ' + ', '.join(flags) if flags else ''}]\n{head}")
    return "\n\n".join(parts)
