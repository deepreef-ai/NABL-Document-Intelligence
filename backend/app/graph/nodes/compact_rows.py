"""Compact results-table extraction for the test_rows profile.

The general extraction prompt asks the model to write ~13 keys and a verbatim
evidence quote for every row, so it writes the document out two or three
times, and generation time grows with every character it writes. MEASURED
2026-11-23 on hp-lab-report.pdf: extraction was 92% of a 61s run.

Here the model is shown numbered lines and answers with positional arrays that
POINT at a line instead of quoting it:

    rows:   [[12, "Haemoglobin", "13.2", "g/dL", "13.0 - 17.0", "Photometry", ""]]
    panels: [[10, "COMPLETE BLOOD COUNT"]]
    header: [[3, "report_date", "12/11/2026"]]

Everything else is derived in code, deterministically:

- page number   — from the line's "--- Page N ---" block, so it cannot be wrong
- evidence      — the referenced line itself, verbatim from the document
- verification  — the test name and result must actually occur on that line
                  (or the line just before/after, for wrapped rows). A row
                  that fails is DROPPED and reported, never passed on. That is
                  stricter than the general path, whose test rows are not
                  evidence-checked at all, and it matches what a sealing
                  service needs: a value the page does not show must not be
                  attested.

The output is the same TestRow / ExtractedField shapes the rest of the graph
already consumes, so nothing downstream changes.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from pydantic import BaseModel, Field

from app.graph.prompts import COMPACT_HEADER_KEYS
from app.graph.schemas import ChunkRecord, ExtractedField, ExtractionStatus, TestRow

_PAGE_MARKER = re.compile(r"^--- Page (\d+) ---$")

#: Lines either side of the cited line searched for the RESULT. Kept tight:
#: the result is the value that gets attested, so it must sit on (or right
#: next to) the line the model pointed at.
_WINDOW_BEFORE = 1
_WINDOW_AFTER = 2

#: Lines either side searched for the TEST NAME, word by word. Long names
#: wrap over several lines and PDF text interleaves the other columns between
#: the pieces. MEASURED 2026-11-23 (high-protein-paneer.pdf, page 2):
#:     64| Copper Hydroxide (Copper 30 Max
#:     65| 8 determined as elemental IFSH/SOP/C/TE/165 mg/kg (Elemental - 0.05 0.12
#:     66| copper) copper)
#: An exact-substring name check refused 34 genuine rows like this one.
_NAME_WINDOW = 3


class CompactReply(BaseModel):
    """Lenient on purpose: cells are checked and coerced in code, so a number
    where text was expected costs nothing instead of the whole chunk."""

    panels: list[list[Any]] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    header: list[list[Any]] = Field(default_factory=list)


def number_lines(chunk: ChunkRecord) -> tuple[str, dict[int, tuple[int, str]]]:
    """Number the lines that belong to this chunk's own pages.

    Returns the prompt text and {line_number: (page, text)}. Lines from
    overlap pages (context carried in from the previous chunk) are shown with
    a "ctx|" prefix and no number, so they can be read but never cited — the
    previous chunk already owns them.
    """
    own_pages = set(chunk.page_numbers)
    current_page: int | None = None
    out: list[str] = []
    lines: dict[int, tuple[int, str]] = {}
    n = 0
    for raw in chunk.text.splitlines():
        text = raw.rstrip()
        marker = _PAGE_MARKER.match(text.strip())
        if marker:
            current_page = int(marker.group(1))
            out.append(text.strip())
            continue
        if not text.strip():
            continue
        if current_page in own_pages:
            n += 1
            lines[n] = (current_page, text.strip())
            out.append(f"{n}| {text.strip()}")
        else:
            out.append(f"ctx| {text.strip()}")
    return "\n".join(out), lines


def _cell(value: Any) -> str:
    if value is None or isinstance(value, bool):
        return ""
    return str(value).strip()


def _fold(text: str) -> str:
    """Comparison form: Unicode-folded, case-folded, all whitespace removed.
    PDF text extraction splits and joins spaces unpredictably ("13 .2",
    "g/ dL"), so spacing is not evidence either way; the characters are."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text).casefold())


def _line_no(value: Any) -> int | None:
    try:
        return int(str(value).strip().lstrip("L"))
    except (TypeError, ValueError):
        return None


def _window(lines: dict[int, tuple[int, str]], n: int,
            before: int = _WINDOW_BEFORE, after: int = _WINDOW_AFTER) -> str:
    page = lines[n][0]
    parts = [
        lines[i][1] for i in range(n - before, n + after + 1)
        if i in lines and lines[i][0] == page
    ]
    return " ".join(parts)


def _name_on_page(name: str, lines: dict[int, tuple[int, str]], n: int) -> bool:
    """Every word of the name appears near the cited line (see _NAME_WINDOW)."""
    near = _fold(_window(lines, n, _NAME_WINDOW, _NAME_WINDOW))
    if _fold(name) in near:
        return True
    words = re.findall(r"[0-9a-z]+", unicodedata.normalize("NFKC", name).casefold())
    return bool(words) and all(w in near for w in words)


def convert(
    reply: CompactReply,
    lines: dict[int, tuple[int, str]],
    chunk: ChunkRecord,
) -> tuple[list[TestRow], list[ExtractedField], list[str]]:
    """Reply -> (verified test rows, header fields, reasons rows were dropped)."""
    rejected: list[str] = []

    panels = sorted(
        (n, _cell(p[1])) for p in reply.panels
        if len(p) >= 2 and (n := _line_no(p[0])) in lines and _cell(p[1])
    )

    def panel_for(n: int) -> str | None:
        name = None
        for line, heading in panels:
            if line > n:
                break
            name = heading
        return name

    tests: list[TestRow] = []
    for row in reply.rows:
        if not row:
            continue
        n = _line_no(row[0])
        cells = [_cell(c) for c in row[1:7]] + [""] * (6 - len(row[1:7]))
        name, result, unit, ref, method, flag = cells
        if not (name or result):
            continue
        if n not in lines:
            rejected.append(f"{name or result!r}: cited line {row[0]!r} is not in this chunk")
            continue
        missing = []
        if name and not _name_on_page(name, lines, n):
            missing.append("test name")
        if result and _fold(result) not in _fold(_window(lines, n)):
            missing.append("result")
        if missing:
            rejected.append(
                f"{name or result!r} = {result!r}: {' and '.join(missing)} not found on "
                f"page {lines[n][0]} line {n}"
            )
            continue
        tests.append(TestRow(
            panel_name=panel_for(n),
            test_name=name or None,
            result=result or None,
            unit=unit or None,
            reference_range=ref or None,
            method=method or None,
            flag=flag or None,
            page_number=lines[n][0],
            exact_source_evidence=lines[n][1],
        ))

    fields: list[ExtractedField] = []
    allowed = set(COMPACT_HEADER_KEYS)
    for item in reply.header:
        if len(item) < 3:
            continue
        n, key, value = _line_no(item[0]), _cell(item[1]), _cell(item[2])
        if key not in allowed or not value or n not in lines:
            continue
        found = _fold(value) in _fold(_window(lines, n))
        fields.append(ExtractedField(
            field_name=key,
            normalized_field_name=key,
            value=value,
            page_number=lines[n][0],
            chunk_id=chunk.chunk_id,
            section_name="Report details",
            exact_source_evidence=lines[n][1],
            confidence_score=1.0 if found else 0.3,
            extraction_status=ExtractionStatus.VERIFIED if found else ExtractionStatus.UNCERTAIN,
            notes="" if found else "value not found on the cited line",
        ))
    return tests, fields, rejected
