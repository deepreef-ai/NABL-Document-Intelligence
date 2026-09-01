"""Extracts numbered-table rows from PyMuPDF/OCR-flattened page text.

PyMuPDF's plain-text mode (and OCR line output) has no real table structure
— a table becomes a flat sequence of lines, one per cell, table structure
gone. But NABL-style lab report tables have a strong, exploitable
regularity: each row starts with its own row number (1, 2, 3, ... as its
own line), the first cell after that is always the parameter/test name, and
the LAST cell is always the limit/requirement — it's the cells *in between*
(method, unit) that vary in count row to row, often collapsing to nothing
or a placeholder "-" for a qualitative test ("Cane Sugar" needs no unit,
"Milk Fat" needs "%"). So cells are assigned from both ends inward, not by
position from the start alone.

Example real row, 4 lines: "1", "Cane Sugar", "-", "Absent", "Shall be
absent" -> parameter="Cane Sugar", method_or_unit=["-"], result="Absent",
limit="Shall be absent".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_ROW_NUMBER = re.compile(r"^(\d{1,3})\.?$")  # "1" or "1." — both seen in real reports

# Different labs' report templates disagree on whether the table's LAST
# column is the limit/requirement (RESULT second-to-last, e.g. a milk
# report's "...RESULTS / REQUIREMENTS") or is itself the result with no
# separate limit column at all (RESULT last, e.g. a food-testing report's
# "...TEST PROTOCOL / RESULT"). Reading the header row actually printed
# just above the table settles it per-document instead of guessing once
# for every report.
_HEADER_KEYWORDS = {
    "parameter", "parameters", "test parameter", "test parameters",
    "method", "methods", "test method", "test protocol", "protocol",
    "unit", "units",
    "result", "results",
    "limit", "limits", "requirement", "requirements", "specification", "standard", "range",
    "s.no", "s. no", "sno", "sl.no", "sl. no", "sl no",
}
_LIMIT_LIKE_HEADER = {"limit", "limits", "requirement", "requirements", "specification", "standard", "range"}
_RESULT_LIKE_HEADER = {"result", "results"}
_HEADER_LOOKBACK_LINES = 8


_TRAILING_NOISE = re.compile(r"[\s:*#]+$")  # footnote markers ("REQUIREMENTS*") and stray punctuation


def _normalize_header_candidate(line: str) -> str:
    return _TRAILING_NOISE.sub("", line.strip()).lower()


def _last_column_is_limit(lines: list[str], first_row_index: int) -> bool:
    """Peek at the lines just before the table's first row for a
    recognizable header block, and use the LAST header label's own meaning.
    Defaults to True (the more commonly observed convention) when no
    header is recognized at all."""
    window = lines[max(0, first_row_index - _HEADER_LOOKBACK_LINES):first_row_index]
    header_lines = [normalized for line in window if (normalized := _normalize_header_candidate(line)) in _HEADER_KEYWORDS]
    if not header_lines:
        return True
    return header_lines[-1] not in _RESULT_LIKE_HEADER

# The widest real row seen (parameter, method, unit, result, limit) is 5
# cells after the row-number line itself. Once a "row" exceeds this by a
# comfortable margin, it isn't a row anymore — it's the table having ended
# and every following line (report remarks, signatures, sample metadata...)
# being wrongly swept into the last row for want of a terminating row N+1
# that will never come, since a table's last row has none.
MAX_CELLS_PER_ROW = 5


@dataclass
class TableRow:
    parameter: str
    result: str
    limit: str | None = None
    method_or_unit: list[str] = field(default_factory=list)
    row_number: int = 0


def _cells_to_row(cells: list[str], row_number: int, last_is_limit: bool = True) -> TableRow | None:
    if len(cells) < 2:
        return None  # just a parameter name with nothing else -> not a usable pair
    parameter = cells[0]
    if len(cells) == 2:
        return TableRow(parameter=parameter, result=cells[1], row_number=row_number)
    if not last_is_limit:
        return TableRow(parameter=parameter, result=cells[-1], method_or_unit=cells[1:-1], row_number=row_number)
    return TableRow(
        parameter=parameter,
        result=cells[-2],
        limit=cells[-1],
        method_or_unit=cells[1:-2],
        row_number=row_number,
    )


def extract_numbered_table_rows(text: str) -> list[TableRow]:
    """Finds ONE contiguous numbered sequence (1, 2, 3, ...) in the text and
    treats every line between consecutive row-number lines as that row's
    cells. Requires the sequence to actually start at 1 and increment by
    exactly 1 each time, specifically to avoid mistaking an unrelated
    standalone number (a page number, a pincode on its own line) elsewhere
    in the document for the start of a table that isn't really there."""
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]

    first_row_index = next(
        (i for i, line in enumerate(lines) if (m := _ROW_NUMBER.match(line)) and int(m.group(1)) == 1),
        None,
    )
    if first_row_index is None:
        return []  # no numbered table in this text at all
    last_is_limit = _last_column_is_limit(lines, first_row_index)

    rows: list[TableRow] = []
    current_cells: list[str] | None = None
    current_row_number: int | None = None
    expected_next = 1

    for line in lines:
        match = _ROW_NUMBER.match(line)
        if match and int(match.group(1)) == expected_next:
            if current_cells is not None:
                row = _cells_to_row(current_cells, current_row_number, last_is_limit)
                if row:
                    rows.append(row)
            current_row_number = expected_next
            current_cells = []
            expected_next += 1
            continue
        if current_cells is not None:
            current_cells.append(line)
            if len(current_cells) > MAX_CELLS_PER_ROW:
                # The table has ended — this "row" is actually the last real
                # row plus everything that came after it in the document.
                # Recover the real row from just the first MAX_CELLS_PER_ROW
                # cells, then stop: nothing past this point belongs to the table.
                row = _cells_to_row(current_cells[:MAX_CELLS_PER_ROW], current_row_number, last_is_limit)
                if row:
                    rows.append(row)
                current_cells = None
                break

    if current_cells is not None:
        row = _cells_to_row(current_cells, current_row_number, last_is_limit)
        if row:
            rows.append(row)

    return rows
