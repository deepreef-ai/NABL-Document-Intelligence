"""LOCKED: a reported value is a number with a sign and a comparator, and a
test row is only accepted when its result is printed on the line it cites."""
from __future__ import annotations

import pytest

from app.documents.value_matching import normalize_for_search, strict_value_key, value_in_source
from app.graph.nodes import helpers
from app.graph.nodes.compact_rows import CompactReply, convert, number_lines
from app.graph.schemas import ChunkRecord, ChunkStatus


@pytest.mark.parametrize("a, b", [("1.5", "0.15"), ("-5", "5"), ("<0.01", "0.01"), ("1.5", "15")])
def test_different_numbers_are_never_the_same_value(a, b):
    """Why this must never break: an attestation that treats 1.5 as 0.15 (a
    10x error), -5 as 5, or "<0.01" (below detection) as a measured 0.01
    certifies a result the lab never reported. Both value keys (conflict
    detection and evidence matching) and the page-text search must stay strict,
    while a pure notation difference (mg/m2 vs mg/m², 1.50 vs 1.5) must NOT
    create a false disagreement that floods a reviewer with noise."""
    for key in (strict_value_key, helpers.strict_value_key):
        assert key(a) != key(b), (key.__module__, a, b)
    assert not value_in_source(b, normalize_for_search(f"Result {a} mg/kg"))
    assert value_in_source(a, normalize_for_search(f"Result {a} mg/kg"))
    for key in (strict_value_key, helpers.strict_value_key):
        assert key("mg/m2") == key("mg/m²")
        assert key("1.50") == key("1.5")


_WRAPPED = (
    "--- Page 2 ---\n"
    "7 Methyl Mercury as element IFSH/SOP/C/TE/181 mg/kg Max 0.25 - 0.10 Not Detected\n"
    "Copper Hydroxide (Copper 30 Max\n"
    "8 determined as elemental IFSH/SOP/C/TE/165 mg/kg (Elemental - 0.05 0.12\n"
    "copper) copper)\n"
)


def _chunk() -> ChunkRecord:
    return ChunkRecord(chunk_id="c1", page_numbers=[2], text=_WRAPPED, char_count=len(_WRAPPED),
                       status=ChunkStatus.PENDING, content_hash="h")


def _convert(rows):
    chunk = _chunk()
    _, lines = number_lines(chunk)
    return convert(CompactReply(rows=rows), lines, chunk)


def test_a_result_not_on_its_cited_line_is_refused():
    """Why this must never break: the model's reply is a claim, not evidence.
    A result that is not printed on the line the model cites (digits swapped,
    a line that does not exist) is an invented
    reading and must be dropped and reported, never shown as verified."""
    tests, _, dropped = _convert([
        [3, "Copper Hydroxide (Copper determined as elemental copper)", "0.21", "mg/kg", "", "", ""],
        [99, "GHOST", "1.0", "", "", "", ""],                               # no such line
    ])
    assert tests == []
    assert len(dropped) == 2 and "result not found" in dropped[0]


def test_a_long_name_wrapped_across_interleaved_lines_is_accepted():
    """Why this must never break: real NABL tables wrap long analyte names
    around other columns. Refusing those rows silently loses genuine results
    (MEASURED: 34 real rows on one report), which is as bad for an attestation
    as accepting a false one. The RESULT stays strict; only the name is lenient."""
    tests, _, dropped = _convert([
        [3, "Copper Hydroxide (Copper determined as elemental copper)",
         "0.12", "mg/kg", "30 Max", "IFSH/SOP/C/TE/165", ""],
    ])
    assert dropped == []
    assert [(t.result, t.page_number) for t in tests] == [("0.12", 2)]
