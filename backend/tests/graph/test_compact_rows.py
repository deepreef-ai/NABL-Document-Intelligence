"""Compact results-table extraction (test_rows profile): numbered lines in,
positional arrays out, every row verified against the line it cites."""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from app.graph.nodes.compact_rows import CompactReply, convert, number_lines
from app.graph.runner import run_document
from app.graph.schemas import ChunkRecord, ChunkStatus, ExtractionStatus


def _chunk(text: str, pages: list[int]) -> ChunkRecord:
    return ChunkRecord(chunk_id="c1", page_numbers=pages, text=text, char_count=len(text),
                       status=ChunkStatus.PENDING, content_hash="h")


_TEXT = (
    "[context from preceding page(s), do not extract as new]\n"
    "--- Page 1 ---\nOld row 9.9 mg/dL\n\n"
    "--- Page 2 ---\n"
    "Report Date: 12/11/2026\n"
    "LIVER FUNCTION TEST, SERUM\n"
    "BILIRUBIN, TOTAL 0.70 mg/dL < 1.1 Diazo\n"
    "ALKALINE\n"
    "PHOSPHATASE 88 U/L 40 - 129 IFCC\n"
)


def _lines():
    return number_lines(_chunk(_TEXT, [2]))


def test_only_the_chunks_own_pages_are_numbered():
    text, lines = _lines()
    assert "ctx| Old row 9.9 mg/dL" in text           # readable, not citable
    assert all(page == 2 for page, _ in lines.values())
    assert lines[3] == (2, "BILIRUBIN, TOTAL 0.70 mg/dL < 1.1 Diazo")


def test_a_row_is_verified_and_gets_page_panel_and_evidence_from_its_line():
    _, lines = _lines()
    reply = CompactReply(panels=[[2, "LIVER FUNCTION TEST, SERUM"]],
                         rows=[[3, "BILIRUBIN, TOTAL", "0.70", "mg/dL", "< 1.1", "Diazo", ""]])
    tests, _, dropped = convert(reply, lines, _chunk(_TEXT, [2]))
    assert dropped == []
    row = tests[0]
    assert (row.result, row.page_number, row.panel_name) == ("0.70", 2, "LIVER FUNCTION TEST, SERUM")
    assert row.exact_source_evidence == "BILIRUBIN, TOTAL 0.70 mg/dL < 1.1 Diazo"


def test_a_value_the_line_does_not_show_is_dropped_and_reported():
    _, lines = _lines()
    reply = CompactReply(rows=[[3, "BILIRUBIN, TOTAL", "0.07", "mg/dL", "", "", ""],   # digits swapped
                               [99, "GHOST", "1.0", "", "", "", ""]])                 # no such line
    tests, _, dropped = convert(reply, lines, _chunk(_TEXT, [2]))
    assert tests == []
    assert len(dropped) == 2 and "not found on page 2 line 3" in dropped[0]


def test_a_name_wrapped_onto_the_previous_line_still_verifies():
    _, lines = _lines()
    reply = CompactReply(rows=[[5, "ALKALINE PHOSPHATASE", 88, "U/L", "40 - 129", "IFCC", ""]])
    tests, _, dropped = convert(reply, lines, _chunk(_TEXT, [2]))
    assert dropped == [] and tests[0].result == "88"            # number coerced to text


def test_header_keys_are_limited_and_checked():
    _, lines = _lines()
    reply = CompactReply(header=[[1, "report_date", "12/11/2026"],
                                 [1, "favourite_colour", "blue"],      # not an allowed key
                                 [1, "patient_name", "Nobody"]])       # not on that line
    _, fields, _ = convert(reply, lines, _chunk(_TEXT, [2]))
    by_name = {f.field_name: f for f in fields}
    assert set(by_name) == {"report_date", "patient_name"}
    assert by_name["report_date"].extraction_status == ExtractionStatus.VERIFIED
    assert by_name["patient_name"].extraction_status == ExtractionStatus.UNCERTAIN


# ----------------------------------------------------------- whole graph


def _compact_reply(user_text: str) -> dict:
    rows, header = [], []
    for line in user_text.splitlines():
        if "|" not in line or line.startswith("ctx|"):
            continue
        n, text = line.split("|", 1)
        if "Protein:" in text:
            rows.append([int(n), "Protein", "3.4", "g/100ml", "", "", ""])
        if "SNF:" in text:
            rows.append([int(n), "SNF", "9.9", "%", "", "", ""])              # page says 8.6
        if "Report No:" in text:
            header.append([int(n), "report_number", "LR-2024-0195"])
    return {"panels": [], "rows": rows, "header": header}


def test_whole_run_keeps_real_rows_and_refuses_invented_ones(simple_pdf, scripted):
    stub = scripted({"dynamic_extraction": _compact_reply})
    result = run_document(simple_pdf, document_id="compact", checkpointer=MemorySaver(),
                          profile="test_rows")

    assert {c["node"] for c in stub.calls} == {"dynamic_extraction"}
    rows = {(t.test_name, t.result) for t in result.tests}
    assert ("Protein", "3.4") in rows
    assert not any(name == "SNF" for name, _ in rows), "a value the page does not show"
    assert any(e.event == "rows_dropped" and "1 row(s)" in e.detail for e in result.audit_log)
    assert any(f.field_name == "report_number" for f in result.extracted_fields)


# ------------------------------------------- names wrapped across columns

_WRAPPED = (
    "--- Page 2 ---\n"
    "7 Methyl Mercury as element IFSH/SOP/C/TE/181 mg/kg Max 0.25 - 0.10 Not Detected\n"
    "Copper Hydroxide (Copper 30 Max\n"
    "8 determined as elemental IFSH/SOP/C/TE/165 mg/kg (Elemental - 0.05 0.12\n"
    "copper) copper)\n"
)


def test_a_name_split_across_interleaved_lines_is_accepted():
    """MEASURED 2026-11-23 on high-protein-paneer.pdf: 34 genuine rows like
    this were refused by an exact-substring name check."""
    chunk = _chunk(_WRAPPED, [2])
    _, lines = number_lines(chunk)
    reply = CompactReply(rows=[[3, "Copper Hydroxide (Copper determined as elemental copper)",
                                "0.12", "mg/kg", "30 Max", "IFSH/SOP/C/TE/165", ""]])
    tests, _, dropped = convert(reply, lines, chunk)
    assert dropped == [] and tests[0].result == "0.12"


def test_the_result_stays_strict_when_the_name_is_lenient():
    chunk = _chunk(_WRAPPED, [2])
    _, lines = number_lines(chunk)
    reply = CompactReply(rows=[[3, "Copper Hydroxide (Copper determined as elemental copper)",
                                "0.21", "mg/kg", "", "", ""]])                  # page says 0.12
    tests, _, dropped = convert(reply, lines, chunk)
    assert tests == [] and "result not found" in dropped[0]


def test_a_page_with_only_test_rows_is_not_sent_back_for_reanalysis(simple_pdf, scripted):
    def rows_only(user_text: str) -> dict:
        rows = []
        for line in user_text.splitlines():
            if "|" in line and not line.startswith("ctx|"):
                n, text = line.split("|", 1)
                if "Fat Content:" in text:
                    rows.append([int(n), "Fat Content", "4.2", "%", "", "", ""])
                if "Protein:" in text:
                    rows.append([int(n), "Protein", "3.4", "g/100ml", "", "", ""])
        return {"panels": [], "rows": rows, "header": []}       # no header fields at all

    stub = scripted({"dynamic_extraction": rows_only})
    result = run_document(simple_pdf, document_id="rows-only-pages", checkpointer=MemorySaver(),
                          profile="test_rows")
    assert len(stub.calls) == 1, "one extraction call per chunk, no re-analysis round"
    assert not any(e.event == "chunks_requeued" for e in result.audit_log)


def test_a_text_only_end_page_does_not_trigger_reanalysis(tmp_path, scripted):
    """MEASURED 2026-11-23: every short report's 'End of Report' page read as
    barren, its chunk was re-read twice and the table rows tripled."""
    from tests.graph.conftest import _write_pdf

    pdf = _write_pdf(str(tmp_path / "two.pdf"), [
        "RESULTS\nProtein: 3.4 g/100ml\n",
        "Note: This report is valid for the tested sample only.\n" * 12 + "End Of Report\n",
    ])

    def rows(user_text: str) -> dict:
        out = []
        for line in user_text.splitlines():
            if "|" in line and not line.startswith("ctx|"):
                n, text = line.split("|", 1)
                if "Protein:" in text:
                    out.append([int(n), "Protein", "3.4", "g/100ml", "", "", ""])
        return {"panels": [], "rows": out, "header": []}

    stub = scripted({"dynamic_extraction": rows})
    result = run_document(pdf, document_id="end-page", checkpointer=MemorySaver(), profile="test_rows")
    assert [t.test_name for t in result.tests] == ["Protein"], "no duplicate rows"
    assert len(stub.calls) == 1
    assert any(e.event == "reanalysis_skipped" for e in result.audit_log)
