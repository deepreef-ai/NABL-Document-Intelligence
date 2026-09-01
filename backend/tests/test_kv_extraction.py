from app.kv_extraction.table_rows import extract_numbered_table_rows


def test_basic_table_with_variable_column_counts():
    # Real shape from an actual dataset document: some rows have no
    # method/unit at all (qualitative "Absent" tests), others have both.
    text = (
        "Test Results:\n"
        "S. NO\nPARAMETERS\nMETHOD\nUNITS\nRESULTS\nREQUIREMENTS*\n"
        "1\nCane Sugar\n-\nAbsent\nShall be absent\n"
        "2\nMilk Fat\nIS 1479 (Part 2):1961\n%\n3.72\nMin. 3.20\n"
    )
    rows = extract_numbered_table_rows(text)

    assert len(rows) == 2
    assert rows[0].parameter == "Cane Sugar"
    assert rows[0].result == "Absent"
    assert rows[0].limit == "Shall be absent"
    assert rows[0].method_or_unit == ["-"]

    assert rows[1].parameter == "Milk Fat"
    assert rows[1].result == "3.72"
    assert rows[1].limit == "Min. 3.20"
    assert rows[1].method_or_unit == ["IS 1479 (Part 2):1961", "%"]


def test_result_last_convention_when_header_has_no_limit_column():
    # A different real report's header: RESULT is the last column, no
    # separate limit/requirement column exists at all.
    text = (
        "RESULTS OF THE ANALYSIS\n"
        "S.NO\nTEST PARAMETERS\nUNIT\nTEST PROTOCOL\nRESULT\n"
        "1\nMoisture\ng/100g\nAOAC 22nd Edition 925.10: 2019\n2.91\n"
        "2\nTotal Ash\ng/100g\nAOAC 22nd Edition 923.03: 2019\n3.70\n"
    )
    rows = extract_numbered_table_rows(text)

    assert len(rows) == 2
    assert rows[0].parameter == "Moisture"
    assert rows[0].result == "2.91"
    assert rows[0].limit is None
    assert rows[1].parameter == "Total Ash"
    assert rows[1].result == "3.70"


def test_row_numbers_with_trailing_period_are_recognized():
    text = "S.NO\nPARAMETER\nRESULT\n1.\nMoisture\n2.91\n2.\nAsh\n3.70\n"
    rows = extract_numbered_table_rows(text)
    assert [r.parameter for r in rows] == ["Moisture", "Ash"]
    assert [r.result for r in rows] == ["2.91", "3.70"]


def test_table_end_is_detected_not_swallowed_into_the_last_row():
    """The actual bug found on real data: without an explicit row N+1, the
    last row would otherwise absorb every remaining line in the document
    (remarks, signatures, sample metadata) into itself."""
    text = (
        "S. NO\nPARAMETERS\nMETHOD\nUNITS\nRESULTS\nREQUIREMENTS*\n"
        "1\nMilk Fat\nIS 1479\n%\n3.72\nMin. 3.20\n"
        "Note: Min. - Minimum\n"
        "*As Per FSS Regulations 2011\n"
        "REMARKS: The sample complies.\n"
        "Verified by\nAuthorised Signatory\n"
    )
    rows = extract_numbered_table_rows(text)

    assert len(rows) == 1
    assert rows[0].parameter == "Milk Fat"
    assert rows[0].result == "3.72"
    assert rows[0].limit == "Min. 3.20"
    assert "Note" not in str(rows[0].method_or_unit) and "REMARKS" not in str(rows[0].limit)


def test_no_table_returns_empty_list():
    text = "Dear Sir,\n\nWe are pleased to enclose the enclosed test report for your review.\n\nRegards,\nLab Director"
    assert extract_numbered_table_rows(text) == []


def test_a_lone_unrelated_number_does_not_start_a_fake_table():
    text = "Page 12 of the annual report was updated.\nThank you for your patience.\n"
    assert extract_numbered_table_rows(text) == []


def test_row_with_only_a_parameter_and_no_further_cells_is_skipped():
    # A row number immediately followed by another row number (or nothing) has no data to pair.
    text = "S.NO\nPARAMETER\nRESULT\n1\n2\nSomething\nValue\n"
    rows = extract_numbered_table_rows(text)
    # Row 1 had zero cells (immediately followed by row 2) -> skipped; row 2 is real.
    assert len(rows) == 1
    assert rows[0].parameter == "Something"
    assert rows[0].result == "Value"
