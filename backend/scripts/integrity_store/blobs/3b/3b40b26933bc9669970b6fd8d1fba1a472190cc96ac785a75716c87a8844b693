from app.documents.chunking import Chunk
from app.documents.geometry import Rect
from app.documents.rule_extraction import extract_identifiers


def test_extracts_gst_pan_tan_from_real_looking_values():
    chunks = [
        Chunk(
            page_number=0,
            text="GSTIN: 27AAECS4821L1ZP\nPAN: AAECS4821L\nTAN: MUMS41287B",
            spans=[
                ("GSTIN: 27AAECS4821L1ZP", Rect(x=0, y=0, w=10, h=5)),
                ("PAN: AAECS4821L", Rect(x=0, y=10, w=10, h=5)),
                ("TAN: MUMS41287B", Rect(x=0, y=20, w=10, h=5)),
            ],
        )
    ]

    fields = extract_identifiers(chunks)
    by_path = {f.field: f for f in fields}

    assert by_path["organisation.gst_number"].value == "27AAECS4821L1ZP"
    assert by_path["organisation.pan_number"].value == "AAECS4821L"
    assert by_path["organisation.tan_number"].value == "MUMS41287B"
    assert all(f.source == "rule_based" and f.confidence == 1.0 for f in fields)


def test_gst_number_does_not_also_false_match_as_a_bare_pan():
    # The GSTIN embeds a PAN-shaped run of characters, but it's one contiguous
    # word with no internal \b boundary — the PAN pattern must not carve a
    # false match out of the middle of a GST number.
    chunks = [Chunk(page_number=0, text="GSTIN 27AAECS4821L1ZP only, no separate PAN here", spans=[])]

    fields = extract_identifiers(chunks)
    by_path = {f.field: f for f in fields}

    assert by_path["organisation.gst_number"].value == "27AAECS4821L1ZP"
    assert "organisation.pan_number" not in by_path


def test_first_match_wins_across_pages():
    chunks = [
        Chunk(page_number=0, text="PAN: AAECS4821L", spans=[]),
        Chunk(page_number=1, text="PAN: ZZZZZ9999Z", spans=[]),
    ]

    fields = extract_identifiers(chunks)

    assert len(fields) == 1
    assert fields[0].value == "AAECS4821L"
    assert fields[0].source_page is None  # no matching span provided to ground() in this fixture


def test_no_identifiers_present_returns_empty_list():
    chunks = [Chunk(page_number=0, text="Nothing identifier-shaped on this page at all.", spans=[])]

    assert extract_identifiers(chunks) == []
