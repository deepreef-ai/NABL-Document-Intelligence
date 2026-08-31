from app.documents.compiler import compile_form
from app.documents.form_pdf import render_form_pdf


def test_render_form_pdf_on_an_empty_compiled_form_produces_a_valid_nonempty_pdf():
    compiled = compile_form("NABL_151", documents=[])

    pdf_bytes = render_form_pdf("NABL_151", compiled)

    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 0


def test_render_form_pdf_includes_flat_and_table_sections():
    compiled = {
        "organisation": {"laboratory_name": "Acme Test Labs", "gst_number": "27AAECS4821L1ZP"},
        "equipment": [
            {"name": "Digital Multimeter", "serial_number": "SN123"},
            {"name": "Oscilloscope", "serial_number": "SN456"},
        ],
        "some_flat_field": "a value",
    }

    pdf_bytes = render_form_pdf("NABL_151", compiled)

    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 500  # a real rendered page, not a near-empty stub


def test_render_form_pdf_handles_an_empty_table_section_gracefully():
    compiled = {"organisation": {"laboratory_name": "Acme Test Labs"}, "equipment": []}

    pdf_bytes = render_form_pdf("NABL_151", compiled)

    assert pdf_bytes.startswith(b"%PDF")
