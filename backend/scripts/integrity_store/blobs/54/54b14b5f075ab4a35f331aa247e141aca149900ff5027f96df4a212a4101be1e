import fitz

from app.documents import pdf_utils


def _make_pdf(text: str | None) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    if text:
        page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_has_text_layer_true_for_born_digital_pdf():
    pdf_bytes = _make_pdf("GST Number: 27ABCDE1234F1Z5")
    assert pdf_utils.has_text_layer(pdf_bytes) is True


def test_has_text_layer_false_for_blank_scanned_pdf():
    pdf_bytes = _make_pdf(None)
    assert pdf_utils.has_text_layer(pdf_bytes) is False


def test_extract_text_and_boxes_finds_the_span_and_its_rect():
    pdf_bytes = _make_pdf("GST Number: 27ABCDE1234F1Z5")
    pages = pdf_utils.extract_text_and_boxes(pdf_bytes)
    assert len(pages) == 1
    joined = " ".join(text for text, _rect in pages[0].spans)
    assert "GST" in joined
    for _text, rect in pages[0].spans:
        assert rect.w > 0
        assert rect.h > 0


def test_rasterize_page_returns_a_png():
    pdf_bytes = _make_pdf("hello")
    png = pdf_utils.rasterize_page(pdf_bytes, 0)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_page_count():
    pdf_bytes = _make_pdf("hello")
    assert pdf_utils.page_count(pdf_bytes) == 1
