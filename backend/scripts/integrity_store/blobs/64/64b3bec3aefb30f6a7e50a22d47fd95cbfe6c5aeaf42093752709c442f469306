from app.documents import local_ocr, pdf_utils
from app.documents.chunking import build_chunks
from app.documents.geometry import Rect
from app.documents.ocr_client import OcrResult
from app.documents.pdf_utils import PageText


def test_page_with_plenty_of_text_is_kept_as_is_no_ocr_call(monkeypatch):
    pages = [PageText(page_number=0, text="a" * 200, spans=[("a" * 200, Rect(x=0, y=0, w=1, h=1))])]

    def fail_rasterize(*a, **k):
        raise AssertionError("should not rasterize a page that already has enough text")

    monkeypatch.setattr(pdf_utils, "rasterize_page", fail_rasterize)

    chunks = build_chunks(b"%PDF-fake", pages)

    assert len(chunks) == 1
    assert chunks[0].text == "a" * 200
    assert chunks[0].page_number == 0


def test_thin_text_page_falls_back_to_local_ocr(monkeypatch):
    pages = [PageText(page_number=0, text="  ", spans=[])]  # below MIN_PAGE_TEXT_CHARS
    monkeypatch.setattr(pdf_utils, "rasterize_page", lambda data, page_number: b"png-bytes")
    monkeypatch.setattr(
        local_ocr,
        "extract_english",
        lambda png: OcrResult(
            text="OCR recovered text",
            lines=["OCR recovered text"],
            confidence=0.9,
            boxes=[Rect(x=0, y=0, w=5, h=5)],
            model_used="paddleocr-en",
            region_count=1,
        ),
    )

    chunks = build_chunks(b"%PDF-fake", pages)

    assert chunks[0].text == "OCR recovered text"
    assert chunks[0].spans == [("OCR recovered text", Rect(x=0, y=0, w=5, h=5))]


def test_thin_text_page_keeps_original_text_when_ocr_fails(monkeypatch):
    pages = [PageText(page_number=0, text="x", spans=[])]
    monkeypatch.setattr(pdf_utils, "rasterize_page", lambda data, page_number: b"png-bytes")

    def fail_ocr(png):
        raise local_ocr.LocalOcrError("paddleocr not installed")

    monkeypatch.setattr(local_ocr, "extract_english", fail_ocr)

    chunks = build_chunks(b"%PDF-fake", pages)

    assert chunks[0].text == "x"  # best-effort: original (thin) text kept, page not lost


def test_thin_text_page_keeps_original_when_ocr_isnt_actually_better(monkeypatch):
    # OCR "succeeds" but returns even less text than the original — original wins.
    pages = [PageText(page_number=0, text="x" * 10, spans=[])]
    monkeypatch.setattr(pdf_utils, "rasterize_page", lambda data, page_number: b"png-bytes")
    monkeypatch.setattr(
        local_ocr,
        "extract_english",
        lambda png: OcrResult(text="", lines=[], confidence=0.0, boxes=[], model_used="paddleocr-en", region_count=0),
    )

    chunks = build_chunks(b"%PDF-fake", pages)

    assert chunks[0].text == "x" * 10


def test_multiple_pages_preserve_order_and_page_numbers(monkeypatch):
    pages = [
        PageText(page_number=0, text="a" * 200, spans=[]),
        PageText(page_number=1, text="b" * 200, spans=[]),
        PageText(page_number=2, text="c" * 200, spans=[]),
    ]

    chunks = build_chunks(b"%PDF-fake", pages)

    assert [c.page_number for c in chunks] == [0, 1, 2]
    assert [c.text for c in chunks] == ["a" * 200, "b" * 200, "c" * 200]
