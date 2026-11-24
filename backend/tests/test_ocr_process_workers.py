"""ocr_worker_mode: threads by default, a spawned process pool when "process",
and any pool failure falls back to {} (sequential OCR). No real OCR runs."""
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import fitz
import pytest

from app.graph.config import get_graph_settings
from app.graph.nodes import preprocess


@pytest.fixture
def scanned_pdf(tmp_path):
    doc = fitz.open()
    for _ in range(3):
        page = doc.new_page()
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40), False)
        pix.clear_with(200)
        page.insert_image(page.rect, pixmap=pix)
    path = str(tmp_path / "scan.pdf")
    doc.save(path)
    return path


@pytest.fixture
def settings(monkeypatch):
    s = get_graph_settings()
    monkeypatch.setattr(s, "ocr_engine", "deepreef_local", raising=False)
    monkeypatch.setattr(s, "ocr_enabled", True, raising=False)
    monkeypatch.setattr(s, "ocr_page_workers", 3, raising=False)
    monkeypatch.setattr(preprocess, "_ocr_scanned_page",
                        lambda data, p, dpi, script: (f"page {p}", 0.9, "", "fake"))
    return s


def _no_process_pool(n):
    raise AssertionError("process pool must not be used in thread mode")


def test_thread_mode_is_default(scanned_pdf, settings, monkeypatch):
    assert type(settings).model_fields["ocr_worker_mode"].default == "thread"
    monkeypatch.setattr(settings, "ocr_worker_mode", "thread", raising=False)
    monkeypatch.setattr(preprocess, "_make_ocr_process_pool", _no_process_pool)
    out = preprocess._prefetch_scanned_pages(scanned_pdf, open(scanned_pdf, "rb").read(), settings, "english")
    assert out == {p: (f"page {p}", 0.9, "", "fake") for p in (1, 2, 3)}


def test_process_mode_uses_process_pool(scanned_pdf, settings, monkeypatch):
    monkeypatch.setattr(settings, "ocr_worker_mode", "process", raising=False)
    made, submitted = [], []

    class FakePool(ThreadPoolExecutor):  # stands in for ProcessPoolExecutor, no spawn
        def submit(self, fn, *args):
            submitted.append(fn)
            return super().submit(fn, *args)

    def factory(n):
        made.append(n)
        return FakePool(max_workers=n)

    monkeypatch.setattr(preprocess, "_make_ocr_process_pool", factory)
    out = preprocess._prefetch_scanned_pages(scanned_pdf, open(scanned_pdf, "rb").read(), settings, "english")
    assert made == [3] and set(submitted) == {preprocess._ocr_page_in_process}
    assert sorted(out) == [1, 2, 3]


def test_broken_process_pool_falls_back_to_sequential(scanned_pdf, settings, monkeypatch):
    monkeypatch.setattr(settings, "ocr_worker_mode", "process", raising=False)

    class BrokenPool(ThreadPoolExecutor):
        def submit(self, fn, *args):
            raise BrokenProcessPool("worker died")

    monkeypatch.setattr(preprocess, "_make_ocr_process_pool", lambda n: BrokenPool(max_workers=n))
    assert preprocess._prefetch_scanned_pages(scanned_pdf, open(scanned_pdf, "rb").read(), settings, "english") == {}
