"""Honest final status + image-region augmentation (2026-11-24)."""
import fitz
from langgraph.checkpoint.memory import MemorySaver

from app.graph.config import get_graph_settings
from app.graph.nodes import preprocess
from app.graph.runner import run_document
from app.graph.schemas import FinalStatus
from tests.graph.conftest import (  # noqa: F401 — fixtures
    _isolated_storage, _no_sleep, scripted, simple_pdf,
)


def _rows(user_text: str) -> dict:
    out = []
    for line in user_text.splitlines():
        if "|" in line and not line.startswith("ctx|"):
            n, text = line.split("|", 1)
            if "Protein:" in text:
                out.append([int(n), "Protein", "3.4", "g/100ml", "", "", ""])
    return {"panels": [], "rows": out, "header": []}


def test_test_rows_with_results_but_no_header_fields_is_not_incomplete(simple_pdf, scripted):
    """MEASURED: image reports with every test row were 'INCOMPLETE' because only
    header fields were counted."""
    scripted({"dynamic_extraction": _rows})
    r = run_document(simple_pdf, document_id="st1", checkpointer=MemorySaver(), profile="test_rows")
    assert r.tests and r.overall_status != FinalStatus.INCOMPLETE, r.final_reasoning


def test_test_rows_with_no_results_is_incomplete(simple_pdf, scripted):
    """A results reader that found no results must not say VALIDATED."""
    scripted({"dynamic_extraction": {"panels": [], "rows": [], "header": []}})
    r = run_document(simple_pdf, document_id="st2", checkpointer=MemorySaver(), profile="test_rows")
    assert r.overall_status == FinalStatus.INCOMPLETE
    assert "no test results" in r.final_reasoning


def test_unreadable_pages_block_validated(simple_pdf, scripted, monkeypatch):
    """MEASURED: a 30-page scan said 'VALIDATED' with 29 pages unreadable."""
    from app.graph.schemas import PageStatus

    real = preprocess.preprocess_document

    def one_page_unreadable(state):
        out = real(state)
        meta = out["page_metadata"][2]
        meta.status = PageStatus.OCR_FAILED
        out["page_text"][2] = ""
        return out

    monkeypatch.setattr("app.graph.graph.preprocess_document", one_page_unreadable, raising=False)
    from app.graph import graph as g
    monkeypatch.setattr(g, "preprocess_document", one_page_unreadable, raising=False)
    scripted({"dynamic_extraction": _rows})
    r = run_document(simple_pdf, document_id="st3", checkpointer=MemorySaver(), profile="test_rows")
    assert r.overall_status != FinalStatus.VALIDATED, r.final_reasoning


def test_image_regions_are_merged_and_tiny_images_ignored(tmp_path):
    doc = fitz.open()
    page = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40), False)
    pix.clear_with(180)
    page.insert_image(fitz.Rect(50, 50, 300, 120), pixmap=pix)    # letterhead band
    page.insert_image(fitz.Rect(250, 60, 400, 110), pixmap=pix)   # overlapping logo
    page.insert_image(fitz.Rect(10, 700, 14, 704), pixmap=pix)    # speck
    regions = preprocess._image_regions(page)
    assert len(regions) == 1 and regions[0].x0 <= 50 and regions[0].x1 >= 400


def test_region_augmentation_is_the_default():
    assert get_graph_settings().ocr_augment_regions_only is True
