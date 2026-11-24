"""E1/E2 wiring: remote OCR engines read every scanned page in one batch, and
a page the remote side fails is left for the local loop (never lost)."""
import fitz
import pytest

from app.documents import deepreef_local
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
def remote(monkeypatch):
    s = get_graph_settings()
    monkeypatch.setattr(deepreef_local, "render_for_ocr", lambda data, p, dpi: f"png{p}".encode())
    return s


def test_service_engine_batches_pages_and_drops_failures(scanned_pdf, remote, monkeypatch):
    from app.documents import ocr_service_client

    monkeypatch.setattr(remote, "ocr_engine", "deepreef_service", raising=False)
    sent = {}

    def fake(images, **kw):
        sent.update(images)
        return {1: deepreef_local.PageOcr(text="a", lines=["Protein 79.77 g/100g"], confidence=0.9,
                                          model_used="m"),
                2: RuntimeError("service down"),
                3: deepreef_local.PageOcr(text="b", lines=["Melamine <0.5"], confidence=0.9, model_used="m")}

    monkeypatch.setattr(ocr_service_client, "extract_pages_remote", fake)
    out = preprocess._prefetch_scanned_pages(scanned_pdf, open(scanned_pdf, "rb").read(), remote, "english")
    assert sorted(sent) == [1, 2, 3], "all pages go in one batch"
    assert sorted(out) == [1, 3], "the failed page is left for the local loop"
    assert out[1][0] == "Protein 79.77 g/100g" and out[1][3].startswith("deepreef-service:en")


def test_lambda_engine_batches_pages_and_drops_failures(scanned_pdf, remote, monkeypatch):
    from app.documents import deepreef_lambda

    monkeypatch.setattr(remote, "ocr_engine", "deepreef_lambda", raising=False)
    monkeypatch.setattr(deepreef_lambda, "extract_pages_lambda",
                        lambda images, **kw: {p: (("Copper 17.16", 0.95, "") if p != 2
                                                  else ("", None, "AccessDenied")) for p in images})
    out = preprocess._prefetch_scanned_pages(scanned_pdf, open(scanned_pdf, "rb").read(), remote, "english")
    assert sorted(out) == [1, 3] and out[3] == ("Copper 17.16", 0.95, "", "deepreef-lambda:en")


def test_a_remote_engine_that_is_down_returns_nothing_not_an_error(scanned_pdf, remote, monkeypatch):
    from app.documents import ocr_service_client

    monkeypatch.setattr(remote, "ocr_engine", "deepreef_service", raising=False)

    def down(images, **kw):
        raise ConnectionError("refused")

    monkeypatch.setattr(ocr_service_client, "extract_pages_remote", down)
    assert preprocess._prefetch_scanned_pages(scanned_pdf, open(scanned_pdf, "rb").read(), remote, "english") == {}
