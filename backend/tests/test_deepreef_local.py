"""deepreef-ocr in-process (English) as the scanned-page OCR engine.
The engine itself is faked: these tests cover routing, fallback, parallel
prefetch and the stand-in/pinned model labelling, not OCR quality."""
import hashlib
import threading

import fitz
import pytest

from app.documents import deepreef_local
from app.graph.config import get_graph_settings
from app.graph.nodes import preprocess


@pytest.fixture
def scanned_pdf(tmp_path):
    """Three pages with no text layer (images only), like a scanned report."""
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
def deepreef_on(monkeypatch):
    s = get_graph_settings()
    monkeypatch.setattr(s, "ocr_engine", "deepreef_local", raising=False)
    monkeypatch.setattr(s, "ocr_page_workers", 3, raising=False)
    return s


def _fake_page(text="45 Copper Hydroxide mg/kg SOP-CHM-27-02 17.16"):
    return deepreef_local.PageOcr(text=text, lines=text.splitlines(), confidence=0.95,
                                  model_used="devanagari_rec.onnx")


def test_english_scan_is_read_by_deepreef_local(scanned_pdf, deepreef_on, monkeypatch):
    monkeypatch.setattr(deepreef_local, "extract_page", lambda image, **kw: _fake_page())
    data = open(scanned_pdf, "rb").read()
    text, conf, error, engine = preprocess._ocr_scanned_page(data, 1, 72, "english")
    assert text.startswith("45 Copper Hydroxide") and error == ""
    assert engine.startswith("deepreef-local:en")


def test_a_failing_deepreef_falls_back_to_rapidocr(scanned_pdf, deepreef_on, monkeypatch):
    def boom(image, **kw):
        raise deepreef_local.DeepreefLocalError("model missing")
    monkeypatch.setattr(deepreef_local, "extract_page", boom)
    monkeypatch.setattr(preprocess, "_ocr_page", lambda data, page, dpi: ("rapid text", 0.9, ""))
    _, _, _, engine = preprocess._ocr_scanned_page(open(scanned_pdf, "rb").read(), 1, 72, "english")
    assert engine == "rapidocr(fallback)"


def test_rapidocr_setting_keeps_the_old_path(scanned_pdf, monkeypatch):
    monkeypatch.setattr(get_graph_settings(), "ocr_engine", "rapidocr", raising=False)
    monkeypatch.setattr(deepreef_local, "extract_page", lambda image, **kw: pytest.fail("not this engine"))
    monkeypatch.setattr(preprocess, "_ocr_page", lambda data, page, dpi: ("rapid text", 0.9, ""))
    _, _, _, engine = preprocess._ocr_scanned_page(open(scanned_pdf, "rb").read(), 1, 72, "english")
    assert engine == "rapidocr"


def test_scanned_pages_are_read_in_parallel(scanned_pdf, deepreef_on, monkeypatch):
    seen_threads, barrier = set(), threading.Barrier(3, timeout=5)

    def fake(image, **kw):
        barrier.wait()                      # all three pages must be in flight at once
        seen_threads.add(threading.current_thread().name)
        return _fake_page()

    monkeypatch.setattr(deepreef_local, "extract_page", fake)
    out = preprocess._prefetch_scanned_pages(scanned_pdf, open(scanned_pdf, "rb").read(),
                                             deepreef_on, "english")
    assert sorted(out) == [1, 2, 3] and len(seen_threads) == 3


def test_no_prefetch_for_other_scripts_or_engines(scanned_pdf, deepreef_on, monkeypatch):
    data = open(scanned_pdf, "rb").read()
    assert preprocess._prefetch_scanned_pages(scanned_pdf, data, deepreef_on, "devanagari") == {}
    monkeypatch.setattr(deepreef_on, "ocr_engine", "rapidocr", raising=False)
    assert preprocess._prefetch_scanned_pages(scanned_pdf, data, deepreef_on, "english") == {}


def test_model_report_tells_pinned_from_stand_in(tmp_path, monkeypatch):
    repo = tmp_path / "deepreef-ocr"
    models = repo / "deepreef_ocr" / "models"
    models.mkdir(parents=True)
    real = b"the pinned model"
    (models / "devanagari_rec.onnx").write_bytes(real)
    (repo / "models.sha256").write_text(f"{hashlib.sha256(real).hexdigest()}  devanagari_rec.onnx\n")
    monkeypatch.setattr(deepreef_local, "_repo_dir", lambda: str(repo))
    monkeypatch.setattr(deepreef_local, "_model_dir", lambda: str(models))

    monkeypatch.setattr(deepreef_local, "_model_report", None)
    assert deepreef_local.model_report()["genuine"] is True

    (models / "devanagari_rec.onnx").write_bytes(b"some other model")
    monkeypatch.setattr(deepreef_local, "_model_report", None)
    report = deepreef_local.model_report()
    assert report["genuine"] is False and report["files"]["devanagari_rec.onnx"] == "stand-in"


# ------------------------------------------- B3/B4: render size, clean-up

def test_render_for_ocr_targets_the_engine_size(scanned_pdf, monkeypatch):
    import numpy as np, cv2
    monkeypatch.setattr(get_graph_settings(), "ocr_render_to_target", True, raising=False)
    png = deepreef_local.render_for_ocr(open(scanned_pdf, "rb").read(), 1, 200)
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    assert max(img.shape[:2]) <= 1536 and max(img.shape[:2]) > 1400   # A4/Letter at ~131 DPI


def test_render_for_ocr_can_be_turned_off(scanned_pdf, monkeypatch):
    monkeypatch.setattr(get_graph_settings(), "ocr_render_to_target", False, raising=False)
    calls = {}
    from app.documents import pdf_utils
    monkeypatch.setattr(pdf_utils, "rasterize_page", lambda data, idx, dpi=200: calls.setdefault("dpi", dpi) and b"png")
    deepreef_local.render_for_ocr(open(scanned_pdf, "rb").read(), 1, 200)
    assert calls["dpi"] == 200


def test_clean_scan_profile_applies_to_pdf_renders_only(monkeypatch):
    seen = []
    import types, sys as _sys
    fake = types.ModuleType("deepreef_ocr.preprocessor")
    fake.decode_and_resize = lambda data, **kw: (seen.append(kw) or ("img", {}))
    monkeypatch.setitem(_sys.modules, "deepreef_ocr.preprocessor", fake)
    monkeypatch.setattr(deepreef_local, "_import_engine", lambda: None)
    monkeypatch.setattr(get_graph_settings(), "ocr_clean_scan_profile", True, raising=False)
    deepreef_local.prepare_image(b"x", rendered_pdf_page=True)
    deepreef_local.prepare_image(b"x", rendered_pdf_page=False)          # a photo keeps every step
    assert seen[0] == deepreef_local._CLEAN_SCAN and seen[1] == {}


def test_an_approved_derived_model_is_genuine_and_labelled(tmp_path, monkeypatch):
    repo = tmp_path / "deepreef-ocr"
    models = repo / "deepreef_ocr" / "models"
    models.mkdir(parents=True)
    int8 = b"int8 build"
    (models / "devanagari_rec.onnx").write_bytes(int8)
    (repo / "models.sha256").write_text("0" * 64 + "  devanagari_rec.onnx\n")       # pinned = something else
    digest = hashlib.sha256(int8).hexdigest()
    monkeypatch.setattr(deepreef_local, "APPROVED_DERIVED_MODELS",
                        {digest: {"file": "devanagari_rec.onnx", "variant": "INT8"}})
    monkeypatch.setattr(deepreef_local, "_repo_dir", lambda: str(repo))
    monkeypatch.setattr(deepreef_local, "_model_dir", lambda: str(models))
    monkeypatch.setattr(deepreef_local, "_model_report", None)
    report = deepreef_local.model_report()
    assert report["genuine"] is True and report["variant"] == "INT8"
    assert report["files"]["devanagari_rec.onnx"] == "approved-INT8"


def test_the_shipped_int8_approval_is_tied_to_the_pinned_model():
    entry = deepreef_local.APPROVED_DERIVED_MODELS[
        "b07bbad325bb181fee14ab004e77bc48b92916a7af40aa8c3b1e468b7f5a7601"]
    assert entry["file"] == "devanagari_rec.onnx" and entry["variant"] == "INT8"
    assert entry["derived_from"].startswith("d6f0a906")
