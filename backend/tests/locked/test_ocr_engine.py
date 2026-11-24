"""LOCKED: scanned-page OCR routing, model approval labelling and the OCR page
cache. The engine is faked; these guard behaviour, not OCR quality."""
from __future__ import annotations

import hashlib

import fitz
import pytest

from app.documents import deepreef_local, ocr_page_cache
from app.graph.config import get_graph_settings
from app.graph.nodes import preprocess


@pytest.fixture
def scanned_pdf_bytes():
    doc = fitz.open()
    page = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40), False)
    pix.clear_with(200)
    page.insert_image(page.rect, pixmap=pix)
    return doc.tobytes()


def test_a_failing_deepreef_falls_back_to_rapidocr_and_the_page_is_kept(scanned_pdf_bytes, monkeypatch):
    """Why this must never break: a missing or crashing local model must not
    turn a scanned page into a silently empty page. Every page of a lab report
    must be read by SOME engine, and the record must say which one."""
    monkeypatch.setattr(get_graph_settings(), "ocr_engine", "deepreef_local", raising=False)

    def boom(image, **kw):
        raise deepreef_local.DeepreefLocalError("model missing")

    monkeypatch.setattr(deepreef_local, "extract_page", boom)
    monkeypatch.setattr(preprocess, "_ocr_page", lambda data, page, dpi: ("Copper 17.16 mg/kg", 0.9, ""))
    text, conf, error, engine = preprocess._ocr_scanned_page(scanned_pdf_bytes, 1, 72, "english")
    assert text == "Copper 17.16 mg/kg" and error == ""
    assert engine == "rapidocr(fallback)"


def _model_repo(tmp_path, monkeypatch, contents: bytes, pinned_sha: str):
    repo = tmp_path / "deepreef-ocr"
    models = repo / "deepreef_ocr" / "models"
    models.mkdir(parents=True)
    (models / "devanagari_rec.onnx").write_bytes(contents)
    (repo / "models.sha256").write_text(f"{pinned_sha}  devanagari_rec.onnx\n")
    monkeypatch.setattr(deepreef_local, "_repo_dir", lambda: str(repo))
    monkeypatch.setattr(deepreef_local, "_model_dir", lambda: str(models))
    monkeypatch.setattr(deepreef_local, "_model_report", None)
    return deepreef_local.model_report()


def test_model_approval_pinned_derived_or_stand_in(tmp_path, monkeypatch):
    """Why this must never break: readings are only attestable when produced by
    the pinned model or an explicitly approved derivative of it. Any other file
    (a swapped, retrained or corrupted model) must be labelled a stand-in, so
    its output can never pass as genuine."""
    pinned = b"the pinned model"
    report = _model_repo(tmp_path / "a", monkeypatch, pinned, hashlib.sha256(pinned).hexdigest())
    assert report["genuine"] is True

    derived = b"int8 build"
    monkeypatch.setattr(deepreef_local, "APPROVED_DERIVED_MODELS",
                        {hashlib.sha256(derived).hexdigest():
                         {"file": "devanagari_rec.onnx", "variant": "INT8"}})
    report = _model_repo(tmp_path / "b", monkeypatch, derived, "0" * 64)
    assert report["genuine"] is True and report["variant"] == "INT8"

    report = _model_repo(tmp_path / "c", monkeypatch, b"some other model", "0" * 64)
    assert report["genuine"] is False
    assert report["files"]["devanagari_rec.onnx"] == "stand-in"


def test_ocr_page_cache_is_keyed_safely_and_never_fails_a_page(tmp_path, monkeypatch):
    """Why this must never break: the cache may only ever return a reading of
    the identical page by the identical model and settings; a new model or a
    changed setting must re-read the page. And the cache is an optimisation:
    a broken cache must never cost a page."""
    s = get_graph_settings()
    monkeypatch.setattr(s, "ocr_engine", "deepreef_local", raising=False)
    monkeypatch.setattr(s, "ocr_page_cache_enabled", True, raising=False)
    monkeypatch.setattr(s, "ocr_page_cache_dir", str(tmp_path / "ocr_cache"), raising=False)
    monkeypatch.setattr(s, "ocr_page_cache_ttl_hours", 168.0, raising=False)
    monkeypatch.setattr(s, "ocr_clean_scan_profile", True, raising=False)
    monkeypatch.setattr(ocr_page_cache, "_model_identity", lambda: "model-a")
    monkeypatch.setattr(ocr_page_cache, "_last_sweep", {})
    calls = []

    def fake_extract(image, **kw):
        calls.append(image)
        return deepreef_local.PageOcr(text="Copper 17.16", lines=["Copper 17.16"], confidence=0.95,
                                      model_used="devanagari_rec.onnx", region_count=1)

    monkeypatch.setattr(deepreef_local, "render_for_ocr", lambda data, page, dpi: data)
    monkeypatch.setattr(deepreef_local, "extract_page", fake_extract)
    read = lambda png=b"page-png": preprocess._ocr_scanned_page(png, 1, 72, "english")  # noqa: E731

    first, second = read(), read()
    assert len(calls) == 1, "identical page, model and settings: one OCR"
    assert first[0] == second[0] == "Copper 17.16"

    monkeypatch.setattr(ocr_page_cache, "_model_identity", lambda: "model-b")
    read()
    assert len(calls) == 2, "a different model must re-read the page"

    monkeypatch.setattr(s, "ocr_clean_scan_profile", False, raising=False)
    read()
    assert len(calls) == 3, "a changed OCR setting must re-read the page"

    blocker = tmp_path / "a-file"
    blocker.write_text("x")
    monkeypatch.setattr(s, "ocr_page_cache_dir", str(blocker / "sub"), raising=False)
    text, _, error, _ = read(b"another-page")
    assert text == "Copper 17.16" and error == "", "an unwritable cache never fails a page"
