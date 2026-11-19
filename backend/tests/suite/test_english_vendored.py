"""Vendored English OCR: the gateway's script-en path runs from files in this
repo (app/ocr_gateway/models/english + vendor/deepreef_preprocessor), with no
external deepreef-ocr checkout, and reads exactly what deepreef-ocr reads.

A. Files     - model + dict present and byte-identical to the recorded sha256.
B. No D:\\    - English OCR through the gateway with every deepreef path pointing
               at a directory that does not exist (realocr).
C. Parity    - vendored vs the external deepreef checkout, same image, same
               output (realocr; skipped when the checkout is absent).
"""
from __future__ import annotations

import hashlib
import json
import sys

import pytest

gw = pytest.importorskip("app.ocr_gateway")

from app.ocr_gateway import engines  # noqa: E402
from app.ocr_gateway import settings as gw_settings  # noqa: E402
from tests.suite import _support as sup  # noqa: E402

#: deepreef-ocr models.sha256 (commit 48ee778); see models/english/README.md.
EXPECTED_SHA256 = {
    "devanagari_rec.onnx": "d6f0a906580e3fa6b324a318718f1f31f268b6ea8ef985f91c2012a37f52c91e",
    "devanagari_dict.txt": "09c7440bfc5477e5c41052304b6b185aff8c4a5e8b2b4c23c1c706f6fe1ee9fc",
}
DATES = ["25/08/2021", "26/08/2021"]


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def _no_aws(monkeypatch):
    import boto3

    monkeypatch.setattr(boto3, "client",
                        lambda *a, **k: pytest.fail("vendored English OCR must not call AWS"))
    gw_settings.get_gateway_settings.cache_clear()
    yield
    gw_settings.get_gateway_settings.cache_clear()


@pytest.fixture
def no_external(monkeypatch, tmp_path):
    """Every route to an external deepreef checkout points at nothing."""
    from app.config import get_settings

    missing = tmp_path / "no_such_deepreef_checkout"
    monkeypatch.setenv("OCR_GATEWAY_DEEPREEF_PATH", str(missing))
    monkeypatch.setattr(get_settings(), "deepreef_ocr_path", str(missing), raising=False)
    gw_settings.get_gateway_settings.cache_clear()
    s = gw_settings.get_gateway_settings()
    assert s.deepreef_path == str(missing)
    assert not engines.resolve_deepreef_dir(s).exists()
    return s


def _scan(kind: str = "clean") -> bytes:
    return sup.degrade(sup.rasterize(sup.text_pdf([sup.REPORT_LINES]), dpi=200), kind)


# ==========================================================================
# A. Files
# ==========================================================================


def test_vendored_model_files_match_recorded_sha256():
    from app.ocr_gateway import english_engine

    assert english_engine.SHA256 == EXPECTED_SHA256
    for name, want in EXPECTED_SHA256.items():
        p = english_engine.MODEL_DIR / name
        assert p.is_file(), f"vendored file missing: {p}"
        assert _sha(p) == want, f"{name} differs from the recorded sha256"
    assert english_engine.available()


def test_vendored_preprocessor_has_no_external_imports():
    from app.ocr_gateway.vendor import deepreef_preprocessor

    assert callable(deepreef_preprocessor.decode_and_resize)
    src = open(deepreef_preprocessor.__file__, encoding="utf-8").read()
    assert "import deepreef_ocr" not in src and "from deepreef_ocr" not in src
    assert "REPLICATED COPY" in src


def test_self_check_reports_vendored_english(no_external):
    rep = gw.self_check()
    assert rep["deepreef_local"]["ok"] is False, "the external checkout is absent here"
    ev = rep["english_vendored"]
    assert ev["ok"] is True and ev["model_used"] == "devanagari_rec.onnx"
    assert all(f["present"] and f["sha256_ok"] for f in ev["files"].values())


def test_english_routes_to_vendored_without_touching_external(no_external, monkeypatch):
    from app.ocr_gateway import english_engine

    monkeypatch.setattr(engines, "_load_deepreef",
                        lambda s: pytest.fail("script en must not load the external checkout"))
    called = []
    monkeypatch.setattr(english_engine, "extract", lambda img, drop_camera_stamps=False: (
        called.append(drop_camera_stamps) or {"text": "a b", "lines": ["a b"], "confidence": 0.9,
                                              "model_used": "devanagari_rec.onnx",
                                              "bounding_boxes": [], "text_lines": "a b"}))
    r = engines.deepreef_local(_scan(), "en", no_external)
    assert called == [False], "keep_dated_lines=True must leave the stamp filter off"
    assert (r.engine, r.model_used, r.text_lines) == ("deepreef_local", "devanagari_rec.onnx", "a b")


@pytest.mark.parametrize("code,vendored_present", [("devanagari", True), ("en", False)],
                         ids=["other-script", "en-vendored-files-missing"])
def test_external_checkout_still_used_otherwise(no_external, monkeypatch, code, vendored_present):
    from app.ocr_gateway import english_engine

    monkeypatch.setattr(engines, "_deepreef", {})  # forget any in-process load
    monkeypatch.setattr(english_engine, "available", lambda: vendored_present)
    with pytest.raises(engines.EngineError, match="deepreef-ocr checkout not found"):
        engines.deepreef_local(_scan(), code, no_external)


# ==========================================================================
# B. Real OCR with no external checkout
# ==========================================================================


@pytest.mark.realocr
def test_english_ocr_works_without_external_checkout(no_external, monkeypatch, trace):
    monkeypatch.setattr(no_external, "enabled", True)
    monkeypatch.setattr(no_external, "keep_dated_lines", True)
    monkeypatch.setattr(no_external, "latin_engines", "deepreef_local")  # no rapidocr fallback
    monkeypatch.setattr(engines, "_load_deepreef",
                        lambda s: pytest.fail("script en must not load the external checkout"))
    with trace.step("gateway read_image (vendored en)"):
        r = gw.read_image(_scan(), "english")
    got = gw.page_text(r)
    sc = sup.score("\n".join(sup.REPORT_LINES), got)
    trace.metric(engine=f"{r.engine}:{r.model_used}", ocr_seconds=r.ocr_seconds, **sc.as_dict())
    assert (r.engine, r.model_used) == ("deepreef_local", "devanagari_rec.onnx")
    assert [a["engine"] for a in r.attempts] == ["deepreef_local"]
    assert r.items and len(r.items) == len(r.lines)
    assert sc.glued_words == 0 and sc.exact_word_recall >= 0.9, f"{sc}\n{got}"
    assert "RaghuRam" in got


@pytest.mark.realocr
def test_vendored_keeps_dated_lines(no_external):
    r = engines.deepreef_local(_scan(), "en", no_external)
    got = gw.page_text(r)
    for d in DATES:
        assert d in got, f"dated line dropped ({d}):\n{got}"
    assert "25/08/2021 10:04" in got and "26/08/2021 15:28" in got, got


# ==========================================================================
# C. Parity with the external deepreef checkout
# ==========================================================================


@pytest.fixture
def external_deepreef(deepreef_path):
    """The real external engine for script en, stamp filters off (as the gateway runs it)."""
    s = gw_settings.GatewaySettings(_env_file=None, deepreef_path=str(deepreef_path))
    sys.dont_write_bytecode = True
    dr = engines._load_deepreef(s)
    if dr["root"].resolve() != deepreef_path.resolve():
        pytest.skip(f"deepreef already loaded in-process from {dr['root']}")
    return s, dr


def _same(a, b) -> bool:
    return json.dumps(a, ensure_ascii=False) == json.dumps(b, ensure_ascii=False)


@pytest.mark.realocr
@pytest.mark.parametrize("kind", ["clean", "noise", "skew_2deg"])
def test_parity_with_external_deepreef(external_deepreef, kind, trace):
    import numpy as np

    from app.ocr_gateway import english_engine

    s, dr = external_deepreef
    png = _scan(kind)
    with trace.step("external deepreef en"):
        img_e, meta_e = dr["decode"](png, max_dim=s.max_dim)
        out_e = dr["engine"].extract(img_e, "en", max_words=s.max_words)
    with trace.step("vendored en"):
        img_v, meta_v = english_engine.decode(png, s.max_dim)
        out_v = english_engine.extract(img_v)
    assert np.array_equal(img_v, img_e) and meta_v == meta_e, "preprocessing differs"
    for key in ("text", "lines", "confidence", "model_used", "region_count",
                "bounding_boxes", "items", "text_lines"):
        assert _same(out_v[key], out_e[key]), f"{key} differs:\n{out_v[key]}\n---\n{out_e[key]}"
    r = engines.deepreef_local(png, "en", s)
    assert (r.lines, r.text_lines, r.confidence) == (out_e["lines"], out_e["text_lines"],
                                                     float(out_e["confidence"]))
    trace.metric(engine="deepreef_local:devanagari_rec.onnx", lines=len(out_v["lines"]))
