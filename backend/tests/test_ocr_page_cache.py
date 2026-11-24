"""OCR page cache (app/documents/ocr_page_cache.py) wired into
preprocess._ocr_scanned_page. Rendering, the engine and the model hash are all
faked: these tests cover keying, TTL, corruption and the on/off switch."""
import glob
import json
import os
import time

import pytest

from app.documents import deepreef_local, ocr_page_cache
from app.graph.config import get_graph_settings
from app.graph.nodes import preprocess


@pytest.fixture
def cache_on(tmp_path, monkeypatch):
    s = get_graph_settings()
    monkeypatch.setattr(s, "ocr_engine", "deepreef_local", raising=False)
    monkeypatch.setattr(s, "ocr_page_cache_enabled", True, raising=False)
    monkeypatch.setattr(s, "ocr_page_cache_dir", str(tmp_path / "ocr_cache"), raising=False)
    monkeypatch.setattr(s, "ocr_page_cache_ttl_hours", 168.0, raising=False)
    monkeypatch.setattr(s, "ocr_render_to_target", True, raising=False)
    monkeypatch.setattr(s, "ocr_clean_scan_profile", True, raising=False)
    monkeypatch.setattr(ocr_page_cache, "_model_identity", lambda: "model-a")
    monkeypatch.setattr(ocr_page_cache, "_last_sweep", {})
    return s


@pytest.fixture
def engine(monkeypatch):
    """Fake render (PNG = the pdf bytes) and a counting fake extract_page."""
    calls = []

    def fake_extract(image, **kw):
        calls.append(image)
        return deepreef_local.PageOcr(text="45 Copper mg/kg 17.16", lines=["45 Copper mg/kg 17.16"],
                                      confidence=0.95, model_used="devanagari_rec.onnx",
                                      region_count=5)

    monkeypatch.setattr(deepreef_local, "render_for_ocr", lambda data, page, dpi: data)
    monkeypatch.setattr(deepreef_local, "extract_page", fake_extract)
    return calls


def _read(png=b"page-png"):
    return preprocess._ocr_scanned_page(png, 1, 72, "english")


def _entries(s):
    return glob.glob(os.path.join(s.ocr_page_cache_dir, "*", "*.json"))


def test_miss_then_hit(cache_on, engine):
    first, second = _read(), _read()
    assert len(engine) == 1
    assert first[0] == second[0] == "45 Copper mg/kg 17.16"
    assert first[1] == second[1] == 0.95
    assert not first[3].endswith("(cache)") and second[3].endswith(" (cache)")
    assert len(_entries(cache_on)) == 1


def test_different_png_is_a_miss(cache_on, engine):
    _read(b"page-1")
    _read(b"page-2")
    assert len(engine) == 2


@pytest.mark.parametrize("field", ["ocr_render_to_target", "ocr_clean_scan_profile"])
def test_changed_setting_is_a_miss(cache_on, engine, monkeypatch, field):
    _read()
    monkeypatch.setattr(cache_on, field, False, raising=False)
    _read()
    assert len(engine) == 2


def test_changed_model_is_a_miss(cache_on, engine, monkeypatch):
    _read()
    monkeypatch.setattr(ocr_page_cache, "_model_identity", lambda: "model-b")
    _read()
    assert len(engine) == 2


def test_expired_entry_is_a_miss_and_removed(cache_on, engine, monkeypatch):
    _read()
    (path,) = _entries(cache_on)
    entry = json.load(open(path))
    entry["created_at"] = time.time() - 200 * 3600
    json.dump(entry, open(path, "w"))
    os.utime(path, None)                       # fresh mtime: only the entry check can catch it
    key = ocr_page_cache.page_key(b"page-png")
    assert ocr_page_cache.get(key) is None
    assert not os.path.exists(path)
    _read()
    assert len(engine) == 2


def test_sweep_removes_old_files(cache_on, engine):
    _read()
    (path,) = _entries(cache_on)
    old = time.time() - 200 * 3600
    os.utime(path, (old, old))
    ocr_page_cache._last_sweep.clear()
    ocr_page_cache.get("ff" * 32)
    assert not os.path.exists(path)


def test_corrupt_entry_falls_back_to_ocr(cache_on, engine):
    _read()
    (path,) = _entries(cache_on)
    open(path, "w").write("{not json")
    text, _, error, label = _read()
    assert text == "45 Copper mg/kg 17.16" and error == "" and not label.endswith("(cache)")
    assert len(engine) == 2
    _read()                                    # the bad entry was replaced by a good one
    assert len(engine) == 2


def test_unwritable_cache_never_fails_a_page(cache_on, engine, monkeypatch, tmp_path):
    blocker = tmp_path / "a-file"
    blocker.write_text("x")
    monkeypatch.setattr(cache_on, "ocr_page_cache_dir", str(blocker / "sub"), raising=False)
    text, _, error, _ = _read()
    assert text and error == ""


def test_disabled_never_reads_or_writes(cache_on, engine, monkeypatch):
    monkeypatch.setattr(cache_on, "ocr_page_cache_enabled", False, raising=False)
    monkeypatch.setattr(ocr_page_cache, "get", lambda key: pytest.fail("read while disabled"))
    _read()
    _read()
    assert len(engine) == 2
    assert not os.path.exists(cache_on.ocr_page_cache_dir)


def test_default_dir_is_next_to_the_graph_db(cache_on, monkeypatch):
    monkeypatch.setattr(cache_on, "ocr_page_cache_dir", "", raising=False)
    expected = os.path.join(os.path.dirname(cache_on.graph_sqlite_path), "ocr_cache")
    assert ocr_page_cache.cache_dir() == expected
