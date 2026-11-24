"""Suite-wide isolation from the developer's backend/.env.

GraphSettings reads backend/.env, so a developer who sets
EXTRACTION_PROFILE=test_rows (or a thinking level) for local testing silently
changed which code path every graph test exercised — MEASURED 2026-11-23: 8
integration tests failed with that .env and passed without it. Tests that
need a non-default value set it themselves with monkeypatch, which runs after
this fixture and therefore wins.
"""
import pytest


@pytest.fixture(autouse=True)
def _default_pipeline_shape(monkeypatch):
    from app.graph.config import get_graph_settings

    settings = get_graph_settings()
    monkeypatch.setattr(settings, "extraction_profile", "full", raising=False)
    monkeypatch.setattr(settings, "extraction_thinking_level", "", raising=False)
    monkeypatch.setattr(settings, "compact_test_rows", True, raising=False)
    monkeypatch.setattr(settings, "ocr_engine", "rapidocr", raising=False)
    # Off for the whole suite: a faked OCR engine must never write fake
    # readings into the developer's real graph_storage/ocr_cache, nor be served
    # an earlier test's reading instead of calling its fake. Cache tests turn
    # it on with a tmp_path directory.
    monkeypatch.setattr(settings, "ocr_page_cache_enabled", False, raising=False)
