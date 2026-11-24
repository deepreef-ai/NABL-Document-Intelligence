"""Docling is warmed at startup only when USE_DOCLING is on.

The warm-up used to run whenever Docling was merely installed — MEASURED
2026-11-24: +262 MB RSS and ~7 s at startup for a feature that is off by
default. These tests pin the gate in app.main._warm_docling without starting a
real thread or loading Docling: threading.Thread is replaced by a stand-in that
runs its target inline, and warm_docling_converter by a recorder.
"""
import pytest

import app.graph.docling_adapter as docling_adapter
import app.main as main
from app.graph.config import get_graph_settings


class _InlineThread:
    """Runs the target synchronously on start() so the test can observe it."""

    def __init__(self, target=None, name=None, daemon=None, **_kwargs):
        self._target = target

    def start(self):
        self._target()


@pytest.fixture
def warm_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(
        docling_adapter, "warm_docling_converter", lambda: calls.append(1) or True
    )
    monkeypatch.setattr(main.threading, "Thread", _InlineThread)
    return calls


def test_flag_off_skips_warmup(monkeypatch, warm_calls, caplog):
    monkeypatch.setattr(get_graph_settings(), "use_docling", False, raising=False)

    with caplog.at_level("INFO", logger="app.main"):
        main._warm_docling()

    assert warm_calls == []
    assert any(
        "docling_warmup_skipped" in r.getMessage() and "USE_DOCLING" in r.getMessage()
        for r in caplog.records
    )


def test_flag_on_warms_converter(monkeypatch, warm_calls):
    monkeypatch.setattr(get_graph_settings(), "use_docling", True, raising=False)

    main._warm_docling()

    assert warm_calls == [1]
