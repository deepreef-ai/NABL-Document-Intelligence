"""Faster extraction: per-node Gemini thinking level + table headers carried
into split chunks. No network: httpx.post is monkeypatched."""
import threading

import httpx

from app.graph.nodes.chunking import _last_table_header, create_chunks
from app.graph.schemas import PageRecord, PageStatus
from app.graph.state import GraphState
from app.llm.providers import GeminiProvider
from app.llm.thinking import (
    current_thinking_level,
    gemini_thinking_config,
    normalise_level,
    use_thinking_level,
)


class _Response:
    status_code = 200
    text = ""

    def json(self):
        return {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]}


def _capture(monkeypatch):
    sent = []

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.append(json)
        return _Response()

    monkeypatch.setattr(httpx, "post", fake_post)
    return sent


# ---------------------------------------------------------------- thinking


def test_no_level_means_no_thinking_config(monkeypatch):
    sent = _capture(monkeypatch)
    GeminiProvider("k", "gemini-3.6-flash", 30.0).generate("s", "u", want_json=True)
    assert "thinkingConfig" not in sent[0]["generationConfig"]


def test_gemini_3_gets_thinking_level(monkeypatch):
    sent = _capture(monkeypatch)
    with use_thinking_level("minimal"):
        GeminiProvider("k", "gemini-3.6-flash", 30.0).generate("s", "u", want_json=True)
    assert sent[0]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "minimal"}


def test_gemini_2_gets_a_token_budget_instead():
    assert gemini_thinking_config("gemini-2.5-flash", "minimal") == {"thinkingBudget": 0}
    assert gemini_thinking_config("models/gemini-2.5-flash", "high") == {"thinkingBudget": -1}


def test_level_is_reset_after_the_block():
    with use_thinking_level("low"):
        assert current_thinking_level() == "low"
    assert current_thinking_level() is None


def test_unknown_or_blank_level_falls_back_to_default():
    assert normalise_level("") is None
    assert normalise_level("turbo") is None
    assert normalise_level(" LOW ") == "low"


def test_parallel_threads_do_not_share_a_level():
    seen = {}
    barrier = threading.Barrier(2)

    def worker(name, level):
        with use_thinking_level(level):
            barrier.wait()          # both threads inside their block at once
            seen[name] = current_thinking_level()

    threads = [threading.Thread(target=worker, args=("a", "minimal")),
               threading.Thread(target=worker, args=("b", None))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen == {"a": "minimal", "b": None}


# ---------------------------------------------------------------- chunking


def _state(pages: dict[int, str]) -> GraphState:
    return GraphState(
        document_id="doc", file_type="pdf", total_pages=len(pages), page_text=pages,
        page_metadata={p: PageRecord(page_number=p, text=t, char_count=len(t),
                                     status=PageStatus.NATIVE_TEXT) for p, t in pages.items()},
    )


def test_header_line_is_recognised():
    text = "ABC Labs\nTest Parameter   Result   Unit   Method\nProtein 12.1 g ISO"
    assert _last_table_header(text) == "Test Parameter   Result   Unit   Method"
    assert _last_table_header("just a sentence about a test") is None


def test_split_page_repeats_the_table_header_in_later_parts(monkeypatch):
    from app.graph.config import get_graph_settings

    monkeypatch.setattr(get_graph_settings(), "max_chunk_chars", 1500, raising=False)
    rows = "\n\n".join(f"Analyte {i}   {i}.5   mg/kg   AOAC 9{i:02d}" for i in range(120))
    page = f"Lab Report\n\nTest Parameter   Result   Unit   Method\n\n{rows}"
    chunks = create_chunks(_state({1: page}))["chunks"]

    assert len(chunks) > 1
    assert "[continuation of page 1" not in chunks[0].text
    for chunk in chunks[1:]:
        assert "Test Parameter   Result   Unit   Method" in chunk.text
        assert "do not extract it as a value" in chunk.text
        assert chunk.text.startswith("--- Page 1 ---")   # page marker still first


def test_split_page_without_a_table_is_unchanged(monkeypatch):
    from app.graph.config import get_graph_settings

    monkeypatch.setattr(get_graph_settings(), "max_chunk_chars", 2000, raising=False)
    chunks = create_chunks(_state({1: "word " * 3000}))["chunks"]
    assert len(chunks) > 1
    assert not any("[continuation" in c.text for c in chunks)


# ------------------------------------------------- tolerant row validation


def test_a_numeric_serial_number_no_longer_rejects_the_chunk():
    """The exact shape that failed 2026-11-23: 's_no': 1 cost a whole chunk."""
    from app.graph.schemas import ChunkExtraction

    parsed = ChunkExtraction.model_validate({
        "chunk_id": "c1",
        "fields": [],
        "tests": [{"s_no": 1, "test_name": "Protein", "result": 12.5, "unit": "g/100g",
                   "page_number": 2, "exact_source_evidence": None}],
    })
    row = parsed.tests[0]
    assert (row.s_no, row.result, row.exact_source_evidence) == ("1", "12.5", "")


def test_text_columns_keep_their_text_and_booleans_are_not_coerced():
    import pytest
    from pydantic import ValidationError

    from app.graph.schemas import TestRow

    assert TestRow.model_validate({"result": "0.70"}).result == "0.70"   # strings untouched
    with pytest.raises(ValidationError):
        TestRow.model_validate({"result": True})                          # not silently "True"


def test_null_descriptive_columns_on_a_field_become_empty_text():
    from app.graph.schemas import ExtractedField

    f = ExtractedField.model_validate({"field_name": "Report No", "value": "LR-1",
                                       "section_name": None, "table_context": None})
    assert (f.section_name, f.table_context) == ("", "")
