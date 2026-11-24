"""The "test_rows" profile: a results-table reader (NABL e-Attestation) skips
the four LLM calls that only serve NABL-form filling, while extraction and
the deterministic evidence checks behave exactly as in a full run."""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from app.graph.runner import run_document
from app.graph.schemas import ConflictResolution, FinalStatus
from tests.graph.conftest import extraction_reply, field
from tests.graph.test_integration import _extract_from_text

import pytest


@pytest.fixture(autouse=True)
def _verbose_extraction(monkeypatch):
    """These tests are about which nodes call the model, using the general
    extraction reply shape; the compact reply shape is covered in
    test_compact_rows.py."""
    from app.graph.config import get_graph_settings

    monkeypatch.setattr(get_graph_settings(), "compact_test_rows", False, raising=False)


_FORM_ONLY_NODES = {"document_analysis", "field_normalization", "conflict_resolution", "form_mapping"}


def _chunk_id(user_text: str) -> str:
    return next((l.split(":", 1)[1].strip() for l in user_text.splitlines()
                 if l.startswith("chunk_id:")), "")


def test_only_extraction_calls_the_model(simple_pdf, scripted):
    stub = scripted({"dynamic_extraction": _extract_from_text})
    result = run_document(simple_pdf, document_id="rows", checkpointer=MemorySaver(),
                          profile="test_rows")

    called = {c["node"] for c in stub.calls}
    assert called == {"dynamic_extraction"}, called
    assert result.extracted_fields, "extraction output is unchanged by the profile"
    assert result.form_mappings == []
    events = {e.event for e in result.audit_log}
    assert {"analysis_skipped", "semantic_merge_skipped", "mapping_skipped"} <= events


def test_full_profile_still_calls_every_node(simple_pdf, scripted):
    stub = scripted({"dynamic_extraction": _extract_from_text})
    run_document(simple_pdf, document_id="full", checkpointer=MemorySaver(), form_id="NABL_151",
                 profile="full")
    assert {"document_analysis", "dynamic_extraction", "form_mapping"} <= {c["node"] for c in stub.calls}


def test_conflicts_go_to_a_person_without_a_model_ruling(simple_pdf, scripted):
    def two_values(user_text: str) -> dict:
        cid = _chunk_id(user_text)
        out = []
        for line in user_text.splitlines():
            if "Report No:" in line:
                out.append(field("Report No", "LR-2024-0195", 1, cid, line.strip()))
            if "Sample Name:" in line:
                out.append(field("Report No", "Cow Milk", 1, cid, line.strip()))
        reply = extraction_reply(cid, out)
        # A results row too: under test_rows a document with no results is now
        # INCOMPLETE (honest status, 2026-11-24), which would mask the conflict.
        reply["tests"] = [{"test_name": "Protein", "result": "3.4", "unit": "g/100ml", "page_number": 2}]
        return reply

    stub = scripted({"dynamic_extraction": two_values})
    result = run_document(simple_pdf, document_id="rows-conflict", checkpointer=MemorySaver(),
                          profile="test_rows")

    assert not ({c["node"] for c in stub.calls} & _FORM_ONLY_NODES)
    assert result.conflicts and all(
        c.resolution == ConflictResolution.MANUAL_REVIEW_REQUIRED for c in result.conflicts
    )
    assert len(result.conflicts[0].candidates) >= 2, "every candidate is preserved"
    assert result.overall_status == FinalStatus.CONFLICTED


def test_extraction_thinking_level_reaches_only_the_extraction_call(simple_pdf, scripted, monkeypatch):
    from app.graph.config import get_graph_settings
    from app.llm.thinking import current_thinking_level

    monkeypatch.setattr(get_graph_settings(), "extraction_thinking_level", "minimal", raising=False)
    seen: dict[str, str | None] = {}

    def extract(user_text: str) -> dict:
        seen["dynamic_extraction"] = current_thinking_level()
        return _extract_from_text(user_text)

    def analysis(user_text: str) -> dict:
        seen["document_analysis"] = current_thinking_level()
        return {"document_type": "lab test report"}

    scripted({"dynamic_extraction": extract, "document_analysis": analysis})
    run_document(simple_pdf, document_id="thinking", checkpointer=MemorySaver(), profile="full")
    assert seen == {"dynamic_extraction": "minimal", "document_analysis": None}


def test_setting_selects_the_profile_when_the_run_does_not(simple_pdf, scripted, monkeypatch):
    from app.graph.config import get_graph_settings

    monkeypatch.setattr(get_graph_settings(), "extraction_profile", "test_rows", raising=False)
    stub = scripted({"dynamic_extraction": _extract_from_text})
    run_document(simple_pdf, document_id="by-setting", checkpointer=MemorySaver())
    assert {c["node"] for c in stub.calls} == {"dynamic_extraction"}
