"""Shared fixtures for the graph tests.

Every LLM call is stubbed. That is not a compromise for test speed — it is the
point of the architecture. Eleven of the sixteen nodes are deterministic, so
the guarantees that matter (no page skipped, no unsupported value filled, no
conflict silently resolved) are all testable without a network call, and the
five LLM nodes are tested against a scripted model so their *handling* of good
and bad replies is what gets exercised.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from app.graph.config import get_graph_settings
from app.graph.llm import LlmOutcome
from app.graph.schemas import LlmCallMetric


# --------------------------------------------------------------------------
# Isolation
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path, monkeypatch):
    """Point the database and checkpointer at a per-test temp directory."""
    from app.graph import database

    settings = get_graph_settings()
    monkeypatch.setattr(settings, "graph_database_url", "", raising=False)
    monkeypatch.setattr(settings, "graph_sqlite_path", str(tmp_path / "graph.db"), raising=False)
    monkeypatch.setattr(settings, "checkpoint_db_path", str(tmp_path / "ckpt.db"), raising=False)
    database.reset_engine()
    yield
    database.reset_engine()


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Backoff is real in production and pointless in tests."""
    monkeypatch.setattr("app.graph.llm._sleep", lambda attempt: None)


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------


def _write_pdf(path: str, pages: list[str]) -> str:
    import fitz

    doc = fitz.open()
    for body in pages:
        page = doc.new_page()
        page.insert_text((72, 100), body, fontsize=11)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def simple_pdf(tmp_path) -> str:
    return _write_pdf(str(tmp_path / "simple.pdf"), [
        "LABORATORY TEST REPORT\n"
        "Report No: LR-2024-0195\n"
        "Sample Name: Cow Milk\n"
        "Collected On: 12/03/2024\n"
        "Fat Content: 4.2 %\n",
        "RESULTS\n"
        "Protein: 3.4 g/100ml\n"
        "SNF: 8.6 %\n"
        "Analyst: R. Menon\n",
    ])


@pytest.fixture
def three_page_pdf(tmp_path) -> str:
    return _write_pdf(str(tmp_path / "three.pdf"), [
        "Page one. Report No: X-1. Value: 10 mg/L.",
        "Page two. Report No: X-1. Value: 12 mg/L.",
        "",  # deliberately blank — must still be accounted for
    ])


@pytest.fixture
def empty_file(tmp_path) -> str:
    p = tmp_path / "empty.pdf"
    p.write_bytes(b"")
    return str(p)


@pytest.fixture
def corrupt_pdf(tmp_path) -> str:
    p = tmp_path / "broken.pdf"
    p.write_bytes(b"not a pdf at all")
    return str(p)


@pytest.fixture
def unsupported_file(tmp_path) -> str:
    p = tmp_path / "notes.txt"
    p.write_text("plain text", encoding="utf-8")
    return str(p)


@pytest.fixture
def encrypted_pdf(tmp_path) -> str:
    import fitz

    path = str(tmp_path / "locked.pdf")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "secret", fontsize=11)
    doc.save(path, encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    doc.close()
    return path


# --------------------------------------------------------------------------
# LLM stubbing
# --------------------------------------------------------------------------


class ScriptedLlm:
    """Returns a canned reply per node, and records what it was asked.

    `replies` maps node name -> either a dict (parsed as the node's output
    model) or an LlmOutcome (to simulate a failure). A node with no entry gets
    an empty-but-valid reply, so a test only has to script the node it cares
    about.
    """

    def __init__(self, replies: dict | None = None):
        self.replies = replies or {}
        self.calls: list[dict] = []

    def __call__(self, *, node, system, user_text, output_model, images=None,
                 image_media_type=None, max_attempts=None, on_metric=None):
        self.calls.append({"node": node, "system": system, "user_text": user_text})
        if on_metric:
            on_metric(LlmCallMetric(node=node, model="stub", provider="stub",
                                    latency_seconds=0.0, input_tokens=10, output_tokens=5))

        reply = self.replies.get(node)
        if isinstance(reply, LlmOutcome):
            return reply
        if callable(reply):
            reply = reply(user_text)
        if reply is None:
            reply = {}
        try:
            parsed = output_model.model_validate(reply)
        except Exception as exc:  # noqa: BLE001
            from app.graph.errors import ErrorType

            return LlmOutcome(ok=False, error_type=ErrorType.LLM_SCHEMA_MISMATCH,
                              error_message=str(exc)[:200])
        return LlmOutcome(ok=True, parsed=parsed, raw=reply, attempts=1)


@pytest.fixture
def scripted(monkeypatch):
    """Install a ScriptedLlm across every module that imported call_structured."""
    def _install(replies: dict | None = None) -> ScriptedLlm:
        stub = ScriptedLlm(replies)
        for module in (
            "app.graph.llm",
            "app.graph.nodes.analysis",
            "app.graph.nodes.extraction",
            "app.graph.nodes.normalize",
            "app.graph.nodes.conflicts",
            "app.graph.nodes.mapping",
        ):
            monkeypatch.setattr(f"{module}.call_structured", stub, raising=False)
        return stub

    return _install


def extraction_reply(chunk_id: str, fields: list[dict]) -> dict:
    return {"chunk_id": chunk_id, "fields": fields, "tables_seen": [], "notes": ""}


def field(name, value, page, chunk_id, evidence, *, status="VERIFIED",
          confidence=0.95, occurrence=0, section="", table_context=""):
    return {
        "field_name": name,
        "normalized_field_name": name.lower().replace(" ", "_"),
        "value": value,
        "normalized_value": value,
        "data_type": "string",
        "page_number": page,
        "chunk_id": chunk_id,
        "section_name": section,
        "table_context": table_context,
        "exact_source_evidence": evidence,
        "confidence_score": confidence,
        "extraction_status": status,
        "occurrence_index": occurrence,
        "notes": "",
    }
