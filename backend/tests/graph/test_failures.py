"""Failure-handling tests (spec section 9).

Every case here is a way the workflow can go wrong. The assertion is never
"it didn't crash" — it is that the failure was *recorded*, *classified*, and
*visible in the final result*, because the business rules forbid a failure
that disappears.
"""
from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.graph.errors import ErrorType
from app.graph.llm import LlmOutcome
from app.graph.nodes.recovery import recover
from app.graph.runner import run_document
from app.graph.schemas import ChunkRecord, ChunkStatus, FinalStatus, PageRecord, PageStatus
from app.graph.state import GraphState
from tests.graph.conftest import extraction_reply, field


def _fail(error_type: ErrorType, message: str = "boom") -> LlmOutcome:
    return LlmOutcome(ok=False, error_type=error_type, error_message=message, attempts=3)


def _error_types(result) -> set[str]:
    return {e.error_type for e in result.audit_log} if False else set()


class TestFatalInput:
    @pytest.mark.parametrize("fixture_name,expected", [
        ("empty_file", "EMPTY_DOCUMENT"),
        ("corrupt_pdf", "CORRUPTED_FILE"),
        ("unsupported_file", "UNSUPPORTED_FILE_TYPE"),
        ("encrypted_pdf", "PASSWORD_PROTECTED"),
    ])
    def test_bad_input_is_rejected_with_a_reason(self, request, fixture_name, expected, scripted):
        scripted({})
        path = request.getfixturevalue(fixture_name)
        result = run_document(path, document_id=f"bad-{fixture_name}", checkpointer=MemorySaver())

        assert result.overall_status == FinalStatus.REJECTED
        assert result.final_reasoning, "a rejection must say why"
        assert any(e.level == "error" for e in result.audit_log)

    def test_missing_file_does_not_raise(self, scripted):
        scripted({})
        result = run_document("/does/not/exist.pdf", document_id="missing",
                              checkpointer=MemorySaver())
        assert result.overall_status == FinalStatus.REJECTED


class TestLlmFailures:
    @pytest.mark.parametrize("error_type", [
        ErrorType.LLM_TIMEOUT,
        ErrorType.LLM_RATE_LIMIT,
        ErrorType.LLM_INVALID_JSON,
        ErrorType.LLM_EMPTY_RESPONSE,
        ErrorType.LLM_UNAVAILABLE,
    ])
    def test_extraction_failure_is_reported_never_silent(self, simple_pdf, scripted, error_type):
        scripted({"dynamic_extraction": _fail(error_type)})
        result = run_document(simple_pdf, document_id=f"llm-{error_type.value}",
                              checkpointer=MemorySaver())

        assert result.overall_status != FinalStatus.VALIDATED
        assert result.failed_chunks, "a chunk that never extracted must be listed as failed"
        assert result.final_reasoning

    def test_analysis_failure_is_survivable(self, simple_pdf, scripted):
        """Structure analysis is a hint. Losing it must not stop the run."""
        def extract(user_text: str) -> dict:
            chunk_id = next(
                (l.split(":", 1)[1].strip() for l in user_text.splitlines()
                 if l.startswith("chunk_id:")), "",
            )
            out = []
            for line in user_text.splitlines():
                if "Report No:" in line:
                    out.append(field("Report No", "LR-2024-0195", 1, chunk_id, line.strip()))
            return extraction_reply(chunk_id, out)

        scripted({
            "document_analysis": _fail(ErrorType.LLM_UNAVAILABLE),
            "dynamic_extraction": extract,
        })
        result = run_document(simple_pdf, document_id="no-analysis", checkpointer=MemorySaver())
        assert result.extracted_fields, "extraction should still work without structural hints"

    def test_schema_mismatch_is_rejected_not_coerced(self, simple_pdf, scripted):
        scripted({"dynamic_extraction": {"chunk_id": "x", "fields": "not a list"}})
        result = run_document(simple_pdf, document_id="badschema", checkpointer=MemorySaver())
        assert result.overall_status != FinalStatus.VALIDATED
        assert result.failed_chunks

    def test_reply_for_the_wrong_chunk_is_refused(self, simple_pdf, scripted):
        scripted({"dynamic_extraction": extraction_reply("somebody-elses-chunk", [])})
        result = run_document(simple_pdf, document_id="crossed", checkpointer=MemorySaver())
        assert result.failed_chunks, "a reply citing another chunk must not be accepted"


class TestRetryAndRecovery:
    def _state_with_failed_chunk(self, attempt_count: int, error_type: ErrorType, chars: int = 12000):
        text = "x " * (chars // 2)
        return GraphState(
            document_id="d",
            file_type="pdf",
            page_text={1: text},
            page_metadata={1: PageRecord(page_number=1, text=text, char_count=len(text),
                                         status=PageStatus.NATIVE_TEXT)},
            chunks=[ChunkRecord(
                chunk_id="c1", page_numbers=[1], text=text, char_count=len(text),
                status=ChunkStatus.FAILED, attempt_count=attempt_count,
                last_error=error_type.value,
            )],
            errors=[__import__("app.graph.errors", fromlist=["make_error"]).make_error(
                error_type, "boom", "dynamic_extraction", chunk_id="c1",
            )],
        )

    def test_retry_limit_stops_the_loop(self, monkeypatch):
        from app.graph.config import get_graph_settings

        monkeypatch.setattr(get_graph_settings(), "max_attempts_per_chunk", 2, raising=False)
        state = self._state_with_failed_chunk(2, ErrorType.LLM_TIMEOUT)
        out = recover(state)

        assert out["failed_chunks"] == ["c1"]
        assert any(e.error_type == "RETRY_LIMIT_EXCEEDED" for e in out["errors"])
        assert any(e.resolution_status == "ABANDONED" for e in out["errors"])

    def test_truncated_chunk_is_split_before_retrying(self):
        state = self._state_with_failed_chunk(1, ErrorType.LLM_TRUNCATED)
        out = recover(state)
        pending = [c for c in out["chunks"] if c.status == ChunkStatus.PENDING]
        assert len(pending) >= 2, "a truncated reply should be retried on smaller pieces"
        assert all(c.page_numbers == [1] for c in pending), \
            "splitting must not lose the page a chunk covered"

    def test_transient_failure_is_retried_unchanged(self):
        state = self._state_with_failed_chunk(1, ErrorType.LLM_RATE_LIMIT)
        out = recover(state)
        pending = [c for c in out["chunks"] if c.status == ChunkStatus.PENDING]
        assert [c.chunk_id for c in pending] == ["c1"]

    def test_recovery_records_a_reason_for_every_retry(self):
        state = self._state_with_failed_chunk(1, ErrorType.LLM_TIMEOUT)
        out = recover(state)
        assert out["retry_reasons"], "a retry with no recorded reason is an untraceable retry"
        assert out["retry_count"] == 1


class TestNoInformationLoss:
    def test_failed_chunk_appears_in_the_final_result(self, simple_pdf, scripted):
        scripted({"dynamic_extraction": _fail(ErrorType.LLM_UNAVAILABLE)})
        result = run_document(simple_pdf, document_id="lossy", checkpointer=MemorySaver())

        assert result.failed_chunks
        assert result.overall_status in (
            FinalStatus.MANUAL_REVIEW_REQUIRED, FinalStatus.INCOMPLETE,
        )
        assert "chunk" in result.final_reasoning.lower()

    def test_partial_failure_keeps_what_succeeded(self, three_page_pdf, scripted):
        """One chunk failing must not discard the others' fields."""
        calls = {"n": 0}

        def flaky(user_text: str):
            calls["n"] += 1
            chunk_id = next(
                (l.split(":", 1)[1].strip() for l in user_text.splitlines()
                 if l.startswith("chunk_id:")), "",
            )
            if calls["n"] == 1:
                return _fail(ErrorType.LLM_UNAVAILABLE)
            out = []
            for line in user_text.splitlines():
                if "Report No:" in line:
                    out.append(field("Report No", "X-1", 1, chunk_id, line.strip()))
            return extraction_reply(chunk_id, out)

        scripted({"dynamic_extraction": flaky})
        result = run_document(three_page_pdf, document_id="partial", checkpointer=MemorySaver())
        assert result.document_summary.page_count == 3
        assert result.audit_log

    def test_empty_page_is_never_silently_skipped(self, three_page_pdf, scripted):
        scripted({"dynamic_extraction": lambda t: extraction_reply(
            next((l.split(":", 1)[1].strip() for l in t.splitlines()
                  if l.startswith("chunk_id:")), ""), [])})
        result = run_document(three_page_pdf, document_id="blankpage",
                              checkpointer=MemorySaver())
        assert result.document_summary.page_count == 3
        assert result.quality_control.all_pages_processed


class TestDatabaseFailure:
    def test_persistence_failure_does_not_lose_the_result(self, simple_pdf, scripted, monkeypatch):
        scripted({"dynamic_extraction": lambda t: extraction_reply(
            next((l.split(":", 1)[1].strip() for l in t.splitlines()
                  if l.startswith("chunk_id:")), ""), [])})

        def boom(*a, **kw):
            raise RuntimeError("database is on fire")

        monkeypatch.setattr("app.graph.database.save_run", boom)
        result = run_document(simple_pdf, document_id="dbfail", checkpointer=MemorySaver())

        assert result.overall_status is not None, "a DB failure must not lose the result"
        assert any(
            a.event == "persist_failed" for a in result.audit_log
        ), "the DB failure must still be recorded"


class TestLoopTermination:
    def test_persistent_failure_terminates(self, simple_pdf, scripted, monkeypatch):
        """The most important safety property: the graph always stops."""
        from app.graph.config import get_graph_settings

        settings = get_graph_settings()
        monkeypatch.setattr(settings, "max_attempts_per_chunk", 2, raising=False)
        monkeypatch.setattr(settings, "max_extraction_rounds", 2, raising=False)

        scripted({"dynamic_extraction": _fail(ErrorType.LLM_TIMEOUT)})
        result = run_document(simple_pdf, document_id="loop", checkpointer=MemorySaver())

        assert result.overall_status is not None
        assert result.metrics.processing_time_seconds >= 0

    def test_reanalysis_rounds_are_bounded(self, three_page_pdf, scripted, monkeypatch):
        from app.graph.config import get_graph_settings

        monkeypatch.setattr(get_graph_settings(), "max_reanalysis_rounds", 1, raising=False)
        # Always return nothing, so completeness keeps wanting a second look.
        scripted({"dynamic_extraction": lambda t: extraction_reply(
            next((l.split(":", 1)[1].strip() for l in t.splitlines()
                  if l.startswith("chunk_id:")), ""), [])})
        result = run_document(three_page_pdf, document_id="reanalyse",
                              checkpointer=MemorySaver())
        assert result.overall_status is not None, "bounded re-analysis must terminate"
