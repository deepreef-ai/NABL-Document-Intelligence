"""Integration tests: the whole graph, end to end, with a scripted model.

These assert the properties the architecture exists to guarantee, at the level
of a complete run rather than a single node:

- every page is accounted for
- a value the document does not contain never reaches the filled form
- a conflict is escalated rather than guessed
- the run is resumable from its checkpoint
- the final payload matches the required output contract exactly
"""
from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.graph.graph import build_graph
from app.graph.runner import get_status, resume_document, run_document
from app.graph.schemas import FinalStatus, MappingStatus
from tests.graph.conftest import extraction_reply, field


def _analysis():
    return {
        "document_type": "lab test report",
        "sections": [{"name": "Results", "start_page": 1, "end_page": 2,
                      "kind": "table", "description": "test results"}],
        "repeated_entities": [], "candidate_fields": ["report_no"],
        "cross_page_references": [], "suspected_duplicates": [], "notes": "",
    }


def _extract_from_text(user_text: str) -> dict:
    """Build a reply whose evidence is genuinely present in the chunk text.

    Deriving the quote from the real text is the point: a stub that invents
    evidence would be rejected by the evidence node, which is correct
    behaviour and would make every integration test fail for the wrong reason.
    """
    chunk_id = ""
    for line in user_text.splitlines():
        if line.startswith("chunk_id:"):
            chunk_id = line.split(":", 1)[1].strip()
            break

    fields = []
    for line in user_text.splitlines():
        if "Report No:" in line:
            fields.append(field("Report No", "LR-2024-0195", 1, chunk_id, line.strip()))
        if "Sample Name:" in line:
            fields.append(field("Sample Name", "Cow Milk", 1, chunk_id, line.strip()))
        if "Protein:" in line:
            fields.append(field("Protein", "3.4 g/100ml", 2, chunk_id, line.strip()))
    return extraction_reply(chunk_id, fields)


class TestHappyPath:
    def test_full_run_produces_a_valid_result(self, simple_pdf, scripted):
        scripted({
            "document_analysis": _analysis(),
            "dynamic_extraction": _extract_from_text,
            "field_normalization": {"groups": [], "notes": ""},
            "conflict_resolution": {"resolutions": []},
            "form_mapping": {"mappings": []},
        })
        result = run_document(simple_pdf, document_id="happy", checkpointer=MemorySaver())

        assert result.overall_status in (
            FinalStatus.VALIDATED, FinalStatus.LOW_CONFIDENCE,
        ), result.final_reasoning
        assert result.document_summary.page_count == 2
        assert result.extracted_fields, "should have extracted something"
        assert result.quality_control.all_pages_processed
        assert not result.quality_control.information_loss_detected
        assert result.metrics.llm_call_count > 0
        assert result.audit_log

    def test_every_page_is_accounted_for(self, three_page_pdf, scripted):
        scripted({"dynamic_extraction": _extract_from_text})
        result = run_document(three_page_pdf, document_id="pages", checkpointer=MemorySaver())
        assert result.document_summary.page_count == 3
        assert result.quality_control.all_pages_processed, result.final_reasoning

    def test_output_matches_the_required_contract(self, simple_pdf, scripted):
        scripted({"dynamic_extraction": _extract_from_text})
        result = run_document(simple_pdf, document_id="contract", checkpointer=MemorySaver())
        payload = result.model_dump(mode="json")

        for key in (
            "overall_status", "document_summary", "extracted_fields", "form_mappings",
            "filled_form", "missing_fields", "conflicts", "failed_chunks",
            "manual_review_items", "quality_control", "final_reasoning",
            "audit_log", "metrics",
        ):
            assert key in payload, f"required top-level key {key!r} missing"

        for key in ("document_type", "page_count", "processed_pages",
                    "processed_chunks", "failed_chunks", "ocr_pages"):
            assert key in payload["document_summary"]

        for key in ("all_pages_processed", "all_chunks_processed",
                    "all_values_have_evidence", "conflicts_resolved",
                    "form_mapping_verified", "information_loss_detected"):
            assert key in payload["quality_control"]

        for key in ("llm_call_count", "retry_count", "processing_time_seconds", "token_usage"):
            assert key in payload["metrics"]

        if payload["extracted_fields"]:
            f = payload["extracted_fields"][0]
            for key in (
                "field_name", "normalized_field_name", "value", "normalized_value",
                "data_type", "page_number", "chunk_id", "section_name",
                "table_context", "exact_source_evidence", "confidence_score",
                "extraction_status", "notes",
            ):
                assert key in f, f"extracted field missing required key {key!r}"


class TestAntiHallucination:
    def test_invented_value_never_reaches_the_form(self, simple_pdf, scripted):
        def invent(user_text: str) -> dict:
            chunk_id = next(
                (l.split(":", 1)[1].strip() for l in user_text.splitlines()
                 if l.startswith("chunk_id:")), "",
            )
            return extraction_reply(chunk_id, [
                field("Bacterial Count", "9999 cfu/ml", 1, chunk_id,
                      "Total Bacterial Count: 9999 cfu/ml"),
            ])

        scripted({
            "dynamic_extraction": invent,
            "form_mapping": {"mappings": [{
                "source_field": "bacterial_count", "target_field": "results.count",
                "source_field_uid": "", "mapping_confidence": 0.95,
                "mapping_reason": "direct", "mapping_status": "MAPPED",
            }]},
        })
        result = run_document(
            simple_pdf, document_id="hallu",
            target_form_schema={"form_id": "T", "fields": [
                {"name": "results.count", "data_type": "string", "required": False,
                 "description": "", "enum_values": [], "repeating": False, "parent": None},
            ], "source": "provided"},
            checkpointer=MemorySaver(),
        )

        assert "9999 cfu/ml" not in str(result.filled_form), \
            "a value absent from the document must never be filled"
        assert result.overall_status != FinalStatus.VALIDATED

    def test_unsupported_value_is_kept_and_flagged(self, simple_pdf, scripted):
        def invent(user_text: str) -> dict:
            chunk_id = next(
                (l.split(":", 1)[1].strip() for l in user_text.splitlines()
                 if l.startswith("chunk_id:")), "",
            )
            return extraction_reply(chunk_id, [
                field("Ghost", "nowhere", 1, chunk_id, "this sentence is not in the document"),
            ])

        scripted({"dynamic_extraction": invent})
        result = run_document(simple_pdf, document_id="ghost", checkpointer=MemorySaver())
        assert result.extracted_fields, "rejected values are preserved for review"
        assert not result.quality_control.all_values_have_evidence


class TestConflicts:
    def test_unresolved_conflict_escalates_and_keeps_every_candidate(self, simple_pdf, scripted):
        def two_values(user_text: str) -> dict:
            chunk_id = next(
                (l.split(":", 1)[1].strip() for l in user_text.splitlines()
                 if l.startswith("chunk_id:")), "",
            )
            out = []
            for line in user_text.splitlines():
                if "Report No:" in line:
                    out.append(field("Report No", "LR-2024-0195", 1, chunk_id, line.strip()))
                if "Sample Name:" in line:
                    out.append(field("Report No", "Cow Milk", 1, chunk_id, line.strip()))
            return extraction_reply(chunk_id, out)

        scripted({
            "dynamic_extraction": two_values,
            "conflict_resolution": {"resolutions": [{
                "normalized_field_name": "report_no", "chosen_field_uid": None,
                "resolution": "MANUAL_REVIEW_REQUIRED",
                "reason": "no candidate better supported", "criteria_used": [],
            }]},
        })
        result = run_document(simple_pdf, document_id="conflict", checkpointer=MemorySaver())

        assert result.conflicts, "differing values must be reported as a conflict"
        assert len(result.conflicts[0].candidates) >= 2, "every candidate is preserved"
        assert result.overall_status == FinalStatus.CONFLICTED
        assert any(i.kind == "conflict" for i in result.manual_review_items)

    def test_confidence_alone_cannot_resolve_a_conflict(self, simple_pdf, scripted):
        def two_values(user_text: str) -> dict:
            chunk_id = next(
                (l.split(":", 1)[1].strip() for l in user_text.splitlines()
                 if l.startswith("chunk_id:")), "",
            )
            out = []
            for line in user_text.splitlines():
                if "Report No:" in line:
                    out.append(field("Report No", "LR-2024-0195", 1, chunk_id, line.strip(),
                                     confidence=0.99))
                if "Sample Name:" in line:
                    out.append(field("Report No", "Cow Milk", 1, chunk_id, line.strip(),
                                     confidence=0.10))
            return extraction_reply(chunk_id, out)

        scripted({
            "dynamic_extraction": two_values,
            "conflict_resolution": {"resolutions": [{
                "normalized_field_name": "report_no",
                "chosen_field_uid": "whatever",
                "resolution": "RESOLVED",
                "reason": "higher confidence",
                "criteria_used": ["confidence"],
            }]},
        })
        result = run_document(simple_pdf, document_id="conf-only", checkpointer=MemorySaver())
        assert result.conflicts[0].resolution.value != "RESOLVED", \
            "business rule: confidence must never be the only criterion"


class TestResumability:
    def test_status_is_readable_from_the_checkpoint(self, simple_pdf, scripted):
        scripted({"dynamic_extraction": _extract_from_text})
        saver = MemorySaver()
        run_document(simple_pdf, document_id="resume-1", checkpointer=saver)

        status = get_status("resume-1", checkpointer=saver)
        assert status is not None
        assert status["document_id"] == "resume-1"
        assert status["final_status"] is not None
        assert status["chunks_total"] > 0

    def test_resume_returns_none_for_an_unknown_document(self):
        assert resume_document("never-existed", checkpointer=MemorySaver()) is None

    def test_human_decision_resolves_a_conflict_on_resume(self, simple_pdf, scripted, monkeypatch):
        from app.graph.config import get_graph_settings

        settings = get_graph_settings()
        monkeypatch.setattr(settings, "interrupt_for_human_review", True, raising=False)

        def two_values(user_text: str) -> dict:
            chunk_id = next(
                (l.split(":", 1)[1].strip() for l in user_text.splitlines()
                 if l.startswith("chunk_id:")), "",
            )
            out = []
            for line in user_text.splitlines():
                if "Report No:" in line:
                    out.append(field("Report No", "LR-2024-0195", 1, chunk_id, line.strip()))
                if "Sample Name:" in line:
                    out.append(field("Report No", "Cow Milk", 1, chunk_id, line.strip()))
            return extraction_reply(chunk_id, out)

        scripted({
            "dynamic_extraction": two_values,
            "conflict_resolution": {"resolutions": [{
                "normalized_field_name": "report_no", "chosen_field_uid": None,
                "resolution": "MANUAL_REVIEW_REQUIRED", "reason": "ambiguous",
                "criteria_used": [],
            }]},
        })

        saver = MemorySaver()
        app = build_graph(saver, interrupt_before_human=True)
        from app.graph.runner import _config, _initial_state

        state = _initial_state(document_id="hitl", file_path=simple_pdf)
        app.invoke(state, config=_config("hitl"))

        snapshot = app.get_state(_config("hitl"))
        assert "human_review" in (snapshot.next or ()), "graph should pause before human review"

        current = snapshot.values
        conflicts = current["conflicts"] if isinstance(current, dict) else current.conflicts
        chosen = conflicts[0].candidates[0].field_uid

        result = resume_document(
            "hitl", human_decisions={"report_no": chosen}, checkpointer=saver,
        )
        assert result is not None
        assert result.conflicts[0].resolution.value == "RESOLVED"
        assert result.conflicts[0].chosen_field_uid == chosen
