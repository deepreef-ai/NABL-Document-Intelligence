"""LangGraph workflow construction, routing and checkpointing.

    document_validation
      ├─(rejected)────────────────────────────────────────────► final_decision
      └─► preprocessing
            ├─(no readable text)───────────────────────────────► final_decision
            └─► document_analysis
                  └─► chunk_management
                        ├─(no chunks)──────────────────────────► final_decision
                        └─► dynamic_extraction ◄──────────────┐
                              └─► response_validation         │
                                    ├─(failures, retries left)─┤
                                    │        retry_recovery ───┘
                                    └─► evidence_validation
                                          └─► field_normalization
                                                └─► conflict_resolution
                                                      └─► completeness_check
                                                            ├─(gaps, rounds left)
                                                            │   targeted_reanalysis ─► dynamic_extraction
                                                            └─► form_mapping
                                                                  └─► form_filling
                                                                        └─► human_review (interrupt)
                                                                              └─► quality_control
                                                                                    └─► final_decision
                                                                                          └─► persistence ─► END

Three loop guards, deliberately independent so one cannot mask the other:

- `extraction_attempts` bounds the extract → validate → recover cycle.
- `reanalysis_count`   bounds the completeness → re-analyse → extract cycle.
- `ChunkRecord.attempt_count` bounds work on any single chunk, and is shared
  between both loops so a chunk cannot ping-pong between them forever.

Every routing decision reads only state — no wall-clock, no randomness — so a
replayed checkpoint takes the same path it took the first time.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import time
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph.config import get_graph_settings
from app.graph.nodes import (
    analyse_document,
    check_completeness,
    create_chunks,
    decide,
    detect_and_resolve_conflicts,
    extract_fields,
    fill_form,
    map_to_form,
    mark_reanalysis_targets,
    normalize_and_merge,
    preprocess_document,
    recover,
    run_quality_control,
    validate_document,
    validate_evidence,
    validate_responses,
)
from app.graph.nodes.helpers import audit
from app.graph.schemas import ChunkStatus, FinalStatus
from app.graph.state import GraphState, WorkflowStatus

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Small nodes defined here (they are graph plumbing, not agents)
# --------------------------------------------------------------------------


def start_run(state: GraphState) -> dict:
    return {
        "current_status": WorkflowStatus.RUNNING,
        "started_at": time.monotonic(),
        "audit_log": audit("start", "workflow_started", f"document {state.document_id}"),
    }


def human_review_gate(state: GraphState) -> dict:
    """Interrupt point. The graph is configured to pause *before* this node.

    When `interrupt_for_human_review` is on and there is something to decide,
    execution stops here and the checkpoint holds everything. A caller resumes
    by writing `human_decisions` into the state and invoking the graph again
    with the same thread id.

    Decisions are applied here rather than in the conflict node so that a
    resumed run does not have to re-adjudicate anything: the human's answer
    simply overrides, and the audit trail records that it was a human who
    decided.
    """
    if not state.human_decisions:
        return {
            "current_node": "human_review",
            "awaiting_human": False,
            "audit_log": audit("human_review", "no_decisions", "continuing without human input"),
        }

    conflicts = [c.model_copy(deep=True) for c in state.conflicts]
    applied = 0
    from app.graph.schemas import ConflictResolution

    for c in conflicts:
        choice = state.human_decisions.get(c.normalized_field_name)
        if not choice:
            continue
        valid = {cand.field_uid for cand in c.candidates}
        if choice in valid:
            c.chosen_field_uid = choice
            c.resolution = ConflictResolution.RESOLVED
            c.reason = "resolved by human reviewer"
            c.criteria_used = ["human_decision"]
            applied += 1

    remaining = [
        i for i in state.manual_review_items
        if not (i.kind == "conflict" and i.reference in state.human_decisions)
    ]

    return {
        "current_node": "human_review",
        "conflicts": conflicts,
        "manual_review_items": remaining,
        "awaiting_human": False,
        "audit_log": audit(
            "human_review", "decisions_applied", f"{applied} conflict(s) resolved by a reviewer",
        ),
    }


def persist(state: GraphState) -> dict:
    """Write the final result. Never fails the run — see nodes/persistence.py."""
    from app.graph.nodes.persistence import persist_result

    return persist_result(state)


# --------------------------------------------------------------------------
# Routers
# --------------------------------------------------------------------------


def route_after_validation(state: GraphState) -> Literal["preprocessing", "final_decision"]:
    if state.final_status == FinalStatus.REJECTED:
        return "final_decision"
    return "preprocessing"


def route_after_preprocess(state: GraphState) -> Literal["document_analysis", "final_decision"]:
    if not any(m.is_readable for m in state.page_metadata.values()):
        return "final_decision"
    return "document_analysis"


def route_after_chunking(state: GraphState) -> Literal["dynamic_extraction", "final_decision"]:
    if not state.chunks:
        return "final_decision"
    if not any(c.status == ChunkStatus.PENDING for c in state.chunks):
        # Every chunk empty: nothing to extract, but the document was read.
        # Completeness and QC still have to run, so go through the normal path.
        return "dynamic_extraction"
    return "dynamic_extraction"


def route_after_response_validation(
    state: GraphState,
) -> Literal["retry_recovery", "evidence_validation"]:
    """Retry only while there is something retryable AND budget for it."""
    settings = get_graph_settings()
    retryable = [
        c for c in state.chunks
        if c.status == ChunkStatus.FAILED and c.attempt_count < settings.max_attempts_per_chunk
    ]
    if retryable and state.extraction_attempts < settings.max_extraction_rounds:
        return "retry_recovery"
    return "evidence_validation"


def route_after_recovery(state: GraphState) -> Literal["dynamic_extraction", "evidence_validation"]:
    if any(c.status == ChunkStatus.PENDING for c in state.chunks):
        return "dynamic_extraction"
    # Recovery abandoned everything it could not fix; carry on with what we have.
    return "evidence_validation"


def route_after_completeness(
    state: GraphState,
) -> Literal["targeted_reanalysis", "form_mapping"]:
    settings = get_graph_settings()
    report = state.completeness_report
    if (
        report is not None
        and report.reanalysis_targets
        and state.reanalysis_count < settings.max_reanalysis_rounds
    ):
        return "targeted_reanalysis"
    return "form_mapping"


def route_after_form_filling(state: GraphState) -> Literal["human_review", "quality_control"]:
    settings = get_graph_settings()
    if not settings.human_review_enabled:
        return "quality_control"
    if settings.interrupt_for_human_review and state.manual_review_items:
        return "human_review"
    if state.human_decisions:
        return "human_review"
    return "quality_control"


# --------------------------------------------------------------------------
# Checkpointer
# --------------------------------------------------------------------------


def build_checkpointer(path: str | None = None):
    """SQLite checkpointer, Postgres-ready.

    Returns a context-managed saver. `check_same_thread=False` because the
    extraction node uses a thread pool and LangGraph may write a checkpoint
    from a worker; the connection is guarded by SQLite's own locking.

    To move to Postgres, install `langgraph-checkpoint-postgres` and swap this
    for `PostgresSaver.from_conn_string(settings.graph_database_url)`. Nothing
    else in the package needs to change — that is why the checkpointer is
    constructed here and nowhere else.
    """
    settings = get_graph_settings()
    if settings.graph_database_url.startswith("postgres"):
        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            return PostgresSaver.from_conn_string(settings.graph_database_url)
        except ImportError:
            log.warning(
                "graph_database_url is Postgres but langgraph-checkpoint-postgres "
                "is not installed; falling back to SQLite"
            )

    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path = path or settings.checkpoint_db_path
    os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------


def build_graph(checkpointer=None, *, interrupt_before_human: bool | None = None):
    """Compile the workflow.

    `checkpointer=None` means an in-memory saver — fine for tests, useless for
    resumability, so the API always passes a real one.
    """
    settings = get_graph_settings()
    builder = StateGraph(GraphState)

    builder.add_node("start", start_run)
    builder.add_node("document_validation", validate_document)
    builder.add_node("preprocessing", preprocess_document)
    builder.add_node("document_analysis", analyse_document)
    builder.add_node("chunk_management", create_chunks)
    builder.add_node("dynamic_extraction", extract_fields)
    builder.add_node("response_validation", validate_responses)
    builder.add_node("retry_recovery", recover)
    builder.add_node("evidence_validation", validate_evidence)
    builder.add_node("field_normalization", normalize_and_merge)
    builder.add_node("conflict_resolution", detect_and_resolve_conflicts)
    builder.add_node("completeness_check", check_completeness)
    builder.add_node("targeted_reanalysis", mark_reanalysis_targets)
    builder.add_node("form_mapping", map_to_form)
    builder.add_node("form_filling", fill_form)
    builder.add_node("human_review", human_review_gate)
    builder.add_node("quality_control", run_quality_control)
    builder.add_node("final_decision", decide)
    builder.add_node("persistence", persist)

    builder.add_edge(START, "start")
    builder.add_edge("start", "document_validation")

    builder.add_conditional_edges("document_validation", route_after_validation, {
        "preprocessing": "preprocessing",
        "final_decision": "final_decision",
    })
    builder.add_conditional_edges("preprocessing", route_after_preprocess, {
        "document_analysis": "document_analysis",
        "final_decision": "final_decision",
    })
    builder.add_edge("document_analysis", "chunk_management")
    builder.add_conditional_edges("chunk_management", route_after_chunking, {
        "dynamic_extraction": "dynamic_extraction",
        "final_decision": "final_decision",
    })

    builder.add_edge("dynamic_extraction", "response_validation")
    builder.add_conditional_edges("response_validation", route_after_response_validation, {
        "retry_recovery": "retry_recovery",
        "evidence_validation": "evidence_validation",
    })
    builder.add_conditional_edges("retry_recovery", route_after_recovery, {
        "dynamic_extraction": "dynamic_extraction",
        "evidence_validation": "evidence_validation",
    })

    builder.add_edge("evidence_validation", "field_normalization")
    builder.add_edge("field_normalization", "conflict_resolution")
    builder.add_edge("conflict_resolution", "completeness_check")

    builder.add_conditional_edges("completeness_check", route_after_completeness, {
        "targeted_reanalysis": "targeted_reanalysis",
        "form_mapping": "form_mapping",
    })
    builder.add_edge("targeted_reanalysis", "dynamic_extraction")

    builder.add_edge("form_mapping", "form_filling")
    builder.add_conditional_edges("form_filling", route_after_form_filling, {
        "human_review": "human_review",
        "quality_control": "quality_control",
    })
    builder.add_edge("human_review", "quality_control")
    builder.add_edge("quality_control", "final_decision")
    builder.add_edge("final_decision", "persistence")
    builder.add_edge("persistence", END)

    pause = settings.interrupt_for_human_review if interrupt_before_human is None else interrupt_before_human
    return builder.compile(
        checkpointer=checkpointer or MemorySaver(),
        interrupt_before=["human_review"] if pause else None,
    )


def mermaid() -> str:
    """The workflow as a mermaid diagram — used by the docs and /graph/diagram."""
    return build_graph().get_graph().draw_mermaid()
