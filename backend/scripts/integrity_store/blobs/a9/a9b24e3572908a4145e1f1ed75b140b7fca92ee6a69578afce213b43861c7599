"""Entry points for running the workflow.

Three of them, because three different callers need different things:

- `run_document()`   — synchronous, run to completion, return the result.
- `stream_document()`— yields after each node, for a progress UI or a log.
- `resume_document()`— continues a paused or crashed run from its checkpoint.

The `thread_id` is the document id. That is the whole resumability story: the
checkpointer keys state by thread, so re-invoking with the same id picks up
exactly where the last run stopped — after a crash, after a human review
pause, or after a deploy in the middle of a 300-page document.

`recursion_limit` is set generously but finitely. The loop guards inside the
graph are the real protection; this is the backstop for a routing bug that
slips past them, and it fails loudly rather than running forever.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Iterator

from app.graph.config import get_graph_settings
from app.graph.graph import build_checkpointer, build_graph
from app.graph.nodes.decision import build_result
from app.graph.schemas import FinalResult, FinalStatus, TargetFormSchema
from app.graph.state import GraphState

log = logging.getLogger(__name__)

#: Node executions per run. A 300-page document is ~40 chunks in one
#: extraction node call, so the ceiling is about loop iterations, not pages.
RECURSION_LIMIT = 120


def new_document_id(prefix: str = "doc") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _initial_state(
    *,
    document_id: str,
    file_path: str,
    form_id: str = "",
    target_form_schema: dict | None = None,
    display_name: str = "",
) -> GraphState:
    schema = None
    if target_form_schema:
        schema = TargetFormSchema.model_validate(target_form_schema)
    elif form_id:
        # A bare form id is enough — the mapping node introspects the model.
        schema = TargetFormSchema(form_id=form_id, fields=[], source="pydantic")

    return GraphState(
        document_id=document_id,
        file_path=file_path,
        display_name=display_name,
        target_form_schema=schema,
        started_at=time.monotonic(),
    )


def _config(thread_id: str) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }


def run_document(
    file_path: str,
    *,
    document_id: str | None = None,
    form_id: str = "",
    target_form_schema: dict | None = None,
    display_name: str = "",
    checkpointer=None,
) -> FinalResult:
    """Process one document start to finish."""
    document_id = document_id or new_document_id()
    saver = checkpointer or build_checkpointer()
    app = build_graph(saver)

    state = _initial_state(
        document_id=document_id, file_path=file_path, display_name=display_name,
        form_id=form_id, target_form_schema=target_form_schema,
    )

    started = time.monotonic()
    try:
        final = app.invoke(state, config=_config(document_id))
    except Exception as exc:  # noqa: BLE001 — a graph-level crash still owes a result
        log.exception("graph run failed for %s", document_id)
        state.final_status = FinalStatus.REJECTED
        state.final_reasoning = f"workflow crashed: {exc}"
        state.error_details = str(exc)[:1000]
        state.metrics.processing_time_seconds = round(time.monotonic() - started, 2)
        return build_result(state)

    return build_result(_as_state(final))


def stream_document(
    file_path: str,
    *,
    document_id: str | None = None,
    form_id: str = "",
    target_form_schema: dict | None = None,
    checkpointer=None,
) -> Iterator[tuple[str, GraphState]]:
    """Yield (node_name, state) after each node completes."""
    document_id = document_id or new_document_id()
    saver = checkpointer or build_checkpointer()
    app = build_graph(saver)
    state = _initial_state(
        document_id=document_id, file_path=file_path,
        form_id=form_id, target_form_schema=target_form_schema,
    )
    for update in app.stream(state, config=_config(document_id), stream_mode="updates"):
        for node_name, partial in update.items():
            yield node_name, partial


def resume_document(
    document_id: str,
    *,
    human_decisions: dict[str, Any] | None = None,
    checkpointer=None,
) -> FinalResult | None:
    """Continue a run from its checkpoint. Returns None if no checkpoint exists.

    `human_decisions` maps a conflicted field's canonical name to the
    `field_uid` of the candidate the reviewer chose. It is written into the
    state before resuming, and the human-review node applies it.
    """
    saver = checkpointer or build_checkpointer()
    app = build_graph(saver)
    config = _config(document_id)

    snapshot = app.get_state(config)
    if snapshot is None or not snapshot.values:
        return None

    if human_decisions:
        app.update_state(config, {"human_decisions": human_decisions, "awaiting_human": False})

    # `None` as input means "carry on from the checkpoint" rather than "start".
    final = app.invoke(None, config=config)
    return build_result(_as_state(final))


def get_status(document_id: str, checkpointer=None) -> dict[str, Any] | None:
    """Cheap progress read straight from the checkpoint — no re-execution."""
    saver = checkpointer or build_checkpointer()
    app = build_graph(saver)
    snapshot = app.get_state(_config(document_id))
    if snapshot is None or not snapshot.values:
        return None

    state = _as_state(snapshot.values)
    return {
        "document_id": state.document_id,
        "current_node": state.current_node,
        "current_status": state.current_status,
        "final_status": state.final_status.value if state.final_status else None,
        "awaiting_human": state.awaiting_human,
        "next_nodes": list(snapshot.next or ()),
        "total_pages": state.total_pages,
        "chunks_total": len(state.chunks),
        "chunks_processed": len(state.processed_chunks),
        "chunks_failed": len(state.failed_chunks),
        "fields_extracted": len(state.active_fields()),
        "llm_calls": state.metrics.llm_call_count,
        "manual_review_items": len(state.manual_review_items),
    }


def _as_state(value) -> GraphState:
    """LangGraph hands back a dict or the model depending on version/mode."""
    if isinstance(value, GraphState):
        return value
    return GraphState.model_validate(value)
