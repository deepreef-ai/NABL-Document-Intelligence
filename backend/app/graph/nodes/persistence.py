"""Persistence node — writes the finished run to the database.

Deliberately the most forgiving node in the graph. By the time it runs, the
document has been read, extracted, validated and decided; every expensive
thing has already happened. Losing that to a database being briefly
unavailable would be the worst trade in the system, so a write failure is
recorded as an error and the run still returns its result to the caller.

The checkpointer is a separate mechanism and has already saved the state after
every node, so a failed write here does not lose the run either — it can be
replayed from its thread id.
"""
from __future__ import annotations

import logging

from app.graph.errors import ErrorType
from app.graph.nodes.decision import build_result
from app.graph.nodes.helpers import audit, err
from app.graph.state import GraphState

log = logging.getLogger(__name__)

NODE = "persistence"


def persist_result(state: GraphState) -> dict:
    from app.graph import database

    try:
        result = build_result(state)
    except Exception as exc:  # noqa: BLE001 — a malformed result is still worth reporting
        log.exception("graph: could not assemble the final result")
        return {
            "current_node": NODE,
            "errors": err(
                ErrorType.NODE_EXCEPTION,
                f"final result could not be assembled: {exc}", NODE,
                recovery_action="run state is preserved in the checkpoint",
            ),
            "audit_log": audit(NODE, "result_assembly_failed", str(exc)[:400], level="error"),
        }

    try:
        run_id = database.save_run(state, result)
    except Exception as exc:  # noqa: BLE001 — see module docstring
        log.warning("graph: persistence failed for %s: %s", state.document_id, exc)
        return {
            "current_node": NODE,
            "errors": err(
                ErrorType.DATABASE_FAILURE,
                f"could not persist run: {exc}", NODE,
                recovery_action="result returned to the caller; replayable from the checkpoint",
                resolution_status="OPEN",
            ),
            "audit_log": audit(
                NODE, "persist_failed",
                f"{exc}"[:400] + " — result still returned", level="error",
            ),
        }

    return {
        "current_node": NODE,
        "audit_log": audit(
            NODE, "persisted",
            f"run {run_id}: {len(result.extracted_fields)} field(s), "
            f"{len(result.form_mappings)} mapping(s), status {result.overall_status.value}",
        ),
    }
