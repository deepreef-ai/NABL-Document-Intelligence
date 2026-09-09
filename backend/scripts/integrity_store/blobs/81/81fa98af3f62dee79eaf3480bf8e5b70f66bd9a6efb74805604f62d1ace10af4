"""Agentic document-processing and dynamic form-filling workflow (LangGraph).

Public surface:

    from app.graph import run_document, resume_document, get_status

Everything else — nodes, prompts, the state model — is internal detail that
callers should not need. The one exception is `GraphState`, exported because
tests and the API's progress endpoint legitimately read it.

Architecture in one paragraph: a shared typed state is threaded through
sixteen nodes. Eleven are deterministic Python and decide everything that can
be decided by counting or comparing; five call an LLM and are used only for
semantic judgement (structure, extraction, name equivalence, conflict
adjudication, form correspondence). Validation gates sit after every LLM stage
and reject rather than repair. Retry, recovery and targeted re-analysis are
separate bounded loops. State is checkpointed after every node, so any run is
resumable from its document id.
"""
from app.graph.config import GraphSettings, get_graph_settings
from app.graph.graph import build_checkpointer, build_graph, mermaid
from app.graph.runner import (
    get_status,
    new_document_id,
    resume_document,
    run_document,
    stream_document,
)
from app.graph.schemas import (
    ExtractedField,
    FinalResult,
    FinalStatus,
    FormMapping,
    TargetFormSchema,
)
from app.graph.state import GraphState

__all__ = [
    "ExtractedField",
    "FinalResult",
    "FinalStatus",
    "FormMapping",
    "GraphSettings",
    "GraphState",
    "TargetFormSchema",
    "build_checkpointer",
    "build_graph",
    "get_graph_settings",
    "get_status",
    "mermaid",
    "new_document_id",
    "resume_document",
    "run_document",
    "stream_document",
]
