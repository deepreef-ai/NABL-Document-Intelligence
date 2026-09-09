"""D. Document Analysis Agent — LLM-backed.

Reads a compact outline of the whole document (the head of each page plus its
detected flags) and describes the structure: sections, tables, repeating
entities, likely fields, cross-page references, and anything that looks stated
twice.

This node produces *hints*, never values. Nothing it returns is trusted as
data — its output is fed to the extraction prompt as context and to
completeness checking as a list of things we expected to find. That distinction
matters: if this node hallucinates a section, the cost is a slightly misleading
hint, not a fabricated field in the output.

Failure here is non-fatal by design. Extraction works without structural
context, just a little blinder, so an analysis failure downgrades to a warning
rather than stopping the run.
"""
from __future__ import annotations

import logging

from app.graph.config import get_graph_settings
from app.graph.errors import ErrorType
from app.graph.llm import call_structured
from app.graph.nodes.helpers import audit, err, page_outline, truncate
from app.graph.prompts import ANALYSIS_SYSTEM, analysis_user
from app.graph.schemas import DocumentStructure
from app.graph.state import GraphState

log = logging.getLogger(__name__)

NODE = "document_analysis"

#: The outline is a digest, not the document. Past this the call gets
#: expensive and the extra pages add little — structure repeats.
_MAX_OUTLINE_CHARS = 30000


def analyse_document(state: GraphState) -> dict:
    settings = get_graph_settings()
    metrics = state.metrics.model_copy(deep=True)

    readable = {p: t for p, t in state.page_text.items() if t.strip()}
    if not readable:
        return {
            "current_node": NODE,
            "document_structure": DocumentStructure(
                document_type="unknown", notes="no readable text to analyse"
            ),
            "audit_log": audit(NODE, "skipped", "no readable pages", level="warning"),
        }

    outline = truncate(page_outline(readable, state.page_metadata), _MAX_OUTLINE_CHARS)

    outcome = call_structured(
        node=NODE,
        system=ANALYSIS_SYSTEM,
        user_text=analysis_user(outline, state.total_pages),
        output_model=DocumentStructure,
        max_attempts=2,
        on_metric=metrics.record,
    )

    if not outcome.ok:
        # Deliberately not fatal: extraction can proceed without structure.
        structure = DocumentStructure(
            document_type="unknown",
            notes=f"structure analysis unavailable: {outcome.error_message[:200]}",
        )
        return {
            "current_node": NODE,
            "document_structure": structure,
            "metrics": metrics,
            "errors": err(
                outcome.error_type or ErrorType.LLM_UNAVAILABLE,
                f"structure analysis failed: {outcome.error_message}",
                NODE, retry_count=outcome.attempts,
                recovery_action="continuing without structural hints",
                resolution_status="RECOVERED",
            ),
            "audit_log": audit(NODE, "analysis_failed",
                               "continuing without structural hints", level="warning"),
        }

    structure: DocumentStructure = outcome.parsed  # type: ignore[assignment]

    # Page numbers the model cites must exist. A hallucinated page range in a
    # hint is harmless on its own, but it would corrupt the completeness
    # report's idea of what we expected to find, so it is scrubbed here.
    valid = set(state.page_text)
    for section in structure.sections:
        if section.start_page not in valid:
            section.start_page = None
        if section.end_page not in valid:
            section.end_page = None

    return {
        "current_node": NODE,
        "document_type": structure.document_type or "unknown",
        "document_structure": structure,
        "metrics": metrics,
        "audit_log": audit(
            NODE, "analysed",
            f"type={structure.document_type!r}, {len(structure.sections)} section(s), "
            f"{len(structure.candidate_fields)} candidate field(s), "
            f"{len(structure.repeated_entities)} repeating entity type(s)",
        ),
    }


def structure_hint(state: GraphState, page_numbers: list[int]) -> str:
    """Short context string for the extraction prompt, scoped to these pages."""
    s = state.document_structure
    if s is None:
        return ""
    bits = [f"document type: {s.document_type}"]
    relevant = [
        sec for sec in s.sections
        if sec.start_page is None
        or any(
            (sec.start_page or 0) <= p <= (sec.end_page or sec.start_page or 0)
            for p in page_numbers
        )
    ]
    if relevant:
        bits.append("sections here: " + "; ".join(
            f"{sec.name} ({sec.kind})" for sec in relevant[:8]
        ))
    if s.repeated_entities:
        bits.append("repeating entities: " + ", ".join(s.repeated_entities[:8]))
    return "\n".join(bits)
