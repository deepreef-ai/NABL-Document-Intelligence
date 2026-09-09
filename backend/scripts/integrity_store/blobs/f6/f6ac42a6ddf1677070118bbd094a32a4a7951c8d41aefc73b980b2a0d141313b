"""Graph nodes, one module per agent or processing stage (spec section 3).

Deterministic nodes (no LLM): validation, preprocess, chunking,
response_validation, recovery, evidence, completeness, form_fill, quality,
decision, persistence.

LLM-backed agents: analysis, extraction, normalize (partly), conflicts
(adjudication only), mapping (second pass only).

That split is the point of the architecture. Anything that can be decided by
comparing strings or counting pages is decided that way, so the model is spent
only on semantic judgement — and so the parts of the system that guarantee
correctness are themselves testable without a network call.
"""
from app.graph.nodes.analysis import analyse_document
from app.graph.nodes.chunking import create_chunks
from app.graph.nodes.completeness import check_completeness, mark_reanalysis_targets
from app.graph.nodes.conflicts import detect_and_resolve_conflicts
from app.graph.nodes.decision import build_result, decide
from app.graph.nodes.evidence import validate_evidence
from app.graph.nodes.extraction import extract_fields
from app.graph.nodes.form_fill import fill_form
from app.graph.nodes.mapping import map_to_form
from app.graph.nodes.normalize import normalize_and_merge
from app.graph.nodes.preprocess import preprocess_document
from app.graph.nodes.quality import run_quality_control
from app.graph.nodes.recovery import recover
from app.graph.nodes.response_validation import validate_responses
from app.graph.nodes.validation import validate_document

__all__ = [
    "analyse_document",
    "build_result",
    "check_completeness",
    "create_chunks",
    "decide",
    "detect_and_resolve_conflicts",
    "extract_fields",
    "fill_form",
    "map_to_form",
    "mark_reanalysis_targets",
    "normalize_and_merge",
    "preprocess_document",
    "recover",
    "run_quality_control",
    "validate_document",
    "validate_evidence",
    "validate_responses",
]
