"""The shared, typed workflow state (spec section 5).

One Pydantic model threaded through every node. Two rules govern it:

1. **Nothing is overwritten that a reviewer might need.** Normalisation adds
   fields rather than replacing them; merging records a MergeRecord rather
   than deleting the merged-away occurrences; conflict resolution keeps every
   candidate. The only things that get replaced are genuinely derived
   summaries (reports), which are recomputed from scratch each time.

2. **Nodes return partial updates, they do not mutate.** LangGraph merges the
   returned dict into the state. Mutating `state` in place works by accident
   with a Pydantic state object and breaks the moment a node runs
   concurrently or a checkpoint is replayed, so every node here returns a
   dict. `helpers.py` has the small append-helpers that make that ergonomic.

List fields that several nodes append to carry an `operator.add` reducer so a
partial update extends rather than replaces. Fields a single node owns
outright (the reports) have no reducer and are replaced wholesale.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from app.graph.schemas import (
    AuditEntry,
    ChunkRecord,
    CompletenessReport,
    DocumentStructure,
    EvidenceResult,
    ExtractedField,
    FieldConflict,
    FinalStatus,
    FormMapping,
    ManualReviewItem,
    MergeRecord,
    Metrics,
    PageRecord,
    ProcessingErrorRecord,
    QualityControlReport,
    TestRow,
    TargetFormSchema,
)


class WorkflowStatus:
    """Coarse lifecycle marker, distinct from the FinalStatus verdict."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    AWAITING_HUMAN = "AWAITING_HUMAN"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class GraphState(BaseModel):
    """Shared state for the document-processing graph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ---- identity / input ------------------------------------------------
    document_id: str
    file_path: str = ""
    file_name: str = ""
    #: What the document is CALLED, as opposed to where it currently sits.
    #: Uploads are staged to a NamedTemporaryFile before the graph reads them,
    #: so file_name is "tmpn0yclbyd.pdf" and the gold-shaped output recorded
    #: that as original_filename. Empty means "no better name than the path".
    display_name: str = ""
    file_type: str = ""
    file_size_bytes: int = 0
    document_type: str = "unknown"

    # ---- pages -----------------------------------------------------------
    total_pages: int = 0
    processed_pages: list[int] = Field(default_factory=list)
    unreadable_pages: list[int] = Field(default_factory=list)
    ocr_pages: list[int] = Field(default_factory=list)
    page_text: dict[int, str] = Field(default_factory=dict)
    page_metadata: dict[int, PageRecord] = Field(default_factory=dict)

    # ---- structure -------------------------------------------------------
    document_structure: DocumentStructure | None = None

    # ---- chunks ----------------------------------------------------------
    chunks: list[ChunkRecord] = Field(default_factory=list)
    processed_chunks: list[str] = Field(default_factory=list)
    failed_chunks: list[str] = Field(default_factory=list)
    retry_count: int = 0
    retry_reasons: Annotated[list[str], operator.add] = Field(default_factory=list)

    # ---- extraction ------------------------------------------------------
    extracted_fields: Annotated[list[ExtractedField], operator.add] = Field(default_factory=list)
    #: Results-table rows, kept structured instead of flattened into
    #: extracted_fields — see structured.py for why that distinction matters.
    tests: Annotated[list[TestRow], operator.add] = Field(default_factory=list)
    normalized_fields: list[ExtractedField] = Field(default_factory=list)
    duplicate_fields: list[MergeRecord] = Field(default_factory=list)
    conflicts: list[FieldConflict] = Field(default_factory=list)
    evidence_validation_results: list[EvidenceResult] = Field(default_factory=list)

    # ---- reports ---------------------------------------------------------
    completeness_report: CompletenessReport | None = None
    quality_control_report: QualityControlReport | None = None

    # ---- form ------------------------------------------------------------
    target_form_schema: TargetFormSchema | None = None
    form_mappings: list[FormMapping] = Field(default_factory=list)
    filled_form: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)

    # ---- human in the loop -----------------------------------------------
    manual_review_items: list[ManualReviewItem] = Field(default_factory=list)
    human_decisions: dict[str, Any] = Field(default_factory=dict)
    awaiting_human: bool = False

    # ---- bookkeeping -----------------------------------------------------
    audit_log: Annotated[list[AuditEntry], operator.add] = Field(default_factory=list)
    errors: Annotated[list[ProcessingErrorRecord], operator.add] = Field(default_factory=list)
    metrics: Metrics = Field(default_factory=Metrics)

    current_node: str = ""
    current_status: str = WorkflowStatus.PENDING
    final_status: FinalStatus | None = None
    final_reasoning: str = ""
    error_details: str = ""

    started_at: float = 0.0

    # ---- loop guards -----------------------------------------------------
    # Counted separately from retry_count so a re-analysis loop and an
    # extraction-retry loop cannot mask each other's runaway.
    reanalysis_count: int = 0
    extraction_attempts: int = 0

    # ---------------------------------------------------------------- views

    def chunk_by_id(self, chunk_id: str) -> ChunkRecord | None:
        for c in self.chunks:
            if c.chunk_id == chunk_id:
                return c
        return None

    def pending_chunks(self) -> list[ChunkRecord]:
        from app.graph.schemas import ChunkStatus

        return [c for c in self.chunks if c.status in (ChunkStatus.PENDING, ChunkStatus.FAILED)]

    def readable_pages(self) -> list[int]:
        return sorted(p for p, rec in self.page_metadata.items() if rec.is_readable)

    def fields_for_page(self, page: int) -> list[ExtractedField]:
        pool = self.normalized_fields or self.extracted_fields
        return [f for f in pool if f.page_number == page]

    def active_fields(self) -> list[ExtractedField]:
        """The best current view: normalised if we have it, raw otherwise."""
        return self.normalized_fields or self.extracted_fields

    def evidence_for(self, field_uid: str) -> EvidenceResult | None:
        for r in self.evidence_validation_results:
            if r.field_uid == field_uid:
                return r
        return None
