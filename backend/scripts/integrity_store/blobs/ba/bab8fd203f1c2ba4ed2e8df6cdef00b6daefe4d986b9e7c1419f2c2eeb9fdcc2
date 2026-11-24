"""Typed output contracts for the agentic extraction graph.

Every LLM-produced structure in this package is parsed into one of these
models before it is allowed to touch the shared state. A model that fails to
validate is a rejected response, never a partially-trusted one — see
nodes/response_validation.py.

Design notes that are easy to get wrong later:

- `value` and `normalized_value` are BOTH kept. Normalisation is lossy by
  definition (a date reformatted, a unit converted), and the original is what
  the reviewer has to be able to check against the page. Same reasoning for
  `field_name` / `normalized_field_name`.
- `exact_source_evidence` is a verbatim substring of the page text, not a
  paraphrase. nodes/evidence.py checks it really is a substring; a paraphrase
  would make evidence validation meaningless.
- Nothing here defaults `confidence_score` to a flattering number. An absent
  confidence is 0.0 and will be treated as unverified.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------


class ExtractionStatus(str, Enum):
    VERIFIED = "VERIFIED"
    UNCERTAIN = "UNCERTAIN"
    CONFLICTED = "CONFLICTED"
    NOT_FOUND = "NOT_FOUND"


class MappingStatus(str, Enum):
    MAPPED = "MAPPED"
    PARTIALLY_MAPPED = "PARTIALLY_MAPPED"
    NOT_FOUND = "NOT_FOUND"
    AMBIGUOUS = "AMBIGUOUS"
    CONFLICTED = "CONFLICTED"
    MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"


class FinalStatus(str, Enum):
    VALIDATED = "VALIDATED"
    INCOMPLETE = "INCOMPLETE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    CONFLICTED = "CONFLICTED"
    MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"
    REJECTED = "REJECTED"


class EvidenceStatus(str, Enum):
    """Outcome of comparing an extracted value against its cited page text."""

    SUPPORTED = "SUPPORTED"          # value found verbatim (after normalisation)
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"  # evidence found, value fuzzy
    UNSUPPORTED = "UNSUPPORTED"      # evidence string absent from the page
    WRONG_PAGE = "WRONG_PAGE"        # evidence exists, but not on the cited page
    NO_EVIDENCE = "NO_EVIDENCE"      # model returned no evidence at all


class PageStatus(str, Enum):
    NATIVE_TEXT = "NATIVE_TEXT"
    OCR = "OCR"
    OCR_FAILED = "OCR_FAILED"
    EMPTY = "EMPTY"
    UNREADABLE = "UNREADABLE"


class ChunkStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"
    SKIPPED_EMPTY = "SKIPPED_EMPTY"


class ConflictResolution(str, Enum):
    RESOLVED = "RESOLVED"
    UNRESOLVED = "UNRESOLVED"
    MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"


# --------------------------------------------------------------------------
# Core extraction contract (spec section 6)
# --------------------------------------------------------------------------


class ExtractedField(BaseModel):
    """One field-value pair discovered in one chunk.

    Identity is (normalized_field_name, page_number, chunk_id, occurrence) —
    NOT field name alone. The same field legitimately appears many times in a
    real document (one `serial_number` per equipment row), and collapsing on
    name would silently destroy every occurrence but one.
    """

    model_config = ConfigDict(extra="forbid")

    field_name: str
    normalized_field_name: str = ""
    value: Any | None = None
    normalized_value: Any | None = None
    data_type: str = "string"
    page_number: int | None = None
    chunk_id: str = ""
    section_name: str = ""
    table_context: str = ""
    exact_source_evidence: str = ""
    confidence_score: float = 0.0
    extraction_status: ExtractionStatus = ExtractionStatus.UNCERTAIN
    notes: str = ""

    # --- populated by later nodes, never by the extraction LLM ------------
    evidence_status: EvidenceStatus | None = None
    occurrence_index: int = 0
    merged_from: list[str] = Field(default_factory=list)
    field_uid: str = ""

    @field_validator("confidence_score")
    @classmethod
    def _clamp(cls, v: float) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0

    @field_validator("field_name", mode="before")
    @classmethod
    def _nonempty(cls, v: Any) -> str:
        # mode="before": the LLM sometimes emits a bare number as a field
        # name (e.g. a cytology finding numbered "3" rather than a label) —
        # MEASURED: 'VU3 cytology (positive) CLN' crashed the whole document
        # with AttributeError: 'int' object has no attribute 'strip'
        # because the old (v or "").strip() called .strip() on that raw int.
        # Coercing to str first turns a crash into a usable (if odd) field
        # name instead of losing the whole chunk's extraction over one field.
        text = "" if v is None else str(v).strip()
        if not text:
            raise ValueError("field_name must not be empty")
        return text

    @property
    def is_usable(self) -> bool:
        """Safe to put into a form without human confirmation."""
        return (
            self.extraction_status == ExtractionStatus.VERIFIED
            and self.evidence_status == EvidenceStatus.SUPPORTED
            and self.value not in (None, "")
        )


class TestRow(BaseModel):
    """One row of a results table — an analyte and everything about it.

    Columns are taken from the hand-authored gold records, where a test is a
    row rather than a scalar. Flattening loses the parts: `ph = "6.5 5-9"`
    fuses the result and the reference range into one string that cannot be
    compared, sorted or validated, and a NABL scope table needs them apart.

    Every column is optional because real reports vary — a qualitative panel
    has no unit, a screening test has no method — and demanding them would
    push the model into inventing values to satisfy the schema.
    """

    model_config = ConfigDict(extra="forbid")

    panel_name: str | None = None       # "LIVER FUNCTION PROFILE, SERUM"
    section: str | None = None          # "Qualitative" / "Quantitative"
    test_name: str | None = None        # "BILIRUBIN, TOTAL"
    result: str | None = None           # "0.70" or "NEG"
    unit: str | None = None             # "mg/dL"
    reference_range: str | None = None  # "< 1.1", "5-9", "NEGATIVE"
    flag: str | None = None             # "High" / "Low" / "Abnormal"
    method: str | None = None           # "COLORIMETRIC DIAZO METHOD"
    sample_date: str | None = None      # for a trended analyte
    specimen: str | None = None
    s_no: str | None = None

    page_number: int | None = None
    exact_source_evidence: str = ""


class ChunkExtraction(BaseModel):
    """What the extraction agent returns for exactly one chunk."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    fields: list[ExtractedField] = Field(default_factory=list)
    #: Results-table rows, kept structured rather than flattened into `fields`.
    tests: list[TestRow] = Field(default_factory=list)
    tables_seen: list[str] = Field(default_factory=list)
    notes: str = ""


# --------------------------------------------------------------------------
# Document structure (analysis agent)
# --------------------------------------------------------------------------


class DocumentSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    start_page: int | None = None
    end_page: int | None = None
    kind: str = "section"  # section | table | list | annexure
    description: str = ""


class DocumentStructure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_type: str = "unknown"
    sections: list[DocumentSection] = Field(default_factory=list)
    repeated_entities: list[str] = Field(default_factory=list)
    candidate_fields: list[str] = Field(default_factory=list)
    cross_page_references: list[str] = Field(default_factory=list)
    suspected_duplicates: list[str] = Field(default_factory=list)
    notes: str = ""


# --------------------------------------------------------------------------
# Pages and chunks
# --------------------------------------------------------------------------


class PageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int
    text: str = ""
    char_count: int = 0
    status: PageStatus = PageStatus.EMPTY
    ocr_applied: bool = False
    ocr_confidence: float | None = None
    has_table: bool = False
    width: float | None = None
    height: float | None = None
    notes: str = ""

    @property
    def is_readable(self) -> bool:
        return self.status in (PageStatus.NATIVE_TEXT, PageStatus.OCR) and self.char_count > 0


class ChunkRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    page_numbers: list[int] = Field(default_factory=list)
    text: str = ""
    char_count: int = 0
    status: ChunkStatus = ChunkStatus.PENDING
    has_table: bool = False
    context_pages: list[int] = Field(default_factory=list)
    attempt_count: int = 0
    last_error: str = ""
    content_hash: str = ""


# --------------------------------------------------------------------------
# Conflicts, merges, evidence
# --------------------------------------------------------------------------


class ConflictCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field_uid: str
    value: Any | None
    page_number: int | None
    chunk_id: str
    section_name: str = ""
    confidence_score: float = 0.0
    evidence: str = ""


class FieldConflict(BaseModel):
    """Same logical field, two or more materially different values.

    Every candidate is preserved. `chosen_field_uid` may be None — an
    unresolved conflict is a legitimate, reportable outcome and is never
    settled by picking the highest confidence on its own.
    """

    model_config = ConfigDict(extra="forbid")

    normalized_field_name: str
    candidates: list[ConflictCandidate] = Field(default_factory=list)
    resolution: ConflictResolution = ConflictResolution.UNRESOLVED
    chosen_field_uid: str | None = None
    reason: str = ""
    criteria_used: list[str] = Field(default_factory=list)


class MergeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical_name: str
    original_names: list[str] = Field(default_factory=list)
    field_uids: list[str] = Field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0


class EvidenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field_uid: str
    status: EvidenceStatus
    cited_page: int | None = None
    found_on_page: int | None = None
    matched_text: str = ""
    similarity: float = 0.0
    reason: str = ""


# --------------------------------------------------------------------------
# Form mapping (spec section 7)
# --------------------------------------------------------------------------


class TargetFormField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    data_type: str = "string"
    required: bool = False
    description: str = ""
    enum_values: list[str] = Field(default_factory=list)
    repeating: bool = False
    parent: str | None = None


class TargetFormSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    form_id: str = "unknown"
    fields: list[TargetFormField] = Field(default_factory=list)
    source: str = "provided"  # provided | pydantic | inferred


class FormMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_field: str
    target_field: str
    mapped_value: Any | None = None
    mapping_confidence: float = 0.0
    source_page: int | None = None
    source_evidence: str = ""
    mapping_reason: str = ""
    mapping_status: MappingStatus = MappingStatus.NOT_FOUND
    source_field_uid: str = ""

    @field_validator("mapping_confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


class CompletenessReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    all_pages_processed: bool = False
    all_chunks_processed: bool = False
    pages_total: int = 0
    pages_readable: int = 0
    pages_unreadable: list[int] = Field(default_factory=list)
    pages_not_in_any_chunk: list[int] = Field(default_factory=list)
    chunks_total: int = 0
    chunks_failed: list[str] = Field(default_factory=list)
    fields_without_evidence: list[str] = Field(default_factory=list)
    pages_with_no_fields: list[int] = Field(default_factory=list)
    information_loss_detected: bool = False
    reanalysis_targets: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    @property
    def critical_failure(self) -> bool:
        return bool(self.pages_not_in_any_chunk) or not self.all_chunks_processed


class QualityControlReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    all_pages_processed: bool = False
    all_chunks_processed: bool = False
    all_values_have_evidence: bool = False
    conflicts_resolved: bool = False
    form_mapping_verified: bool = False
    information_loss_detected: bool = False
    schema_valid: bool = False
    required_fields_present: bool = False
    issues: list[str] = Field(default_factory=list)
    critical_issues: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.critical_issues


class ManualReviewItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str  # conflict | low_confidence | unsupported | ambiguous_mapping | failed_chunk
    reference: str
    summary: str
    page_number: int | None = None
    candidates: list[Any] = Field(default_factory=list)
    suggested_action: str = ""


class AuditEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(default_factory=_utcnow)
    node: str
    event: str
    detail: str = ""
    page_number: int | None = None
    chunk_id: str | None = None
    level: str = "info"  # info | warning | error


class ProcessingErrorRecord(BaseModel):
    """Spec section 9: every failure is recorded with this shape."""

    model_config = ConfigDict(extra="forbid")

    error_type: str
    error_message: str
    node: str
    page_number: int | None = None
    chunk_id: str | None = None
    retry_count: int = 0
    recovery_action: str = ""
    resolution_status: str = "OPEN"  # OPEN | RECOVERED | ABANDONED
    timestamp: datetime = Field(default_factory=_utcnow)


class LlmCallMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node: str
    model: str
    provider: str
    latency_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    ok: bool = True
    error: str = ""
    attempt: int = 1


class Metrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_call_count: int = 0
    retry_count: int = 0
    processing_time_seconds: float = 0.0
    token_usage: dict[str, int] = Field(default_factory=lambda: {"input": 0, "output": 0, "total": 0})
    node_durations: dict[str, float] = Field(default_factory=dict)
    calls: list[LlmCallMetric] = Field(default_factory=list)

    def record(self, m: LlmCallMetric) -> None:
        self.calls.append(m)
        self.llm_call_count += 1
        self.token_usage["input"] += m.input_tokens
        self.token_usage["output"] += m.output_tokens
        self.token_usage["total"] += m.input_tokens + m.output_tokens


# --------------------------------------------------------------------------
# Final output (spec section 8)
# --------------------------------------------------------------------------


class DocumentSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_type: str | None = None
    page_count: int = 0
    processed_pages: int = 0
    processed_chunks: int = 0
    failed_chunks: int = 0
    ocr_pages: int = 0


class QualityControlOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    all_pages_processed: bool = False
    all_chunks_processed: bool = False
    all_values_have_evidence: bool = False
    conflicts_resolved: bool = False
    form_mapping_verified: bool = False
    information_loss_detected: bool = False


class FinalResult(BaseModel):
    """Exactly the JSON shape required by spec section 8."""

    model_config = ConfigDict(extra="forbid")

    overall_status: FinalStatus
    document_summary: DocumentSummary
    extracted_fields: list[ExtractedField] = Field(default_factory=list)
    #: Results-table rows, structured rather than flattened.
    tests: list[TestRow] = Field(default_factory=list)
    #: The same content in the shape the hand-authored gold records use —
    #: lab_info / patient_info / sample_info / report_info / tests[] — so it
    #: can be diffed against ground truth without a translation layer.
    structured_document: dict[str, Any] = Field(default_factory=dict)
    form_mappings: list[FormMapping] = Field(default_factory=list)
    filled_form: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    conflicts: list[FieldConflict] = Field(default_factory=list)
    failed_chunks: list[str] = Field(default_factory=list)
    manual_review_items: list[ManualReviewItem] = Field(default_factory=list)
    quality_control: QualityControlOutput = Field(default_factory=QualityControlOutput)
    final_reasoning: str = ""
    audit_log: list[AuditEntry] = Field(default_factory=list)
    metrics: Metrics = Field(default_factory=Metrics)
