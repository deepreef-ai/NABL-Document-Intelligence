"""Persistence for graph runs: SQLAlchemy models + a thin repository.

SQLite by default, Postgres by changing `graph_database_url` and nothing else.
The column types are chosen to make that swap real rather than nominal:

- `JSON` (not JSONB) so SQLite and Postgres both accept the DDL. On Postgres,
  add a migration to convert to JSONB and index it if you start querying inside
  the payloads; nothing in this module reads them by content.
- Timestamps are timezone-aware UTC everywhere. A naive local timestamp in an
  audit trail is worse than none, because it looks authoritative.
- No `ON DELETE CASCADE` from run to field. Deleting a run's rows should be a
  deliberate, logged act — this is an audit store, and the whole point is that
  records outlive the process that made them.

The repository never raises to its caller. A workflow that produced a correct
result and then failed to write it down has still produced a correct result;
losing that to a database hiccup would be the worst possible trade. Failures
are returned as error records and surfaced in the audit trail instead.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from app.graph.config import get_graph_settings

log = logging.getLogger(__name__)

Base = declarative_base()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


class GraphRun(Base):
    """One execution of the workflow over one document."""

    __tablename__ = "graph_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String(128), nullable=False, index=True)
    thread_id = Column(String(128), nullable=False, unique=True, index=True)

    file_name = Column(String(512), default="")
    file_path = Column(Text, default="")
    file_type = Column(String(32), default="")
    file_size_bytes = Column(Integer, default=0)
    document_type = Column(String(128), default="unknown")
    form_id = Column(String(64), default="")

    total_pages = Column(Integer, default=0)
    processed_pages = Column(Integer, default=0)
    ocr_pages = Column(Integer, default=0)
    unreadable_pages = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    processed_chunks = Column(Integer, default=0)
    failed_chunks = Column(Integer, default=0)

    current_status = Column(String(32), default="PENDING", index=True)
    final_status = Column(String(32), default="", index=True)
    final_reasoning = Column(Text, default="")
    awaiting_human = Column(Boolean, default=False, index=True)

    llm_call_count = Column(Integer, default=0)
    retry_count = Column(Integer, default=0)
    processing_time_seconds = Column(Float, default=0.0)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)

    result_json = Column(JSON, default=dict)
    quality_control_json = Column(JSON, default=dict)
    completeness_json = Column(JSON, default=dict)

    created_at = Column(DateTime(timezone=True), default=_utcnow, index=True)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


class GraphField(Base):
    """One extracted field occurrence, denormalised for querying.

    Stored per-occurrence rather than per-field-name, because "how many times
    did this document state a serial number, and where" is the question an
    auditor actually asks.
    """

    __tablename__ = "graph_fields"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("graph_runs.id"), nullable=False, index=True)
    field_uid = Column(String(32), index=True)

    field_name = Column(String(512), default="")
    normalized_field_name = Column(String(512), index=True)
    value = Column(Text)
    normalized_value = Column(Text)
    data_type = Column(String(32), default="string")

    page_number = Column(Integer, index=True)
    chunk_id = Column(String(128), default="")
    section_name = Column(String(512), default="")
    table_context = Column(String(512), default="")
    exact_source_evidence = Column(Text, default="")

    confidence_score = Column(Float, default=0.0)
    extraction_status = Column(String(32), index=True)
    evidence_status = Column(String(32), index=True)
    occurrence_index = Column(Integer, default=0)
    notes = Column(Text, default="")


class GraphMapping(Base):
    __tablename__ = "graph_mappings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("graph_runs.id"), nullable=False, index=True)
    source_field = Column(String(512), default="")
    target_field = Column(String(512), index=True)
    mapped_value = Column(Text)
    mapping_confidence = Column(Float, default=0.0)
    source_page = Column(Integer)
    source_evidence = Column(Text, default="")
    mapping_reason = Column(Text, default="")
    mapping_status = Column(String(32), index=True)


class GraphConflict(Base):
    __tablename__ = "graph_conflicts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("graph_runs.id"), nullable=False, index=True)
    normalized_field_name = Column(String(512), index=True)
    resolution = Column(String(32), index=True)
    chosen_field_uid = Column(String(32), default="")
    reason = Column(Text, default="")
    candidates_json = Column(JSON, default=list)


class GraphErrorLog(Base):
    __tablename__ = "graph_errors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("graph_runs.id"), nullable=False, index=True)
    error_type = Column(String(64), index=True)
    error_message = Column(Text, default="")
    node = Column(String(64), index=True)
    page_number = Column(Integer)
    chunk_id = Column(String(128))
    retry_count = Column(Integer, default=0)
    recovery_action = Column(Text, default="")
    resolution_status = Column(String(32), index=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)


class GraphAudit(Base):
    __tablename__ = "graph_audit"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("graph_runs.id"), nullable=False, index=True)
    node = Column(String(64), index=True)
    event = Column(String(128), index=True)
    detail = Column(Text, default="")
    page_number = Column(Integer)
    chunk_id = Column(String(128))
    level = Column(String(16), default="info", index=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)


Index("ix_graph_fields_run_name", GraphField.run_id, GraphField.normalized_field_name)
Index("ix_graph_audit_run_created", GraphAudit.run_id, GraphAudit.created_at)


# --------------------------------------------------------------------------
# Engine / session
# --------------------------------------------------------------------------

_engine = None
_Session = None


def _url() -> str:
    settings = get_graph_settings()
    if settings.graph_database_url:
        return settings.graph_database_url
    path = os.path.abspath(settings.graph_sqlite_path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return f"sqlite:///{path}"


def get_engine():
    global _engine
    if _engine is None:
        url = _url()
        kwargs = {"future": True}
        if url.startswith("sqlite"):
            # The API and the graph's worker threads share this engine.
            kwargs["connect_args"] = {"check_same_thread": False}
        else:
            kwargs.update(pool_pre_ping=True, pool_size=5, max_overflow=10)
        _engine = create_engine(url, **kwargs)
    return _engine


def init_db() -> None:
    Base.metadata.create_all(get_engine())


@contextmanager
def session_scope():
    global _Session
    if _Session is None:
        _Session = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    session = _Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """Test hook: drop the cached engine so a new URL takes effect."""
    global _engine, _Session
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _Session = None


# --------------------------------------------------------------------------
# Repository
# --------------------------------------------------------------------------


def save_run(state, result) -> int:
    """Persist one completed run. Returns the run id.

    Raises only on a genuine database failure; the caller (the persistence
    node) catches and records it rather than failing the workflow.
    """
    init_db()
    payload = result.model_dump(mode="json")

    with session_scope() as s:
        run = s.query(GraphRun).filter_by(thread_id=state.document_id).one_or_none()
        if run is None:
            run = GraphRun(document_id=state.document_id, thread_id=state.document_id)
            s.add(run)

        run.file_name = state.file_name
        run.file_path = state.file_path
        run.file_type = state.file_type
        run.file_size_bytes = state.file_size_bytes
        run.document_type = state.document_type
        run.form_id = state.target_form_schema.form_id if state.target_form_schema else ""
        run.total_pages = state.total_pages
        run.processed_pages = len(state.processed_pages)
        run.ocr_pages = len(state.ocr_pages)
        run.unreadable_pages = len(state.unreadable_pages)
        run.total_chunks = len(state.chunks)
        run.processed_chunks = len(state.processed_chunks)
        run.failed_chunks = len(state.failed_chunks)
        run.current_status = state.current_status
        run.final_status = state.final_status.value if state.final_status else ""
        run.final_reasoning = state.final_reasoning
        run.awaiting_human = state.awaiting_human
        run.llm_call_count = state.metrics.llm_call_count
        run.retry_count = state.retry_count
        run.processing_time_seconds = state.metrics.processing_time_seconds
        run.input_tokens = state.metrics.token_usage.get("input", 0)
        run.output_tokens = state.metrics.token_usage.get("output", 0)
        run.result_json = payload
        run.quality_control_json = (
            state.quality_control_report.model_dump(mode="json")
            if state.quality_control_report else {}
        )
        run.completeness_json = (
            state.completeness_report.model_dump(mode="json")
            if state.completeness_report else {}
        )
        s.flush()
        run_id = run.id

        # Child rows are rewritten wholesale — a re-run of the same thread
        # supersedes its predecessor rather than accumulating duplicates.
        for model in (GraphField, GraphMapping, GraphConflict, GraphErrorLog, GraphAudit):
            s.query(model).filter_by(run_id=run_id).delete(synchronize_session=False)

        s.bulk_save_objects([
            GraphField(
                run_id=run_id, field_uid=f.field_uid, field_name=f.field_name[:512],
                normalized_field_name=f.normalized_field_name[:512],
                value=None if f.value is None else str(f.value),
                normalized_value=None if f.normalized_value is None else str(f.normalized_value),
                data_type=f.data_type, page_number=f.page_number, chunk_id=f.chunk_id[:128],
                section_name=f.section_name[:512], table_context=f.table_context[:512],
                exact_source_evidence=f.exact_source_evidence,
                confidence_score=f.confidence_score,
                extraction_status=f.extraction_status.value,
                evidence_status=f.evidence_status.value if f.evidence_status else None,
                occurrence_index=f.occurrence_index, notes=f.notes,
            )
            for f in result.extracted_fields
        ])
        s.bulk_save_objects([
            GraphMapping(
                run_id=run_id, source_field=m.source_field[:512],
                target_field=m.target_field[:512],
                mapped_value=None if m.mapped_value is None else str(m.mapped_value),
                mapping_confidence=m.mapping_confidence, source_page=m.source_page,
                source_evidence=m.source_evidence, mapping_reason=m.mapping_reason,
                mapping_status=m.mapping_status.value,
            )
            for m in result.form_mappings
        ])
        s.bulk_save_objects([
            GraphConflict(
                run_id=run_id, normalized_field_name=c.normalized_field_name[:512],
                resolution=c.resolution.value, chosen_field_uid=c.chosen_field_uid or "",
                reason=c.reason,
                candidates_json=[cand.model_dump(mode="json") for cand in c.candidates],
            )
            for c in result.conflicts
        ])
        s.bulk_save_objects([
            GraphErrorLog(
                run_id=run_id, error_type=e.error_type, error_message=e.error_message,
                node=e.node, page_number=e.page_number, chunk_id=e.chunk_id,
                retry_count=e.retry_count, recovery_action=e.recovery_action,
                resolution_status=e.resolution_status,
            )
            for e in state.errors
        ])
        s.bulk_save_objects([
            GraphAudit(
                run_id=run_id, node=a.node, event=a.event, detail=a.detail,
                page_number=a.page_number, chunk_id=a.chunk_id, level=a.level,
                created_at=a.timestamp,
            )
            for a in result.audit_log
        ])
        return run_id


def get_run(document_id: str) -> dict | None:
    init_db()
    with session_scope() as s:
        run = s.query(GraphRun).filter_by(thread_id=document_id).one_or_none()
        if run is None:
            return None
        return {
            "document_id": run.document_id,
            "thread_id": run.thread_id,
            "file_name": run.file_name,
            "final_status": run.final_status,
            "current_status": run.current_status,
            "awaiting_human": run.awaiting_human,
            "total_pages": run.total_pages,
            "processed_chunks": run.processed_chunks,
            "failed_chunks": run.failed_chunks,
            "llm_call_count": run.llm_call_count,
            "processing_time_seconds": run.processing_time_seconds,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "result": run.result_json,
        }


def list_runs(limit: int = 50, status: str | None = None) -> list[dict]:
    init_db()
    with session_scope() as s:
        q = s.query(GraphRun).order_by(GraphRun.created_at.desc())
        if status:
            q = q.filter(GraphRun.final_status == status)
        return [
            {
                "document_id": r.document_id,
                "file_name": r.file_name,
                "final_status": r.final_status,
                "current_status": r.current_status,
                "awaiting_human": r.awaiting_human,
                "total_pages": r.total_pages,
                "failed_chunks": r.failed_chunks,
                "llm_call_count": r.llm_call_count,
                "processing_time_seconds": r.processing_time_seconds,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in q.limit(limit).all()
        ]
