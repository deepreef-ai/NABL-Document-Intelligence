"""A document's stored status must reflect what the graph pipeline decided.

Regression: a REJECTED / INCOMPLETE-with-nothing run came back from the
adapter as an ordinary PipelineResult and the router stamped it "extracted".
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.documents.grounding import FieldResult, PipelineResult
from app.models import Application, Document
from app.routers import documents as docs_router


@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    app = Application(form_type="x") if "form_type" in Application.__table__.c else Application()
    session.add(app)
    session.flush()
    doc = Document(application_id=app.id, filename="a.pdf", content_type="application/pdf",
                   storage_path="p")
    session.add(doc)
    session.commit()
    yield session, doc
    session.close()


def _run(db, monkeypatch, **kw):
    session, doc = db
    fields = kw.pop("fields", [FieldResult("report_no", "LR-1", 0.9, 1, None, source="llm")])
    result = PipelineResult(doc_type="other", doc_confidence=0.5, extraction_source="pdf",
                            fields=fields, **kw)
    monkeypatch.setattr(docs_router, "process_document", lambda *a, **k: result)
    docs_router._run_pipeline(session, doc, b"x", "english")
    session.refresh(doc)
    return doc


def test_rejected_run_is_failed(db, monkeypatch):
    doc = _run(db, monkeypatch, fields=[], overall_status="REJECTED",
               extraction_warnings=["Extraction rejected: corrupt pdf"])
    assert doc.status == "failed"
    assert "rejected" in doc.error.lower()


def test_incomplete_with_no_fields_is_failed(db, monkeypatch):
    doc = _run(db, monkeypatch, fields=[], overall_status="INCOMPLETE",
               extraction_warnings=["Partial read: nothing extracted"])
    assert doc.status == "failed"


@pytest.mark.parametrize("st", ["CONFLICTED", "LOW_CONFIDENCE", "MANUAL_REVIEW_REQUIRED", "INCOMPLETE"])
def test_unresolved_run_needs_review_and_keeps_fields(db, monkeypatch, st):
    doc = _run(db, monkeypatch, overall_status=st, extraction_warnings=["check me"])
    assert doc.status == "needs_review"
    assert [f.field_path for f in doc.fields] == ["report_no"]


def test_clean_run_is_extracted(db, monkeypatch):
    assert _run(db, monkeypatch, overall_status="VALIDATED").status == "extracted"


def test_legacy_result_without_status_is_extracted(db, monkeypatch):
    assert _run(db, monkeypatch).status == "extracted"


def test_poll_reports_needs_review_as_done(db):
    session, doc = db
    doc.status = "needs_review"
    session.commit()
    assert docs_router.document_status(doc.id, session)["done"] is True
