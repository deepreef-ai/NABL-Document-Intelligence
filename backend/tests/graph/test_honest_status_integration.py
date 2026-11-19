"""Real graph -> adapter -> router: a rejected document must be stored as failed."""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.documents.pipeline import process_document
from app.graph.adapter import run_graph_pipeline
from app.models import Application, Document
from app.routers import documents as docs_router


def test_adapter_carries_the_graph_verdict(corrupt_pdf, scripted):
    scripted({})
    result = run_graph_pipeline(corrupt_pdf, document_id="hs-1")
    assert result.overall_status == "REJECTED"


def test_rejected_document_is_stored_failed_end_to_end(corrupt_pdf, scripted):
    scripted({})
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    app = Application(form_type="NABL_151")
    db.add(app)
    db.flush()
    doc = Document(application_id=app.id, filename="broken.pdf",
                   content_type="application/pdf", storage_path="p")
    db.add(doc)
    db.commit()

    data = open(corrupt_pdf, "rb").read()
    docs_router._run_pipeline(db, doc, data, "english")
    db.refresh(doc)
    assert doc.status == "failed"
    assert doc.error
