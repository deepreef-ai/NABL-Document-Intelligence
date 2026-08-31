from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.documents.compiler import compile_form
from app.documents.extraction_report import ExtractionReport
from app.documents.form_pdf import render_form_pdf
from app.models import Application, Document, ExtractedField

router = APIRouter(tags=["review"])


class FieldPatch(BaseModel):
    value: str | None = None
    accepted: bool | None = None


def _get_application(db: Session, application_id: str) -> Application:
    application = db.get(Application, application_id)
    if application is None:
        raise HTTPException(404, f"application {application_id!r} not found")
    return application


@router.get("/applications/{application_id}/form")
def get_compiled_form(application_id: str, accepted_only: bool = False, db: Session = Depends(get_db)):
    application = _get_application(db, application_id)
    documents = db.query(Document).filter(Document.application_id == application_id).all()
    threshold = get_settings().confidence_threshold

    return {
        "form_type": application.form_type,
        "form": compile_form(application.form_type, documents, accepted_only=accepted_only),
        "confidence_threshold": threshold,
        "documents": [
            {
                "id": d.id,
                "filename": d.filename,
                "doc_type": d.doc_type,
                "extraction_source": d.extraction_source,
                "status": d.status,
                "fields": [
                    {
                        "id": f.id,
                        "field_path": f.field_path,
                        "value": f.value,
                        "confidence": f.confidence,
                        "needs_review": f.confidence < threshold,
                        "source_page": f.source_page,
                        "source_bbox": f.source_bbox,
                        "accepted": f.accepted,
                    }
                    for f in d.fields
                ],
            }
            for d in documents
        ],
    }


def _find_missing_flat_fields(compiled: dict, prefix: str = "") -> list[str]:
    """Walks the compiled dict for flat (non-list) fields left at their
    default None/empty — repeating sections (lists) are skipped since
    "missing" doesn't mean the same thing there (we don't know how many
    records *should* exist)."""
    missing = []
    for key, value in compiled.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            missing.extend(_find_missing_flat_fields(value, prefix=f"{path}."))
        elif isinstance(value, list):
            continue
        elif value in (None, ""):
            missing.append(path)
    return missing


@router.get("/applications/{application_id}/extraction-report")
def get_extraction_report(application_id: str, db: Session = Depends(get_db)):
    """Traceability alongside the compiled form: every extracted field with
    its source and confidence, what's still missing, what conflicted across
    sources (rule-based vs. LLM, or two different pages), and what Pydantic
    validation rejected — see documents/extraction_report.py."""
    application = _get_application(db, application_id)
    documents = db.query(Document).filter(Document.application_id == application_id).all()

    report = ExtractionReport()
    compiled = compile_form(application.form_type, documents, report=report)
    report.missing_fields = _find_missing_flat_fields(compiled)
    report.extracted_fields = [
        {
            "field_path": f.field_path,
            "value": f.value,
            "confidence": f.confidence,
            "source": f.source,
            "source_document": d.filename,
            "source_page": f.source_page,
            "accepted": f.accepted,
        }
        for d in documents
        for f in d.fields
        if f.value
    ]
    return report.to_dict()


@router.get("/applications/{application_id}/form.pdf")
def get_form_pdf(application_id: str, db: Session = Depends(get_db)):
    application = _get_application(db, application_id)
    documents = db.query(Document).filter(Document.application_id == application_id).all()
    compiled = compile_form(application.form_type, documents)
    pdf_bytes = render_form_pdf(application.form_type, compiled)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{application.form_type}.pdf"'},
    )


@router.patch("/documents/{document_id}/fields/{field_id}")
def patch_field(document_id: str, field_id: str, body: FieldPatch, db: Session = Depends(get_db)):
    field = db.get(ExtractedField, field_id)
    if field is None or field.document_id != document_id:
        raise HTTPException(404, f"field {field_id!r} not found on document {document_id!r}")

    if body.value is not None:
        field.value = body.value
        field.confidence = 1.0  # a human-entered value is authoritative
    if body.accepted is not None:
        field.accepted = body.accepted

    db.add(field)
    db.commit()
    db.refresh(field)
    return {
        "id": field.id,
        "field_path": field.field_path,
        "value": field.value,
        "confidence": field.confidence,
        "accepted": field.accepted,
    }
