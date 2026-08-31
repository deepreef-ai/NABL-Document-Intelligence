from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from fastapi.responses import Response
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.db import get_db
from app.documents import pdf_utils
from app.documents.pipeline import process_document
from app.documents.storage import LocalFileStorage
from app.llm.base import redact_known_secrets
from app.models import Application, Document, ExtractedField

router = APIRouter(tags=["documents"])
storage = LocalFileStorage()

UNLOCKED_STATUSES = {"unlocked", "reviewing", "complete"}


def _get_application(db: Session, application_id: str) -> Application:
    application = db.get(Application, application_id)
    if application is None:
        raise HTTPException(404, f"application {application_id!r} not found")
    return application


def _get_document(db: Session, document_id: str) -> Document:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(404, f"document {document_id!r} not found")
    return document


def _run_pipeline(db: Session, document: Document, data: bytes, script: str) -> None:
    document.status = "processing"
    db.add(document)
    db.commit()

    try:
        result = process_document(
            data,
            document.filename,
            document.content_type,
            script=script,
            form_type=document.application.form_type,
            document_id=document.id,
        )
    except Exception as exc:  # noqa: BLE001 — surface as a failed document, not a 500
        document.status = "failed"
        # redact_known_secrets is defense-in-depth: every LLM provider already
        # scrubs its own API key out of an error at the point of failure (see
        # llm/base.py's classify_http_error), but this is the one place any
        # exception in the whole pipeline ends up stored and returned straight
        # to the frontend, so it gets a second, independent pass.
        document.error = redact_known_secrets(str(exc))
        db.add(document)
        db.commit()
        return

    document.doc_type = result.doc_type
    document.extraction_source = result.extraction_source
    document.status = "extracted"
    # Non-fatal: some section/chunk of a whole-form extraction couldn't be
    # completed (e.g. every LLM provider was rate-limited at that moment)
    # while the rest succeeded — see documents/pipeline.py's
    # extraction_warnings. The document is still "extracted" with whatever
    # DID come back, but the gap is recorded rather than silently hidden.
    document.error = redact_known_secrets("; ".join(result.extraction_warnings)) or None

    db.query(ExtractedField).filter(ExtractedField.document_id == document.id).delete()
    for f in result.fields:
        db.add(
            ExtractedField(
                document_id=document.id,
                field_path=f.field,
                value=f.value,
                confidence=f.confidence,
                source_page=f.source_page,
                source_bbox=f.source_bbox.model_dump() if f.source_bbox else None,
                source=f.source,
            )
        )
    db.add(document)
    db.commit()


@router.post("/applications/{application_id}/documents")
async def upload_document(
    application_id: str,
    file: UploadFile,
    script: str = Form("english"),
    db: Session = Depends(get_db),
):
    application = _get_application(db, application_id)
    if application.status not in UNLOCKED_STATUSES:
        raise HTTPException(409, "eligibility prerequisites are not all satisfied yet")

    data = await file.read()
    key = storage.put(data, file.filename or "upload")
    document = Document(
        application_id=application_id,
        filename=file.filename or key,
        content_type=file.content_type or "application/octet-stream",
        storage_path=key,
    )
    db.add(document)
    if application.status == "unlocked":
        application.status = "reviewing"
        db.add(application)
    db.commit()
    db.refresh(document)

    # A local-model chunked extraction (see documents/pipeline.py's whole-form
    # path) can run for minutes — _run_pipeline does blocking network calls
    # (httpx, not an async client), and this is an `async def` endpoint (only
    # because of `await file.read()` above), so calling it directly would
    # block this worker's entire event loop and starve every other request,
    # not just this one, for the whole duration. Offload it to a thread.
    # (reextract_document below needs no such wrap — it's a plain `def`
    # endpoint, which Starlette already dispatches to a thread automatically.)
    await run_in_threadpool(_run_pipeline, db, document, data, script)
    db.refresh(document)
    return _serialize(document)


@router.get("/applications/{application_id}/documents")
def list_documents(application_id: str, db: Session = Depends(get_db)):
    _get_application(db, application_id)
    docs = db.query(Document).filter(Document.application_id == application_id).all()
    return [_serialize(d) for d in docs]


@router.get("/documents/{document_id}/render")
def render_document(document_id: str, page: int = 0, db: Session = Depends(get_db)):
    document = _get_document(db, document_id)
    data = storage.get(document.storage_path)
    if "pdf" in document.content_type or document.filename.lower().endswith(".pdf"):
        png = pdf_utils.rasterize_page(data, page)
        return Response(content=png, media_type="image/png")
    if document.content_type.startswith("image/"):
        return Response(content=data, media_type=document.content_type)
    raise HTTPException(415, "this document type has no visual rendering (e.g. DOCX)")


@router.post("/documents/{document_id}/reextract")
def reextract_document(document_id: str, script: str = Form("english"), db: Session = Depends(get_db)):
    document = _get_document(db, document_id)
    data = storage.get(document.storage_path)
    _run_pipeline(db, document, data, script)
    db.refresh(document)
    return _serialize(document)


def _serialize(document: Document) -> dict:
    return {
        "id": document.id,
        "application_id": document.application_id,
        "filename": document.filename,
        "content_type": document.content_type,
        "doc_type": document.doc_type,
        "extraction_source": document.extraction_source,
        "status": document.status,
        "error": document.error,
        "fields": [
            {
                "id": f.id,
                "field_path": f.field_path,
                "value": f.value,
                "confidence": f.confidence,
                "source_page": f.source_page,
                "source_bbox": f.source_bbox,
                "source": f.source,
                "accepted": f.accepted,
            }
            for f in document.fields
        ],
    }
