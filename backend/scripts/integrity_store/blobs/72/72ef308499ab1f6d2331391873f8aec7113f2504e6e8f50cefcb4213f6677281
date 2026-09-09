"""FastAPI endpoints for the agentic extraction workflow.

Design decisions worth stating:

- **Processing runs in a background task, not in the request.** A 300-page
  document takes minutes; holding an HTTP connection open for that is a
  liability at both ends. The POST returns a document id immediately and the
  caller polls `/graph/documents/{id}/status`, which reads the checkpoint and
  costs nothing.
- **Status comes from the checkpointer, results from the database.** A run in
  flight has no database row yet, and a finished run should not require the
  graph to be rebuilt to read. Two stores, two purposes.
- **Uploads are written to disk before validation.** The validation node needs
  a real path, and streaming to disk bounds memory for a large PDF. The file
  is written under a per-document directory so cleanup is one `rmtree`.
- **The review endpoint takes decisions, not values.** A reviewer chooses among
  candidates the document actually contains, by `field_uid`. There is no API
  path that injects an arbitrary value into a form, because that would be a
  hole straight through the evidence guarantee.
"""
from __future__ import annotations

import logging
import os
import shutil
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from app.graph import database
from app.graph.config import get_graph_settings
from app.graph.graph import mermaid
from app.graph.runner import get_status, new_document_id, resume_document, run_document

log = logging.getLogger(__name__)
router = APIRouter(prefix="/graph", tags=["agentic-extraction"])

UPLOAD_ROOT = os.path.join("graph_storage", "uploads")


# --------------------------------------------------------------------------
# Payloads
# --------------------------------------------------------------------------


class SubmitResponse(BaseModel):
    document_id: str
    status: str = "ACCEPTED"
    poll: str


class ReviewDecision(BaseModel):
    """One reviewer choice: which candidate wins for one conflicted field."""

    field_name: str = Field(description="the conflict's normalized_field_name")
    chosen_field_uid: str = Field(description="field_uid of the candidate to accept")


class ReviewRequest(BaseModel):
    decisions: list[ReviewDecision] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _save_upload(upload: UploadFile, document_id: str) -> str:
    settings = get_graph_settings()
    ext = os.path.splitext(upload.filename or "")[1].lower()
    if ext and ext not in settings.extensions:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported file type {ext!r}; supported: {sorted(settings.extensions)}",
        )

    folder = os.path.join(UPLOAD_ROOT, document_id)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, os.path.basename(upload.filename or f"upload{ext or '.bin'}"))

    written = 0
    limit = settings.max_file_size_bytes
    with open(path, "wb") as fh:
        while chunk := upload.file.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                fh.close()
                shutil.rmtree(folder, ignore_errors=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"file exceeds the {settings.max_file_size_mb} MB limit",
                )
            fh.write(chunk)

    if written == 0:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    return path


def _process(path: str, document_id: str, form_id: str, schema: dict | None) -> None:
    try:
        run_document(path, document_id=document_id, form_id=form_id, target_form_schema=schema)
    except Exception:  # noqa: BLE001 — a background task must never die silently
        log.exception("graph: background processing failed for %s", document_id)


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


@router.post("/documents", response_model=SubmitResponse, status_code=202)
async def submit_document(
    background: BackgroundTasks,
    file: UploadFile = File(..., description="PDF, DOCX or image"),
    form_id: str = Form("", description="target form id, e.g. NABL_151"),
    target_form_schema: str = Form("", description="optional inline JSON form schema"),
) -> SubmitResponse:
    """Accept a document and start processing it in the background."""
    import json

    document_id = new_document_id()
    path = _save_upload(file, document_id)

    schema = None
    if target_form_schema.strip():
        try:
            schema = json.loads(target_form_schema)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400,
                                detail=f"target_form_schema is not valid JSON: {exc}") from exc

    background.add_task(_process, path, document_id, form_id, schema)
    return SubmitResponse(
        document_id=document_id,
        poll=f"/graph/documents/{document_id}/status",
    )


@router.get("/documents/{document_id}/status")
def document_status(document_id: str) -> dict[str, Any]:
    """Live progress, read from the checkpoint. Cheap enough to poll."""
    live = get_status(document_id)
    if live is not None:
        return {"source": "checkpoint", **live}

    stored = database.get_run(document_id)
    if stored is not None:
        return {"source": "database", **{k: v for k, v in stored.items() if k != "result"}}

    raise HTTPException(status_code=404, detail=f"no run found for {document_id!r}")


@router.get("/documents/{document_id}/result")
def document_result(document_id: str) -> dict[str, Any]:
    """The full spec-section-8 payload for a finished run."""
    stored = database.get_run(document_id)
    if stored is None:
        live = get_status(document_id)
        if live is not None:
            raise HTTPException(
                status_code=409,
                detail=f"run {document_id!r} is still in progress "
                       f"(node {live.get('current_node')!r}); poll /status",
            )
        raise HTTPException(status_code=404, detail=f"no run found for {document_id!r}")
    return stored["result"] or {}


@router.get("/documents")
def list_documents(
    limit: int = Query(50, ge=1, le=500),
    status: str | None = Query(None, description="filter by final status"),
) -> dict[str, Any]:
    runs = database.list_runs(limit=limit, status=status)
    return {"count": len(runs), "runs": runs}


@router.get("/documents/{document_id}/review")
def review_queue(document_id: str) -> dict[str, Any]:
    """Everything a human is being asked to decide, with the evidence."""
    stored = database.get_run(document_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"no run found for {document_id!r}")
    result = stored["result"] or {}
    return {
        "document_id": document_id,
        "overall_status": result.get("overall_status"),
        "manual_review_items": result.get("manual_review_items", []),
        "conflicts": result.get("conflicts", []),
        "missing_fields": result.get("missing_fields", []),
        "failed_chunks": result.get("failed_chunks", []),
    }


@router.post("/documents/{document_id}/review")
def submit_review(document_id: str, payload: ReviewRequest) -> dict[str, Any]:
    """Apply reviewer decisions and resume the run from its checkpoint."""
    if not payload.decisions:
        raise HTTPException(status_code=400, detail="no decisions supplied")

    decisions = {d.field_name: d.chosen_field_uid for d in payload.decisions}
    result = resume_document(document_id, human_decisions=decisions)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"no resumable checkpoint for {document_id!r} — it may have been pruned",
        )
    return result.model_dump(mode="json")


@router.post("/documents/{document_id}/retry")
def retry_document(document_id: str) -> dict[str, Any]:
    """Resume a crashed or interrupted run without reprocessing what succeeded."""
    result = resume_document(document_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"no resumable checkpoint for {document_id!r}")
    return result.model_dump(mode="json")


@router.delete("/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, str]:
    """Remove the uploaded file. The audit record is deliberately retained."""
    folder = os.path.join(UPLOAD_ROOT, document_id)
    if os.path.isdir(folder):
        shutil.rmtree(folder, ignore_errors=True)
        return {"status": "deleted", "detail": "source file removed; audit record retained"}
    raise HTTPException(status_code=404, detail=f"no upload found for {document_id!r}")


@router.get("/diagram")
def diagram() -> dict[str, str]:
    """The compiled workflow as mermaid — useful for docs and debugging."""
    return {"format": "mermaid", "diagram": mermaid()}


@router.get("/health")
def health(check_llm: bool = Query(False, description="also probe the LLM providers")) -> dict:
    settings = get_graph_settings()
    out: dict[str, Any] = {
        "status": "ok",
        "graph_enabled": settings.graph_pipeline_enabled,
        "provider_order": settings.provider_order,
        "max_chunk_chars": settings.max_chunk_chars,
        "human_review_enabled": settings.human_review_enabled,
    }
    try:
        database.init_db()
        out["database"] = "ok"
    except Exception as exc:  # noqa: BLE001
        out["status"] = "degraded"
        out["database"] = str(exc)[:200]

    if check_llm:
        from app.graph.llm import probe_providers

        try:
            out["providers"] = probe_providers()
            if all(v != "ok" for v in out["providers"].values()):
                out["status"] = "degraded"
        except Exception as exc:  # noqa: BLE001
            out["status"] = "degraded"
            out["providers"] = {"error": str(exc)[:200]}
    return out
