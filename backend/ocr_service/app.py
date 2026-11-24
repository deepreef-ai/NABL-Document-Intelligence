"""OCR as a separate HTTP service, so OCR can run on a stronger host.

Run (from backend/):
    .venv/bin/uvicorn ocr_service.app:app --host 0.0.0.0 --port 8100

WHY (2026-11-24): deepreef-ocr loads its ONNX models once per process and keeps
one engine per worker thread. Loading them is the slow part, so a long-lived
server that keeps the models warm and sits on a machine with more cores beats
loading them inside every pipeline worker on a laptop. The pipeline talks to it
through app/documents/ocr_service_client.py.

Endpoints:
    GET  /health    -> {"status": "ok", "model": deepreef_local.model_report()}
    POST /ocr/page  -> {"text", "lines", "confidence", "model_used", "region_count"}
        body: {"image_base64": str, "rendered_pdf_page": bool}
"""
from __future__ import annotations

import base64
import binascii
import logging

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.documents import deepreef_local

logger = logging.getLogger(__name__)

app = FastAPI(title="NABL OCR service")


class OcrPageRequest(BaseModel):
    image_base64: str
    rendered_pdf_page: bool = False


class OcrPageResponse(BaseModel):
    text: str
    lines: list[str]
    confidence: float | None
    model_used: str
    region_count: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": deepreef_local.model_report()}


@app.post("/ocr/page", response_model=OcrPageResponse)
async def ocr_page(req: OcrPageRequest) -> OcrPageResponse:
    try:
        image_bytes = base64.b64decode(req.image_base64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="image_base64 is not valid base64")
    try:
        # WHY: extract_page is CPU-bound and blocking; running it in the thread
        # pool lets several pages OCR side by side (deepreef_local keeps one
        # engine per thread) instead of serialising on the event loop.
        # Looked up via the module at call time so tests can monkeypatch it.
        page = await run_in_threadpool(
            deepreef_local.extract_page,
            image_bytes,
            rendered_pdf_page=req.rendered_pdf_page,
        )
    except Exception as exc:  # noqa: BLE001
        # WHY: log only the exception type — never page text or image data,
        # which may hold confidential lab results.
        logger.error("OCR failed: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail=f"OCR failed: {type(exc).__name__}")
    return OcrPageResponse(
        text=page.text,
        lines=list(page.lines),
        confidence=page.confidence,
        model_used=page.model_used,
        region_count=page.region_count,
    )
