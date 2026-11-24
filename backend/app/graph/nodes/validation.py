"""B. Document Validation Node — deterministic, no LLM.

The first gate. Everything downstream assumes it is looking at a readable
document of a supported type, and this is the only node that earns that
assumption. It is deliberately paranoid: an encrypted PDF that reports a page
count but yields empty text on every page looks, three nodes later, exactly
like a scanned document whose OCR failed — and the two need completely
different responses.

A failure here is fatal. There is no partial result worth keeping from a file
we cannot open, so this is the one place the graph short-circuits straight to
the final decision with REJECTED.
"""
from __future__ import annotations

import logging
import os

from app.graph.config import get_graph_settings
from app.graph.errors import ErrorType
from app.graph.nodes.helpers import audit, err
from app.graph.schemas import FinalStatus
from app.graph.state import GraphState, WorkflowStatus

log = logging.getLogger(__name__)

NODE = "document_validation"


def _fail(reason: str, error_type: ErrorType, state: GraphState, **kw) -> dict:
    return {
        "current_node": NODE,
        "current_status": WorkflowStatus.FAILED,
        "final_status": FinalStatus.REJECTED,
        "final_reasoning": reason,
        "error_details": reason,
        "errors": err(error_type, reason, NODE, resolution_status="ABANDONED", **kw),
        "audit_log": audit(NODE, "validation_failed", reason, level="error"),
    }


def validate_document(state: GraphState) -> dict:
    settings = get_graph_settings()
    path = state.file_path

    # --- existence and readability ---------------------------------------
    if not path:
        return _fail("No file path was supplied.", ErrorType.INVALID_FILE, state)
    if not os.path.exists(path):
        return _fail(f"File does not exist: {path}", ErrorType.INVALID_FILE, state)
    if not os.path.isfile(path):
        return _fail(f"Path is not a file: {path}", ErrorType.INVALID_FILE, state)
    if not os.access(path, os.R_OK):
        return _fail(f"File is not readable: {path}", ErrorType.INVALID_FILE, state)

    # --- size -------------------------------------------------------------
    size = os.path.getsize(path)
    if size == 0:
        return _fail("File is empty (0 bytes).", ErrorType.EMPTY_DOCUMENT, state)
    if size > settings.max_file_size_bytes:
        return _fail(
            f"File is {size / 1024 / 1024:.1f} MB, over the {settings.max_file_size_mb} MB limit.",
            ErrorType.FILE_TOO_LARGE, state,
        )

    # --- extension --------------------------------------------------------
    ext = os.path.splitext(path)[1].lower()
    if ext not in settings.extensions:
        return _fail(
            f"Unsupported file type {ext!r}. Supported: {sorted(settings.extensions)}",
            ErrorType.UNSUPPORTED_FILE_TYPE, state,
        )

    # --- readable content -------------------------------------------------
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
            data_len = size
    except OSError as exc:
        return _fail(f"Could not read file: {exc}", ErrorType.INVALID_FILE, state)

    file_type = ext.lstrip(".")
    total_pages = 0
    warnings: list[str] = []

    if ext == ".pdf":
        if not head.startswith(b"%PDF"):
            return _fail(
                "File has a .pdf extension but no PDF header — it is corrupted or misnamed.",
                ErrorType.CORRUPTED_FILE, state,
            )
        try:
            import fitz

            with fitz.open(path) as doc:
                if doc.needs_pass or doc.is_encrypted:
                    # is_encrypted stays true even after a successful empty-password
                    # unlock, so needs_pass is the one that actually blocks us.
                    if doc.needs_pass:
                        return _fail(
                            "PDF is password-protected and cannot be opened.",
                            ErrorType.PASSWORD_PROTECTED, state,
                        )
                    warnings.append("PDF is encrypted but opened with an empty password.")
                total_pages = doc.page_count
                if total_pages == 0:
                    return _fail("PDF contains no pages.", ErrorType.EMPTY_DOCUMENT, state)
                if total_pages > settings.max_pages:
                    return _fail(
                        f"PDF has {total_pages} pages, over the {settings.max_pages} limit.",
                        ErrorType.FILE_TOO_LARGE, state,
                    )
        except Exception as exc:  # noqa: BLE001 — any open failure is corruption to us
            return _fail(f"PDF could not be opened: {exc}", ErrorType.CORRUPTED_FILE, state)

    elif ext == ".docx":
        # A .docx is a zip. An unopenable zip is a corrupt document, and
        # catching it here is much clearer than a KeyError inside python-docx.
        import zipfile

        if not zipfile.is_zipfile(path):
            return _fail(
                "File has a .docx extension but is not a valid DOCX container.",
                ErrorType.CORRUPTED_FILE, state,
            )
        total_pages = 1  # no real page concept; preprocessing paginates it

    else:  # image
        try:
            from PIL import Image

            with Image.open(path) as img:
                img.verify()
            total_pages = 1
        except Exception as exc:  # noqa: BLE001
            return _fail(f"Image could not be decoded: {exc}", ErrorType.CORRUPTED_FILE, state)

    entries = audit(
        NODE, "validated",
        f"{os.path.basename(path)} ({file_type}, {data_len / 1024:.0f} KB, {total_pages} page(s))",
    )
    for w in warnings:
        entries += audit(NODE, "validation_warning", w, level="warning")

    return {
        "current_node": NODE,
        "current_status": WorkflowStatus.RUNNING,
        "file_type": file_type,
        "file_name": os.path.basename(path),
        "file_size_bytes": data_len,
        "total_pages": total_pages,
        "audit_log": entries,
    }
