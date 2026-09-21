import logging
import threading

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.db import init_db
from app.llm.base import LlmNotConfigured, LlmProviderError
from app.observability import configure_logging
from app.routers import chat, documents, graph_documents, review, wizard

settings = get_settings()

log = logging.getLogger(__name__)

app = FastAPI(title="NABL Document Intelligence")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(LlmNotConfigured)
async def handle_llm_not_configured(request: Request, exc: LlmNotConfigured) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(LlmProviderError)
async def handle_llm_provider_error(request: Request, exc: LlmProviderError) -> JSONResponse:
    # Every configured provider failed (rate-limited, outage, etc.) — 503, not
    # a 500, since this is an upstream/config issue, not a bug in this app.
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.on_event("startup")
def on_startup() -> None:
    # First: the app had no logging configuration, so every log.info() in this
    # codebase - the existing diagnostics as well as the new stage timings -
    # was being discarded. Installed before anything else so startup itself is
    # visible.
    configure_logging()

    init_db()

    # The graph keeps its own audit store (runs, fields, mappings,
    # conflicts, errors, audit), separate from the app's operational
    # tables: different lifecycle, different retention.
    from app.graph.database import init_db as init_graph_db

    init_graph_db()

    _warm_docling()

    # Documents whose extraction died with a previous process. Reconciled
    # here, before anything new is accepted — see _reap_orphaned_documents.
    _reap_orphaned_documents()



def _warm_docling() -> None:
    """Build the Docling converter ahead of the first upload.

    On a background thread, not inline: the build takes ~30s (MEASURED
    2026-09-21), and blocking startup for that long would make the server
    unreachable — including /health, which deployment tooling waits on.

    A request arriving mid-warm-up does not race: get_converter() holds a lock,
    so the upload waits on the build already in progress instead of starting a
    second one. Worst case it waits as long as it does today; best case, and in
    the normal case, the converter is ready before the first upload arrives.
    """
    from app.graph.docling_adapter import warm_docling_converter

    threading.Thread(
        target=warm_docling_converter,
        name="docling-warmup",
        daemon=True,  # never hold up interpreter shutdown
    ).start()


def _reap_orphaned_documents() -> None:
    """Fail any document left mid-extraction by a process that is now gone.

    Extraction runs as an in-process background task (see documents.py's
    upload endpoint), so it dies with the process that started it: a restart,
    a crash, a Ctrl-C between `status = "processing"` and the pipeline's final
    commit leaves a row claiming to be in progress with nothing in the world
    working on it. Nothing ever reclaimed those rows, so they accumulated —
    eleven of them on this installation, the oldest three weeks old.

    The cost was not the stale rows themselves but what the UI told people
    about them. The upload screen polls such a document for thirty minutes and
    then says it has NOT failed and that reloading will pick up the result.
    Both halves are false, and a reviewer waiting on a document that will
    never arrive has no way to discover that.

    Marking them failed is honest and, unlike deleting them, keeps the stored
    file and the row so Re-extract can recover the document in one click.

    SAFE ONLY AT STARTUP, and only while this is a single API process: nothing
    else can be mid-extraction at the moment this runs, because the tasks live
    inside this process and it has not begun serving yet. Running it at any
    other time — or alongside a second worker — would fail a document that is
    genuinely being worked on. A multi-process deployment needs a real job
    queue with leases instead; this is the honest fix for the design that is
    actually here.
    """
    from app.db import SessionLocal
    from app.models import Document

    db = SessionLocal()
    try:
        orphans = db.query(Document).filter(Document.status == "processing").all()
        if not orphans:
            return
        for document in orphans:
            document.status = "failed"
            document.error = (
                "Extraction stopped when the server restarted, so this document was never "
                "finished. Nothing was lost — the uploaded file is still here. Re-extract to run it again."
            )
            db.add(document)
        db.commit()
        log.warning(
            "reaped %d document(s) left mid-extraction by a previous process", len(orphans)
        )
    finally:
        db.close()


@app.get("/health")
def health() -> dict:
    from app.graph.docling_adapter import converter_is_warm

    # docling_warm lets an operator (and the startup test) see whether the
    # first upload will pay the converter build. "status" stays exactly as it
    # was: existing callers assert on that key.
    return {"status": "ok", "docling_warm": converter_is_warm()}


app.include_router(wizard.router)
app.include_router(documents.router)
app.include_router(review.router)
app.include_router(chat.router)
app.include_router(graph_documents.router)
