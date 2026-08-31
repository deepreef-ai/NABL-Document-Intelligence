from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.db import init_db
from app.llm.base import LlmNotConfigured, LlmProviderError
from app.routers import chat, documents, review, wizard

settings = get_settings()

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
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


app.include_router(wizard.router)
app.include_router(documents.router)
app.include_router(review.router)
app.include_router(chat.router)
