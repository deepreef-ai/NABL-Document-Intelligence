"""Stage-level timing instrumentation.

Why this exists
---------------
Before this module the pipeline reported only its own end-to-end duration, so
"an upload takes minutes" could not be attributed to Docling, OCR, the LLM, or
persistence. Tuning anything — the timeout, the concurrency — was guesswork.

What it emits
-------------
One INFO line per stage, in the project's existing ``key=value`` log style::

    stage=docling_parse job=8f2c1a4e duration=4.62s pages=3
    stage=llm_extraction job=8f2c1a4e duration=18.40s chunks=4 workers=4

Never emitted: document text, field values, file paths, credentials. Only a
stage name, a correlation id, a duration and small numeric facts, so these
lines are safe to ship to a log aggregator.

Correlation id
--------------
``job`` ties every stage of one document together. It is the document id where
there is one, so a log line joins straight back to a row in the database.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

log = logging.getLogger(__name__)

# A ContextVar rather than a parameter threaded through twenty call sites:
# the graph nodes are called by LangGraph, not by us, so there is nowhere to
# pass an id. ContextVars also survive into ThreadPoolExecutor workers when
# the context is copied, and are per-request rather than global.
_job_id: ContextVar[str] = ContextVar("job_id", default="-")


def new_job_id() -> str:
    """A short correlation id for one processing job."""
    return uuid.uuid4().hex[:8]


def set_job_id(job_id: str) -> str:
    """Bind a correlation id to the current context. Returns it."""
    _job_id.set(job_id)
    return job_id


def get_job_id() -> str:
    return _job_id.get()


@contextmanager
def job_context(job_id: str | None = None) -> Iterator[str]:
    """Bind a correlation id for the duration of a block."""
    token = _job_id.set(job_id or new_job_id())
    try:
        yield _job_id.get()
    finally:
        _job_id.reset(token)


def _timings_enabled() -> bool:
    """Read the setting lazily so tests can toggle it."""
    try:
        from app.config import get_settings

        return bool(get_settings().log_stage_timings)
    except Exception:  # noqa: BLE001 — instrumentation must never break a request
        return True


def _format_fields(fields: dict[str, Any]) -> str:
    parts = []
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:.2f}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


@contextmanager
def stage(name: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Time a pipeline stage and log its duration.

    Extra facts can be attached inside the block, so a stage can report what
    it only learns while running::

        with stage("llm_extraction") as st:
            results = extract(...)
            st["chunks"] = len(results)

    A failing stage still logs, with ``outcome=error``: a stage that crashed
    after 40s is exactly the one worth seeing. The exception is re-raised
    unchanged — this wrapper never swallows a failure.
    """
    extra: dict[str, Any] = dict(fields)
    started = time.perf_counter()
    outcome = "ok"
    try:
        yield extra
    except BaseException as exc:  # noqa: BLE001 — logged, then re-raised
        outcome = "error"
        extra.setdefault("error", type(exc).__name__)
        raise
    finally:
        duration = time.perf_counter() - started
        if _timings_enabled():
            suffix = _format_fields(extra)
            log.info(
                "stage=%s job=%s duration=%.2fs outcome=%s%s",
                name,
                get_job_id(),
                duration,
                outcome,
                f" {suffix}" if suffix else "",
            )


def configure_logging(level: str | None = None) -> None:
    """Install a root handler so application logs are actually emitted.

    Without this, every ``log.info`` in this project is discarded: uvicorn
    configures only its own loggers and the root logger defaults to WARNING.
    Safe to call more than once — a second call does not duplicate handlers,
    which would otherwise double every line under ``--reload``.
    """
    from app.config import get_settings

    resolved = (level or get_settings().log_level).upper()
    numeric = getattr(logging, resolved, logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric)

    if not any(getattr(h, "_nabl_handler", False) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        handler._nabl_handler = True  # type: ignore[attr-defined]
        root.addHandler(handler)

    # These are chatty at INFO and drown the stage lines. Docling in
    # particular prints a progress bar per page.
    for noisy in ("httpx", "httpcore", "botocore", "boto3", "urllib3", "docling", "PIL"):
        logging.getLogger(noisy).setLevel(max(numeric, logging.WARNING))
