"""Error taxonomy for the graph (spec section 9).

Every failure the workflow can hit has a named type here, so an error is
always classifiable rather than a free-text string. `ProcessingErrorRecord`
in schemas.py is the thing that gets stored; this module names the kinds and
says which of them are *critical* — meaning the graph must not report success
while one is unresolved.

A deliberate choice: almost nothing in this package raises. Nodes return an
error record and let the router decide, because a raised exception in a
LangGraph node aborts the run and loses everything the earlier nodes did.
The exceptions below exist only for the two cases where continuing is
genuinely meaningless (an unreadable file, an exhausted retry budget with no
partial result to keep).
"""
from __future__ import annotations

from enum import Enum

from app.graph.schemas import ProcessingErrorRecord


class ErrorType(str, Enum):
    # --- input / file ---------------------------------------------------
    INVALID_FILE = "INVALID_FILE"
    CORRUPTED_FILE = "CORRUPTED_FILE"
    EMPTY_DOCUMENT = "EMPTY_DOCUMENT"
    PASSWORD_PROTECTED = "PASSWORD_PROTECTED"
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"

    # --- pages / OCR ----------------------------------------------------
    MISSING_PAGES = "MISSING_PAGES"
    OCR_FAILURE = "OCR_FAILURE"
    POOR_QUALITY_TEXT = "POOR_QUALITY_TEXT"
    PAGE_EXTRACTION_FAILED = "PAGE_EXTRACTION_FAILED"

    # --- chunking -------------------------------------------------------
    EMPTY_CHUNK = "EMPTY_CHUNK"
    OVERSIZED_CHUNK = "OVERSIZED_CHUNK"
    DUPLICATE_CHUNK = "DUPLICATE_CHUNK"
    MISSING_CHUNK = "MISSING_CHUNK"
    PAGE_NOT_CHUNKED = "PAGE_NOT_CHUNKED"

    # --- LLM ------------------------------------------------------------
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_RATE_LIMIT = "LLM_RATE_LIMIT"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    LLM_INVALID_JSON = "LLM_INVALID_JSON"
    LLM_SCHEMA_MISMATCH = "LLM_SCHEMA_MISMATCH"
    LLM_INCOMPLETE_RESPONSE = "LLM_INCOMPLETE_RESPONSE"
    LLM_TRUNCATED = "LLM_TRUNCATED"
    LLM_UNRELATED_RESPONSE = "LLM_UNRELATED_RESPONSE"
    LLM_DUPLICATE_RESPONSE = "LLM_DUPLICATE_RESPONSE"
    LLM_EMPTY_RESPONSE = "LLM_EMPTY_RESPONSE"

    # --- content --------------------------------------------------------
    HALLUCINATED_VALUE = "HALLUCINATED_VALUE"
    MISSING_EVIDENCE = "MISSING_EVIDENCE"
    INCORRECT_PAGE_REFERENCE = "INCORRECT_PAGE_REFERENCE"
    DUPLICATE_FIELD = "DUPLICATE_FIELD"
    CONFLICTING_FIELD = "CONFLICTING_FIELD"

    # --- form -----------------------------------------------------------
    MISSING_TARGET_FORM_FIELD = "MISSING_TARGET_FORM_FIELD"
    AMBIGUOUS_MAPPING = "AMBIGUOUS_MAPPING"
    UNSUPPORTED_MAPPING_VALUE = "UNSUPPORTED_MAPPING_VALUE"

    # --- workflow -------------------------------------------------------
    INCOMPLETE_PROCESSING = "INCOMPLETE_PROCESSING"
    RETRY_LIMIT_EXCEEDED = "RETRY_LIMIT_EXCEEDED"
    DATABASE_FAILURE = "DATABASE_FAILURE"
    INFORMATION_LOSS = "INFORMATION_LOSS"
    NODE_EXCEPTION = "NODE_EXCEPTION"


#: Errors that must block a VALIDATED verdict no matter what else succeeded.
CRITICAL_ERRORS: frozenset[ErrorType] = frozenset({
    ErrorType.INVALID_FILE,
    ErrorType.CORRUPTED_FILE,
    ErrorType.EMPTY_DOCUMENT,
    ErrorType.PASSWORD_PROTECTED,
    ErrorType.UNSUPPORTED_FILE_TYPE,
    ErrorType.FILE_TOO_LARGE,
    ErrorType.MISSING_PAGES,
    ErrorType.PAGE_NOT_CHUNKED,
    ErrorType.MISSING_CHUNK,
    ErrorType.INFORMATION_LOSS,
    ErrorType.INCOMPLETE_PROCESSING,
})

#: Errors that mean "stop the workflow now"; nothing downstream can help.
FATAL_ERRORS: frozenset[ErrorType] = frozenset({
    ErrorType.INVALID_FILE,
    ErrorType.CORRUPTED_FILE,
    ErrorType.EMPTY_DOCUMENT,
    ErrorType.PASSWORD_PROTECTED,
    ErrorType.UNSUPPORTED_FILE_TYPE,
    ErrorType.FILE_TOO_LARGE,
})

#: Transient LLM conditions worth an automatic retry with backoff.
RETRYABLE_ERRORS: frozenset[ErrorType] = frozenset({
    ErrorType.LLM_TIMEOUT,
    ErrorType.LLM_RATE_LIMIT,
    ErrorType.LLM_UNAVAILABLE,
    ErrorType.LLM_INVALID_JSON,
    ErrorType.LLM_SCHEMA_MISMATCH,
    ErrorType.LLM_INCOMPLETE_RESPONSE,
    ErrorType.LLM_TRUNCATED,
    ErrorType.LLM_EMPTY_RESPONSE,
    ErrorType.LLM_UNRELATED_RESPONSE,
})


def make_error(
    error_type: ErrorType,
    message: str,
    node: str,
    *,
    page_number: int | None = None,
    chunk_id: str | None = None,
    retry_count: int = 0,
    recovery_action: str = "",
    resolution_status: str = "OPEN",
) -> ProcessingErrorRecord:
    return ProcessingErrorRecord(
        error_type=error_type.value,
        error_message=message[:2000],
        node=node,
        page_number=page_number,
        chunk_id=chunk_id,
        retry_count=retry_count,
        recovery_action=recovery_action,
        resolution_status=resolution_status,
    )


def is_critical(record: ProcessingErrorRecord) -> bool:
    try:
        return ErrorType(record.error_type) in CRITICAL_ERRORS
    except ValueError:
        return False


def is_fatal(record: ProcessingErrorRecord) -> bool:
    try:
        return ErrorType(record.error_type) in FATAL_ERRORS
    except ValueError:
        return False


def is_retryable(record: ProcessingErrorRecord) -> bool:
    try:
        return ErrorType(record.error_type) in RETRYABLE_ERRORS
    except ValueError:
        return False


class FatalDocumentError(Exception):
    """The document cannot be processed at all. Raised only by validation."""

    def __init__(self, error_type: ErrorType, message: str):
        super().__init__(message)
        self.error_type = error_type
        self.message = message


class LlmUnavailable(Exception):
    """Every configured provider refused. Carries the per-provider detail."""

    def __init__(self, message: str, provider_errors: dict[str, str] | None = None):
        super().__init__(message)
        self.provider_errors = provider_errors or {}
