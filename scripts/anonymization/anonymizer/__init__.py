"""Standalone PII anonymiser for laboratory reports.

Deliberately self-contained: this package does not import from the NABL
Document Intelligence application, and nothing in the application imports from
here. It is a separate utility that happens to live in the same repository, so
it can be run against a Drive folder on a machine where the rest of the system
is not installed.
"""
from .config import AnonymizerConfig  # noqa: F401
from .report import (  # noqa: F401
    STATUS_ERROR,
    STATUS_FAILED_REVIEW,
    STATUS_OK,
    STATUS_SKIPPED,
)

__version__ = "1.0.0"
__all__ = [
    "AnonymizerConfig",
    "STATUS_OK",
    "STATUS_FAILED_REVIEW",
    "STATUS_ERROR",
    "STATUS_SKIPPED",
    "__version__",
]
