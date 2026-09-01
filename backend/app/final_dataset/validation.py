"""Validates each assembled FinalDocumentRecord before it's allowed into
the final dataset — a record that fails is EXCLUDED from the output (never
written) and recorded as a validation error in dataset_summary.json, rather
than writing something known to be broken.
"""
from __future__ import annotations

import re
from pathlib import Path

from app.final_dataset.models import FinalDocumentRecord
from app.schema_discovery.domains import CANONICAL_DOMAINS

_ID_PATTERN = re.compile(r"^LR_\d{6}$")


def validate_record(record: FinalDocumentRecord) -> list[str]:
    errors = []
    if not record.document_id or not _ID_PATTERN.match(record.document_id):
        errors.append(f"invalid document_id: {record.document_id!r}")
    if record.annotation_status != "approved":
        errors.append(f"annotation_status is not 'approved': {record.annotation_status!r}")
    if record.domain not in CANONICAL_DOMAINS:
        errors.append(f"invalid domain: {record.domain!r}")
    if not isinstance(record.fields, dict):
        errors.append("'fields' is not an object")
    if not isinstance(record.tests, list):
        errors.append("'tests' is not a list")
    else:
        for i, row in enumerate(record.tests):
            if not isinstance(row, dict) or not row.get("test_name"):
                errors.append(f"tests[{i}] is not a well-formed test row")
    if not isinstance(record.page_count, int) or record.page_count < 1:
        errors.append(f"invalid page_count: {record.page_count!r}")
    if record.source_path and not Path(record.source_path).is_file():
        errors.append(f"source_path not found: {record.source_path!r}")
    return errors
