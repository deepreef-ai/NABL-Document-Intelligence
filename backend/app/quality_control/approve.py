"""The enforcement point for "do not allow an invalid annotation to become
approved": approving a label is only ever done through here, and it always
re-runs QC fresh against the label's CURRENT content before allowing the
transition — it never trusts a stale `qc.passed` flag left over from an
earlier run, since the label (or its source data) could have changed since.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.quality_control.checks import ALL_CHECKS, HARD_ISSUE_CODES, check_valid_document_id
from app.quality_control.models import QCContext
from app.quality_control.pipeline import DEFAULT_OCR_CONFIDENCE_THRESHOLD, REQUIRED_FIELD_COUNT, load_normalized_documents
from app.labeling.schema_hints import build_domain_hints, load_master_schema
from app.schema_discovery.domains import CANONICAL_DOMAINS


class ApprovalRefused(RuntimeError):
    """Raised instead of approving — the label fails QC as it stands now."""


def approve_document(
    document_id: str,
    labeled_dir: Path,
    normalized_dir: Path,
    master_schema_dir: Path,
    ocr_confidence_threshold: float = DEFAULT_OCR_CONFIDENCE_THRESHOLD,
) -> dict:
    label_path = labeled_dir / "labels" / f"{document_id}.json"
    label = json.loads(label_path.read_text(encoding="utf-8"))

    doc_jsons = load_normalized_documents(normalized_dir)
    hints = build_domain_hints(load_master_schema(master_schema_dir))
    domain = label.get("domain") if label.get("domain") in CANONICAL_DOMAINS else "other"

    ctx = QCContext(
        doc_json=doc_jsons.get(document_id),
        required_fields=hints.get(domain, {}).get("fields", [])[:REQUIRED_FIELD_COUNT],
        ocr_confidence_threshold=ocr_confidence_threshold,
    )
    issues = check_valid_document_id(label, document_id, ctx)
    for check in ALL_CHECKS:
        issues.extend(check(label, ctx))

    hard_issues = [i for i in issues if i.code in HARD_ISSUE_CODES]
    if hard_issues:
        raise ApprovalRefused(
            f"{document_id} fails QC and cannot be approved: "
            + "; ".join(f"{i.code}: {i.message}" for i in hard_issues)
        )
    # Soft issues (missing_required_fields, invalid_values, etc.) don't block
    # approval — a human approving despite a documented warning is exactly
    # the "human verification" this stage exists to allow — but they're
    # still recorded so the approval isn't silently hiding them.

    label["annotation_status"] = "approved"
    label["qc"] = {"passed": True, "issues": [i.to_json_dict() for i in issues]}
    label_path.write_text(json.dumps(label, indent=2, ensure_ascii=False), encoding="utf-8")
    return label


def approve_all_passing(
    labeled_dir: Path,
    normalized_dir: Path,
    master_schema_dir: Path,
    ocr_confidence_threshold: float = DEFAULT_OCR_CONFIDENCE_THRESHOLD,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Batch convenience over approve_document: tries every label not
    already "approved", one at a time, through the exact same fresh-recheck
    gate — this is NOT a bulk bypass of it. Returns (approved_ids,
    [(document_id, refusal_reason), ...]) so a caller can see exactly what
    happened to each one rather than a single pass/fail count."""
    approved_ids: list[str] = []
    refused: list[tuple[str, str]] = []
    for label_path in sorted((labeled_dir / "labels").glob("*.json")):
        document_id = label_path.stem
        current = json.loads(label_path.read_text(encoding="utf-8"))
        if current.get("annotation_status") == "approved":
            continue
        try:
            approve_document(document_id, labeled_dir, normalized_dir, master_schema_dir, ocr_confidence_threshold)
            approved_ids.append(document_id)
        except ApprovalRefused as exc:
            refused.append((document_id, str(exc)))
    return approved_ids, refused
