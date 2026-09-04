"""Targeted, batched recovery for fields validation flagged.

Two rules that the old retry pass broke:

1. Never resend the whole document. The old whole-form path re-asked per
   SECTION, so a 20-page form with gaps in three sections cost three calls
   carrying that section's full retrieved text each time.
2. Batch across chunks. Gaps in sections 2, 4 and 7 are ONE call with six
   fields, not three calls with two each — the old code keyed its batching on
   the exact source-text string, so two sections with gaps could never share
   a call even when both would have fitted comfortably.

Vision escalation is deliberately narrow (spec §11). A low OCR confidence
alone is NOT sufficient: most low-confidence pages are low-confidence for
reasons that a second look won't fix (a faint scan of a page whose fields we
already read correctly). A page image is sent only when it is BOTH relevant
and there is a concrete reason to believe the text is what failed us.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.documents import call_budget as cb

log = logging.getLogger(__name__)

_SYSTEM = (
    "You are re-attempting to extract specific fields a first pass could not "
    "establish, or established with a value the source does not support. "
    "Read the evidence again carefully. Return a field ONLY if the evidence "
    "actually supports a value — omit any field you still cannot find. Do "
    "not guess, and do not repeat a value you cannot see in the evidence."
)

_JSON_INSTRUCTION = (
    'Respond with ONLY a JSON object: {"fields": {"<field>": "<value>", ...}}. '
    "Include only fields you found. Use an empty object if you found none."
)


@dataclass
class RecoveryRequest:
    """What needs recovering, and the evidence for it."""

    fields: list[str] = field(default_factory=list)
    evidence: str = ""
    page_numbers: list[int] = field(default_factory=list)
    images: list[bytes] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.fields


def needs_vision(
    pages: list,
    relevant_page_numbers: list[int],
    has_suspicious: bool,
    ocr_conflict: bool = False,
) -> list[int]:
    """Page numbers whose IMAGE is worth sending, per the escalation policy.

    Returns [] in the common case. A page qualifies when it is relevant AND
    at least one of: its OCR confidence is low, its OCR text conflicts with
    another extraction, or validation flagged something suspicious that came
    from an OCR'd page (i.e. the transcription is a plausible culprit).
    """
    relevant = set(relevant_page_numbers)
    out: list[int] = []
    for page in pages:
        if page.page_number not in relevant:
            continue
        if page.page.extraction_method != "ocr":
            # A born-digital page's text is exact; re-reading it as an image
            # cannot resolve anything the text didn't already say.
            continue
        if page.needs_visual_check or ocr_conflict or has_suspicious:
            out.append(page.page_number)
    return out


def build_request(
    missing: list[str],
    suspicious: list[str],
    pages: list,
    relevant_page_numbers: list[int],
    max_chars: int,
    include_images: bool = False,
) -> RecoveryRequest:
    """ONE request covering every gap across every chunk."""
    wanted = list(dict.fromkeys([*missing, *suspicious]))
    if not wanted:
        return RecoveryRequest()

    relevant = set(relevant_page_numbers)
    parts: list[str] = []
    used_pages: list[int] = []
    images: list[bytes] = []
    budget = max_chars
    for page in pages:
        if page.page_number not in relevant or not (page.text or "").strip():
            continue
        piece = f"--- Page {page.page_number} ---\n{page.text}"
        if len(piece) > budget:
            piece = piece[:budget]
        if not piece:
            break
        parts.append(piece)
        used_pages.append(page.page_number)
        budget -= len(piece)
        if include_images and page.image is not None:
            images.append(page.image)
        if budget <= 0:
            break

    return RecoveryRequest(
        fields=wanted, evidence="\n\n".join(parts), page_numbers=used_pages, images=images,
    )


def recover(chain, request: RecoveryRequest, budget: cb.CallBudget) -> dict:
    """Spends at most ONE call. Returns {} when the budget refuses it."""
    if request.is_empty:
        return {}
    category = cb.VISION if request.images else cb.RECOVERY
    if not budget.spend(category):
        log.info("recovery skipped: %s", budget.stop_reason)
        return {}

    prompt = (
        f"Fields to recover: {', '.join(request.fields)}\n\n"
        f"Evidence:\n\n{request.evidence}\n\n{_JSON_INSTRUCTION}"
    )
    try:
        result = chain.generate_json(
            _SYSTEM,
            prompt,
            image_media_type="image/png" if request.images else None,
            images=request.images or None,
        )
    except Exception as exc:  # noqa: BLE001 — recovery is a bonus pass; never sink the document
        log.warning("recovery call failed: %s", exc)
        return {}
    if not isinstance(result, dict):
        return {}
    fields = result.get("fields")
    if not isinstance(fields, dict):
        return {}
    return {
        k: v for k, v in fields.items()
        if k in set(request.fields) and isinstance(v, (str, int, float)) and str(v).strip()
    }
