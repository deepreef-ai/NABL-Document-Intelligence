"""THE shared document-extraction path, used by BOTH the live upload
pipeline (documents/pipeline.py) and the offline benchmark
(scripts/generate_predictions_and_score.py).

It exists because those two used to be separate implementations with
different prompts, different inputs and different output shapes, so the
benchmark's metrics did not describe what the product actually did — the
benchmark sent a scanned page as an image and never OCR'd it, the UI OCR'd
it and never sent the image, and only the benchmark ever produced structured
`tests` rows at all (the UI returned flat fields, so a results table lost
its structure completely).

The input contract, identical for both callers:

    every page contributes TEXT   — PyMuPDF's text layer where the page has
                                    one, OCR where it does not
    every page contributes AN IMAGE

so the model always gets both the transcription and the page itself and can
cross-check one against the other. Nothing is dropped for lack of a text
layer, and nothing is reduced to a possibly-degraded OCR transcription with
the original page thrown away.

Deliberately NOT shared: the provider chain. The benchmark passes a
Nova-only chain because its whole job is to measure one model; the live app
passes the full fallback chain because its job is to answer the request.
Prompt, page handling and output shape are identical either way, which is
what makes the numbers comparable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.config import get_settings
from app.documents import local_ocr, pdf_utils
from app.documents.ocr_client import SUPPORTED_SCRIPTS, OcrClient

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# A page whose text layer yields fewer than this many characters is treated
# as having no usable text and is OCR'd instead. Same threshold
# documents/pipeline.py and chunking.py use, so a page's classification
# never depends on which entry point read it.
MIN_PAGE_TEXT_CHARS = 20

SYSTEM_PROMPT = (
    "You are an information-extraction system for laboratory/testing reports "
    "(milk, food, chemical, medical, and similar documents). Extract EVERY "
    "key-value pair actually present in the document — never invent, guess, "
    "or infer a value that isn't really there; if unsure, leave it out.\n\n"
    "Each page is given to you as text AND as an image. The text is a "
    "machine transcription and may be imperfect or mis-ordered; the image is "
    "the page itself. Where they disagree, trust the image.\n\n"
    "Respond with ONLY a JSON object of this exact shape:\n"
    '{"fields": {"<snake_case_key>": "<value exactly as written>", ...}, '
    '"tests": [{"test_name": "<name>", "result": "<value as written>", '
    '"unit": "<unit or null>", "reference_range": "<range or null>"}, ...]}\n\n'
    '"fields" holds every header/metadata value (names, dates, addresses, '
    "report numbers, sample details, letterhead/lab identity, accreditation "
    "numbers, page markers, sampling details, footnotes) using natural "
    'snake_case keys derived from the document\'s own printed labels. '
    '"tests" holds every row of a results table. Use an empty list for '
    '"tests" if the document has no tabular results.'
)


@dataclass
class DocumentPayload:
    """What gets sent to the model for one document."""

    text_blocks: list[str] = field(default_factory=list)
    images: list[bytes] = field(default_factory=list)
    media_type: str = "image/png"
    # Non-fatal per-page problems (e.g. OCR unavailable for one page) — the
    # caller decides whether to surface them; extraction still runs.
    warnings: list[str] = field(default_factory=list)

    @property
    def user_text(self) -> str:
        if not self.text_blocks:
            return (
                f"(No page in this document has an extractable text layer — "
                f"read the {len(self.images)} attached page image(s).)"
            )
        return "\n\n".join(self.text_blocks)


def _ocr_page(png: bytes, script: str, ocr_client: OcrClient | None) -> str:
    """Text for a page with no usable text layer. Same routing the live
    pipeline already uses: deepreef-ocr's Lambda for the scripts it has
    models for, local RapidOCR (in-process, no AWS call) for English."""
    if script in SUPPORTED_SCRIPTS:
        client = ocr_client or OcrClient()
        return client.extract(png, script).text
    return local_ocr.extract_english(png).text


def build_payload(
    data: bytes,
    suffix: str,
    script: str = "english",
    ocr_client: OcrClient | None = None,
    dpi: int | None = None,
) -> DocumentPayload:
    """Text AND an image for every page. `suffix` is the source file's
    extension (".pdf", ".jpg", ...) so a raw image file and a PDF can share
    one entry point."""
    dpi = dpi or get_settings().page_image_dpi
    payload = DocumentPayload()
    suffix = suffix.lower()

    if suffix in IMAGE_EXTENSIONS:
        payload.media_type = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
        payload.images.append(data)
        try:
            text = _ocr_page(data, script, ocr_client)
            if text.strip():
                payload.text_blocks.append(f"--- Page 1 (OCR) ---\n{text}")
        except Exception as exc:  # noqa: BLE001 — the image still goes to the model
            log.warning("unified_extraction: OCR failed for an image file, sending image only: %s", exc)
            payload.warnings.append(f"OCR failed, page sent as image only: {exc}")
        return payload

    if suffix != ".pdf":
        raise ValueError(f"unsupported file type: {suffix}")

    pages = {p.page_number: p for p in pdf_utils.extract_text_and_boxes(data)}
    for i in range(pdf_utils.page_count(data)):
        page = pages.get(i)
        layer_text = page.text if page else ""

        # Rasterized for EVERY page, born-digital included — the image is
        # what lets the model resolve a form layout whose text layer comes
        # out label-block-then-value-block (see AH3500627 in the dataset,
        # where "Specimen:" and its value are pages apart in reading order).
        try:
            payload.images.append(pdf_utils.rasterize_page(data, i, dpi=dpi))
        except Exception as exc:  # noqa: BLE001 — text alone is still worth sending
            log.warning("unified_extraction: could not rasterize page %d: %s", i, exc)
            payload.warnings.append(f"page {i + 1} could not be rendered as an image: {exc}")

        if len(layer_text.strip()) >= MIN_PAGE_TEXT_CHARS:
            payload.text_blocks.append(f"--- Page {i + 1} ---\n{layer_text}")
            continue

        # No usable text layer: OCR it rather than relying on the image alone.
        if not payload.images:
            payload.warnings.append(f"page {i + 1} has no text layer and could not be rendered")
            continue
        try:
            ocr_text = _ocr_page(payload.images[-1], script, ocr_client)
            if ocr_text.strip():
                payload.text_blocks.append(f"--- Page {i + 1} (OCR) ---\n{ocr_text}")
        except Exception as exc:  # noqa: BLE001 — one unreadable page must not lose the rest
            log.warning("unified_extraction: OCR failed for page %d, image only: %s", i, exc)
            payload.warnings.append(f"page {i + 1} OCR failed, sent as image only: {exc}")

    return payload


def extract(chain, payload: DocumentPayload) -> dict:
    """One call, text + every page image. `chain` is any LlmChain — see this
    module's docstring on why the chain is the one thing NOT shared."""
    result = chain.generate_json(
        SYSTEM_PROMPT,
        payload.user_text,
        image_media_type=payload.media_type,
        images=payload.images or None,
    )
    if not isinstance(result, dict):
        return {"fields": {}, "tests": []}
    fields = result.get("fields")
    tests = result.get("tests")
    return {
        # Empty-valued keys are dropped. MEASURED 2026-09-03 on
        # 001_Lab-report.png: once OCR text is included, the model sees the
        # form's LABEL block ("Client Code", "Tel", "Fax", "CIN", ...) whose
        # values sit in a separate block it can't confidently pair, so it
        # emits the key with "" rather than inventing a value — correct
        # behaviour per the prompt, but a key with no value is not a field.
        # Dropped here, at the one shared boundary, so neither the benchmark
        # nor the review UI has to filter it separately.
        "fields": {
            k: v for k, v in fields.items() if isinstance(v, (str, int, float)) and str(v).strip()
        } if isinstance(fields, dict) else {},
        "tests": [t for t in tests if isinstance(t, dict)] if isinstance(tests, list) else [],
    }
