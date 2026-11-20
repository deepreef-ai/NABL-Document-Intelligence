"""Post-anonymisation verification — reading the output back as an attacker would.

The anonymiser's own belief that it redacted something is worth very little.
This pass throws that belief away and re-examines the file that will actually
be handed over, through two independent channels:

* **Text layer.** Catches the failure where a box was drawn but the glyphs
  survive underneath — still selectable, still copyable, invisible to the eye.
* **OCR of the rendered page.** Catches PII that was never detected in the
  first place, and pixels that were somehow left legible.

Only self-validating identifiers fail a document. Context-driven detections are
too noisy to gate on: a page containing the word "Patient" near any capitalised
word would fail every report, and a verification step that always fails teaches
its operator to ignore it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from .config import AnonymizerConfig
from .detect import detect_in_text
from .ocr_engine import OcrEngine, OcrUnavailable
from .protect import ProtectedVocabulary

log = logging.getLogger(__name__)


@dataclass
class VerificationFinding:
    kind: str
    channel: str          # "text_layer" | "ocr"
    page: int
    sample: str = ""      # populated only when include_pii_samples is on


@dataclass
class VerificationResult:
    checked: bool = True
    passed: bool = True
    findings: list[VerificationFinding] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def summary(self) -> str:
        if not self.checked:
            return "verification skipped"
        if self.error:
            return f"verification incomplete: {self.error}"
        if self.passed:
            return "no disqualifying PII found in the output"
        kinds = sorted({f.kind for f in self.findings})
        return f"possible PII remains: {', '.join(kinds)}"


def _sample(value: str, include: bool) -> str:
    """A non-disclosing fingerprint of a finding, unless samples are enabled."""
    if not include:
        return f"<{len(value)} chars>"
    head = value[:2]
    return f"{head}… ({len(value)} chars)"


def _scan(text: str, cfg: AnonymizerConfig, vocab: ProtectedVocabulary,
          channel: str, page: int) -> list[VerificationFinding]:
    findings = []
    for kind, value in detect_in_text(text, cfg.detection, vocab):
        if kind not in cfg.verification.failing_kinds:
            continue
        findings.append(VerificationFinding(
            kind=kind, channel=channel, page=page,
            sample=_sample(value, cfg.include_pii_samples),
        ))
    return findings


def verify_pdf(
    path, cfg: AnonymizerConfig, vocab: ProtectedVocabulary, ocr: OcrEngine | None,
) -> VerificationResult:
    result = VerificationResult(checked=True)
    if not cfg.verification.enabled:
        return VerificationResult(checked=False)

    try:
        doc = fitz.open(path)
    except Exception as exc:  # noqa: BLE001
        return VerificationResult(checked=True, passed=False, error=f"cannot reopen output: {exc}")

    try:
        for index in range(doc.page_count):
            page = doc[index]
            if cfg.verification.check_text_layer:
                if "text_layer" not in result.channels:
                    result.channels.append("text_layer")
                result.findings += _scan(page.get_text("text"), cfg, vocab, "text_layer", index)

            if cfg.verification.check_ocr and ocr is not None:
                if "ocr" not in result.channels:
                    result.channels.append("ocr")
                try:
                    zoom = cfg.ocr.dpi / 72.0
                    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    lines = ocr.read(np.array(image))
                    text = "\n".join(t for _b, t, _s in lines)
                    result.findings += _scan(text, cfg, vocab, "ocr", index)
                except OcrUnavailable as exc:
                    # An OCR failure means we could not check, which is not the
                    # same as passing — recorded so the status reflects it.
                    result.error = str(exc)
                    log.warning("verification OCR unavailable on page %d: %s", index, exc)
    finally:
        doc.close()

    result.passed = not result.findings and not result.error
    return result


def verify_image(
    path, cfg: AnonymizerConfig, vocab: ProtectedVocabulary, ocr: OcrEngine | None,
) -> VerificationResult:
    if not cfg.verification.enabled:
        return VerificationResult(checked=False)
    result = VerificationResult(checked=True)
    if ocr is None or not cfg.verification.check_ocr:
        result.error = "image verification requires OCR, which is disabled"
        result.passed = False
        return result
    try:
        image = Image.open(path)
        image.load()
        lines = ocr.read(np.array(image.convert("RGB")))
        text = "\n".join(t for _b, t, _s in lines)
        result.channels.append("ocr")
        result.findings += _scan(text, cfg, vocab, "ocr", 0)
    except OcrUnavailable as exc:
        result.error = str(exc)
    except Exception as exc:  # noqa: BLE001
        result.error = f"cannot reopen output image: {exc}"
    result.passed = not result.findings and not result.error
    return result
