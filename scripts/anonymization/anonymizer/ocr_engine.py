"""OCR, behind one lazily-loaded interface.

RapidOCR loads ONNX models on first use, which costs seconds and a few hundred
megabytes. A run over a folder of born-digital PDFs must not pay that, so the
engine is constructed on the first scanned page and never before.

This module is deliberately standalone — it does not import the host
application's OCR code. The anonymiser is meant to be runnable against a Drive
folder on a machine that does not have the rest of the system set up.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)


class OcrUnavailable(RuntimeError):
    """Raised when OCR is needed but the engine cannot be loaded."""


class OcrEngine:
    """Thin wrapper returning (quad, text, score) triples."""

    def __init__(self) -> None:
        self._engine = None

    def _load(self):
        if self._engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as exc:  # pragma: no cover - import-time environment issue
                raise OcrUnavailable(
                    "rapidocr-onnxruntime is not installed; install it or run with --no-ocr"
                ) from exc
            log.info("loading OCR models (first scanned page)")
            self._engine = RapidOCR()
        return self._engine

    def read(self, image: "np.ndarray") -> list[tuple[Sequence[Sequence[float]], str, float]]:
        """Recognise one page image. Returns [] when the page has no text."""
        engine = self._load()
        try:
            result, _elapsed = engine(image)
        except Exception as exc:  # noqa: BLE001 — a bad page must not stop the batch
            raise OcrUnavailable(f"OCR failed on this page: {exc}") from exc
        if not result:
            return []
        out = []
        for entry in result:
            # RapidOCR returns [box, text, score]; guard the shape rather than
            # trusting it, since versions have differed here.
            if len(entry) < 3:
                continue
            box, text, score = entry[0], entry[1], entry[2]
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0
            out.append((box, str(text), score))
        return out


class NullOcrEngine(OcrEngine):
    """Used with --no-ocr: scanned pages are reported, never silently passed."""

    def read(self, image):  # noqa: D102
        raise OcrUnavailable("OCR is disabled (--no-ocr) but this page has no text layer")
