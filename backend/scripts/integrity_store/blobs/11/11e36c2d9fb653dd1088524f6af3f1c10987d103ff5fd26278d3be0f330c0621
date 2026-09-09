"""Local OCR for English/Latin-script scans. Preprocessing (orientation
correction, resize, illumination/contrast normalization) is reused
in-process from the deepreef-ocr sibling repo's own checkout (see
config.py's deepreef_ocr_path) rather than duplicated here — imported
lazily, at call time, from that repo's `preprocessor.py`.

Recognition, however, calls `rapidocr_onnxruntime.RapidOCR` DIRECTLY rather
than going through deepreef-ocr's `DeepReefOCR`/`engine.py`: that repo's
SCRIPTS table only ships baked recognition models for Devanagari/Arabic/
Tamil/Telugu/Kannada — see documents/ocr_client.py's SUPPORTED_SCRIPTS
comment: "notably no English/Latin model. Callers must route English scans
elsewhere." Calling RapidOCR with no rec_model_path override makes it fall
back to its own bundled default English/Latin recognition model — no
baked file, no AWS call, no Docker, nothing to download.

The import is deliberately lazy (deferred until the first real OCR call,
not done at module load) so a missing/misconfigured sibling checkout only
breaks the preprocessing step specifically — not the whole backend's
startup.
"""
import os
import sys
from functools import lru_cache

from app.config import get_settings
from app.documents.geometry import quad_to_rect
from app.documents.ocr_client import OcrResult


class LocalOcrError(RuntimeError):
    pass


def _deepreef_ocr_dir() -> str:
    # backend/app/documents/local_ocr.py -> backend/, then the configured
    # path (default ../../deepreef-ocr/deepreef-ocr) from there.
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.normpath(os.path.join(backend_dir, get_settings().deepreef_ocr_path))


def _ensure_importable() -> None:
    repo_dir = _deepreef_ocr_dir()
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)


@lru_cache
def _rapidocr_engine():
    from rapidocr_onnxruntime import RapidOCR

    # No rec_model_path override: RapidOCR loads its own bundled default
    # English/Latin recognition model, mirroring exactly how deepreef-ocr's
    # own engine.py would call it for a script with no baked override (see
    # that file's get_engine(), which only adds rec_model_path when a path
    # is configured).
    return RapidOCR()


def _decode_and_resize(image_bytes: bytes):
    _ensure_importable()
    from deepreef_ocr.preprocessor import decode_and_resize

    return decode_and_resize(image_bytes)


def extract_english(image_bytes: bytes) -> OcrResult:
    try:
        image, _meta = _decode_and_resize(image_bytes)
        result, _elapsed = _rapidocr_engine()(image)
    except Exception as exc:  # noqa: BLE001 — preprocessor/RapidOCR/import raise several distinct error types
        raise LocalOcrError(f"local OCR call failed: {exc}") from exc

    lines, confidences, boxes = [], [], []
    for box, text, conf in (result or []):
        boxes.append(quad_to_rect([[int(pt[0]), int(pt[1])] for pt in box]))
        lines.append(text)
        confidences.append(float(conf))

    return OcrResult(
        text=" ".join(lines),
        lines=lines,
        confidence=round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        boxes=boxes,
        model_used="rapidocr-default-en",
        region_count=len(lines),
    )
