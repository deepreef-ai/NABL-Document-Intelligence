"""Local OCR for English/Latin-script scans — runs the REAL deepreef-ocr
pipeline in-process, imported directly from that sibling repo's own
checkout (see config.py's deepreef_ocr_path) rather than a copy kept inside
this project, so there's exactly one place that code lives. The import
adds that repo's root to sys.path at call time and then imports its actual
`engine.py` (DeepReefOCR, the RapidOCR wrapper) and `preprocessor.py`
(decode_and_resize) — the same two-step pipeline (preprocess, then OCR)
that repo's own app.py chains together for every real request.

For English specifically, DeepReefOCR needs no baked .onnx file at all: its
SCRIPTS table maps "english" to RapidOCR's own bundled default recognition
model. That's what makes this a zero-extra-file, in-process path — no AWS
call, no Docker, nothing to download — unlike Devanagari/Arabic/Tamil/
Telugu/Kannada, which still go through the real deployed Lambda
(documents/ocr_client.py) since baking those models in here would
duplicate 15MB+ of binaries this project has no other use for.

The import is deliberately lazy (deferred until the first real OCR call,
not done at module load) so a missing/misconfigured sibling checkout only
breaks English OCR specifically, the same way the previous PaddleOCR-based
version deferred its own import — not the whole backend's startup.
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
def _engine():
    _ensure_importable()
    from deepreef_ocr.engine import DeepReefOCR

    # warm="english": that repo's own default (warm="devanagari") would try
    # to eagerly load a model file that only exists inside ITS checkout's
    # deepreef_ocr/models/ for baked non-English scripts — irrelevant here.
    return DeepReefOCR(warm="english")


def _decode_and_resize(image_bytes: bytes):
    _ensure_importable()
    from deepreef_ocr.preprocessor import decode_and_resize

    return decode_and_resize(image_bytes)


def extract_english(image_bytes: bytes) -> OcrResult:
    try:
        image, _meta = _decode_and_resize(image_bytes)
        ocr_data = _engine().extract(image, script="english")
    except Exception as exc:  # noqa: BLE001 — preprocessor/RapidOCR/import raise several distinct error types
        raise LocalOcrError(f"deepreef-ocr call failed: {exc}") from exc

    boxes = [quad_to_rect(quad) for quad in ocr_data["bounding_boxes"]]
    return OcrResult(
        text=ocr_data["text"],
        lines=ocr_data["lines"],
        confidence=ocr_data["confidence"],
        boxes=boxes,
        model_used=ocr_data["model_used"],
        region_count=ocr_data["region_count"],
    )
