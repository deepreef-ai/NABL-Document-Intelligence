"""deepreef-ocr run in-process, English only.

Uses the team's own engine (deepreef-ai/deepreef-ocr, `deepreef_ocr.engine.
DeepReefOCR`) as a library instead of calling its Lambda. For English its
recogniser is `devanagari_rec.onnx` (PP-OCRv4 Devanagari, whose dictionary
holds all printable ASCII) driven by RapidOCR 1.4.4 — the same library the old
path used, but configured the way deepreef-ocr measured to be fastest:

- ONNX threads pinned per engine (OCR_NUM_THREADS, default 2) instead of
  RapidOCR's "every core", so several pages can run side by side without
  oversubscribing the CPU (deepreef-ocr measured ~4x throughput from this);
- no 180-degree classifier for English (NO_CLS_SCRIPTS);
- `text_lines`: boxes grouped into rows in reading order, so a table row comes
  out as one line ("45 Copper Hydroxide mg/kg SOP-CHM-27-02 17.16") instead of
  one line per cell, which is what the extractor and the row check need;
- no word cap (`max_words=0`): the engine defaults to 80 words for signboards,
  and a lab report page has hundreds.

One engine per worker thread: the RapidOCR object keeps per-call state, so a
shared instance is not guaranteed safe across threads. Each instance loads the
detector and recogniser once and is reused for every page that thread reads.
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys
import threading
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

SCRIPT = "en"
_MODEL_FILES = ("devanagari_rec.onnx", "devanagari_dict.txt")

_local = threading.local()
_import_lock = threading.Lock()
_model_report: dict | None = None


class DeepreefLocalError(RuntimeError):
    pass


@dataclass
class PageOcr:
    text: str                       # row-grouped, one table row per line
    lines: list[str] = field(default_factory=list)
    confidence: float | None = None
    model_used: str = ""
    ocr_ms: float = 0.0
    region_count: int = 0


def _settings():
    from app.config import get_settings
    from app.graph.config import get_graph_settings

    return get_settings(), get_graph_settings()


def _repo_dir() -> str:
    app_settings, _ = _settings()
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.normpath(os.path.join(backend_dir, app_settings.deepreef_ocr_path))


def _model_dir() -> str:
    _, graph_settings = _settings()
    configured = (graph_settings.deepreef_model_dir or "").strip()
    return configured or os.path.join(_repo_dir(), "deepreef_ocr", "models")


def _import_engine():
    """Import deepreef_ocr.engine with the thread count fixed BEFORE import —
    the engine reads OCR_NUM_THREADS once, at module import."""
    with _import_lock:
        repo = _repo_dir()
        if not os.path.isdir(os.path.join(repo, "deepreef_ocr")):
            raise DeepreefLocalError(f"deepreef-ocr checkout not found at {repo} (DEEPREEF_OCR_PATH)")
        if repo not in sys.path:
            sys.path.insert(0, repo)
        _, graph_settings = _settings()
        os.environ.setdefault("OCR_NUM_THREADS", str(max(1, graph_settings.ocr_threads_per_page)))
        from deepreef_ocr import engine

        return engine


#: Second approved checksum(s): models DERIVED from a pinned deepreef-ocr model
#: and validated here. A file matching one of these is accepted as genuine and
#: labelled with its variant; anything else that is not pinned is a STAND-IN.
#: To add one, record exactly how it was produced and the validation result.
APPROVED_DERIVED_MODELS: dict[str, dict[str, str]] = {
    "b07bbad325bb181fee14ab004e77bc48b92916a7af40aa8c3b1e468b7f5a7601": {
        "file": "devanagari_rec.onnx",
        "variant": "INT8",
        "derived_from": "d6f0a906580e3fa6b324a318718f1f31f268b6ea8ef985f91c2012a37f52c91e",
        "how": "onnxruntime.quantization.quantize_dynamic(weight_type=QUInt8, "
               "op_types_to_quantize=['MatMul','Gemm']); convs stay fp32; opset 7 -> 11",
        # MEASURED 2026-11-24, 29 scanned pages (benchmarks/int8/REPORT.md):
        # rows 267/268 identical to fp32, 2/1004 OCR lines differ (logo text
        # only, no values), 3-6% faster recognition in alternating runs.
        "validated": "2026-11-24",
    },
}


def model_report() -> dict:
    """Which model files are present and whether they are approved.

    deepreef-ocr pins every model by sha256 (models.sha256). A file matching a
    pinned hash is "pinned"; one matching APPROVED_DERIVED_MODELS is
    "approved-<variant>" (e.g. the validated INT8 build). Anything else is a
    stand-in, and every reading made with it is labelled so — never passed off
    as the real engine's output.
    """
    global _model_report
    if _model_report is not None:
        return _model_report
    model_dir = _model_dir()
    pinned: dict[str, str] = {}
    try:
        with open(os.path.join(_repo_dir(), "models.sha256"), encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) == 2:
                    pinned[parts[1]] = parts[0]
    except OSError:
        pass
    files = {}
    for name in _MODEL_FILES:
        path = os.path.join(model_dir, name)
        if not os.path.isfile(path):
            files[name] = "missing"
            continue
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
        derived = APPROVED_DERIVED_MODELS.get(digest)
        if pinned.get(name) == digest:
            files[name] = "pinned"
        elif derived and derived["file"] == name:
            files[name] = f"approved-{derived['variant']}"
        else:
            files[name] = "stand-in"
    rec = files.get("devanagari_rec.onnx", "missing")
    _model_report = {
        "model_dir": model_dir,
        "files": files,
        "genuine": rec == "pinned" or rec.startswith("approved-"),
        "variant": rec.split("-", 1)[1] if rec.startswith("approved-") else ("fp32" if rec == "pinned" else ""),
    }
    log.info("deepreef-local: models %s", _model_report)
    return _model_report


def _engine():
    eng = getattr(_local, "engine", None)
    if eng is None:
        engine_mod = _import_engine()
        report = model_report()
        if report["files"].get("devanagari_rec.onnx") == "missing":
            raise DeepreefLocalError(
                f"devanagari_rec.onnx not found in {report['model_dir']} — "
                "copy the pinned model set (deepreef-ocr `make models`)"
            )
        eng = engine_mod.DeepReefOCR(model_dir=report["model_dir"], warm=None)
        _local.engine = eng
    return eng


#: deepreef-ocr's preprocessor resizes every image to this longest side.
_TARGET_MAX_DIM = 1536

#: Clean-up steps that exist for CAMERA PHOTOS of boards (uneven light, low
#: contrast, blur, huge JPEGs). A PDF page rendered by PyMuPDF has none of
#: those problems. MEASURED 2026-11-24 on a dense scanned page: default
#: clean-up 233 ms, without these 57 ms.
_CLEAN_SCAN = dict(enable_illumination=False, enable_contrast=False,
                   enable_sharpen=False, enable_compression=False)


def render_for_ocr(pdf_bytes: bytes, page_number: int, max_dpi: int) -> bytes:
    """Render a PDF page for OCR (page_number is 1-based).

    With OCR_RENDER_TO_TARGET on, the page is rendered straight at the size the
    OCR engine uses (longest side <= 1536 px, ~131 DPI for A4) instead of at
    max_dpi and then shrunk by the preprocessor — the same pixels the engine
    would end up with, without rendering and resampling ~2.3x more of them.
    """
    import fitz

    from app.documents import pdf_utils

    _, graph_settings = _settings()
    if not graph_settings.ocr_render_to_target:
        return pdf_utils.rasterize_page(pdf_bytes, page_number - 1, dpi=max_dpi)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[page_number - 1]
        longest_inches = max(page.rect.width, page.rect.height) / 72.0
        dpi = max(72, min(max_dpi, int(_TARGET_MAX_DIM / longest_inches)))
        return page.get_pixmap(dpi=dpi).tobytes("png")
    finally:
        doc.close()


def prepare_image(image_bytes: bytes, *, rendered_pdf_page: bool = False):
    """deepreef-ocr's preprocessor; the clean-scan profile for PDF renders."""
    _import_engine()
    from deepreef_ocr.preprocessor import decode_and_resize

    _, graph_settings = _settings()
    options = _CLEAN_SCAN if (rendered_pdf_page and graph_settings.ocr_clean_scan_profile) else {}
    image, _meta = decode_and_resize(image_bytes, **options)
    return image


def extract_page(image_bytes: bytes, *, rendered_pdf_page: bool = False) -> PageOcr:
    """OCR one page image with deepreef-ocr's English engine.

    rendered_pdf_page=True means the image came from render_for_ocr (a clean
    digital rendering), so photo-oriented clean-up can be skipped.
    """
    try:
        engine = _engine()
        image = prepare_image(image_bytes, rendered_pdf_page=rendered_pdf_page)
        out = engine.extract(image, SCRIPT, max_words=0)
    except DeepreefLocalError:
        raise
    except Exception as exc:  # noqa: BLE001 — preprocessor/ONNX raise many types
        raise DeepreefLocalError(f"deepreef-local OCR failed: {exc}") from exc

    text = (out.get("text_lines") or "").strip() or (out.get("text") or "")
    model = out.get("model_used", "")
    report = model_report()
    if not report["genuine"]:
        model = f"{model} (STAND-IN, not the pinned model)"
    elif report["variant"] not in ("", "fp32"):
        model = f"{model} ({report['variant']}, approved)"
    return PageOcr(
        text=text,
        lines=[ln for ln in text.splitlines() if ln.strip()],
        confidence=out.get("confidence"),
        model_used=model,
        ocr_ms=float(out.get("ocr_ms") or 0.0),
        region_count=int(out.get("region_count") or 0),
    )
