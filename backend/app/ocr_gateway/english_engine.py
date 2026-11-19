"""English OCR with the vendored PP-OCRv5 recogniser (no external checkout).

A replica of exactly what deepreef-ocr's ``DeepReefOCR.extract(img, "en")``
does (deepreef_ocr/engine.py at 48ee778), using files kept in this repo:

* model: ``models/english/devanagari_rec.onnx`` + ``devanagari_dict.txt``
  (copies of deepreef's, sha256-pinned below);
* preprocessing: ``vendor/deepreef_preprocessor.decode_and_resize`` (verbatim copy);
* RapidOCR built as ``get_engine`` builds it: ``rec_model_path``,
  ``rec_keys_path`` and intra/inter-op threads = ``OCR_NUM_THREADS`` (default 2);
  every other setting, including the text detector, is RapidOCR's bundled default;
* called with ``use_cls=False`` (``en`` is in deepreef's ``NO_CLS_SCRIPTS``);
* output: the same response dict (``text``, ``lines``, ``confidence``,
  ``bounding_boxes``, ``items``, ``text_lines`` ...) built the same way
  (``_per_line`` / ``_reading_rows`` replicated below), ``model_used`` =
  ``devanagari_rec.onnx``.

The one deliberate difference from deepreef's defaults: the camera-stamp
filter (which drops every line containing a dd/mm/yyyy date) is OFF unless
the caller asks for it; the gateway asks only when keep_dated_lines=False,
which is what the external path does too.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict

log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent / "models" / "english"
REC_MODEL = "devanagari_rec.onnx"
REC_KEYS = "devanagari_dict.txt"
#: Pinned to deepreef-ocr models.sha256 (see models/english/README.md).
SHA256 = {
    REC_MODEL: "d6f0a906580e3fa6b324a318718f1f31f268b6ea8ef985f91c2012a37f52c91e",
    REC_KEYS: "09c7440bfc5477e5c41052304b6b185aff8c4a5e8b2b4c23c1c706f6fe1ee9fc",
}
SCRIPT = "en"

# Same env var and default as deepreef's engine.NUM_THREADS.
NUM_THREADS = max(1, int(os.getenv("OCR_NUM_THREADS", "2")))

# deepreef engine._CAMERA_STAMP, unchanged (only used when drop_camera_stamps=True).
_CAMERA_STAMP = re.compile(
    r"gps\s*map|map\s*camera|timestamp\s*camera|\bgoogle\b|\blat(itude)?\b[\s:.]*-?\d|\blong(itude)?\b[\s:.]*-?\d"
    r"|gmt\s*[+-]\s*\d|\b\d{1,2}[:.]\d{2}\s*(am|pm)\b|\b[0-9a-z]{4}\+[0-9a-z]{2,3}\b"
    r"|\b(mon|tues|wednes|thurs|fri|satur|sun)day\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    re.IGNORECASE)


def model_paths() -> tuple[Path, Path]:
    return MODEL_DIR / REC_MODEL, MODEL_DIR / REC_KEYS


def available() -> bool:
    """True when both vendored files are present (cheap; no hashing)."""
    rec, keys = model_paths()
    return rec.is_file() and keys.is_file()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def status(verify_hash: bool = True) -> dict:
    """Health of the vendored files, for self_check. Never raises."""
    files = {}
    for name, want in SHA256.items():
        p = MODEL_DIR / name
        entry: dict = {"present": p.is_file()}
        if p.is_file() and verify_hash:
            try:
                got = _sha256(p)
                entry["sha256_ok"] = got == want
            except OSError as exc:
                entry["sha256_ok"] = False
                entry["error"] = str(exc)
        files[name] = entry
    ok = all(f["present"] and f.get("sha256_ok", True) for f in files.values())
    return {"ok": ok, "path": str(MODEL_DIR), "model_used": REC_MODEL, "files": files,
            "detail": "vendored English model present" + (" and sha256 verified" if verify_hash else "")
            if ok else "vendored English model missing or sha256 mismatch"}


# --- replicated from deepreef_ocr/engine.py ---------------------------------

def _reading_rows(boxes) -> list:
    """Indices of `boxes` grouped into rows, in reading order: a row is boxes whose
    vertical centres fall inside each other's vertical span; rows top to bottom,
    boxes left to right within a row. Box = 4 [x, y] points."""
    spans = []
    for i, b in enumerate(boxes):
        ys, xs = [p[1] for p in b], [p[0] for p in b]
        spans.append((i, min(ys), max(ys), (min(ys) + max(ys)) / 2, (min(xs) + max(xs)) / 2))
    rows: list = []
    for i, top, bottom, cy, cx in sorted(spans, key=lambda s: s[3]):
        row = rows[-1] if rows else None
        if row and any(t <= cy <= b or top <= c <= bottom for _, t, b, c, _ in row):
            row.append((i, top, bottom, cy, cx))
        else:
            rows.append([(i, top, bottom, cy, cx)])
    return [[i for i, *_ in sorted(row, key=lambda s: s[4])] for row in rows]


def _per_line(lines, confidences, boxes, sources, script) -> Dict[str, Any]:
    rows = _reading_rows(boxes)
    return {
        "items": [{"text": lines[i], "confidence": round(confidences[i], 4),
                   "script": "latin" if sources[i] == "latin" else script, "source": sources[i],
                   "box": boxes[i]}
                  for row in rows for i in row],
        "text_lines": "\n".join(" ".join(lines[i] for i in row) for row in rows),
    }


# --- engine -------------------------------------------------------------------

class EnglishModelMissing(RuntimeError):
    pass


_lock = threading.Lock()
_engine: Any = None


def get_engine():
    """The RapidOCR instance, built once per process (thread-safe)."""
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                rec, keys = model_paths()
                if not rec.is_file():
                    raise EnglishModelMissing(f"missing vendored model {rec}")
                from rapidocr_onnxruntime import RapidOCR

                kwargs = {
                    "rec_model_path": str(rec),
                    "intra_op_num_threads": NUM_THREADS,
                    "inter_op_num_threads": NUM_THREADS,
                }
                if keys.is_file():
                    kwargs["rec_keys_path"] = str(keys)
                t0 = time.perf_counter()
                _engine = RapidOCR(**kwargs)
                log.info("vendored English OCR loaded (%s) in %.2fs, threads=%d",
                         REC_MODEL, time.perf_counter() - t0, NUM_THREADS)
    return _engine


def decode(image_bytes: bytes, max_dim: int):
    from app.ocr_gateway.vendor.deepreef_preprocessor import decode_and_resize

    return decode_and_resize(image_bytes, max_dim=max_dim)


def extract(image, drop_camera_stamps: bool = False) -> Dict[str, Any]:
    """deepreef ``DeepReefOCR.extract(image, "en")`` on a decoded BGR array."""
    engine = get_engine()
    t0 = time.perf_counter()
    result, _elapse = engine(image, use_cls=False)
    ocr_ms = round((time.perf_counter() - t0) * 1000, 2)

    lines, confidences, boxes = [], [], []
    for box, text, conf in (result or []):
        if drop_camera_stamps and _CAMERA_STAMP.search(text or ""):
            continue
        boxes.append([[int(pt[0]), int(pt[1])] for pt in box])
        lines.append(text)
        confidences.append(float(conf))

    return {
        "text": " ".join(lines),
        "lines": lines,
        "confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        "script": SCRIPT,
        "model_used": REC_MODEL,
        "region_count": len(lines),
        "bounding_boxes": boxes,
        "ocr_ms": ocr_ms,
        **_per_line(lines, confidences, boxes, ["rapidocr"] * len(lines), SCRIPT),
    }


def read(image_bytes: bytes, max_dim: int, drop_camera_stamps: bool = False) -> tuple[dict, dict]:
    """Bytes -> (ocr_data, image metadata), same contract as the deepreef path."""
    img, meta = decode(image_bytes, max_dim)
    return extract(img, drop_camera_stamps=drop_camera_stamps), meta
