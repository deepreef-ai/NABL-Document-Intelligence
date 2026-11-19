"""OCR engines behind the gateway. Each takes (image_bytes, script_code,
settings) and returns an EngineResult or raises."""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.config import _BACKEND_DIR, get_settings
from app.documents.geometry import quad_to_rect
from app.ocr_gateway.settings import GatewaySettings

log = logging.getLogger(__name__)

_STAMP_ENV = ("OCR_STAMP_FILTER_RAPIDOCR", "OCR_DROP_CAMERA_STAMPS")
_STAMP_ATTRS = ("STAMP_FILTER_RAPIDOCR", "DROP_CAMERA_STAMPS")


@dataclass
class EngineResult:
    """Attribute-compatible with app.documents.ocr_client.OcrResult."""

    text: str
    lines: list[str]
    confidence: float
    boxes: list
    model_used: str
    region_count: int
    engine: str
    text_lines: str = ""
    items: list[dict] = field(default_factory=list)
    quality_flag: str | None = None
    blur_score: float | None = None
    ocr_seconds: float = 0.0

    @classmethod
    def from_legacy(cls, r: Any, engine: str) -> "EngineResult":
        return cls(
            text=r.text,
            lines=list(r.lines),
            confidence=r.confidence,
            boxes=list(r.boxes),
            model_used=r.model_used,
            region_count=r.region_count,
            engine=engine,
            text_lines="\n".join(r.lines),
        )


class EngineError(RuntimeError):
    pass


def _from_deepreef(ocr_data: dict, meta: dict | None, engine: str) -> EngineResult:
    """Map deepreef's response contract (ocr_data + image metadata)."""
    lines = list(ocr_data.get("lines") or [])
    meta = meta or {}
    blur = meta.get("blur_score")
    return EngineResult(
        text=ocr_data.get("text") or "",
        lines=lines,
        confidence=float(ocr_data.get("confidence") or 0.0),
        boxes=[quad_to_rect(b) for b in ocr_data.get("bounding_boxes") or []],
        model_used=ocr_data.get("model_used") or "",
        region_count=int(ocr_data.get("region_count") or len(lines)),
        engine=engine,
        text_lines=ocr_data.get("text_lines") or "\n".join(lines),
        items=list(ocr_data.get("items") or []),
        quality_flag=meta.get("quality_flag"),
        blur_score=float(blur) if blur is not None else None,
    )


# --- deepreef_local ---------------------------------------------------------

def resolve_deepreef_dir(s: GatewaySettings) -> Path:
    """The deepreef-ocr checkout root (the dir that contains deepreef_ocr/)."""
    raw = s.deepreef_path or get_settings().deepreef_ocr_path
    path = Path(raw)
    return path if path.is_absolute() else (_BACKEND_DIR / path).resolve()


_lock = threading.Lock()
_deepreef: dict[str, Any] = {}  # {"decode": fn, "engine": DeepReefOCR, "module": engine module}


def _disable_stamp_filters(engine_module: Any) -> None:
    for attr in _STAMP_ATTRS:
        setattr(engine_module, attr, False)
    if any(getattr(engine_module, a) for a in _STAMP_ATTRS):  # pragma: no cover - defensive
        raise EngineError("could not disable deepreef camera-stamp filters")


def _load_deepreef(s: GatewaySettings) -> dict[str, Any]:
    with _lock:
        if not _deepreef:
            root = resolve_deepreef_dir(s)
            if not (root / "deepreef_ocr" / "engine.py").is_file():
                raise EngineError(
                    f"deepreef-ocr checkout not found at {root} (no deepreef_ocr/engine.py); "
                    "set OCR_GATEWAY_DEEPREEF_PATH or DEEPREEF_OCR_PATH to the checkout root"
                )
            if s.keep_dated_lines:
                # engine.py reads these at import time; process env only.
                for name in _STAMP_ENV:
                    os.environ[name] = "0"
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            prev = sys.dont_write_bytecode
            sys.dont_write_bytecode = True
            try:
                from deepreef_ocr import engine as engine_module
                from deepreef_ocr.preprocessor import decode_and_resize
            except Exception as exc:  # noqa: BLE001 - import can fail many ways
                raise EngineError(f"cannot import deepreef_ocr from {root}: {exc}") from exc
            finally:
                sys.dont_write_bytecode = prev
            _deepreef.update(
                decode=decode_and_resize,
                module=engine_module,
                engine=engine_module.DeepReefOCR(warm=None),
                root=root,
            )
            log.info("deepreef_local loaded from %s", root)
        if s.keep_dated_lines:
            # Also covers a module imported earlier by someone else.
            _disable_stamp_filters(_deepreef["module"])
        return _deepreef


def deepreef_local(image_bytes: bytes, code: str, s: GatewaySettings) -> EngineResult:
    if not image_bytes:
        raise EngineError("empty image")
    dr = _load_deepreef(s)
    try:
        img, meta = dr["decode"](image_bytes, max_dim=s.max_dim)
    except Exception as exc:  # noqa: BLE001
        raise EngineError(f"could not decode image: {exc}") from exc
    try:
        out = dr["engine"].extract(img, code, max_words=s.max_words)
    except Exception as exc:  # noqa: BLE001
        raise EngineError(f"deepreef extract failed: {exc}") from exc
    return _from_deepreef(out, meta, "deepreef_local")


# --- lambda -------------------------------------------------------------------

def _fit_payload(image_bytes: bytes, limit: int) -> str:
    """Base64 of the image, re-encoded as JPEG (q85, then 0.8x steps) if too big."""
    encoded = base64.b64encode(image_bytes).decode("ascii")
    if len(encoded) <= limit:
        return encoded
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise EngineError(f"payload too large and image not decodable: {exc}") from exc
    for _ in range(12):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        if len(encoded) <= limit:
            return encoded
        w, h = img.size
        if min(w, h) < 64:
            break
        img = img.resize((max(1, int(w * 0.8)), max(1, int(h * 0.8))))
    raise EngineError(f"image does not fit the Lambda payload limit ({limit} bytes)")


def lambda_engine(image_bytes: bytes, code: str, s: GatewaySettings) -> EngineResult:
    import boto3
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import BotoCoreError, ClientError

    if not image_bytes:
        raise EngineError("empty image")
    payload = {
        "image_base64": _fit_payload(image_bytes, s.lambda_payload_limit_bytes),
        "script": code,
        "max_dim": s.max_dim,
        "max_words": s.max_words,
    }
    client = boto3.client(
        "lambda",
        region_name=s.lambda_region,
        config=BotoConfig(connect_timeout=s.lambda_timeout_seconds,
                          read_timeout=s.lambda_timeout_seconds),
    )
    try:
        response = client.invoke(
            FunctionName=s.lambda_function,
            Payload=json.dumps(payload).encode("utf-8"),
        )
    except (BotoCoreError, ClientError) as exc:
        raise EngineError(f"Lambda call failed: {exc}") from exc
    raw = response["Payload"].read()
    if response.get("FunctionError"):
        raise EngineError(f"Lambda errored: {raw.decode('utf-8', 'replace')[:500]}")
    envelope = json.loads(raw)
    body = envelope.get("body", envelope)
    parsed = json.loads(body) if isinstance(body, str) else body
    if not parsed.get("success"):
        raise EngineError(f"Lambda returned an error: {parsed.get('error')}")
    ocr_data = parsed.get("ocr_data") or {}
    if not ocr_data.get("model_used"):
        raise EngineError("engine did not report model_used")
    return _from_deepreef(ocr_data, parsed.get("image"), "lambda")


# --- rapidocr (legacy) ------------------------------------------------------------

def rapidocr(image_bytes: bytes, code: str, s: GatewaySettings) -> EngineResult:
    from app.documents.local_ocr import extract_english

    return EngineResult.from_legacy(extract_english(image_bytes), "rapidocr")


ENGINES: dict[str, Callable[[bytes, str, GatewaySettings], EngineResult]] = {
    "deepreef_local": deepreef_local,
    "lambda": lambda_engine,
    "rapidocr": rapidocr,
}
