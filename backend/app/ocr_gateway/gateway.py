"""One entry point for reading an image with the right OCR engine.

Why this exists:
- The legacy path (documents/local_ocr.py) runs RapidOCR's bundled default
  model, which glues words together ("DIAGNOSTICREPORT", "25/08/202110:04").
- deepreef-ocr's ``en`` script (PP-OCRv5 Devanagari model, which reads all
  ASCII) keeps word spacing, so it is the preferred English engine.
- deepreef's camera-stamp filter drops every line containing a dd/mm/yyyy
  date. Lab reports are full of dates, so the filter is disabled in-process
  (env vars before import + module attributes; deepreef's files are untouched).
- The Lambda cannot disable that filter per request, so English defaults to
  the local engine; other scripts go to the Lambda first.
Every engine failure falls through to the next; ``attempts`` records each.
"""
from __future__ import annotations

import logging
import time

from app.ocr_gateway import engines
from app.ocr_gateway.engines import EngineResult, resolve_deepreef_dir
from app.ocr_gateway.settings import GatewaySettings, get_gateway_settings

log = logging.getLogger(__name__)

_ALIASES: dict[str, str] = {
    "": "en", "english": "en", "en": "en", "latin": "en",
    "hindi": "devanagari", "marathi": "devanagari", "nepali": "devanagari", "devanagari": "devanagari",
    "urdu": "arabic", "arabic": "arabic",
    "tamil": "ta", "ta": "ta",
    "telugu": "te", "te": "te",
    "kannada": "ka", "ka": "ka",
    "punjabi": "pa", "gurmukhi": "pa", "pa": "pa",
    "odia": "or", "oriya": "or", "or": "or",
    "bengali": "bn", "bangla": "bn", "bn": "bn",
    "assamese": "as", "as": "as",
    "gujarati": "gu", "gu": "gu",
    "malayalam": "ml", "ml": "ml",
}


class OcrGatewayError(RuntimeError):
    def __init__(self, message: str, attempts: list[dict] | None = None):
        super().__init__(message)
        self.attempts: list[dict] = attempts or []


def resolve_script(script: str | None) -> str:
    key = (script or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _ALIASES.get(key, "en")


def engine_order(code: str, s: GatewaySettings) -> list[str]:
    raw = s.latin_engines if code == "en" else s.other_engines
    return [n for n in (x.strip() for x in raw.split(",")) if n in engines.ENGINES]


def read_image(image_bytes: bytes, script: str | None) -> EngineResult:
    if not image_bytes:
        raise OcrGatewayError("empty image bytes", [])
    s = get_gateway_settings()
    code = resolve_script(script)
    attempts: list[dict] = []
    for name in engine_order(code, s):
        t0 = time.perf_counter()
        try:
            result = engines.ENGINES[name](image_bytes, code, s)  # looked up per call
            reason = "" if (result.text or "").strip() or page_text(result).strip() else "empty text"
        except Exception as exc:  # noqa: BLE001 - every engine failure falls through
            result, reason = None, f"{type(exc).__name__}: {exc}"
        seconds = round(time.perf_counter() - t0, 3)
        attempts.append({"engine": name, "ok": not reason, "reason": reason, "seconds": seconds})
        if reason:
            log.warning("ocr_gateway engine=%s script=%s failed in %.2fs: %s", name, code, seconds, reason)
            continue
        result.ocr_seconds = seconds
        result.attempts = attempts  # type: ignore[attr-defined]
        result.script_code = code  # type: ignore[attr-defined]
        log.info("ocr_gateway engine=%s script=%s model_used=%s seconds=%.2f quality_flag=%s",
                 name, code, result.model_used, seconds, result.quality_flag)
        return result
    detail = "; ".join(f"{a['engine']}: {a['reason']}" for a in attempts) or "no engines configured"
    raise OcrGatewayError(f"all OCR engines failed for script={code!r}: {detail}", attempts)


def page_text(result: EngineResult) -> str:
    return result.text_lines or "\n".join(result.lines)


def self_check() -> dict:
    """Configuration health; never invokes the Lambda, never raises."""
    report: dict = {"enabled": False,
                    "deepreef_local": {"ok": False, "path": "", "detail": ""},
                    "lambda": {"configured": False, "function": ""}}
    try:
        s = get_gateway_settings()
        report["enabled"] = s.enabled
        root = resolve_deepreef_dir(s)
        ok = (root / "deepreef_ocr" / "engine.py").is_file()
        report["deepreef_local"] = {
            "ok": ok,
            "path": str(root),
            "detail": "deepreef_ocr/engine.py found" if ok else
            "deepreef_ocr/engine.py missing; set OCR_GATEWAY_DEEPREEF_PATH or DEEPREEF_OCR_PATH",
        }
    except Exception as exc:  # noqa: BLE001
        report["deepreef_local"]["detail"] = f"{type(exc).__name__}: {exc}"
    try:
        s = get_gateway_settings()
        fn = s.lambda_function
        report["lambda"] = {"configured": bool(fn), "function": fn or "",
                            "region": s.lambda_region}
    except Exception:  # noqa: BLE001
        pass
    return report
