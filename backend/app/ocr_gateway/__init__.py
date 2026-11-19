"""OCR gateway: engine selection, spacing/date-preserving OCR, fallback."""
from app.ocr_gateway.engines import ENGINES, EngineError, EngineResult
from app.ocr_gateway.gateway import (
    OcrGatewayError,
    engine_order,
    page_text,
    read_image,
    resolve_script,
    self_check,
)
from app.ocr_gateway.settings import GatewaySettings, get_gateway_settings

__all__ = [
    "ENGINES", "EngineError", "EngineResult", "GatewaySettings", "OcrGatewayError",
    "engine_order", "get_gateway_settings", "page_text", "read_image", "resolve_script",
    "self_check",
]
