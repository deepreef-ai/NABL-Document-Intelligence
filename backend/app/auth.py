"""Optional API-key protection.

Disabled unless the API_KEY env var / setting is non-empty, in which case
behaviour is exactly as if this module did not exist. When enabled, every
request must carry a matching X-API-Key header, except health and docs
endpoints and CORS preflight (OPTIONS) requests.
"""
import hmac

from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import get_settings

OPEN_PATHS = {"/health", "/graph/health", "/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}


async def api_key_middleware(request: Request, call_next):
    expected = (get_settings().api_key or "").strip()
    if not expected or request.method == "OPTIONS" or request.url.path in OPEN_PATHS:
        return await call_next(request)
    supplied = request.headers.get("x-api-key", "")
    # compare_digest on bytes: str comparison raises on non-ASCII input.
    if not hmac.compare_digest(supplied.encode(), expected.encode()):
        return JSONResponse(status_code=401, content={"detail": "invalid or missing API key"})
    return await call_next(request)
