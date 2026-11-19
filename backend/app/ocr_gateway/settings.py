"""Settings for the OCR gateway (env prefix ``OCR_GATEWAY_``)."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config import _ENV_FILE


class GatewaySettings(BaseSettings):
    """Overridable via ``OCR_GATEWAY_*`` env vars or backend/.env."""

    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_prefix="OCR_GATEWAY_", extra="ignore")

    # The rollback lever: False keeps the legacy OCR path byte-for-byte.
    enabled: bool = False
    max_dim: int = 2048
    max_words: int = 1000
    # Comma-separated engine names, tried in order (see engines.ENGINES).
    latin_engines: str = "deepreef_local,rapidocr"
    other_engines: str = "lambda,deepreef_local,rapidocr"
    # Empty -> app.config Settings.deepreef_ocr_path. Relative -> against backend/.
    deepreef_path: str = ""
    # Turn deepreef's camera-stamp filters off IN-PROCESS so dated lines survive.
    keep_dated_lines: bool = True
    lambda_payload_limit_bytes: int = 5_500_000
    # The deployed deepreef-ocr Lambda (renamed from akash-ocr upstream, #39).
    # Only the gateway uses these; the legacy ocr_client keeps its own setting.
    lambda_function: str = "deepreef-ocr"
    lambda_region: str = "ap-south-1"
    lambda_timeout_seconds: float = 60.0


@lru_cache
def get_gateway_settings() -> GatewaySettings:
    return GatewaySettings()
