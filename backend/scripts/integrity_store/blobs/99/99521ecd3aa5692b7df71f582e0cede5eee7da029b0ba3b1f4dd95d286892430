import base64
import json
from dataclasses import dataclass

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from app.config import get_settings
from app.documents.geometry import Rect, quad_to_rect

# deepreef-ocr (dev branch, see deepreef_ocr/engine.py SCRIPTS table) only
# ships recognition models for these scripts — notably no English/Latin
# model. Callers must route English scans elsewhere (see documents/pipeline.py).
SUPPORTED_SCRIPTS = {"devanagari", "arabic", "ta", "te", "ka"}


@dataclass
class OcrResult:
    text: str
    lines: list[str]
    confidence: float
    boxes: list[Rect]  # aligned index-for-index with `lines`
    model_used: str
    region_count: int


class OcrError(RuntimeError):
    pass


class OcrClient:
    """Adapter for deepreef-ocr's Lambda contract. Invokes the real akash-ocr
    Lambda directly via the AWS SDK — the same mechanism pcsapaiv2 production
    uses (there is no public Function URL). Requires AWS credentials in the
    environment with lambda:InvokeFunction on that function."""

    def __init__(self, function_name: str | None = None, region: str | None = None,
                 timeout: float | None = None):
        settings = get_settings()
        self.function_name = function_name or settings.ocr_lambda_function_name
        self.timeout = timeout or settings.ocr_timeout_seconds
        self._lambda = boto3.client(
            "lambda",
            region_name=region or settings.ocr_lambda_region,
            config=BotoConfig(connect_timeout=self.timeout, read_timeout=self.timeout),
        )

    def extract(self, image_bytes: bytes, script: str) -> OcrResult:
        if script not in SUPPORTED_SCRIPTS:
            raise OcrError(
                f"deepreef-ocr has no recognition model for script={script!r}; "
                f"supported: {sorted(SUPPORTED_SCRIPTS)}"
            )
        payload = {
            "image_base64": base64.b64encode(image_bytes).decode("ascii"),
            "script": script,
        }
        try:
            response = self._lambda.invoke(
                FunctionName=self.function_name,
                Payload=json.dumps(payload).encode("utf-8"),
            )
        except (BotoCoreError, ClientError) as exc:
            raise OcrError(f"deepreef-ocr call failed: {exc}") from exc

        raw = response["Payload"].read()
        if response.get("FunctionError"):
            raise OcrError(f"deepreef-ocr Lambda errored: {raw.decode('utf-8', 'replace')}")

        # The Lambda always returns {statusCode, headers, body}; `body` is
        # itself a JSON string (see that repo's Makefile `invoke` target).
        envelope = json.loads(raw)
        body = envelope.get("body", envelope)
        parsed = json.loads(body) if isinstance(body, str) else body

        if not parsed.get("success"):
            raise OcrError(f"deepreef-ocr returned an error: {parsed.get('error')}")

        ocr_data = parsed["ocr_data"]
        boxes = [quad_to_rect(box) for box in ocr_data.get("bounding_boxes", [])]
        return OcrResult(
            text=ocr_data["text"],
            lines=ocr_data["lines"],
            confidence=ocr_data["confidence"],
            boxes=boxes,
            model_used=ocr_data["model_used"],
            region_count=ocr_data["region_count"],
        )
