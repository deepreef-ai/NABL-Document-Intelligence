"""Parallel per-page OCR via the team's deepreef-ocr Lambda (`akash-ocr`).

2026-11-24: one synchronous invoke per page, all pages fanned out at once on
a thread pool. The work is network-bound (the Lambda does the OCR), so total
wall time is roughly the slowest single page rather than the sum of pages.

WHY this needs care before wiring in:
  * The caller's IAM identity must be granted lambda:InvokeFunction on
    akash-ocr (ap-south-1); without it every page comes back AccessDenied,
    which we report per page instead of raising.
  * Page images leave this host and are sent to the team's own AWS account
    for OCR. That is a data-egress decision, not just a performance one.
"""

import base64
import json
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

# Lambda's synchronous-invoke request payload cap. base64 inflates images by
# ~1/3, so a ~4.5 MB page image is already over; check the encoded size up
# front rather than paying for a round trip that AWS will reject.
MAX_SYNC_PAYLOAD_BYTES = 6 * 1024 * 1024

PageResult = tuple[str, float | None, str]


def _parse_response(response: dict) -> PageResult:
    raw = response["Payload"].read()
    if response.get("FunctionError"):
        return "", None, f"deepreef-ocr Lambda errored: {raw.decode('utf-8', 'replace')}"

    # Same envelope handling as ocr_client.OcrClient: the Lambda returns
    # {statusCode, headers, body} where `body` is itself a JSON string.
    envelope = json.loads(raw)
    body = envelope.get("body", envelope) if isinstance(envelope, dict) else envelope
    parsed = json.loads(body) if isinstance(body, str) else body

    if not isinstance(parsed, dict) or not parsed.get("success"):
        err = parsed.get("error") if isinstance(parsed, dict) else parsed
        return "", None, f"deepreef-ocr returned an error: {err}"

    ocr_data = parsed.get("ocr_data") or {}
    # text_lines keeps reading order with one table row per line, which the
    # downstream extractor parses far better than the flat `text` field.
    text_lines = ocr_data.get("text_lines")
    if text_lines:
        text = "\n".join(str(line) for line in text_lines)
    else:
        text = ocr_data.get("text") or ""
    confidence = ocr_data.get("confidence")
    return text, (float(confidence) if confidence is not None else None), ""


def _invoke_page(client, function_name: str, image: bytes) -> PageResult:
    payload = {
        "image_base64": base64.b64encode(image).decode("ascii"),
        # Always English for NABL scans; max_words=0 disables the Lambda's
        # default 80-word cap, which would otherwise truncate full pages.
        "script": "en",
        "max_words": 0,
    }
    body = json.dumps(payload).encode("utf-8")
    if len(body) > MAX_SYNC_PAYLOAD_BYTES:
        return "", None, (
            f"page payload {len(body)} bytes exceeds Lambda sync limit "
            f"{MAX_SYNC_PAYLOAD_BYTES} bytes"
        )
    try:
        response = client.invoke(FunctionName=function_name, Payload=body)
        return _parse_response(response)
    except (BotoCoreError, ClientError) as exc:
        return "", None, f"deepreef-ocr call failed: {exc}"
    except Exception as exc:  # one bad page must never sink the document
        return "", None, f"deepreef-ocr page failed: {type(exc).__name__}: {exc}"


def extract_pages_lambda(
    images: dict[int, bytes],
    *,
    function_name: str,
    region: str,
    max_workers: int = 16,
    timeout: float = 60,
    client=None,
) -> dict[int, PageResult]:
    """OCR every page concurrently. Returns page -> (text, confidence, error);
    error is "" on success. Never raises for a single page's failure."""
    if not images:
        return {}
    if client is None:
        # One shared client: boto3 clients are thread-safe, and the pool must
        # be at least max_workers wide or the fan-out serialises on sockets.
        client = boto3.client(
            "lambda",
            region_name=region,
            config=BotoConfig(
                max_pool_connections=max_workers,
                connect_timeout=timeout,
                read_timeout=timeout,
                retries={"max_attempts": 2},
            ),
        )
    workers = max(1, min(max_workers, len(images)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            page: pool.submit(_invoke_page, client, function_name, image)
            for page, image in images.items()
        }
        return {page: fut.result() for page, fut in futures.items()}
