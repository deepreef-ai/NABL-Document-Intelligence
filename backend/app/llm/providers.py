import base64
import time

import boto3
import httpx
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from app.llm.base import (
    LlmProvider,
    LlmProviderError,
    LlmRateLimitError,
    LlmTimeoutError,
    classify_boto_error,
    classify_http_error,
)


class NovaProvider(LlmProvider):
    """Amazon Nova on AWS Bedrock, via the Converse API — boto3, not httpx.
    Authenticates via the ambient AWS credential chain (same one
    documents/ocr_client.py already relies on for the OCR Lambda) — no
    separate API key setting. The only LLM provider in this app: see
    config.py's llm_provider_order/chunked_extraction_provider_order.

    `model` MUST be a region-prefixed inference-profile ID (e.g.
    "us.amazon.nova-2-lite-v1:0"), not the bare model ID
    ("amazon.nova-2-lite-v1:0") — MEASURED 2026-09-02: Bedrock rejects
    on-demand invocation of this model generation by its bare ID with
    ValidationException ("Retry your request with the ID or ARN of an
    inference profile that contains this model")."""

    def __init__(self, model: str, region: str, timeout: float, name: str = "nova", max_tokens: int = 8192):
        self.name = name
        self.model = model
        self.region = region
        self.timeout = timeout
        # MEASURED 2026-09-02: with no maxTokens set, Bedrock's own default
        # output cap silently truncated a dense document's JSON reply
        # mid-object ("...\"voice\": \"(800) 634-" — cut off, not a real
        # value), which then failed to parse as JSON at all. A document
        # with a genuinely long results table (hundreds of test rows) needs
        # real headroom here, not just enough for a short chat reply.
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        # Constructed lazily (not at __init__ time) so building the provider
        # object itself never touches the network/credential chain — only
        # the first real generate() call does, consistent with how a
        # missing/misconfigured provider elsewhere in this chain only fails
        # at call time, not at startup.
        if self._client is None:
            config = BotoConfig(connect_timeout=self.timeout, read_timeout=self.timeout)
            self._client = boto3.client("bedrock-runtime", region_name=self.region, config=config)
        return self._client

    def generate(
        self,
        system: str,
        user_text: str,
        image: bytes | None = None,
        image_media_type: str | None = None,
        want_json: bool = False,
    ) -> str:
        content: list[dict] = [{"text": user_text}]
        if image is not None:
            image_format = (image_media_type or "image/png").split("/")[-1]
            if image_format == "jpg":
                image_format = "jpeg"
            content.append({"image": {"format": image_format, "source": {"bytes": image}}})

        try:
            response = self._get_client().converse(
                modelId=self.model,
                system=[{"text": system}],
                messages=[{"role": "user", "content": content}],
                inferenceConfig={"temperature": 0.2, "maxTokens": self.max_tokens},
            )
        except (BotoCoreError, ClientError) as exc:
            raise classify_boto_error(self.name, exc) from exc

        try:
            parts = response["output"]["message"]["content"]
            return "".join(part.get("text", "") for part in parts)
        except (KeyError, IndexError) as exc:
            raise LlmProviderError(f"{self.name} returned an unexpected response shape: {response}") from exc


class GeminiProvider(LlmProvider):
    """Google Gemini via the REST API (httpx, already a dependency — no
    google-generativeai SDK needed for one endpoint).

    Added as a SECOND provider so the accuracy benchmark can keep running when
    Bedrock is unavailable — MEASURED 2026-09-03/04: every Bedrock model, on
    both Converse and InvokeModel, began returning
    ValidationException("Operation not allowed") account-wide while the
    credentials themselves stayed valid. LlmChain already falls through to the
    next provider on failure, so listing this after "nova" in
    LLM_PROVIDER_ORDER makes that outage a degradation instead of a stoppage.

    Chosen over Groq specifically because documents/pipeline.py and
    documents/app.py send the page IMAGE alongside the OCR text (that pairing
    is what fixed the redacted-field hallucination), and Gemini's vision
    support covers that directly.

    NOTE the benchmark caveat: a Gemini run measures GEMINI. It does not
    validate a prompt change for Nova — only a Nova run does that.
    """

    _ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    # Retries for 429/503 only. 3 attempts with 10s then 20s waits covers a
    # passing capacity spike without stalling a run for minutes.
    _MAX_ATTEMPTS = 3
    _RETRY_WAIT_SECONDS = 10.0

    def __init__(self, api_key: str, model: str, timeout: float, name: str = "gemini", max_tokens: int = 8192):
        self.name = name
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens

    def generate(
        self,
        system: str,
        user_text: str,
        image: bytes | None = None,
        image_media_type: str | None = None,
        want_json: bool = False,
    ) -> str:
        parts: list[dict] = [{"text": user_text}]
        if image is not None:
            parts.append({
                "inline_data": {
                    "mime_type": image_media_type or "image/png",
                    "data": base64.b64encode(image).decode("ascii"),
                }
            })

        generation_config: dict = {"temperature": 0.2, "maxOutputTokens": self.max_tokens}
        if want_json:
            # Gemini's own JSON mode. json_utils.parse_json_object still runs on
            # the result — same defensive contract every provider has, since a
            # truncated reply is still unparseable however it was requested.
            generation_config["responseMimeType"] = "application/json"

        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config,
        }

        # Transient capacity (503 "experiencing high demand") and rate limits
        # are absorbed HERE rather than reported to the chain, because the
        # chain's cooldown is per-provider and persists across calls: in a
        # sequential batch run one spike would otherwise put the only provider
        # into a growing cooldown and every later document would be SKIPPED
        # without an attempt. MEASURED 2026-09-04: that cascade took a 51-
        # document run down to 9 successes. Only a still-failing provider after
        # these retries is escalated, which is genuinely worth cooling down for.
        last: LlmProviderError | None = None
        for attempt in range(self._MAX_ATTEMPTS):
            try:
                # Key goes in a header, never the URL — an error message built
                # from a URL would otherwise leak it (see classify_http_error).
                response = httpx.post(
                    self._ENDPOINT.format(model=self.model),
                    json=payload,
                    headers={"x-goog-api-key": self.api_key},
                    timeout=self.timeout,
                )
            except httpx.TimeoutException as exc:
                raise LlmTimeoutError(f"{self.name} timed out after {self.timeout}s") from exc
            except httpx.HTTPError as exc:
                raise LlmProviderError(f"{self.name} request failed: {type(exc).__name__}") from exc

            if response.status_code == 200:
                break
            error = classify_http_error(self.name, response.status_code, response.text[:400])
            if not isinstance(error, LlmRateLimitError) or attempt == self._MAX_ATTEMPTS - 1:
                raise error
            last = error
            time.sleep(self._RETRY_WAIT_SECONDS * (attempt + 1))
        else:  # pragma: no cover - loop always breaks or raises
            raise last or LlmProviderError(f"{self.name} exhausted retries")

        try:
            data = response.json()
            candidate = data["candidates"][0]
            # A reply cut off by the token cap still has parts worth parsing;
            # surface the reason instead when there is nothing at all, so the
            # failure reads as "truncated" rather than "unexpected shape".
            parts_out = candidate.get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts_out)
            if not text:
                raise LlmProviderError(
                    f"{self.name} returned no text (finishReason="
                    f"{candidate.get('finishReason')!r})"
                )
            return text
        except LlmProviderError:
            raise
        except (KeyError, IndexError, ValueError) as exc:
            raise LlmProviderError(f"{self.name} returned an unexpected response shape: {response.text[:300]}") from exc
