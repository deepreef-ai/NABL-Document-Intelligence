import httpx


class LlmProviderError(RuntimeError):
    """This one provider failed to answer — the chain should try the next one."""


class LlmNotConfigured(RuntimeError):
    """No provider in the chain has an API key configured."""


class LlmRateLimitError(LlmProviderError):
    """The provider answered with 429 — too many requests right now. Transient;
    the router backs off exponentially and retries later rather than treating
    this as a permanent outage."""


class LlmAuthError(LlmProviderError):
    """The provider answered with 401/403 — the API key is missing, wrong, or
    revoked. Retrying won't help until the credential itself is fixed, so the
    router disables this provider instead of hammering it."""


class LlmQuotaError(LlmProviderError):
    """The provider answered with 402 — a billing/credits problem on that
    account. Same treatment as LlmAuthError: disable, don't retry."""


class LlmBadRequestError(LlmProviderError):
    """The provider answered with 400 — this specific request was malformed
    (an unsupported parameter, a prompt that tripped a provider-specific
    validation rule). Not evidence the provider itself is down, so the router
    does NOT back off or disable it for future calls — it just moves on to
    the next provider for this one call."""


class LlmTimeoutError(LlmProviderError):
    """The provider didn't respond within the configured timeout."""


_REDACTED = "[REDACTED]"


def redact(text: str, *secrets: str) -> str:
    """Strip any of `secrets` (e.g. an API key) out of `text` before it can
    reach a log line, a stored Document.error, or a frontend-visible error
    message. Silently skips falsy secrets (an unset key) rather than erroring."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, _REDACTED)
    return text


def redact_known_secrets(text: str) -> str:
    """Last-resort safety net: scrub every currently configured provider API
    key out of `text`. Each provider already redacts its own key at the exact
    point of failure (see classify_http_error below) — this exists for
    anywhere else an exception's str() might end up in front of a user (e.g.
    Document.error) without having to thread every provider's key through by
    hand. Local import to avoid a module-load-order dependency on app.config."""
    from app.config import get_settings

    settings = get_settings()
    secrets = (
        settings.gemini_api_key,
        settings.groq_api_key,
        settings.huggingface_api_key,
        settings.cerebras_api_key,
        settings.openrouter_api_key,
    )
    return redact(text, *secrets)


def classify_http_error(name: str, exc: httpx.HTTPError, *secrets: str) -> LlmProviderError:
    """Turn an httpx exception into the right typed LlmProviderError subclass
    based on its HTTP status code (if any), with every given secret scrubbed
    out of the message first. Shared by every provider in llm/providers.py so
    the status-code-to-behavior mapping lives in exactly one place."""
    message = redact(str(exc), *secrets)
    if isinstance(exc, httpx.TimeoutException):
        return LlmTimeoutError(f"{name} timed out: {message}")
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 429:
            return LlmRateLimitError(f"{name} rate-limited (429): {message}")
        if status == 402:
            return LlmQuotaError(f"{name} billing/quota error (402): {message}")
        if status in (401, 403):
            return LlmAuthError(f"{name} auth error ({status}): {message}")
        if status == 400:
            return LlmBadRequestError(f"{name} bad request (400): {message}")
    return LlmProviderError(f"{name} call failed: {message}")


def classify_boto_error(name: str, exc: Exception) -> LlmProviderError:
    """Bedrock's equivalent of classify_http_error above — boto3/botocore
    raise a ClientError carrying an AWS error CODE rather than an HTTP
    status, so the mapping is code-based instead. No secrets to redact here:
    Nova (see providers.py's NovaProvider) authenticates via the ambient AWS
    credential chain, never a key embedded in the request or its error text."""
    from botocore.exceptions import ClientError, ConnectTimeoutError, ReadTimeoutError

    message = str(exc)
    if isinstance(exc, (ReadTimeoutError, ConnectTimeoutError)):
        return LlmTimeoutError(f"{name} timed out: {message}")
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("ThrottlingException", "TooManyRequestsException"):
            return LlmRateLimitError(f"{name} rate-limited ({code}): {message}")
        if code in ("AccessDeniedException", "UnrecognizedClientException", "UnauthorizedException"):
            return LlmAuthError(f"{name} auth error ({code}): {message}")
        if code == "ValidationException":
            return LlmBadRequestError(f"{name} bad request ({code}): {message}")
        if code == "ModelTimeoutException":
            return LlmTimeoutError(f"{name} timed out ({code}): {message}")
    return LlmProviderError(f"{name} call failed: {message}")


class LlmProvider:
    name: str

    def generate(
        self,
        system: str,
        user_text: str,
        image: bytes | None = None,
        image_media_type: str | None = None,
        want_json: bool = False,
        images: list[bytes] | None = None,
    ) -> str:
        """Return the raw text reply. Callers ask for JSON in the prompt and
        parse it themselves (see llm/json_utils.py) — a plain text-in/text-out
        contract is the one thing every provider (Gemini, Groq, HF router,
        anything OpenAI-chat-shaped) can do the same way, which is what makes
        a fallback chain across them possible without per-provider branching
        at every call site.

        `want_json` is a per-call hint, not a provider-level constant: a plain
        chat reply (LlmChain.generate_text) must NOT force JSON mode — some
        providers (Groq/OpenAI-shaped `response_format: json_object`) reject
        the request with a 400 if the prompt doesn't itself mention "json".

        `image` is the single-image case (one scanned page, one photo);
        `images` carries several pages of the SAME document in order, for a
        multi-page scan. A provider that can only take one image per request
        must RAISE for a multi-image call rather than quietly sending the
        first page — silently dropping pages is precisely the bug this
        parameter exists to fix. Pass one or the other, not both."""
        raise NotImplementedError
