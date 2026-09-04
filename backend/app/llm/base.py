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


def classify_boto_error(name: str, exc: Exception) -> LlmProviderError:
    """Bedrock/botocore raise a ClientError carrying an AWS error CODE rather
    than an HTTP status, so the mapping is code-based. No secrets to redact
    here: Nova (see providers.py's NovaProvider) authenticates via the
    ambient AWS credential chain, never a key embedded in the request or its
    error text."""
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


def classify_http_error(name: str, status: int, body: str) -> LlmProviderError:
    """Same mapping as classify_boto_error, for a REST provider. `body` is the
    server's response text — it is included because a 400 from these APIs
    usually explains what about the request was rejected, which is otherwise
    invisible. The caller MUST NOT pass anything containing the API key: keyed
    providers put the key in the query string or a header, so build error text
    from the status and body only, never from the request URL."""
    if status == 429:
        return LlmRateLimitError(f"{name} rate-limited (429): {body}")
    if status in (401, 403):
        return LlmAuthError(f"{name} auth error ({status}): {body}")
    if status == 402:
        return LlmQuotaError(f"{name} quota/billing error (402): {body}")
    if status == 400:
        return LlmBadRequestError(f"{name} bad request (400): {body}")
    if status in (408, 504):
        return LlmTimeoutError(f"{name} timed out ({status}): {body}")
    if status in (503, 529):
        # "This model is currently experiencing high demand" — transient
        # capacity, not a broken request. Same treatment as a 429 so the
        # router backs off and a later document in the same batch can
        # succeed, instead of the whole run dying on a passing spike.
        return LlmRateLimitError(f"{name} temporarily unavailable ({status}): {body}")
    return LlmProviderError(f"{name} call failed ({status}): {body}")


class LlmProvider:
    name: str

    def generate(
        self,
        system: str,
        user_text: str,
        image: bytes | None = None,
        image_media_type: str | None = None,
        want_json: bool = False,
    ) -> str:
        """Return the raw text reply. Callers ask for JSON in the prompt and
        parse it themselves (see llm/json_utils.py) — a plain text-in/text-out
        contract that would let a future second provider join the chain
        without per-provider branching at any call site.

        `want_json` is a per-call hint, not a provider-level constant: a plain
        chat reply (LlmChain.generate_text) must NOT force JSON mode."""
        raise NotImplementedError
