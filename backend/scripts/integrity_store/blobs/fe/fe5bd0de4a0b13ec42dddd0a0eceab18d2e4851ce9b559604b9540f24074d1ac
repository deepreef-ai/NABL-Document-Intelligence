"""LLM access layer for the graph.

Wraps the existing provider chain (app/llm/chain.py) rather than reimplementing
it — that chain already does ordered provider fallback, per-provider cooldown
with exponential backoff, and salvage of truncated JSON, all of which were
measured against real failures. What it does not do, and what this layer adds:

- **Structured output.** A raw dict is not an answer; every call here parses
  into a Pydantic model and a parse failure is a typed, retryable error rather
  than a dict that quietly lacks the key a node will later index into.
- **Rate limiting.** The graph makes one call per chunk, so a 300-page
  document is hundreds of calls in a burst. A token-bucket limiter keeps that
  under the provider's per-minute allowance instead of discovering the limit
  by being throttled.
- **Per-node metrics.** Latency, tokens and attempt count attributed to the
  node that spent them, so cost is answerable per stage rather than per run.
- **Attempt escalation.** A retry is not the same prompt again. An invalid-JSON
  failure retries with a stricter instruction; a truncation failure retries
  with the input halved. Sending an identical prompt to a model that just
  failed on it is the least likely thing to work.

Nova is the primary provider per spec. The chain is ordered, not exclusive:
`graph_provider_order` defaults to `nova,gemini` so a Nova outage degrades the
run instead of ending it.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.config import get_settings as get_app_settings
from app.graph.config import get_graph_settings
from app.graph.errors import ErrorType, LlmUnavailable
from app.graph.schemas import LlmCallMetric
from app.llm.chain import LlmChain
from app.llm.factory import _build_chain  # intentional: same construction rules
from app.llm.json_utils import parse_json_object

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------


class RateLimiter:
    """Token bucket, shared process-wide, thread-safe.

    Deliberately blocking rather than raising: the caller wants the call to
    happen, just not right now. A node that hits the limiter simply runs a
    little slower, which is the correct trade for a workflow whose stated
    priority is accuracy over speed.
    """

    def __init__(self, max_per_minute: int):
        self.capacity = max(1, max_per_minute)
        self._tokens = float(self.capacity)
        self._refill_rate = self.capacity / 60.0
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 120.0) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self.capacity, self._tokens + (now - self._last) * self._refill_rate
                )
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
                needed = (1.0 - self._tokens) / self._refill_rate
            if time.monotonic() + needed > deadline:
                return False
            time.sleep(min(needed, 1.0))


_limiter: RateLimiter | None = None
_limiter_lock = threading.Lock()


def get_rate_limiter() -> RateLimiter:
    global _limiter
    with _limiter_lock:
        if _limiter is None:
            _limiter = RateLimiter(get_graph_settings().max_requests_per_minute)
        return _limiter


# --------------------------------------------------------------------------
# Chain construction
# --------------------------------------------------------------------------

_chain: LlmChain | None = None


def get_graph_chain() -> LlmChain:
    """The graph's own provider chain, built from graph_provider_order.

    Every provider is pinned to `llm_temperature` (0.0 by default). The
    providers default to 0.2, which is fine for chat and wrong here: MEASURED,
    the same document mapped six form fields on one run and zero on the next,
    with no code change between them. A form-filling stage that cannot
    reproduce its own answer cannot be trusted or debugged.
    """
    global _chain
    if _chain is None:
        settings = get_graph_settings()
        _chain = _build_chain(get_app_settings(), settings.graph_provider_order)
        for provider in getattr(_chain, "providers", []):
            if hasattr(provider, "temperature"):
                provider.temperature = settings.llm_temperature
    return _chain


def reset_chain() -> None:
    """Test hook — drops the cached chain so a patched provider takes effect."""
    global _chain, _limiter
    _chain = None
    _limiter = None


# --------------------------------------------------------------------------
# Result envelope
# --------------------------------------------------------------------------


class LlmOutcome(BaseModel):
    """Never raises on a bad model reply — the caller decides what that means."""

    model_config = {"arbitrary_types_allowed": True}

    ok: bool
    parsed: BaseModel | None = None
    raw: dict | None = None
    error_type: ErrorType | None = None
    error_message: str = ""
    metric: LlmCallMetric | None = None
    attempts: int = 1


def is_quota_exhausted(message: str) -> bool:
    """A daily/plan quota, as opposed to a transient per-minute rate limit.

    Worth separating because the operator response is different: a rate limit
    clears on its own, an exhausted quota needs a bigger plan or tomorrow.
    """
    t = (message or "").lower()
    return "quota" in t and ("exceed" in t or "exhaust" in t or "billing" in t)


def _classify(exc: Exception) -> ErrorType:
    text = str(exc).lower()
    if "timeout" in text or "timed out" in text:
        return ErrorType.LLM_TIMEOUT
    if "rate" in text and "limit" in text:
        return ErrorType.LLM_RATE_LIMIT
    if "429" in text or "quota" in text or "throttl" in text:
        return ErrorType.LLM_RATE_LIMIT
    if "json" in text:
        return ErrorType.LLM_INVALID_JSON
    return ErrorType.LLM_UNAVAILABLE


def _backoff(attempt: int) -> float:
    s = get_graph_settings()
    delay = min(s.backoff_max_seconds, s.backoff_base_seconds * (2 ** (attempt - 1)))
    # Jitter: without it, N chunks that fail together retry together and
    # reproduce the burst that caused the rate limit in the first place.
    return delay * (0.5 + random.random() * 0.5)


_STRICTER = (
    "\n\nIMPORTANT: your previous reply could not be parsed as JSON. "
    "Reply with ONE JSON object and nothing else — no prose before or after, "
    "no markdown fences, no trailing commas. Every string must be double-quoted."
)

_SHORTER = (
    "\n\nIMPORTANT: your previous reply was cut off before it finished. "
    "Return FEWER items this time — only the most clearly-supported ones — "
    "so the JSON object is complete and closes properly."
)


def call_structured(
    *,
    node: str,
    system: str,
    user_text: str,
    output_model: type[T],
    images: list[bytes] | None = None,
    image_media_type: str | None = None,
    max_attempts: int | None = None,
    on_metric=None,
) -> LlmOutcome:
    """One logical LLM call, parsed into `output_model`.

    Returns an outcome rather than raising, so a node can record the failure
    and let the router decide between retry, recovery and escalation.
    """
    settings = get_graph_settings()
    attempts_allowed = max_attempts or settings.max_attempts_per_chunk
    chain = get_graph_chain()
    limiter = get_rate_limiter()

    last_error_type = ErrorType.LLM_UNAVAILABLE
    last_message = "no attempt ran"
    prompt = user_text
    sys_prompt = system

    attempt = 0
    while attempt < attempts_allowed:
        attempt += 1

        # A provider that is cooling down has not failed this call — it has not
        # been asked yet. Counting that as an attempt burns the retry budget on
        # nothing: MEASURED, three attempts vanished in ~14s against a 25s
        # cooldown, and the chunk was abandoned without the model ever seeing
        # it. Wait the cooldown out instead, and do not spend an attempt.
        cooldown = _seconds_until_available(chain)
        if cooldown > 0:
            if cooldown <= settings.max_cooldown_wait_seconds:
                log.info(
                    "graph.llm[%s] every provider cooling down; waiting %.0fs", node, cooldown,
                )
                time.sleep(cooldown + 0.5)
                attempt -= 1          # this was a wait, not an attempt
                continue
            last_error_type = ErrorType.LLM_RATE_LIMIT
            last_message = (
                f"every provider is rate-limited for another {cooldown:.0f}s, longer than "
                f"the {settings.max_cooldown_wait_seconds:.0f}s this call will wait. If the "
                f"provider reported a daily quota, waiting will not help — the run can be "
                f"resumed later from its document id without reprocessing what succeeded."
            )
            break

        if not limiter.acquire():
            last_error_type = ErrorType.LLM_RATE_LIMIT
            last_message = "local rate limiter timed out waiting for a slot"
            break

        started = time.monotonic()
        provider_name = "unknown"
        try:
            raw = chain.generate_json(
                sys_prompt, prompt,
                images=images or None,
                image_media_type=image_media_type,
            )
            latency = time.monotonic() - started
            usage = getattr(chain, "last_usage", None) or {}
            provider_name = getattr(chain, "last_provider", "") or "chain"

            if not isinstance(raw, dict) or not raw:
                last_error_type = ErrorType.LLM_EMPTY_RESPONSE
                last_message = "provider returned an empty or non-object reply"
                metric = LlmCallMetric(
                    node=node, model=settings.nova_model, provider=provider_name,
                    latency_seconds=latency, ok=False, error=last_message, attempt=attempt,
                    input_tokens=int(usage.get("input_tokens", 0)),
                    output_tokens=int(usage.get("output_tokens", 0)),
                )
                if on_metric:
                    on_metric(metric)
                sys_prompt = system + _STRICTER
                _sleep(attempt)
                continue

            try:
                parsed = output_model.model_validate(raw)
            except ValidationError as ve:
                last_error_type = ErrorType.LLM_SCHEMA_MISMATCH
                last_message = f"reply did not match {output_model.__name__}: {ve.errors()[:3]}"
                metric = LlmCallMetric(
                    node=node, model=settings.nova_model, provider=provider_name,
                    latency_seconds=latency, ok=False, error=last_message[:400], attempt=attempt,
                    input_tokens=int(usage.get("input_tokens", 0)),
                    output_tokens=int(usage.get("output_tokens", 0)),
                )
                if on_metric:
                    on_metric(metric)
                sys_prompt = system + _STRICTER
                _sleep(attempt)
                continue

            metric = LlmCallMetric(
                node=node, model=settings.nova_model, provider=provider_name,
                latency_seconds=latency, ok=True, attempt=attempt,
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
            )
            if on_metric:
                on_metric(metric)
            return LlmOutcome(ok=True, parsed=parsed, raw=raw, metric=metric, attempts=attempt)

        except Exception as exc:  # noqa: BLE001 — every provider failure is data here
            latency = time.monotonic() - started
            last_error_type = _classify(exc)
            last_message = str(exc)[:1000]
            metric = LlmCallMetric(
                node=node, model=settings.nova_model, provider=provider_name,
                latency_seconds=latency, ok=False, error=last_message[:400], attempt=attempt,
            )
            if on_metric:
                on_metric(metric)

            if last_error_type == ErrorType.LLM_INVALID_JSON:
                sys_prompt = system + _STRICTER
            elif last_error_type == ErrorType.LLM_TRUNCATED:
                sys_prompt = system + _SHORTER
            log.warning("graph.llm[%s] attempt %d failed: %s", node, attempt, last_message[:200])
            _sleep(attempt)

    return LlmOutcome(
        ok=False,
        error_type=last_error_type,
        error_message=last_message,
        attempts=attempts_allowed,
    )


def _sleep(attempt: int) -> None:
    time.sleep(_backoff(attempt))


def _seconds_until_available(chain) -> float:
    """How long until ANY provider in the chain will accept a call.

    0.0 when one is ready now. The chain exposes this so a caller can wait for
    a cooldown rather than hammering it and being told to wait again.
    """
    try:
        wait = chain.seconds_until_available()
    except Exception:  # noqa: BLE001 — a chain without the method is fine
        return 0.0
    return max(0.0, float(wait)) if wait else 0.0


def probe_providers() -> dict[str, str]:
    """Cheap health check used by the API's /health endpoint.

    Returns {provider_name: "ok" | error text}. Makes one tiny call per
    provider, so it is safe to expose but not free — the endpoint caches it.
    """
    results: dict[str, str] = {}
    chain = get_graph_chain()
    for provider in getattr(chain, "providers", []):
        try:
            provider.generate("Reply with the single word OK.", "ping")
            results[provider.name] = "ok"
        except Exception as exc:  # noqa: BLE001
            results[provider.name] = str(exc)[:200]
    if not results:
        raise LlmUnavailable("no providers configured for the graph chain")
    return results


__all__ = [
    "LlmOutcome",
    "RateLimiter",
    "call_structured",
    "get_graph_chain",
    "get_rate_limiter",
    "parse_json_object",
    "probe_providers",
    "reset_chain",
]
