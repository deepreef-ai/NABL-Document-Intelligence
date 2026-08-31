import logging
import math
import time
from dataclasses import dataclass

from app.llm.base import (
    LlmAuthError,
    LlmNotConfigured,
    LlmProvider,
    LlmProviderError,
    LlmQuotaError,
    LlmRateLimitError,
    LlmTimeoutError,
)
from app.llm.json_utils import parse_json_object

log = logging.getLogger(__name__)

_NOT_CONFIGURED_MESSAGE = (
    "No LLM provider is configured — set at least one of GEMINI_API_KEY, "
    "GROQ_API_KEY, HUGGINGFACE_API_KEY in backend/.env."
)

# 429 / timeout backoff: doubles each consecutive failure, capped. Kept as
# separate bases (a rate limit usually clears faster than a struggling
# provider recovers from timing out) but share one cap and one growth curve.
_RATE_LIMIT_BASE_SECONDS = 5.0
_TIMEOUT_BASE_SECONDS = 15.0
_BACKOFF_CAP_SECONDS = 300.0


@dataclass
class _ProviderStatus:
    """Per-provider cooldown state, held for the lifetime of one LlmChain
    instance — which is itself a process-lifetime singleton (see
    llm/factory.py's @lru_cache). Not persisted anywhere: a process restart
    clears it, which is fine since this is a live "is this provider currently
    answering" signal, not a record anything else needs to read."""

    available_at: float = 0.0
    consecutive_failures: int = 0
    disabled_reason: str | None = None

    def is_available(self, now: float) -> bool:
        return now >= self.available_at

    def note_success(self) -> None:
        self.available_at = 0.0
        self.consecutive_failures = 0
        self.disabled_reason = None

    def back_off(self, base_seconds: float, now: float) -> None:
        self.consecutive_failures += 1
        delay = min(base_seconds * (2 ** (self.consecutive_failures - 1)), _BACKOFF_CAP_SECONDS)
        self.available_at = now + delay

    def disable(self, reason: str) -> None:
        # Not a timed cooldown: a bad credential or an empty billing account
        # doesn't fix itself on a timer, so this provider is skipped for the
        # rest of the process's life rather than retried at all.
        self.available_at = math.inf
        self.disabled_reason = reason


class LlmChain:
    """Tries each configured provider in order and falls through to the next
    on any failure (network error, rate limit, unexpected response shape, or
    — for generate_json — a reply that isn't parseable JSON). Only raises once
    every available provider has been tried, per 'use all of them, fall back
    if one fails' resilience against free-tier rate limits.

    Also a status-aware router: each provider carries its own cooldown state,
    updated from the *kind* of failure it just had —
      - LlmRateLimitError (HTTP 429): exponential backoff, doubling each
        consecutive hit, capped at 5 minutes. Reset the moment a call to that
        provider succeeds again.
      - LlmTimeoutError: the same idea, tracked with its own counter — a
        rate-limited provider and a slow/unreachable one are different
        failure modes worth backing off independently.
      - LlmQuotaError (402) / LlmAuthError (401/403): disabled outright.
        Retrying a bad credential or an empty billing account on a timer
        doesn't help — nothing changes until a human fixes it.
      - LlmBadRequestError (400), or any other/bare exception (kept for
        backward compatibility with callers that raise a plain Exception):
        no state change. A 400 is almost always something about *this* call
        rather than the provider being down, so it isn't punished on a
        later, different call.
    A provider already known to be cooling down or disabled is skipped up
    front, before spending a network round-trip re-discovering that."""

    def __init__(self, providers: list[LlmProvider]):
        self.providers = providers
        self._status: dict[str, _ProviderStatus] = {p.name: _ProviderStatus() for p in providers}

    def _record_failure(self, provider: LlmProvider, exc: Exception, now: float) -> None:
        status = self._status[provider.name]
        if isinstance(exc, LlmRateLimitError):
            status.back_off(_RATE_LIMIT_BASE_SECONDS, now)
            log.warning("llm chain: %s rate-limited, backing off %.0fs", provider.name, status.available_at - now)
        elif isinstance(exc, LlmTimeoutError):
            status.back_off(_TIMEOUT_BASE_SECONDS, now)
            log.warning("llm chain: %s timed out, backing off %.0fs", provider.name, status.available_at - now)
        elif isinstance(exc, (LlmQuotaError, LlmAuthError)):
            status.disable(type(exc).__name__)
            log.error("llm chain: disabling %s (%s)", provider.name, exc)
        # LlmBadRequestError and anything else: no state change.

    def _usable_providers(self) -> tuple[list[LlmProvider], list[str]]:
        now = time.monotonic()
        usable, skip_notes = [], []
        for p in self.providers:
            status = self._status[p.name]
            if status.is_available(now):
                usable.append(p)
            elif status.disabled_reason:
                skip_notes.append(f"{p.name}: disabled ({status.disabled_reason})")
            else:
                skip_notes.append(f"{p.name}: cooling down for {status.available_at - now:.0f}s more")
        return usable, skip_notes

    def generate_text(
        self, system: str, user_text: str, image: bytes | None = None, image_media_type: str | None = None
    ) -> str:
        if not self.providers:
            raise LlmNotConfigured(_NOT_CONFIGURED_MESSAGE)
        usable, errors = self._usable_providers()
        for provider in usable:
            try:
                result = provider.generate(system, user_text, image, image_media_type)
            except Exception as exc:  # noqa: BLE001 - any failure falls through to the next provider
                self._record_failure(provider, exc, time.monotonic())
                errors.append(f"{provider.name}: {exc}")
                continue
            self._status[provider.name].note_success()
            return result
        raise LlmProviderError("every configured LLM provider failed: " + " | ".join(errors))

    def generate_json(
        self, system: str, user_text: str, image: bytes | None = None, image_media_type: str | None = None
    ) -> dict:
        if not self.providers:
            raise LlmNotConfigured(_NOT_CONFIGURED_MESSAGE)
        usable, errors = self._usable_providers()
        for provider in usable:
            try:
                raw = provider.generate(system, user_text, image, image_media_type, want_json=True)
                result = parse_json_object(raw)
            except Exception as exc:  # noqa: BLE001
                self._record_failure(provider, exc, time.monotonic())
                errors.append(f"{provider.name}: {exc}")
                continue
            self._status[provider.name].note_success()
            return result
        raise LlmProviderError("every configured LLM provider failed: " + " | ".join(errors))
