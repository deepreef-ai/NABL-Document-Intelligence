from functools import lru_cache

from app.config import Settings, get_settings
from app.llm.chain import LlmChain
from app.llm.providers import GeminiProvider, GroqProvider, NovaProvider

# Each builder returns the providers for one NAME in the order string, and an
# empty list when that provider isn't configured — so an unconfigured name is
# skipped rather than producing a provider that fails on every call.
#
# A builder returns a LIST because one name can expand to several providers:
# Gemini and Groq free-tier quotas are per-key (MEASURED 2026-09-04: Gemini
# allows 20 requests/day per project per model), so several keys are several
# allowances. Each key becomes its own link — gemini, gemini-2, gemini-3 —
# and LlmChain moves to the next when one is exhausted, which is exactly the
# fallthrough it already does for a failing provider.
#
# Nova is gated on nova_model (an inference-profile ID) rather than an API key
# because it authenticates via the ambient AWS credential chain (see
# NovaProvider's docstring); the others are normal keyed REST providers.


def _numbered(base: str, index: int) -> str:
    """First key keeps the plain name so logs/metrics stay recognisable when
    only one key is configured (the common case)."""
    return base if index == 0 else f"{base}-{index + 1}"


def _nova(s: Settings) -> list:
    if not s.nova_model:
        return []
    return [NovaProvider(s.nova_model, s.nova_region, s.llm_timeout_seconds, max_tokens=s.nova_max_tokens)]


def _gemini(s: Settings) -> list:
    return [
        GeminiProvider(
            key, s.gemini_model, s.llm_timeout_seconds,
            name=_numbered("gemini", i), max_tokens=s.gemini_max_tokens,
        )
        for i, key in enumerate(s.gemini_api_key_list)
    ]


def _groq(s: Settings) -> list:
    return [
        GroqProvider(
            key, s.groq_model, s.llm_timeout_seconds,
            name=_numbered("groq", i), max_tokens=s.groq_max_tokens,
        )
        for i, key in enumerate(s.groq_api_key_list)
    ]


_BUILDERS = {"nova": _nova, "gemini": _gemini, "groq": _groq}


def _build_chain(settings: Settings, order: str) -> LlmChain:
    names = [name.strip() for name in order.split(",") if name.strip()]
    providers = []
    for name in names:
        builder = _BUILDERS.get(name)
        if builder:
            providers.extend(builder(settings))
    return LlmChain(providers)


@lru_cache
def get_llm_chain() -> LlmChain:
    settings = get_settings()
    return _build_chain(settings, settings.llm_provider_order)


@lru_cache
def get_chunked_extraction_chain() -> LlmChain:
    """documents/extractor.py's whole-form, page-by-page extraction uses this
    instead of get_llm_chain() — kept as a separate setting/cache entry in
    case a many-section whole-form run ever needs a different provider order
    than everything else (see config.py's chunked_extraction_provider_order)."""
    settings = get_settings()
    return _build_chain(settings, settings.chunked_extraction_provider_order)
