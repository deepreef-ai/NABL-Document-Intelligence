from functools import lru_cache

from app.config import Settings, get_settings
from app.llm.chain import LlmChain
from app.llm.providers import GeminiProvider, NovaProvider

# Each builder returns None when that provider isn't configured, so an
# unconfigured name in the order string is skipped rather than producing a
# provider that fails on every call. Nova is gated on nova_model (an
# inference-profile ID) rather than an API key because it authenticates via
# the ambient AWS credential chain (see NovaProvider's docstring); Gemini is
# a normal keyed REST provider, so it's gated on its key.
_BUILDERS = {
    "nova": lambda s: NovaProvider(
        s.nova_model, s.nova_region, s.llm_timeout_seconds, max_tokens=s.nova_max_tokens,
    ) if s.nova_model else None,
    "gemini": lambda s: GeminiProvider(
        s.gemini_api_key, s.gemini_model, s.llm_timeout_seconds, max_tokens=s.gemini_max_tokens,
    ) if s.gemini_api_key else None,
}


def _build_chain(settings: Settings, order: str) -> LlmChain:
    names = [name.strip() for name in order.split(",") if name.strip()]
    providers = []
    for name in names:
        builder = _BUILDERS.get(name)
        provider = builder(settings) if builder else None
        if provider is not None:
            providers.append(provider)
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
