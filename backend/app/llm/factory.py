from functools import lru_cache

from app.config import Settings, get_settings
from app.llm.chain import LlmChain
from app.llm.providers import GeminiProvider, OllamaProvider, OpenAiCompatibleProvider

# Only Gemini gets vision — see GeminiProvider's docstring for why. Groq, the
# HF router, and Ollama are text-only here; a document routed to the vision
# fallback (documents/pipeline.py) simply skips straight past them to the
# next provider that supports images, or to Gemini if it's earlier in the
# order. Ollama has no API key to gate on — a missing/unreachable local
# server just fails fast and the chain falls through, same as any other
# provider failure.
_BUILDERS = {
    "gemini": lambda s: GeminiProvider(s.gemini_api_key, s.gemini_model, s.llm_timeout_seconds) if s.gemini_api_key else None,
    "groq": lambda s: OpenAiCompatibleProvider(
        "groq", "https://api.groq.com/openai/v1", s.groq_api_key, s.groq_model, s.llm_timeout_seconds,
        supports_vision=False, supports_json_mode=True,
    ) if s.groq_api_key else None,
    "huggingface": lambda s: OpenAiCompatibleProvider(
        "huggingface", "https://router.huggingface.co/v1", s.huggingface_api_key, s.huggingface_model, s.llm_timeout_seconds,
        supports_vision=False, supports_json_mode=False,
    ) if s.huggingface_api_key else None,
    "cerebras": lambda s: OpenAiCompatibleProvider(
        "cerebras", "https://api.cerebras.ai/v1", s.cerebras_api_key, s.cerebras_model, s.llm_timeout_seconds,
        supports_vision=False, supports_json_mode=True,
    ) if s.cerebras_api_key else None,
    "openrouter": lambda s: OpenAiCompatibleProvider(
        "openrouter", "https://openrouter.ai/api/v1", s.openrouter_api_key, s.openrouter_model, s.llm_timeout_seconds,
        supports_vision=False, supports_json_mode=True,
    ) if s.openrouter_api_key else None,
    # Extra free-tier OpenAI-compatible providers — same class, different
    # base URL/model, each independently optional. See config.py's comment
    # on why model names here are defaults to verify, not guarantees.
    "sambanova": lambda s: OpenAiCompatibleProvider(
        "sambanova", "https://api.sambanova.ai/v1", s.sambanova_api_key, s.sambanova_model, s.llm_timeout_seconds,
        supports_vision=False, supports_json_mode=True,
    ) if s.sambanova_api_key else None,
    "mistral": lambda s: OpenAiCompatibleProvider(
        "mistral", "https://api.mistral.ai/v1", s.mistral_api_key, s.mistral_model, s.llm_timeout_seconds,
        supports_vision=False, supports_json_mode=True,
    ) if s.mistral_api_key else None,
    "nvidia": lambda s: OpenAiCompatibleProvider(
        "nvidia", "https://integrate.api.nvidia.com/v1", s.nvidia_api_key, s.nvidia_model, s.llm_timeout_seconds,
        # Conservative: NIM hosts many different underlying models and JSON
        # mode support isn't uniform across them the way it is for a single
        # vendor's own API — a wrong "True" here isn't dangerous (a 400 just
        # falls through to the next provider without disabling this one, see
        # llm/chain.py), but there's no strong reason to assume "yes" either.
        supports_vision=False, supports_json_mode=False,
    ) if s.nvidia_api_key else None,
    "github": lambda s: OpenAiCompatibleProvider(
        "github", "https://models.github.ai/inference", s.github_token, s.github_model, s.llm_timeout_seconds,
        supports_vision=False, supports_json_mode=False,
    ) if s.github_token else None,
    # Key rotation: a second/third key for a provider already above is a
    # separate quota, not a separate provider — same class/base_url/model
    # settings, just a different credential and a distinct chain-order name
    # (see GeminiProvider's `name` param and config.py's comment) so
    # llm/chain.py's per-provider cooldown tracks each key independently.
    "gemini_2": lambda s: GeminiProvider(s.gemini_api_key_2, s.gemini_model, s.llm_timeout_seconds, name="gemini_2") if s.gemini_api_key_2 else None,
    "gemini_3": lambda s: GeminiProvider(s.gemini_api_key_3, s.gemini_model, s.llm_timeout_seconds, name="gemini_3") if s.gemini_api_key_3 else None,
    "groq_2": lambda s: OpenAiCompatibleProvider(
        "groq_2", "https://api.groq.com/openai/v1", s.groq_api_key_2, s.groq_model, s.llm_timeout_seconds,
        supports_vision=False, supports_json_mode=True,
    ) if s.groq_api_key_2 else None,
    "ollama": lambda s: OllamaProvider(
        s.ollama_base_url, s.ollama_model, s.ollama_timeout_seconds,
        num_thread=s.ollama_num_thread, num_predict=s.ollama_num_predict,
    ),
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
    instead of get_llm_chain() — Ollama first, since a 15+ page document is
    exactly the case that hits the cloud providers' rate limits and hard
    request-size caps hardest (see config.py's chunked_extraction_provider_order)."""
    settings = get_settings()
    return _build_chain(settings, settings.chunked_extraction_provider_order)
