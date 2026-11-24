"""Provider-chain assembly: Nova (gated on an inference-profile ID rather
than an API key) plus Gemini, the keyed fallback that keeps extraction running
when Bedrock is unavailable."""
from types import SimpleNamespace

from app.llm.factory import _build_chain
from app.llm.providers import GeminiProvider, GroqProvider, NovaProvider

_DEFAULTS = dict(
    nova_model="", nova_region="us-east-1", nova_max_tokens=8192,
    gemini_api_key="", gemini_model="gemini-3.6-flash", gemini_max_tokens=8192,
    groq_api_key="", groq_model="meta-llama/llama-4-scout-17b-16e-instruct", groq_max_tokens=8192,
    llm_timeout_seconds=30.0,
)


def _settings(**overrides):
    values = {**_DEFAULTS, **overrides}
    # the real Settings exposes these as computed properties
    values["gemini_api_key_list"] = [k.strip() for k in values["gemini_api_key"].split(",") if k.strip()]
    values["groq_api_key_list"] = [k.strip() for k in values["groq_api_key"].split(",") if k.strip()]
    return SimpleNamespace(**values)


def test_blank_model_is_skipped_entirely():
    chain = _build_chain(_settings(), "nova")
    assert chain.providers == []  # no model configured -> nothing built, no crash


def test_nova_is_skipped_with_no_model_configured_and_builds_once_set():
    # Unlike a typical provider, gated on an inference-profile ID rather
    # than an API key — Nova authenticates via the ambient AWS credential
    # chain, not a credential stored in settings.
    assert _build_chain(_settings(), "nova").providers == []

    s = _settings(nova_model="us.amazon.nova-2-lite-v1:0", nova_region="ap-south-1")
    chain = _build_chain(s, "nova")
    assert len(chain.providers) == 1
    provider = chain.providers[0]
    assert isinstance(provider, NovaProvider)
    assert provider.model == "us.amazon.nova-2-lite-v1:0"
    assert provider.region == "ap-south-1"


def test_unknown_provider_name_in_order_is_skipped_not_an_error():
    s = _settings(nova_model="us.amazon.nova-2-lite-v1:0")
    chain = _build_chain(s, "gemini,nova,ollama")
    assert [p.name for p in chain.providers] == ["nova"]


def test_gemini_is_skipped_without_a_key_and_builds_once_set():
    assert _build_chain(_settings(), "gemini").providers == []

    chain = _build_chain(_settings(gemini_api_key="k", gemini_model="gemini-3.6-flash"), "gemini")
    assert len(chain.providers) == 1
    provider = chain.providers[0]
    assert isinstance(provider, GeminiProvider)
    assert provider.model == "gemini-3.6-flash"
    assert provider.name == "gemini"


def test_order_string_controls_the_fallback_sequence():
    """LlmChain tries providers in order and falls through on failure, so
    "nova,gemini" is what turns a Bedrock outage into a degradation rather
    than a stoppage."""
    s = _settings(nova_model="us.amazon.nova-2-lite-v1:0", gemini_api_key="k")

    chain = _build_chain(s, "nova,gemini")
    assert [p.name for p in chain.providers] == ["nova", "gemini"]

    chain = _build_chain(s, "gemini,nova")
    assert [p.name for p in chain.providers] == ["gemini", "nova"]

    # An unconfigured name in the order is skipped, not fatal.
    chain = _build_chain(_settings(gemini_api_key="k"), "nova,gemini")
    assert [p.name for p in chain.providers] == ["gemini"]


def test_several_gemini_keys_each_become_their_own_fallback_link():
    """Gemini's free-tier quota is per project (MEASURED 2026-09-04: 20
    requests/day per project per model), so a second key is a second
    allowance — and only helps if it is a SEPARATE link the chain can fall
    through to when the first is exhausted."""
    chain = _build_chain(_settings(gemini_api_key="key-one, key-two , key-three"), "gemini")
    assert [p.name for p in chain.providers] == ["gemini", "gemini-2", "gemini-3"]
    assert [p.api_key for p in chain.providers] == ["key-one", "key-two", "key-three"]
    assert all(isinstance(p, GeminiProvider) for p in chain.providers)


def test_groq_is_skipped_without_a_key_and_builds_once_set():
    assert _build_chain(_settings(), "groq").providers == []
    chain = _build_chain(_settings(groq_api_key="gsk_x"), "groq")
    assert len(chain.providers) == 1
    assert isinstance(chain.providers[0], GroqProvider)
    assert chain.providers[0].name == "groq"


def test_the_full_fallback_chain_assembles_in_order():
    s = _settings(
        nova_model="us.amazon.nova-2-lite-v1:0",
        gemini_api_key="k1,k2",
        groq_api_key="gsk_x",
    )
    chain = _build_chain(s, "nova,gemini,groq")
    assert [p.name for p in chain.providers] == ["nova", "gemini", "gemini-2", "groq"]


def test_an_unconfigured_provider_in_the_order_is_skipped_not_fatal():
    chain = _build_chain(_settings(groq_api_key="gsk_x"), "nova,gemini,groq")
    assert [p.name for p in chain.providers] == ["groq"]
