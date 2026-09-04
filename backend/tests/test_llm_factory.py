"""Provider-chain assembly: Nova (gated on an inference-profile ID rather
than an API key) plus Gemini, the keyed fallback that keeps extraction running
when Bedrock is unavailable."""
from types import SimpleNamespace

from app.llm.factory import _build_chain
from app.llm.providers import GeminiProvider, NovaProvider

_DEFAULTS = dict(
    nova_model="", nova_region="us-east-1", nova_max_tokens=8192,
    gemini_api_key="", gemini_model="gemini-2.5-flash", gemini_max_tokens=8192,
    llm_timeout_seconds=30.0,
)


def _settings(**overrides):
    return SimpleNamespace(**{**_DEFAULTS, **overrides})


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

    chain = _build_chain(_settings(gemini_api_key="k", gemini_model="gemini-2.5-flash"), "gemini")
    assert len(chain.providers) == 1
    provider = chain.providers[0]
    assert isinstance(provider, GeminiProvider)
    assert provider.model == "gemini-2.5-flash"
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
