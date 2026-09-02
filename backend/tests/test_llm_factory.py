"""Provider-chain assembly: which providers get built from which settings,
and — the actual point of key rotation — that a second/third key for the
same provider gets its own distinct identity so llm/chain.py tracks its
cooldown independently rather than colliding with the first key's."""
from types import SimpleNamespace

from app.llm.factory import _build_chain
from app.llm.providers import GeminiProvider, NovaProvider, OllamaProvider, OpenAiCompatibleProvider

_DEFAULTS = dict(
    gemini_api_key="", gemini_model="gemini-2.5-flash",
    gemini_api_key_2="", gemini_api_key_3="",
    groq_api_key="", groq_model="groq-model", groq_api_key_2="",
    huggingface_api_key="", huggingface_model="hf-model",
    cerebras_api_key="", cerebras_model="cerebras-model",
    openrouter_api_key="", openrouter_model="openrouter-model",
    sambanova_api_key="", sambanova_model="sambanova-model",
    mistral_api_key="", mistral_model="mistral-model",
    nvidia_api_key="", nvidia_model="nvidia-model",
    github_token="", github_model="github-model",
    nova_model="", nova_region="us-east-1", nova_max_tokens=8192,
    llm_timeout_seconds=30.0,
    ollama_base_url="http://localhost:11434", ollama_model="qwen2.5:3b",
    ollama_timeout_seconds=210.0, ollama_num_thread=0, ollama_num_predict=1024,
)


def _settings(**overrides):
    return SimpleNamespace(**{**_DEFAULTS, **overrides})


def test_blank_keys_are_skipped_entirely():
    chain = _build_chain(_settings(), "gemini,gemini_2,gemini_3,groq,groq_2,sambanova,mistral,nvidia,github")
    assert chain.providers == []  # every key blank -> nothing built, no crash


def test_each_new_provider_builds_when_its_key_is_set():
    s = _settings(sambanova_api_key="sn-key", mistral_api_key="mi-key", nvidia_api_key="nv-key", github_token="gh-token")
    chain = _build_chain(s, "sambanova,mistral,nvidia,github")
    names = [p.name for p in chain.providers]
    assert names == ["sambanova", "mistral", "nvidia", "github"]
    assert all(isinstance(p, OpenAiCompatibleProvider) for p in chain.providers)


def test_second_and_third_gemini_keys_get_distinct_names_for_independent_cooldown():
    """This is the actual mechanism key rotation depends on: llm/chain.py
    keys its per-provider cooldown state by `.name`. Two GeminiProvider
    instances that both reported as "gemini" would collide into one shared
    cooldown entry, so a 429 on key 1 would incorrectly cool down key 2 as
    well — defeating the entire point of having a second key."""
    s = _settings(gemini_api_key="key-1", gemini_api_key_2="key-2", gemini_api_key_3="key-3")
    chain = _build_chain(s, "gemini,gemini_2,gemini_3")

    assert [p.name for p in chain.providers] == ["gemini", "gemini_2", "gemini_3"]
    assert len(set(p.name for p in chain.providers)) == 3  # no collisions
    assert all(isinstance(p, GeminiProvider) for p in chain.providers)
    # Each instance actually carries its OWN key, not a shared/overwritten one.
    assert [p.api_key for p in chain.providers] == ["key-1", "key-2", "key-3"]
    # And each gets its own independent cooldown slot in the router.
    assert set(chain._status.keys()) == {"gemini", "gemini_2", "gemini_3"}


def test_second_groq_key_is_independent_of_the_first():
    s = _settings(groq_api_key="key-1", groq_api_key_2="key-2")
    chain = _build_chain(s, "groq,groq_2")
    assert [p.name for p in chain.providers] == ["groq", "groq_2"]
    assert [p.api_key for p in chain.providers] == ["key-1", "key-2"]


def test_rotation_key_left_blank_is_skipped_even_if_the_base_key_is_set():
    s = _settings(gemini_api_key="key-1", gemini_api_key_2="")
    chain = _build_chain(s, "gemini,gemini_2")
    assert [p.name for p in chain.providers] == ["gemini"]


def test_ollama_always_builds_regardless_of_keys():
    chain = _build_chain(_settings(), "ollama")
    assert len(chain.providers) == 1
    assert isinstance(chain.providers[0], OllamaProvider)


def test_nova_is_skipped_with_no_model_configured_and_builds_once_set():
    # Unlike every other provider, gated on an inference-profile ID rather
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
