"""Each provider must turn an HTTP failure into the right typed exception
(so llm/chain.py's router can act on it) and must never leak the API key
into that exception's message — see llm/base.py's classify_http_error."""
import httpx
import pytest

from app.llm.base import (
    LlmAuthError,
    LlmBadRequestError,
    LlmProviderError,
    LlmQuotaError,
    LlmRateLimitError,
    LlmTimeoutError,
)
from app.llm.providers import GeminiProvider, OllamaProvider, OpenAiCompatibleProvider

SECRET = "sk-super-secret-key-do-not-leak"


def _status_response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=httpx.Request("POST", "https://example.test/x"), json={"error": "boom"})


@pytest.mark.parametrize(
    "status_code,expected_exc",
    [(429, LlmRateLimitError), (402, LlmQuotaError), (401, LlmAuthError), (403, LlmAuthError), (400, LlmBadRequestError), (500, LlmProviderError)],
)
def test_gemini_classifies_status_codes(monkeypatch, status_code, expected_exc):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _status_response(status_code))
    provider = GeminiProvider(SECRET, "gemini-2.5-flash", 30.0)

    with pytest.raises(expected_exc) as excinfo:
        provider.generate("system", "hello")
    assert SECRET not in str(excinfo.value)


def test_gemini_sends_key_as_header_not_query_param(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None, **kwargs):
        captured["url"] = url
        captured["headers"] = headers
        return httpx.Response(200, request=httpx.Request("POST", url), json={
            "candidates": [{"content": {"parts": [{"text": "hi"}]}}]
        })

    monkeypatch.setattr(httpx, "post", fake_post)
    GeminiProvider(SECRET, "gemini-2.5-flash", 30.0).generate("system", "hello")

    assert SECRET not in captured["url"]
    assert "key" not in captured["url"]
    assert captured["headers"]["x-goog-api-key"] == SECRET


@pytest.mark.parametrize(
    "status_code,expected_exc",
    [(429, LlmRateLimitError), (402, LlmQuotaError), (401, LlmAuthError), (403, LlmAuthError), (400, LlmBadRequestError), (500, LlmProviderError)],
)
def test_openai_compatible_provider_classifies_status_codes(monkeypatch, status_code, expected_exc):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _status_response(status_code))
    provider = OpenAiCompatibleProvider("groq", "https://api.groq.com/openai/v1", SECRET, "model", 30.0)

    with pytest.raises(expected_exc) as excinfo:
        provider.generate("system", "hello")
    assert SECRET not in str(excinfo.value)


def test_ollama_sends_num_thread_and_num_predict(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None, **kwargs):
        captured["options"] = json["options"]
        return httpx.Response(200, request=httpx.Request("POST", url), json={"message": {"content": "hi"}})

    monkeypatch.setattr(httpx, "post", fake_post)

    # num_thread=0 means "resolve to every logical core at call time".
    OllamaProvider("http://localhost:11434", "qwen2.5:3b", 210.0, num_thread=0, num_predict=1024).generate(
        "system", "hello"
    )
    assert captured["options"]["num_thread"] > 0
    assert captured["options"]["num_predict"] == 1024

    # An explicit non-zero value is passed through untouched.
    OllamaProvider("http://localhost:11434", "qwen2.5:3b", 210.0, num_thread=6, num_predict=512).generate(
        "system", "hello"
    )
    assert captured["options"]["num_thread"] == 6
    assert captured["options"]["num_predict"] == 512


def test_ollama_classifies_status_codes_with_no_secret_to_leak(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _status_response(429))
    provider = OllamaProvider("http://localhost:11434", "qwen2.5:3b", 300.0)

    with pytest.raises(LlmRateLimitError):
        provider.generate("system", "hello")


def test_timeout_is_classified_as_llm_timeout_error(monkeypatch):
    def raise_timeout(*a, **k):
        raise httpx.ReadTimeout("timed out", request=httpx.Request("POST", "https://example.test/x"))

    monkeypatch.setattr(httpx, "post", raise_timeout)
    provider = GeminiProvider(SECRET, "gemini-2.5-flash", 30.0)

    with pytest.raises(LlmTimeoutError) as excinfo:
        provider.generate("system", "hello")
    assert SECRET not in str(excinfo.value)


def test_redact_scrubs_secret_from_arbitrary_text():
    from app.llm.base import redact

    assert redact(f"error talking to https://x?key={SECRET}", SECRET) == "error talking to https://x?key=[REDACTED]"
    # Falsy secrets (an unset key) are skipped rather than erroring.
    assert redact("no secret here", "", None) == "no secret here"


def test_redact_known_secrets_pulls_every_configured_provider_key(monkeypatch):
    """This is the last-resort pass applied in routers/documents.py before an
    exception's text is stored as Document.error / returned to the frontend —
    it must catch a leaked key regardless of which provider it belonged to,
    not just the one that happened to fail."""
    from types import SimpleNamespace

    from app.llm import base

    fake_settings = SimpleNamespace(
        gemini_api_key="gemini-secret",
        groq_api_key="groq-secret",
        huggingface_api_key="",
        cerebras_api_key="cerebras-secret",
        openrouter_api_key="",
    )
    monkeypatch.setattr("app.config.get_settings", lambda: fake_settings)

    text = "call failed: gemini-secret and also cerebras-secret leaked, but groq-secret too"
    cleaned = base.redact_known_secrets(text)

    assert "gemini-secret" not in cleaned
    assert "groq-secret" not in cleaned
    assert "cerebras-secret" not in cleaned
