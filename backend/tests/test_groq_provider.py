"""GroqProvider — third link in the fallback chain, on Groq's
OpenAI-compatible endpoint. Nothing here touches the network."""
import base64
import json

import httpx
import pytest

from app.llm.base import (
    LlmAuthError,
    LlmBadRequestError,
    LlmProviderError,
    LlmRateLimitError,
    LlmTimeoutError,
)
from app.llm.providers import GroqProvider


class _Response:
    def __init__(self, status_code=200, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload or {})

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def _ok(text="hello", finish="stop"):
    return _Response(payload={"choices": [{"message": {"content": text}, "finish_reason": finish}]})


def _capture(monkeypatch, response):
    sent = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.update(url=url, body=json, headers=headers, timeout=timeout)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(httpx, "post", fake_post)
    return sent


def test_returns_the_reply_text(monkeypatch):
    _capture(monkeypatch, _ok("extracted"))
    assert GroqProvider("k", "m", 30.0).generate("sys", "user") == "extracted"


def test_image_is_sent_as_an_inline_data_url(monkeypatch):
    """The image+OCR-text pairing is what fixed the redacted-field
    hallucination, so it has to survive the OpenAI content-parts shape."""
    sent = _capture(monkeypatch, _ok())
    GroqProvider("k", "m", 30.0).generate("s", "u", image=b"PNGBYTES", image_media_type="image/jpeg")

    parts = sent["body"]["messages"][1]["content"]
    assert parts[0] == {"type": "text", "text": "u"}
    url = parts[1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"PNGBYTES"


def test_text_only_call_sends_no_image_part(monkeypatch):
    sent = _capture(monkeypatch, _ok())
    GroqProvider("k", "m", 30.0).generate("s", "u")
    assert sent["body"]["messages"][1]["content"] == [{"type": "text", "text": "u"}]


def test_system_prompt_is_its_own_message(monkeypatch):
    sent = _capture(monkeypatch, _ok())
    GroqProvider("k", "m", 30.0).generate("SYSTEM", "u")
    assert sent["body"]["messages"][0] == {"role": "system", "content": "SYSTEM"}


def test_json_mode_is_only_requested_when_asked(monkeypatch):
    sent = _capture(monkeypatch, _ok("{}"))
    GroqProvider("k", "m", 30.0).generate("s", "u", want_json=True)
    assert sent["body"]["response_format"] == {"type": "json_object"}

    sent = _capture(monkeypatch, _ok())
    GroqProvider("k", "m", 30.0).generate("s", "u")
    assert "response_format" not in sent["body"]


@pytest.mark.parametrize(
    "status,expected",
    [(429, LlmRateLimitError), (401, LlmAuthError), (400, LlmBadRequestError), (500, LlmProviderError)],
)
def test_http_statuses_map_to_the_routers_typed_errors(monkeypatch, status, expected):
    monkeypatch.setattr("app.llm.providers.time.sleep", lambda _s: None)
    _capture(monkeypatch, _Response(status_code=status, text="boom"))
    with pytest.raises(expected):
        GroqProvider("k", "m", 30.0).generate("s", "u")


def test_a_timeout_is_reported_as_a_timeout(monkeypatch):
    _capture(monkeypatch, httpx.ReadTimeout("too slow"))
    with pytest.raises(LlmTimeoutError):
        GroqProvider("k", "m", 30.0).generate("s", "u")


def test_the_api_key_never_appears_in_an_error_message(monkeypatch):
    """Errors land in the benchmark's failure summaries, which are written to
    a shared drive — the key must not travel with them."""
    secret = "gsk_SUPER_SECRET"
    sent = _capture(monkeypatch, _Response(status_code=400, text="bad request"))
    with pytest.raises(LlmProviderError) as excinfo:
        GroqProvider(secret, "m", 30.0).generate("s", "u")
    assert secret not in str(excinfo.value)
    assert secret not in sent["url"]
    assert sent["headers"]["Authorization"] == f"Bearer {secret}"


def test_a_passing_rate_limit_is_retried_and_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls["n"] += 1
        return _Response(status_code=429, text="slow down") if calls["n"] < 3 else _ok("recovered")

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("app.llm.providers.time.sleep", lambda _s: None)
    assert GroqProvider("k", "m", 30.0).generate("s", "u") == "recovered"
    assert calls["n"] == 3


def test_an_empty_reply_reports_its_finish_reason(monkeypatch):
    _capture(monkeypatch, _ok(text="", finish="length"))
    with pytest.raises(LlmProviderError, match="length"):
        GroqProvider("k", "m", 30.0).generate("s", "u")


def test_an_unexpected_body_raises_a_provider_error_not_a_crash(monkeypatch):
    _capture(monkeypatch, _Response(payload={"nope": True}))
    with pytest.raises(LlmProviderError):
        GroqProvider("k", "m", 30.0).generate("s", "u")
