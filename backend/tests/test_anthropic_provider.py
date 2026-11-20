"""AnthropicProvider — fourth link in the fallback chain, on the Anthropic
Messages API. Nothing here touches the network."""
import json
from types import SimpleNamespace

import httpx
import pytest

from app.llm import providers as providers_module
from app.llm.base import LlmAuthError, LlmBadRequestError, LlmProviderError, LlmRateLimitError
from app.llm.chain import LlmChain
from app.llm.factory import _build_chain
from app.llm.providers import AnthropicProvider, GroqProvider


class _Response:
    def __init__(self, status_code=200, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload or {})

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def _ok(text="hello"):
    return _Response(payload={"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"})


def _script(monkeypatch, responses):
    sent = []
    queue = list(responses)

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.append(dict(url=url, body=json, headers=headers, timeout=timeout))
        return queue.pop(0)

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(providers_module.time, "sleep", lambda s: None)
    return sent


def test_success_sends_expected_request(monkeypatch):
    sent = _script(monkeypatch, [_ok("extracted")])
    out = AnthropicProvider("k", "claude-x", 30.0, max_tokens=123).generate("sys", "user")
    assert out == "extracted"
    req = sent[0]
    assert req["url"] == "https://api.anthropic.com/v1/messages"
    assert req["headers"]["x-api-key"] == "k"
    assert req["headers"]["anthropic-version"] == "2023-06-01"
    assert req["body"]["model"] == "claude-x"
    assert req["body"]["max_tokens"] == 123
    assert req["body"]["system"] == "sys"
    assert req["body"]["messages"] == [{"role": "user", "content": [{"type": "text", "text": "user"}]}]


def test_json_mode_instructs_and_strips_fences(monkeypatch):
    sent = _script(monkeypatch, [_ok('```json\n{"a": 1}\n```')])
    out = AnthropicProvider("k", "m", 30.0).generate("sys", "u", want_json=True)
    assert json.loads(out) == {"a": 1}
    assert "JSON" in sent[0]["body"]["system"]


def test_images_become_base64_blocks(monkeypatch):
    sent = _script(monkeypatch, [_ok()])
    AnthropicProvider("k", "m", 30.0).generate("s", "u", image=b"\x01", image_media_type="image/jpeg", images=[b"\x02"])
    blocks = sent[0]["body"]["messages"][0]["content"]
    assert blocks[0]["source"]["media_type"] == "image/jpeg"
    assert blocks[1]["type"] == "image" and blocks[-1]["type"] == "text"


def test_429_then_success(monkeypatch):
    sent = _script(monkeypatch, [_Response(429, text="slow down"), _ok("fine")])
    assert AnthropicProvider("k", "m", 30.0).generate("s", "u") == "fine"
    assert len(sent) == 2


def test_529_exhausts_retries_as_rate_limit(monkeypatch):
    _script(monkeypatch, [_Response(529, text="overloaded")] * 3)
    with pytest.raises(LlmRateLimitError):
        AnthropicProvider("k", "m", 30.0).generate("s", "u")


def test_401_is_not_retried(monkeypatch):
    sent = _script(monkeypatch, [_Response(401, text="bad key")])
    with pytest.raises(LlmAuthError):
        AnthropicProvider("k", "m", 30.0).generate("s", "u")
    assert len(sent) == 1


def test_400_is_not_retried(monkeypatch):
    sent = _script(monkeypatch, [_Response(400, text="prompt is too long")])
    with pytest.raises(LlmBadRequestError):
        AnthropicProvider("k", "m", 30.0).generate("s", "u")
    assert len(sent) == 1


def test_empty_content_raises(monkeypatch):
    _script(monkeypatch, [_Response(payload={"content": [], "stop_reason": "max_tokens"})])
    with pytest.raises(LlmProviderError):
        AnthropicProvider("k", "m", 30.0).generate("s", "u")


def _settings(key):
    return SimpleNamespace(
        anthropic_api_key_list=[k.strip() for k in key.split(",") if k.strip()],
        anthropic_model="m", anthropic_max_tokens=77, llm_timeout_seconds=5.0,
    )


def test_missing_key_builds_nothing():
    assert _build_chain(_settings(""), "anthropic").providers == []


def test_multiple_keys_build_numbered_providers():
    chain = _build_chain(_settings("a,b"), "anthropic")
    assert [p.name for p in chain.providers] == ["anthropic", "anthropic-2"]
    assert [p.api_key for p in chain.providers] == ["a", "b"]
    assert chain.providers[0].max_tokens == 77


def test_chain_falls_through_from_failing_groq_to_anthropic(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        if "anthropic" in url:
            return _ok('{"ok": true}')
        return _Response(401, text="revoked")

    monkeypatch.setattr(httpx, "post", fake_post)
    chain = LlmChain([GroqProvider("g", "m", 5.0), AnthropicProvider("a", "m", 5.0)])
    assert chain.generate_json("s", "u") == {"ok": True}
