"""GeminiProvider — the keyed REST fallback that keeps extraction and the
accuracy benchmark running when Bedrock is unavailable. Nothing here touches
the network; httpx.post is monkeypatched."""
import base64
import json

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
from app.llm.providers import GeminiProvider


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
    return _Response(payload={"candidates": [{"content": {"parts": [{"text": text}]}}]})


def _capture(monkeypatch, response):
    """Swaps httpx.post and records the call it received."""
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
    provider = GeminiProvider("key-123", "gemini-2.5-flash", 30.0)
    assert provider.generate("sys", "user") == "extracted"


def test_multi_part_replies_are_joined(monkeypatch):
    _capture(monkeypatch, _Response(payload={
        "candidates": [{"content": {"parts": [{"text": "abc"}, {"text": "def"}]}}]
    }))
    assert GeminiProvider("k", "m", 30.0).generate("s", "u") == "abcdef"


def test_image_is_sent_inline_as_base64_with_its_media_type(monkeypatch):
    """The whole image+text pairing depends on this reaching the API."""
    sent = _capture(monkeypatch, _ok())
    GeminiProvider("k", "m", 30.0).generate("s", "u", image=b"PNGBYTES", image_media_type="image/jpeg")

    parts = sent["body"]["contents"][0]["parts"]
    assert parts[0] == {"text": "u"}
    assert parts[1]["inline_data"]["mime_type"] == "image/jpeg"
    assert base64.b64decode(parts[1]["inline_data"]["data"]) == b"PNGBYTES"


def test_text_only_call_sends_no_image_part(monkeypatch):
    sent = _capture(monkeypatch, _ok())
    GeminiProvider("k", "m", 30.0).generate("s", "u")
    assert sent["body"]["contents"][0]["parts"] == [{"text": "u"}]


def test_json_mode_is_only_requested_when_asked(monkeypatch):
    sent = _capture(monkeypatch, _ok("{}"))
    GeminiProvider("k", "m", 30.0).generate("s", "u", want_json=True)
    assert sent["body"]["generationConfig"]["responseMimeType"] == "application/json"

    sent = _capture(monkeypatch, _ok())
    GeminiProvider("k", "m", 30.0).generate("s", "u")
    assert "responseMimeType" not in sent["body"]["generationConfig"]


def test_system_prompt_and_token_cap_are_passed_through(monkeypatch):
    sent = _capture(monkeypatch, _ok())
    GeminiProvider("k", "m", 30.0, max_tokens=1234).generate("SYSTEM", "u")
    assert sent["body"]["system_instruction"]["parts"][0]["text"] == "SYSTEM"
    assert sent["body"]["generationConfig"]["maxOutputTokens"] == 1234


@pytest.mark.parametrize(
    "status,expected",
    [
        (429, LlmRateLimitError),
        (401, LlmAuthError),
        (403, LlmAuthError),
        (402, LlmQuotaError),
        (400, LlmBadRequestError),
        (504, LlmTimeoutError),
        (500, LlmProviderError),
    ],
)
def test_http_statuses_map_to_the_routers_typed_errors(monkeypatch, status, expected):
    """llm/chain.py routes on the exception TYPE — a 429 must back off, a 401
    must disable the provider, a 400 must do neither."""
    _capture(monkeypatch, _Response(status_code=status, text="boom"))
    with pytest.raises(expected):
        GeminiProvider("k", "m", 30.0).generate("s", "u")


def test_a_timeout_is_reported_as_a_timeout(monkeypatch):
    _capture(monkeypatch, httpx.ReadTimeout("too slow"))
    with pytest.raises(LlmTimeoutError):
        GeminiProvider("k", "m", 30.0).generate("s", "u")


def test_the_api_key_never_appears_in_an_error_message(monkeypatch):
    """The key goes in a header, not the query string — an error built from the
    request URL would leak it into logs and into the benchmark's failure
    summaries, which get written to a shared drive."""
    secret = "AIzaSy-SUPER-SECRET-KEY"
    sent = _capture(monkeypatch, _Response(status_code=400, text="bad request"))
    with pytest.raises(LlmProviderError) as excinfo:
        GeminiProvider(secret, "m", 30.0).generate("s", "u")
    assert secret not in str(excinfo.value)
    assert secret not in sent["url"]
    assert sent["headers"]["x-goog-api-key"] == secret


def test_a_truncated_reply_reports_its_finish_reason(monkeypatch):
    """An empty parts list means the cap cut the reply off before any text —
    saying so beats 'unexpected response shape', which sends you looking for a
    parsing bug that isn't there."""
    _capture(monkeypatch, _Response(payload={"candidates": [{"finishReason": "MAX_TOKENS", "content": {}}]}))
    with pytest.raises(LlmProviderError, match="MAX_TOKENS"):
        GeminiProvider("k", "m", 30.0).generate("s", "u")


def test_an_unexpected_body_raises_a_provider_error_not_a_crash(monkeypatch):
    _capture(monkeypatch, _Response(payload={"nope": True}))
    with pytest.raises(LlmProviderError):
        GeminiProvider("k", "m", 30.0).generate("s", "u")


def test_transient_capacity_errors_are_retryable_not_fatal(monkeypatch):
    """Google returns 503 "experiencing high demand" on capacity spikes —
    MEASURED 2026-09-04, it killed 6 consecutive documents in a batch run.
    It must route like a 429 (back off, try again later), not like a hard
    failure, so a passing spike doesn't sink an entire benchmark run."""
    monkeypatch.setattr("app.llm.providers.time.sleep", lambda _s: None)
    for status in (503, 529):
        _capture(monkeypatch, _Response(status_code=status, text="high demand"))
        with pytest.raises(LlmRateLimitError):
            GeminiProvider("k", "m", 30.0).generate("s", "u")


def test_a_passing_capacity_spike_is_retried_and_succeeds(monkeypatch):
    """503 must be absorbed in-provider. The chain's cooldown is per-provider
    and persists across calls, so escalating a spike would put the only
    provider into a growing cooldown and skip every later document in a batch
    without attempting it — MEASURED: that cascade took a 51-document run down
    to 9 successes."""
    calls = {"n": 0}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            return _Response(status_code=503, text="high demand")
        return _ok("recovered")

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("app.llm.providers.time.sleep", lambda _s: None)  # don't really wait
    assert GeminiProvider("k", "m", 30.0).generate("s", "u") == "recovered"
    assert calls["n"] == 3


def test_a_sustained_outage_still_escalates_after_the_retries(monkeypatch):
    """A provider that is genuinely down SHOULD reach the chain, so it can
    cool down / fall through to the next provider."""
    monkeypatch.setattr("app.llm.providers.time.sleep", lambda _s: None)
    _capture(monkeypatch, _Response(status_code=503, text="high demand"))
    with pytest.raises(LlmRateLimitError):
        GeminiProvider("k", "m", 30.0).generate("s", "u")


def test_a_non_transient_error_is_not_retried(monkeypatch):
    """A 400 is about THIS request — retrying it just wastes time."""
    calls = {"n": 0}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls["n"] += 1
        return _Response(status_code=400, text="bad request")

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(LlmBadRequestError):
        GeminiProvider("k", "m", 30.0).generate("s", "u")
    assert calls["n"] == 1
