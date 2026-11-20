import pytest

from app.llm.base import LlmNotConfigured, LlmProvider, LlmProviderError
from app.llm.chain import LlmChain
from app.llm.json_utils import JsonParseError, parse_json_object


class FakeProvider(LlmProvider):
    def __init__(self, name, reply=None, error=None):
        self.name = name
        self.reply = reply
        self.error = error
        self.calls = 0

    def generate(self, system, user_text, image=None, image_media_type=None, want_json=False, images=None):
        self.calls += 1
        if self.error:
            raise self.error
        return self.reply


def test_generate_text_falls_through_to_the_next_provider_on_failure():
    failing = FakeProvider("gemini", error=RuntimeError("rate limited"))
    working = FakeProvider("groq", reply="hello")
    chain = LlmChain([failing, working])

    assert chain.generate_text(system="s", user_text="u") == "hello"
    assert failing.calls == 1
    assert working.calls == 1


def test_generate_json_treats_unparseable_reply_as_a_provider_failure():
    bad_json = FakeProvider("gemini", reply="not json at all")
    good_json = FakeProvider("groq", reply='{"doc_type": "legal_proof", "confidence": 0.9}')
    chain = LlmChain([bad_json, good_json])

    result = chain.generate_json(system="s", user_text="u")

    assert result == {"doc_type": "legal_proof", "confidence": 0.9}
    assert good_json.calls == 1


def test_no_providers_configured_raises_llm_not_configured():
    chain = LlmChain([])
    with pytest.raises(LlmNotConfigured):
        chain.generate_text(system="s", user_text="u")


def test_all_providers_failing_raises_llm_provider_error():
    chain = LlmChain([FakeProvider("gemini", error=RuntimeError("boom"))])
    with pytest.raises(LlmProviderError, match="gemini: boom"):
        chain.generate_text(system="s", user_text="u")


def test_parse_json_object_salvages_a_markdown_fenced_reply():
    text = '```json\n{"a": 1}\n```'
    assert parse_json_object(text) == {"a": 1}


def test_parse_json_object_salvages_prose_wrapped_json():
    text = 'Sure, here you go:\n{"a": 1}\nHope that helps!'
    assert parse_json_object(text) == {"a": 1}


def test_parse_json_object_raises_when_nothing_json_like_is_present():
    with pytest.raises(JsonParseError):
        parse_json_object("no json here")


def test_cooldown_wait_cap_covers_the_backoff_cap_the_chain_can_impose():
    """The wait ceiling must not sit below the backoff ceiling.

    graph.llm waits out a provider cooldown only while it is <=
    max_cooldown_wait_seconds, and abandons the call otherwise. The chain's
    own exponential backoff is capped at _BACKOFF_CAP_SECONDS. If the wait cap
    is the smaller of the two, every cooldown that lands between them is one
    the chain created and then refused to wait for, and the chunk is dropped
    with no LLM ever seeing it.

    MEASURED: with a 240s wait cap against the 300s backoff cap, cooldowns of
    287s and 297s abandoned both chunks of a 17-page PDF (graph run 137) even
    though all 17 pages had been read successfully.
    """
    from app.graph.config import GraphSettings
    from app.llm.chain import _BACKOFF_CAP_SECONDS

    assert GraphSettings().max_cooldown_wait_seconds >= _BACKOFF_CAP_SECONDS
