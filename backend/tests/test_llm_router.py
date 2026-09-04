"""Status-aware routing in LlmChain: a provider's own cooldown state should
determine whether it's even tried again, without needing a fresh HTTP round
trip to rediscover a failure mode we already know about."""
import app.llm.chain as chain_module
from app.llm.base import (
    LlmAuthError,
    LlmBadRequestError,
    LlmProviderError,
    LlmProvider,
    LlmQuotaError,
    LlmRateLimitError,
    LlmTimeoutError,
)
from app.llm.chain import LlmChain


class FakeProvider(LlmProvider):
    def __init__(self, name, errors=None, reply="ok"):
        self.name = name
        self.errors = list(errors or [])  # one exception (or None) consumed per call, then repeats the last
        self.reply = reply
        self.calls = 0

    def generate(self, system, user_text, image=None, image_media_type=None, want_json=False):
        self.calls += 1
        outcome = self.errors[min(self.calls, len(self.errors)) - 1] if self.errors else None
        if outcome is not None:
            raise outcome
        return self.reply


class FakeClock:
    """A controllable stand-in for time.monotonic()."""

    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _install_clock(monkeypatch, start=1000.0):
    clock = FakeClock(start)
    monkeypatch.setattr(chain_module.time, "monotonic", clock)
    return clock


def test_429_backs_off_and_is_skipped_until_cooldown_elapses(monkeypatch):
    clock = _install_clock(monkeypatch)
    limited = FakeProvider("groq", errors=[LlmRateLimitError("429")])
    backup = FakeProvider("ollama", reply="backup answer")
    chain = LlmChain([limited, backup])

    assert chain.generate_text(system="s", user_text="u") == "backup answer"
    assert limited.calls == 1

    # Still within the backoff window: groq must be skipped without a call.
    clock.advance(1.0)
    assert chain.generate_text(system="s", user_text="u") == "backup answer"
    assert limited.calls == 1  # not retried yet

    # Past the backoff window: groq is tried again.
    clock.advance(30.0)
    limited.errors = []  # simulate it recovering
    limited.reply = "groq is back"
    assert chain.generate_text(system="s", user_text="u") == "groq is back"
    assert limited.calls == 2


def test_429_backoff_doubles_on_repeated_failures(monkeypatch):
    clock = _install_clock(monkeypatch)
    limited = FakeProvider("groq", errors=[LlmRateLimitError("1"), LlmRateLimitError("2"), LlmRateLimitError("3")])
    backup = FakeProvider("ollama", reply="backup")
    chain = LlmChain([limited, backup])

    chain.generate_text(system="s", user_text="u")  # failure #1 -> 5s backoff
    first_available_at = chain._status["groq"].available_at

    clock.advance(5.0)  # exactly at the boundary, provider is available again
    chain.generate_text(system="s", user_text="u")  # failure #2 -> 10s backoff (doubled)
    second_delay = chain._status["groq"].available_at - clock.now

    assert second_delay > (first_available_at - 1000.0)  # grew relative to the first backoff


def test_402_and_401_403_disable_the_provider_outright(monkeypatch):
    clock = _install_clock(monkeypatch)
    for exc in [LlmQuotaError("402"), LlmAuthError("401"), LlmAuthError("403")]:
        broken = FakeProvider("gemini", errors=[exc])
        backup = FakeProvider("ollama", reply="backup")
        chain = LlmChain([broken, backup])

        chain.generate_text(system="s", user_text="u")
        assert broken.calls == 1

        # Even after a very long time, a disabled provider is never retried.
        clock.advance(10 ** 9)
        chain.generate_text(system="s", user_text="u")
        assert broken.calls == 1  # still 1 — never called again


def test_400_bad_request_is_not_punished_on_a_later_call(monkeypatch):
    clock = _install_clock(monkeypatch)
    flaky = FakeProvider("groq", errors=[LlmBadRequestError("400")])
    backup = FakeProvider("ollama", reply="backup")
    chain = LlmChain([flaky, backup])

    assert chain.generate_text(system="s", user_text="u") == "backup"
    assert flaky.calls == 1

    # A 400 sets no cooldown at all — the very next call tries it again.
    flaky.errors = []
    flaky.reply = "groq answers fine this time"
    assert chain.generate_text(system="s", user_text="u") == "groq answers fine this time"
    assert flaky.calls == 2


def test_timeout_backs_off_independently_of_rate_limit_counter(monkeypatch):
    clock = _install_clock(monkeypatch)
    slow = FakeProvider("cerebras", errors=[LlmTimeoutError("timed out")])
    backup = FakeProvider("ollama", reply="backup")
    chain = LlmChain([slow, backup])

    chain.generate_text(system="s", user_text="u")
    assert slow.calls == 1

    clock.advance(1.0)
    chain.generate_text(system="s", user_text="u")
    assert slow.calls == 1  # still cooling down

    clock.advance(60.0)
    slow.errors = []
    slow.reply = "back now"
    assert chain.generate_text(system="s", user_text="u") == "back now"


def test_success_resets_a_providers_backoff_counter(monkeypatch):
    clock = _install_clock(monkeypatch)
    provider = FakeProvider("groq", errors=[LlmRateLimitError("1")], reply="ok")
    backup = FakeProvider("ollama", reply="backup")
    chain = LlmChain([provider, backup])

    chain.generate_text(system="s", user_text="u")  # fails once, backs off
    clock.advance(10.0)
    provider.errors = []
    chain.generate_text(system="s", user_text="u")  # succeeds, should reset the counter
    assert chain._status["groq"].consecutive_failures == 0
    assert chain._status["groq"].available_at == 0.0


def test_all_providers_cooling_down_raises_with_a_clear_reason(monkeypatch):
    clock = _install_clock(monkeypatch)
    a = FakeProvider("gemini", errors=[LlmAuthError("401")])
    b = FakeProvider("groq", errors=[LlmRateLimitError("429")])
    chain = LlmChain([a, b])

    # First call: both providers are still tried once each, and both fail —
    # gemini gets disabled, groq gets backed off.
    try:
        chain.generate_text(system="s", user_text="u")
        assert False, "expected the first call to fail too"
    except LlmProviderError:
        pass
    assert a.calls == 1 and b.calls == 1

    # Second call: both should be skipped up front, without another attempt.
    try:
        chain.generate_text(system="s", user_text="u")
        assert False, "expected the second call to fail as well"
    except LlmProviderError as exc:
        message = str(exc)

    assert a.calls == 1 and b.calls == 1  # neither was actually retried
    assert "gemini" in message and "disabled" in message
    assert "groq" in message and "cooling down" in message




def test_seconds_until_available_lets_a_batch_caller_wait_instead_of_skipping(monkeypatch):
    """The chain fails fast for interactive callers, but a batch job should
    sleep through a cooldown rather than abandon work — MEASURED 2026-09-04:
    one rate-limit cascade cost 37 of 51 benchmark documents, each failing
    instantly with "cooling down for Ns more" without an API attempt."""
    clock = _install_clock(monkeypatch)
    chain = LlmChain([FakeProvider("gemini", errors=[LlmRateLimitError("429")])])

    assert chain.seconds_until_available() is None  # healthy: usable right now

    try:
        chain.generate_text(system="s", user_text="u")
    except LlmProviderError:
        pass

    wait = chain.seconds_until_available()
    assert wait is not None and 0 < wait <= 5.0  # first 429 -> ~5s backoff

    clock.advance(wait + 1)
    assert chain.seconds_until_available() is None  # waiting it out restores it


def test_seconds_until_available_is_infinite_when_waiting_cannot_help(monkeypatch):
    """A disabled provider (bad key, dead billing) never recovers on a timer,
    so a batch caller must not sit waiting for it."""
    import math
    _install_clock(monkeypatch)
    chain = LlmChain([FakeProvider("gemini", errors=[LlmAuthError("401")])])
    try:
        chain.generate_text(system="s", user_text="u")
    except LlmProviderError:
        pass
    assert chain.seconds_until_available() == math.inf
    assert LlmChain([]).seconds_until_available() == math.inf
