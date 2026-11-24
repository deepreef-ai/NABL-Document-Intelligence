"""Per-call "thinking" (reasoning) level for providers that support one.

Why a context variable rather than a `generate()` argument: the level is a
property of the *caller* (which graph node is asking), while the call passes
through LlmChain, which tries several providers that mostly do not support
the setting. Threading a new parameter through the chain and every provider
signature would break the plain text-in/text-out contract llm/base.py keeps.

A ContextVar is also safe under the graph's ThreadPoolExecutor: each worker
thread has its own context, and the value is set and reset around one
synchronous provider call on that same thread, so parallel chunk extractions
never see each other's level.

Levels follow Gemini 3's `thinkingLevel` vocabulary: minimal, low, medium,
high. None means "provider default" — exactly the behaviour before this
module existed.

MEASURED 2026-11-23 (gemini-3.6-flash, 14-row synthetic lab report): default
spent 989 thinking tokens and 7.8s; "minimal" and "low" spent 0 and ~5s, and
all three returned 14/14 test rows.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

VALID_LEVELS = ("minimal", "low", "medium", "high")

_level: ContextVar[str | None] = ContextVar("llm_thinking_level", default=None)


def normalise_level(level: str | None) -> str | None:
    """Blank/unknown -> None (provider default). Never raises: a typo in .env
    must not take extraction down, it just falls back to today's behaviour."""
    if not level:
        return None
    level = level.strip().lower()
    return level if level in VALID_LEVELS else None


def current_thinking_level() -> str | None:
    return _level.get()


@contextmanager
def use_thinking_level(level: str | None) -> Iterator[None]:
    token = _level.set(normalise_level(level))
    try:
        yield
    finally:
        _level.reset(token)


#: Gemini 2.x has no thinkingLevel; it takes a token budget instead.
#: 0 disables thinking (Flash models only), -1 is "dynamic" (the default).
_GEMINI_2_BUDGETS = {"minimal": 0, "low": 1024, "medium": 8192, "high": -1}


def gemini_thinking_config(model: str, level: str | None) -> dict | None:
    """The generationConfig.thinkingConfig for `model`, or None to omit it."""
    if level is None:
        return None
    name = model.rsplit("/", 1)[-1]
    if name.startswith("gemini-2"):
        return {"thinkingBudget": _GEMINI_2_BUDGETS[level]}
    return {"thinkingLevel": level}
