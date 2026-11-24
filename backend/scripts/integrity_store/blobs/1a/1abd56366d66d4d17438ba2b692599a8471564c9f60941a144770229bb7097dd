import json
import re


class JsonParseError(RuntimeError):
    pass


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _salvage_truncated(text: str) -> dict | None:
    """Rebuild a usable object out of a reply that was cut off mid-generation.

    MEASURED 2026-09-03: high-protein-paneer.pdf has a 273-row results table
    and lands right at Nova's output cap — the same request returned
    stopReason 'end_turn' at 17,786 output tokens on one attempt and
    'max_tokens' at 32,768 on the next. A truncated reply has unbalanced
    brackets, so both json.loads and the brace-scan below fail and the WHOLE
    document is lost, table and header fields alike. Recovering the rows that
    did arrive is strictly better than discarding 250-odd good rows over the
    one that got cut in half.

    Walks the text tracking string/escape state so that braces and commas
    inside string values are not mistaken for structure, remembers the last
    position at which a complete value ended, cuts there and closes whatever
    containers are still open.
    """
    start = text.find("{")
    if start == -1:
        return None

    stack: list[str] = []
    in_string = escaped = False
    cut: tuple[int, list[str]] | None = None

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            cut = (i + 1, list(stack))   # a complete value ends just past here
        elif ch == ",":
            cut = (i, list(stack))       # everything before the comma is complete

    if cut is None:
        return None
    end, still_open = cut
    closers = "".join("}" if c == "{" else "]" for c in reversed(still_open))
    try:
        parsed = json.loads(text[start:end] + closers)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_json_object(text: str) -> dict:
    """Free-tier models don't always honor 'JSON only' as strictly as we'd
    like — salvage a markdown-fenced or prose-wrapped object instead of
    failing the whole provider over cosmetic noise."""
    text = text.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    salvaged = _salvage_truncated(text)
    if salvaged is not None:
        return salvaged

    raise JsonParseError(f"no JSON object found in model output: {text[:200]!r}")
