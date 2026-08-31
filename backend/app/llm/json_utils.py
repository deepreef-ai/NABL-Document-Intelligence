import json
import re


class JsonParseError(RuntimeError):
    pass


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


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

    raise JsonParseError(f"no JSON object found in model output: {text[:200]!r}")
