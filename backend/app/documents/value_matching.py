"""Strict "is this extracted value really on the page?" check.

The old check stripped every space and asked whether the value was a plain
substring of the page text, so "5" was "found" in "-5" / "0.15", and "1.5" in
"0.15". Here:

  * a purely numeric value (optional sign / < > comparator) is compared AS A
    NUMBER against whole numeric tokens on the page — never a piece of a
    longer number, and the sign / comparator must agree
    (1.5 != 0.15, -5 != 5, <0.01 != 0.01; 1,00,000 == 100000; 7 == 007).
  * any other value (words, ranges, "13.2 g/dL") is searched for with
    case / whitespace / dash-style ignored, but it may not start or end in
    the middle of a number.
"""
import re
from decimal import Decimal, InvalidOperation
from functools import lru_cache

_DASH_VARIANTS = re.compile("[‐‑‒–—―−]")
_WHITESPACE = re.compile(r"\s+")
_COMPARATOR_FOLD = str.maketrans({"≤": "<=", "≥": ">="})

_NUM = r"\d+(?:,\d+)*(?:\.\d+)?"
_SIGN = r"(?:<=|>=|<|>|-|\+)"
# One numeric token on the page. Must not start inside a longer word/number
# (so "15" in "0.15" or "a15" is not a token) nor end inside one.
_PAGE_NUMBER = re.compile(
    rf"(?<![\w.,])(?P<sign>{_SIGN}(?=\d)|(?:<=|>=|<|>)\s)?(?P<num>{_NUM})(?![\w]|\.\d|,\d)"
)
_VALUE_NUMBER = re.compile(rf"\s*(?P<sign>{_SIGN})?\s*(?P<num>{_NUM})\s*")


def normalize_for_search(value) -> str:
    """Lowercases, unifies dash variants and comparator glyphs, collapses
    whitespace to single spaces (kept — token boundaries matter for numbers)."""
    if value is None:
        return ""
    text = _DASH_VARIANTS.sub("-", str(value).strip().lower()).translate(_COMPARATOR_FOLD)
    return _WHITESPACE.sub(" ", text)


def _to_decimal(num: str) -> Decimal | None:
    try:
        return Decimal(num.replace(",", ""))
    except InvalidOperation:
        return None


@lru_cache(maxsize=32)
def _page_numbers(source_norm: str) -> frozenset:
    found = set()
    for m in _PAGE_NUMBER.finditer(source_norm):
        d = _to_decimal(m.group("num"))
        if d is not None:
            found.add(((m.group("sign") or "").strip(), d))
    return frozenset(found)


@lru_cache(maxsize=32)
def _compact(source_norm: str) -> str:
    return source_norm.replace(" ", "")


def value_in_source(value, source_norm: str) -> bool:
    """True only when `value` is really in the (normalized) page text."""
    value_norm = normalize_for_search(value)
    if not value_norm or not source_norm:
        return False

    m = _VALUE_NUMBER.fullmatch(value_norm)
    if m:
        d = _to_decimal(m.group("num"))
        return d is not None and ((m.group("sign") or ""), d) in _page_numbers(source_norm)

    needle = re.escape(value_norm.replace(" ", ""))
    prefix = r"(?<![\d.,])" if value_norm[0].isdigit() else ""
    suffix = r"(?!\d|[.,]\d)" if value_norm[-1].isdigit() else ""
    return re.search(prefix + needle + suffix, _compact(source_norm)) is not None
