"""Value equivalence for scoring: when do two differently-written values mean
the same thing?

compare.py's _normalize_value already folds representation noise — case, all
whitespace, separator punctuation, unicode dashes, trailing sentence
punctuation. This adds the layers above that, where the two strings differ but
the READING does not:

  numeric      3.5 == 3.50 == 3.5000,  5 == 5.0,  0.35 == .35
  range flags  "0.35 High" == "0.35"   (the flag is derived from the result
                                        and the reference range, both already
                                        compared on their own)
  synonyms     Male == M,  Positive == +ve,  Absent == Not Detected
  dates        02 Nov 2020 == 2020-11-02 == 02/11/2020

Deliberately NOT equivalent — these differ in reading, not in spelling:

  6.5   vs 65       a decimal point is meaning, not punctuation
  40-129 vs 40129   a range is not a number
  09:55:20 vs 03/02/2019 09:55:20    a dropped date is a dropped value

Kept OUT of compare.py on purpose. That module scores the existing
predictions_* runs; changing its rules would silently move every historical
number and make predictions_v2's 65.1% incomparable with anything scored
afterwards. This is additive and opt-in.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from app.benchmark.compare import _normalize_field_value, _normalize_value

# A bare number, optionally followed by a unit or a range flag. Anchored, so an
# internal dash ("40-129") never parses — a range is not a number.
_NUMBER_WITH_SUFFIX = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+))([a-z%°/]*)$")

# An out-of-range marker printed beside a result ("0.35 High", "4.2 L"). It is
# derived from the result and the reference range — both compared on their own
# — so it carries no independent information, and which side chose to
# transcribe it must not decide the score. Two CONTRADICTING flags are a real
# disagreement and still count as a mismatch.
_RANGE_FLAGS = {
    "high": "high", "h": "high",
    "low": "low", "l": "low",
    "abnormal": "abnormal", "a": "abnormal",
    "critical": "critical",
    "borderline": "borderline",
    "normal": "normal", "n": "normal",
}

# Values that are the same reading spelled differently. Grouped rather than
# aliased pairwise so membership is obvious and adding a spelling is one edit.
#
# A qualitative result is ONE binary axis however the lab words it, confirmed
# for this corpus: positive / present / detected / reactive all say the same
# thing, and so do negative / absent / not detected / non-reactive. Splitting
# "absent" from "negative" (an earlier version did) scored a correct reading as
# wrong whenever the label and the report chose different wording for it.
#
# The opposing groups stay opposed, obviously — "present" never equals
# "absent" — so a genuine mis-reading is still caught.
_SYNONYM_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"male", "m"}),
    frozenset({"female", "f"}),
    frozenset({"positive", "pos", "+ve", "+", "reactive", "present", "detected"}),
    frozenset({
        # Hyphenated forms listed separately: _normalize_value keeps "-"
        # (it carries numeric meaning in ranges), so "Non-Reactive"
        # normalizes to "non-reactive", not "nonreactive".
        "negative", "neg", "-ve", "nonreactive", "non-reactive",
        "notreactive", "not-reactive",
        "absent", "notdetected", "nildetected", "notseen",
        # "nd" is ambiguous in the wild — Not Detected here, but some labs
        # write it for "Not Done", which is a genuinely different reading.
        # Kept because this corpus uses it for Not Detected; drop it if a
        # lab that means Not Done is ever added.
        "nd",
    }),
    frozenset({"yes", "y"}),
    frozenset({"no", "n"}),
)
_SYNONYM_OF: dict[str, int] = {
    spelling: index for index, group in enumerate(_SYNONYM_GROUPS) for spelling in group
}

# Day-first, matching the corpus: an Indian lab report writes 02/03/2020 as
# 2 March. Month-first is never attempted, because trying both would silently
# equate two genuinely different dates whenever the day is <= 12.
_DATE_FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d",
    "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",
    "%d %b %Y", "%d-%b-%Y", "%d/%b/%Y", "%d %B %Y", "%d-%B-%Y",
    "%b %d %Y", "%B %d %Y",
    "%d-%m-%y", "%d/%m/%y",
)
_TIME_SUFFIXES = ("", " %H:%M", " %H:%M:%S", " %H:%M:%S %p", " %I:%M %p", " %I:%M:%S %p")


def _as_number(normalized: str) -> tuple[Decimal, str] | None:
    """(value, suffix) for a bare number with an optional trailing unit/flag."""
    match = _NUMBER_WITH_SUFFIX.match(normalized)
    if not match:
        return None
    try:
        return Decimal(match.group(1)), match.group(2)
    except InvalidOperation:
        return None


def _numbers_agree(left: str, right: str) -> bool:
    a, b = _as_number(left), _as_number(right)
    if a is None or b is None:
        return False
    if a[0] != b[0]:
        return False

    left_suffix, right_suffix = a[1], b[1]
    if left_suffix == right_suffix:
        return True
    # A range flag on one side only is not a disagreement; a unit is.
    left_flag = _RANGE_FLAGS.get(left_suffix, "" if not left_suffix else None)
    right_flag = _RANGE_FLAGS.get(right_suffix, "" if not right_suffix else None)
    if left_flag is None or right_flag is None:
        return False
    return not (left_flag and right_flag and left_flag != right_flag)


def _synonyms_agree(left: str, right: str) -> bool:
    group = _SYNONYM_OF.get(left)
    return group is not None and group == _SYNONYM_OF.get(right)


def _as_datetime(raw) -> tuple[date, bool, datetime] | None:
    """(date, carries_a_time, full) parsed from the RAW value.

    Raw, not normalized: normalization strips the separators a date format is
    made of. Returns None for anything that isn't a full date — a bare time
    ("09:55:20") must never match a datetime that contains it.
    """
    if raw is None:
        return None
    text = re.sub(r"\s+", " ", str(raw).strip())
    if not text:
        return None
    for fmt in _DATE_FORMATS:
        for time_suffix in _TIME_SUFFIXES:
            try:
                parsed = datetime.strptime(text, fmt + time_suffix)
            except ValueError:
                continue
            return parsed.date(), bool(time_suffix), parsed
    return None


def _dates_agree(left_raw, right_raw) -> bool:
    left, right = _as_datetime(left_raw), _as_datetime(right_raw)
    if left is None or right is None:
        return False
    if left[0] != right[0]:
        return False
    # One side carrying a time the other lacks is a dropped value, not a
    # formatting difference — same reasoning compare.py applies to truncation.
    if left[1] != right[1]:
        return False
    return left[2] == right[2]


def values_equal(gold, predicted, key: str = "") -> bool:
    """True when the two values are the same reading.

    `key` is passed through to compare.py's _normalize_field_value so the
    page-label rule ("Page 4 of 15" == "4 of 15") still applies to the page
    field. Every check below is additive: anything that matched before this
    module existed still matches.
    """
    gold_norm = _normalize_field_value(key, gold) if key else _normalize_value(gold)
    pred_norm = _normalize_field_value(key, predicted) if key else _normalize_value(predicted)

    if gold_norm is None or pred_norm is None:
        return gold_norm == pred_norm  # both absent counts as agreement
    if gold_norm == pred_norm:
        return True
    return (
        _numbers_agree(gold_norm, pred_norm)
        or _synonyms_agree(gold_norm, pred_norm)
        or _dates_agree(gold, predicted)
    )
