"""The detection engine: positioned words in, PII spans out.

Two families of detector run over every line, and they fail in different
directions, which is why both exist:

* **Pattern detectors** match a shape and then prove it — Aadhaar by Verhoeff
  checksum, PAN and IFSC by structure. They need no surrounding context, and
  they are what the verification pass treats as disqualifying, because a
  checksum-valid Aadhaar in an "anonymised" file is unambiguously a failure.

* **Label detectors** read "Patient Name:" and take what follows. They are the
  only way to find a name, and they are the source of nearly every false
  positive, which is why each candidate must clear `ProtectedVocabulary`
  before it is accepted, and why a column-gap heuristic stops a value running
  into the next field.

No LLM is involved at any point.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .config import DetectionConfig
from .patterns import (
    ADDRESS_LABELS,
    AGE_SEX_LABELS,
    DOB_LABELS,
    DOCTOR_NAME_LABELS,
    KIND_TO_FLAG,
    NAME_LABEL_NEGATIVE_PREFIXES,
    PATIENT_ID_LABELS,
    PATIENT_NAME_LABELS,
    PATTERNS,
    VALIDATORS,
    is_aadhaar,
    looks_like_null,
    normalise_label,
)
from .protect import ProtectedVocabulary
from .textsource import PageWords, Word, group_by_line

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    kind: str
    text: str
    words: tuple[Word, ...]
    page: int
    rule: str
    source: str
    confidence: float = 1.0

    @property
    def bboxes(self) -> list[tuple[float, float, float, float]]:
        return [w.bbox for w in self.words]


# Label groups, paired with the detection kind they produce and the config
# flag that enables them.
_LABEL_GROUPS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("PATIENT_NAME", PATIENT_NAME_LABELS, "patient_name"),
    ("DOCTOR_NAME", DOCTOR_NAME_LABELS, "doctor_name"),
    ("ADDRESS", ADDRESS_LABELS, "address"),
    ("DOB", DOB_LABELS, "date_of_birth"),
    ("PATIENT_ID", PATIENT_ID_LABELS, "patient_ids"),
    ("AGE_SEX", AGE_SEX_LABELS, "age_sex"),
)

# Every label string, used as a stop condition so one field's value cannot run
# into the next field's label.
_ALL_LABELS = frozenset(
    lbl for _, labels, _ in _LABEL_GROUPS for lbl in labels
) | frozenset({
    "test", "result", "unit", "units", "method", "reference", "range",
    "specimen", "sample", "collected", "received", "reported", "status",
    "department", "sex", "gender", "age", "date", "time", "bill no", "invoice",
})


def _line_text_with_offsets(words: list[Word]) -> tuple[str, list[tuple[int, int, Word]]]:
    """Join a line, remembering where each word landed in the joined string."""
    parts, spans, cursor = [], [], 0
    for w in words:
        if parts:
            cursor += 1  # the single space we are about to add
        parts.append(w.text)
        spans.append((cursor, cursor + len(w.text), w))
        cursor += len(w.text)
    return " ".join(parts), spans


def _words_for_span(
    spans: list[tuple[int, int, Word]], start: int, end: int
) -> tuple[Word, ...]:
    """Every word overlapping [start, end) in the joined line text."""
    return tuple(w for s, e, w in spans if s < end and e > start)


def _detect_patterns(
    line_words: list[Word], cfg: DetectionConfig, vocab: ProtectedVocabulary,
) -> list[Detection]:
    text, spans = _line_text_with_offsets(line_words)
    found: list[Detection] = []

    for kind, pattern in PATTERNS.items():
        flag = KIND_TO_FLAG.get(kind)
        if flag and not getattr(cfg, flag, False):
            continue
        for m in pattern.finditer(text):
            value = m.group()

            if kind == "AADHAAR":
                if not is_aadhaar(value, cfg.aadhaar_require_checksum):
                    continue
            validator = VALIDATORS.get(kind)
            if validator and not validator(value):
                continue

            protected, reason = vocab.is_protected(kind, value)
            if protected:
                log.debug("pattern %s rejected (%s): %r", kind, reason, value)
                continue

            matched = _words_for_span(spans, m.start(), m.end())
            if not matched:
                continue
            found.append(Detection(
                kind=kind, text=value, words=matched, page=matched[0].page,
                rule=f"pattern:{kind.lower()}", source=matched[0].source,
                confidence=min(w.confidence for w in matched),
            ))
    return found


def _match_label_at(
    line_words: list[Word], index: int, labels: tuple[str, ...],
) -> int:
    """Length in tokens of a label starting at `index`, or 0 for no match.

    Tries the longest window first so "patient name" wins over "patient".
    """
    for window in (4, 3, 2, 1):
        if index + window > len(line_words):
            continue
        phrase = normalise_label(" ".join(w.cleaned for w in line_words[index:index + window]))
        if phrase in labels:
            return window
    return 0


def _is_label_boundary(word: Word) -> bool:
    """Does this token begin another field's label?"""
    cleaned = normalise_label(word.cleaned)
    return bool(cleaned) and cleaned in _ALL_LABELS


def _median_char_width(line_words: list[Word]) -> float:
    widths = [
        (w.bbox[2] - w.bbox[0]) / max(len(w.text), 1)
        for w in line_words if w.text.strip()
    ]
    if not widths:
        return 0.0
    widths.sort()
    return widths[len(widths) // 2]


def _collect_value(
    line_words: list[Word], start: int, max_tokens: int,
) -> list[Word]:
    """Take the value that follows a label, stopping at the field boundary.

    Three stop conditions, all of them cheap and all of them necessary:
    the next field's label, the token budget, and a horizontal gap wide enough
    to be a column break. Without the gap rule, "Patient Name: Ramesh Kumar
    Age: 45" on one visual line would swallow the age as part of the name on a
    layout where the label boundary is missed.
    """
    taken: list[Word] = []
    gap_limit = _median_char_width(line_words) * 6 or None

    for i in range(start, min(start + max_tokens, len(line_words))):
        word = line_words[i]
        if _is_label_boundary(word):
            break
        if taken and gap_limit is not None:
            if word.bbox[0] - taken[-1].bbox[2] > gap_limit:
                break
        if word.cleaned in ("|", "/", "\\") and taken:
            break
        taken.append(word)

    # Drop trailing separator tokens the value does not own.
    while taken and not any(c.isalnum() for c in taken[-1].text):
        taken.pop()
    return taken


def _detect_labels(
    line_words: list[Word], cfg: DetectionConfig, vocab: ProtectedVocabulary,
    extra_patient_labels: tuple[str, ...] = (),
) -> list[Detection]:
    found: list[Detection] = []
    i = 0
    while i < len(line_words):
        matched_here = False
        for kind, labels, flag in _LABEL_GROUPS:
            if not getattr(cfg, flag, False):
                continue
            search = labels + extra_patient_labels if kind == "PATIENT_NAME" else labels
            width = _match_label_at(line_words, i, search)
            if not width:
                continue

            # "Test Name" / "Method Name" are laboratory columns, not identity.
            if kind == "PATIENT_NAME" and i > 0:
                prev = normalise_label(line_words[i - 1].cleaned)
                if prev in NAME_LABEL_NEGATIVE_PREFIXES:
                    continue

            budget = (cfg.max_value_tokens_address if kind == "ADDRESS"
                      else cfg.max_value_tokens_name)
            value_words = _collect_value(line_words, i + width, budget)
            if not value_words:
                i += width
                matched_here = True
                break

            value = " ".join(w.text for w in value_words)
            if looks_like_null(value):
                i += width + len(value_words)
                matched_here = True
                break

            protected, reason = vocab.is_protected(kind, value)
            if protected:
                log.debug("label %s rejected (%s): %r", kind, reason, value)
                i += width
                matched_here = True
                break

            found.append(Detection(
                kind=kind, text=value, words=tuple(value_words),
                page=value_words[0].page, rule=f"label:{kind.lower()}",
                source=value_words[0].source,
                confidence=min(w.confidence for w in value_words),
            ))
            i += width + len(value_words)
            matched_here = True
            break
        if not matched_here:
            i += 1
    return found


def detect_page(
    page: PageWords,
    cfg: DetectionConfig,
    vocab: ProtectedVocabulary,
    extra_patient_labels: tuple[str, ...] = (),
) -> list[Detection]:
    """Every detection on one page, de-duplicated by covered word."""
    detections: list[Detection] = []
    for line_words in group_by_line(page.words):
        detections.extend(_detect_patterns(line_words, cfg, vocab))
        detections.extend(_detect_labels(line_words, cfg, vocab, extra_patient_labels))

    # A value can be caught by both families — "UHID: ABCDE1234F" is a patient
    # id by label and a PAN by pattern. Keep the pattern hit, which is the more
    # specific claim, and drop the label hit covering the same words.
    by_words: dict[tuple, Detection] = {}
    for d in sorted(detections, key=lambda d: (0 if d.rule.startswith("pattern") else 1)):
        key = tuple(sorted((w.page, w.line, w.order) for w in d.words))
        if key not in by_words:
            by_words[key] = d
    return list(by_words.values())


def detect_in_text(text: str, cfg: DetectionConfig, vocab: ProtectedVocabulary) -> list[tuple[str, str]]:
    """Pattern-only scan of raw text, for the verification pass.

    Verification has no boxes and does not need them — it only has to answer
    "is a valid identifier still present?".
    """
    hits: list[tuple[str, str]] = []
    for kind, pattern in PATTERNS.items():
        flag = KIND_TO_FLAG.get(kind)
        if flag and not getattr(cfg, flag, False):
            continue
        for m in pattern.finditer(text):
            value = m.group()
            if kind == "AADHAAR" and not is_aadhaar(value, cfg.aadhaar_require_checksum):
                continue
            validator = VALIDATORS.get(kind)
            if validator and not validator(value):
                continue
            protected, _ = vocab.is_protected(kind, value)
            if protected:
                continue
            hits.append((kind, value))
    return hits
