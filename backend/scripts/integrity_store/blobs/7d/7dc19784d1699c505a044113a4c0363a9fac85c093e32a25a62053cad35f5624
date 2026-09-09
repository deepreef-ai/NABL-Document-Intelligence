""""Is this page's PyMuPDF text actually usable, or is it noise?" — the one
decision every PDF page makes before choosing pymupdf-text vs OCR.

Deliberately NOT just a character-count check (a page containing only a
header, a page number, or a watermark's stray characters can easily clear
20-30 raw characters while containing no real content) — three independent,
configurable signals must all pass:
  1. enough raw characters at all
  2. most of those characters are actually alphanumeric, not symbol/
     whitespace noise
  3. enough distinct "word-shaped" tokens (guards against e.g. "12  34  56"
     or a wall of punctuation passing the first two checks)
"""
from dataclasses import dataclass


@dataclass
class TextQualityThresholds:
    min_chars: int = 30
    min_alnum_ratio: float = 0.5
    min_word_count: int = 3


DEFAULT_THRESHOLDS = TextQualityThresholds()


def is_meaningful_page_text(text: str, thresholds: TextQualityThresholds = DEFAULT_THRESHOLDS) -> bool:
    stripped = (text or "").strip()
    if len(stripped) < thresholds.min_chars:
        return False

    alnum_count = sum(1 for c in stripped if c.isalnum())
    if alnum_count / len(stripped) < thresholds.min_alnum_ratio:
        return False

    word_count = sum(1 for w in stripped.split() if any(c.isalpha() for c in w))
    if word_count < thresholds.min_word_count:
        return False

    return True
