"""Repairs a specific, verified RapidOCR failure mode: dense paragraph text
on scans/images is sometimes recognized as one long unbroken run of glued-
together words with NO space characters at all — confirmed present at the
individual OCR-LINE level (i.e. it is RapidOCR's own word-boundary
detection failing, not this project's line-joining), across both raw
images and scanned/mixed PDF pages.

Only touches alphabetic runs long enough to be UNAMBIGUOUSLY multiple
words. An ordinary English word essentially never exceeds ~20 characters
("electroencephalography" is 23) — past that threshold, resegmenting is
safe. wordninja's dictionary is general English, not medical/technical —
it visibly mangles domain terms ("pseudohyphal" -> "pseudo hy ph al") — so
this is a best-effort mitigation for the worst, most damaging failures
(whole sentences fused into one 80+ character token), not a complete fix.
Below the threshold, text is left untouched rather than risk corrupting a
real (if unusual) single word.
"""
from __future__ import annotations

import re

import wordninja

MIN_GLUED_RUN_LENGTH = 20
_GLUED_RUN = re.compile(rf"[A-Za-z]{{{MIN_GLUED_RUN_LENGTH},}}")


def repair_glued_words(text: str) -> str:
    if not text:
        return text
    return _GLUED_RUN.sub(lambda m: " ".join(wordninja.split(m.group(0))), text)
