"""Builders and metrics shared by the suite.

Ground truth is generated, never hand-typed: a document is built from known
text (born-digital PDF), rendered to pixels (a "scan"), optionally degraded,
read back through the real pipeline, and compared with the text it started
from. That round trip is what makes the OCR assertions exact rather than
"looks plausible".

Real dataset files are only ever READ (see conftest.dataset_file). Nothing
from the dataset is copied into the repository.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Ground-truth content: a lab report with the traps that break exact text.
# --------------------------------------------------------------------------

#: Every line is something a NABL reviewer needs verbatim. CamelCase names,
#: dates with times, decimals, ranges with spaced dashes, Indian digit
#: grouping and "<" qualifiers are the cases OCR and post-processing mangle.
REPORT_LINES = [
    "LABORATORY TEST REPORT",
    "Patient Name: RaghuRam",
    "Referred By: Dr. McDonald",
    "Sample ID: NABL-TC-1234",
    "Collected: 25/08/2021 10:04",
    "Reported: 26/08/2021 15:28",
    "Hemoglobin 13.2 g/dL 13.0 - 17.0",
    "Creatinine 0.15 mg/dL 0.60 - 1.20",
    "Total Count 1,00,000 /cumm",
    "TSH < 0.01 uIU/mL",
]

DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
NUM_RE = re.compile(r"(?<![\w.])\d[\d,]*\.?\d*(?![\w])")


# --------------------------------------------------------------------------
# Document builders (PyMuPDF base-14 fonts: identical on every OS)
# --------------------------------------------------------------------------


def text_pdf(pages: list[list[str]], *, width: float = 595, height: float = 340,
             fontsize: float = 12) -> bytes:
    """Born-digital PDF: its text layer IS the ground truth."""
    import fitz

    doc = fitz.open()
    for lines in pages:
        page = doc.new_page(width=width, height=height)
        y = 40.0
        for line in lines:
            page.insert_text((40, y), line, fontsize=fontsize, fontname="helv")
            y += fontsize * 2.0
    out = doc.tobytes()
    doc.close()
    return out


def rasterize(pdf_bytes: bytes, page_index: int = 0, dpi: int = 200) -> bytes:
    import fitz

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        pix = doc[page_index].get_pixmap(dpi=dpi)
        return pix.tobytes("png")


def scanned_pdf(images: list[bytes | None]) -> bytes:
    """Image-only PDF (no text layer). `None` makes a genuinely blank page."""
    import fitz

    doc = fitz.open()
    for img in images:
        page = doc.new_page(width=595, height=340)
        if img:
            page.insert_image(page.rect, stream=img)
    out = doc.tobytes()
    doc.close()
    return out


def degrade(png: bytes, kind: str) -> bytes:
    """Apply one real-world scan defect. `clean` returns the input unchanged."""
    if kind == "clean":
        return png
    import numpy as np
    from PIL import Image, ImageFilter

    im = Image.open(io.BytesIO(png)).convert("RGB")
    if kind == "skew_2deg":
        im = im.rotate(2, expand=True, fillcolor="white", resample=Image.BICUBIC)
    elif kind == "rotated_90":
        im = im.rotate(90, expand=True, fillcolor="white")
    elif kind == "low_dpi_100":
        w, h = im.size
        im = im.resize((w // 2, h // 2), Image.BILINEAR)
    elif kind == "noise":
        arr = np.asarray(im).astype(np.int16)
        rng = np.random.default_rng(7)  # fixed seed: same "noise" every run
        arr = np.clip(arr + rng.normal(0, 22, arr.shape), 0, 255).astype(np.uint8)
        im = Image.fromarray(arr)
    elif kind == "blur":
        im = im.filter(ImageFilter.GaussianBlur(1.6))
    elif kind == "jpeg_q25":
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=25)
        return buf.getvalue()
    else:
        raise ValueError(kind)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def tokens(text: str) -> list[str]:
    return text.split()


@dataclass
class TextScore:
    exact_word_recall: float
    glued_words: int
    dates_expected: int
    dates_found: int
    numbers_recall: float
    missing: list[str] = field(default_factory=list)
    glued_examples: list[str] = field(default_factory=list)

    @property
    def date_recall(self) -> float:
        return self.dates_found / self.dates_expected if self.dates_expected else 1.0

    def as_dict(self) -> dict:
        return {
            "exact_word_recall": round(self.exact_word_recall, 3),
            "date_recall": round(self.date_recall, 3),
            "numbers_recall": round(self.numbers_recall, 3),
            "glued_words": self.glued_words,
        }


def score(expected_text: str, got_text: str) -> TextScore:
    """Exact, case-sensitive comparison of what a document says vs. what we read.

    * exact_word_recall: expected whitespace tokens found verbatim (multiset).
    * glued_words: read tokens that are 2+ consecutive expected tokens fused
      together ("PatientName:", "25/08/202110:04") — the RaghuRam failure mode
      in reverse, and the single most common OCR defect on this dataset.
    * dates / numbers: the values a NABL form actually needs, counted exactly.
    """
    exp, got = tokens(expected_text), tokens(got_text)
    pool: dict[str, int] = {}
    for t in got:
        pool[t] = pool.get(t, 0) + 1
    hit, missing = 0, []
    for t in exp:
        if pool.get(t, 0) > 0:
            pool[t] -= 1
            hit += 1
        else:
            missing.append(t)

    fused = set()
    for i in range(len(exp) - 1):
        for j in range(i + 2, min(i + 4, len(exp)) + 1):
            fused.add("".join(exp[i:j]))
    glued = [t for t in got if t in fused]

    exp_dates = DATE_RE.findall(expected_text)
    got_tokens = set(got)
    dates_found = sum(1 for d in exp_dates if d in got_tokens)

    exp_nums = NUM_RE.findall(expected_text)
    got_nums = NUM_RE.findall(got_text)
    npool: dict[str, int] = {}
    for n in got_nums:
        npool[n] = npool.get(n, 0) + 1
    nhit = 0
    for n in exp_nums:
        if npool.get(n, 0) > 0:
            npool[n] -= 1
            nhit += 1

    return TextScore(
        exact_word_recall=hit / len(exp) if exp else 1.0,
        glued_words=len(glued),
        dates_expected=len(exp_dates),
        dates_found=dates_found,
        numbers_recall=nhit / len(exp_nums) if exp_nums else 1.0,
        missing=missing[:12],
        glued_examples=glued[:6],
    )


# --------------------------------------------------------------------------
# A scripted LLM, so the end-to-end flow runs with no network call
# --------------------------------------------------------------------------


class RecordingLlm:
    """Answers every LLM node with an empty-but-valid reply and records the
    text it was shown. Extraction quality is not under test here; what is
    under test is that the text OCR produced actually reaches the model."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, *, node, system, user_text, output_model, images=None,
                 image_media_type=None, max_attempts=None, on_metric=None):
        from app.graph.llm import LlmOutcome

        self.calls.append({"node": node, "user_text": user_text})
        try:
            parsed = output_model.model_validate({})
        except Exception:  # noqa: BLE001 — some outputs need a chunk id
            parsed = None
        if parsed is None:
            import re as _re

            cid = _re.search(r"chunk[_ ]id[\"':\s]+([\w-]+)", user_text or "")
            try:
                parsed = output_model.model_validate(
                    {"chunk_id": cid.group(1) if cid else "c1", "fields": [],
                     "tables_seen": [], "notes": ""})
            except Exception as exc:  # noqa: BLE001
                from app.graph.errors import ErrorType

                return LlmOutcome(ok=False, error_type=ErrorType.LLM_SCHEMA_MISMATCH,
                                  error_message=str(exc)[:200])
        return LlmOutcome(ok=True, parsed=parsed, raw={}, attempts=1)

    def text_seen(self, node: str = "dynamic_extraction") -> str:
        return "\n".join(c["user_text"] or "" for c in self.calls if c["node"] == node)
