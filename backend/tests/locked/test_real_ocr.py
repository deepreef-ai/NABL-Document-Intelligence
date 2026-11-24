"""LOCKED integration on a REAL scanned lab report with the REAL deepreef
engine. Skipped (not failed) when the PDF, its answer key or the models are
absent; otherwise it runs by default."""
from __future__ import annotations

import json
import os
import re

import pytest

_KEY = "/Users/mihirpandey/Desktop/untitled folder 2/benchmarks/ingredient_lab_reports.key.json"
_PDF = ("/Users/mihirpandey/Downloads/users_fPIDy8u7DyToson4gMJsRmt2Ni93_qr_codes_K3fSi8lr_"
        "1789556040156_1784809847505_3_Ingredient_Lab_Reports-_23.07.26.pdf")


def _token_on(line: str, value: str) -> bool:
    """`value` appears as a whole token (so "0.70" is not found inside "10.70")."""
    return re.search(rf"(?<![\w.]){re.escape(value)}(?![\d])", line) is not None


@pytest.mark.slow
def test_real_scanned_pages_keep_every_expected_result_on_one_line():
    """Why this must never break: row verification works line by line, so a
    result the OCR detaches from its analyte name (or misreads) is a lost
    result on an attested report. On two real pages every key row must appear
    with its name and its result on one OCR line, and the count of "<0.01" /
    "17.16" lines must match the answer key exactly."""
    if not (os.path.exists(_KEY) and os.path.exists(_PDF)):
        pytest.skip("real report or answer key not present")
    from app.documents import deepreef_local

    try:
        deepreef_local._engine()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"deepreef models do not load: {exc}")

    key = json.load(open(_KEY, encoding="utf-8"))
    pdf = open(_PDF, "rb").read()

    def lines(page_no: int) -> list[str]:
        png = deepreef_local.render_for_ocr(pdf, page_no, 200)
        return deepreef_local.extract_page(png, rendered_pdf_page=True).lines

    page7 = lines(7)
    missing = []
    for name, result in key["exact"]["7"]:
        words = [w.lower() for w in re.findall(r"\w+", name)]
        if not any(all(re.search(rf"\b{re.escape(w)}\b", ln.lower()) for w in words) and _token_on(ln, result) for ln in page7):
            missing.append((name, result))
    assert not missing, f"page 7 rows not on one OCR line: {missing}\n" + "\n".join(page7)

    page17 = lines(17)
    for value, expected in key["count_pages"]["17"]["values"].items():
        got = sum(1 for ln in page17 if _token_on(ln, value))
        assert got == expected, f"page 17: {got} lines with {value!r}, key says {expected}\n" + "\n".join(page17)
