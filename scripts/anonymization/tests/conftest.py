"""Test fixtures. Puts the standalone package on sys.path and builds sample
documents in memory, so the suite needs no checked-in binary fixtures and no
access to real patient data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

import fitz  # noqa: E402  (PyMuPDF, imported after the path fix)

# A synthetic report carrying both things the tool must get right: identifiers
# that have to disappear, and laboratory values that have to survive.
# The Aadhaar number below is Verhoeff-valid and entirely made up.
PII_AADHAAR = "2341 2345 6783"
PII_PAN = "ABCPK1234F"
PII_EMAIL = "ramesh.kumar@example.com"
PII_PHONE = "9876543210"
PII_NAME = "Ramesh Kumar"

LAB_LINES = [
    "Haemoglobin        12.4    g/dL     13.0-17.0",
    "Glucose Fasting    92      mg/dL    70-100",
    "Test Name: Complete Blood Count",
    "Sample Type: Serum      Method: Photometry",
]


def make_text_pdf(path: Path, *, with_pii: bool = True) -> Path:
    """A born-digital PDF with a real text layer."""
    doc = fitz.open()
    page = doc.new_page()
    y = 60
    lines = ["ACME DIAGNOSTICS LABORATORY", "NABL Accredited Testing Laboratory", ""]
    if with_pii:
        lines += [
            f"Patient Name: {PII_NAME}",
            f"UHID: UH0099123    Age/Sex: 45/M",
            f"Address: 14 Nehru Road, Pune 411001",
            f"Mobile: {PII_PHONE}    Email: {PII_EMAIL}",
            f"Aadhaar: {PII_AADHAAR}    PAN: {PII_PAN}",
            "Referred By: Dr Anita Sharma",
            "",
        ]
    lines += ["TEST                RESULT  UNIT     REFERENCE"] + LAB_LINES
    for line in lines:
        page.insert_text((50, y), line, fontsize=10, fontname="helv")
        y += 16
    doc.save(path)
    doc.close()
    return path


def make_image_pdf(path: Path) -> Path:
    """A PDF whose page carries an image and no extractable text."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1200, 500), "white")
    draw = ImageDraw.Draw(img)
    draw.text((40, 60), f"Patient Name: {PII_NAME}", fill="black")
    draw.text((40, 100), f"Aadhaar: {PII_AADHAAR}", fill="black")
    draw.text((40, 140), "Haemoglobin 12.4 g/dL", fill="black")

    png = path.with_suffix(".fixture.png")
    img.save(png)
    doc = fitz.open()
    page = doc.new_page(width=600, height=250)
    page.insert_image(fitz.Rect(0, 0, 600, 250), filename=str(png))
    doc.save(path)
    doc.close()
    png.unlink(missing_ok=True)
    return path


@pytest.fixture
def text_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "in" / "report.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    return make_text_pdf(path)


@pytest.fixture
def scanned_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "in" / "scan.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    return make_image_pdf(path)


@pytest.fixture
def input_root(tmp_path: Path) -> Path:
    root = tmp_path / "in"
    (root / "sub").mkdir(parents=True, exist_ok=True)
    make_text_pdf(root / "report.pdf")
    make_text_pdf(root / "sub" / "nested.pdf")
    return root


@pytest.fixture
def output_root(tmp_path: Path) -> Path:
    return tmp_path / "out"
