"""Real-OCR accuracy: round trips with generated ground truth + the real dataset.

Every case runs the REAL preprocessing node (`preprocess_document`) with the
REAL OCR engine; nothing is faked. Two kinds of assertion live here:

* BASELINE (hard): what the current path does today. A floor is the value
  MEASURED on this machine minus ~0.10, so a regression fails loudly while
  normal run-to-run jitter does not.
* TARGET (xfail strict): what Fix 1 must deliver. Each is a known gap today;
  when a fix closes it the test XPASSes, which strict mode reports as a failure
  so the xfail is removed deliberately rather than silently.

The engine is a parameter (`ENGINE_MODES`) so the same cases can be re-run for
another OCR path later; this file covers only the legacy path.

Slow by nature (~6-10 s per synthetic page, ~50 s for the dataset PNG, which
also gets a chart second pass). Each document is OCR'd ONCE per run and the
result is cached for the target assertions.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from tests.suite._support import (
    REPORT_LINES, degrade, rasterize, scanned_pdf, score, text_pdf,
)

pytestmark = pytest.mark.realocr

ENGINE_MODES = ["legacy"]

TRUTH = "\n".join(REPORT_LINES)

# Floors = measured - ~0.10 (exact_word_recall) on the legacy path, measured
# 2026-11-19 on the dev CPU. Measured values (exact / dates / numbers / glued / s):
#   clean        0.923 / 1.00 / 1.000 /  1 / 9.1  ("LABORATORYTESTREPORT")
#   skew_2deg    0.897 / 1.00 / 1.000 /  1 / 5.7
#   low_dpi_100  0.128 / 0.00 / 0.526 / 10 / 6.0
#   noise        1.000 / 1.00 / 1.000 /  0 / 7.6
#   jpeg_q25     0.333 / 0.00 / 0.789 /  7 / 6.0
#   blur         0.000 / 0.00 / 0.474 /  7 / 6.8  (every line glued)
#   rotated_90   0.410 / 0.00 / 0.789 /  8 / 7.3  (no orientation fix)
FLOORS = {  # kind: (exact_word_recall floor, numbers_recall floor)
    "clean": (0.82, 0.90),
    "skew_2deg": (0.79, 0.90),
    "low_dpi_100": (0.02, 0.42),
    "noise": (0.90, 0.90),
    "jpeg_q25": (0.23, 0.68),
    "blur": (0.00, 0.37),
    "rotated_90": (0.31, 0.68),
}
KINDS = list(FLOORS)

# Targets that fail today, per engine mode. Anything not listed is asserted
# hard. (On legacy, "all dates kept on clean" and "RaghuRam byte-exact" already
# pass, so they are ordinary assertions.)
KNOWN_GAPS = {
    "legacy": {
        "clean-no_glued_words": "legacy RapidOCR glues 'LABORATORYTESTREPORT'",
        "rotated_90-readable": "no orientation correction: 0.41 exact-word recall",
        "png-dates_exact": "legacy OCR glues '25/08/202110:04'",
        "png-heading_not_glued": "legacy OCR reads 'DIAGNOSTICREPORT'",
    },
}

# Verified by eye on 001_Lab-report.png (see SUITE_SPEC).
PNG_NAME = "001_Lab-report.png"
PNG_ANCHORS = ["DIAGNOSTIC REPORT", "PATIENT ID", "25/08/2021 10:04", "25/08/2021 10:07",
               "25/08/2021 15:28", "BILIRUBIN, TOTAL", "0.70", "mg/dL",
               "LACTATE DEHYDROGENASE", "239", "ALBUMIN", "Page 4 Of 15"]
PNG_DATES = ["25/08/2021 10:04", "25/08/2021 10:07", "25/08/2021 15:28"]
# Measured legacy (anchors joined): exact 0.23, dates 0.0, numbers 0.74, glued 6,
# ~49 s (includes the chart second pass).
PNG_FLOOR = 0.13

# Born-digital dataset PDF whose page 1 has a real text layer (1599 chars).
DIGITAL_NAME = "Chahal-Black-Mustard-Oil-Nutritional-Label-Test-NABL-May2026.pdf"
# Measured legacy: exact 0.512, dates 1.0, numbers 0.938, glued 26, ~30 s.
DIGITAL_FLOOR = 0.41

_CACHE: dict[tuple[str, str], dict] = {}


# --------------------------------------------------------------------------
# Helpers / fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def ocr_mode(request, monkeypatch, deepreef_path):
    mode = request.param
    if mode == "legacy":
        return mode  # deepreef_path already makes legacy local OCR importable
    pytest.skip(f"engine mode {mode!r} is not covered by this file")


def _metric(trace, **values) -> None:
    """trace.metric + publish to the report now. conftest attaches metrics at
    fixture TEARDOWN, after the call-phase report was already logged, so
    without this the OCR accuracy table stays empty. Same dict objects, so the
    teardown copy changes nothing."""
    trace.metric(**values)
    props = trace.item.user_properties
    if not any(k == "metrics" for k, _ in props):
        props += [("steps", trace.steps), ("metrics", trace.metrics)]


def _gap(request, mode: str, key: str) -> None:
    reason = KNOWN_GAPS.get(mode, {}).get(key)
    if reason:
        request.applymarker(pytest.mark.xfail(strict=True, reason=reason))


def _engine(out: dict, mode: str) -> str:
    for e in out.get("audit_log", []):
        if e.event == "page_ocr" and e.detail.startswith("read by "):
            return f"{mode}-{e.detail[len('read by '):]}"
    return f"{mode}-rapidocr"


def _run(trace, path: Path, file_type: str) -> tuple[dict, float]:
    from app.graph.nodes.preprocess import preprocess_document
    from app.graph.state import GraphState

    state = GraphState(document_id="t", file_path=str(path), file_type=file_type,
                       script="english")
    with trace.step(f"preprocess ({file_type}, real OCR)"):
        t0 = time.perf_counter()
        out = preprocess_document(state)
        dt = time.perf_counter() - t0
    return out, dt


def _cached(mode: str, case: str, build):
    if (mode, case) not in _CACHE:
        _CACHE[(mode, case)] = build()
    return _CACHE[(mode, case)]


def _round_trip(mode, kind, trace, tmp_path) -> dict:
    def build():
        with trace.step(f"build scan ({kind})"):
            png = degrade(rasterize(text_pdf([REPORT_LINES]), dpi=200), kind)
            path = tmp_path / f"{kind}.pdf"
            path.write_bytes(scanned_pdf([png]))
        out, dt = _run(trace, path, "pdf")
        text = out["page_text"].get(1, "")
        return {"out": out, "text": text, "seconds": dt, "score": score(TRUTH, text)}
    return _cached(mode, kind, build)


def _png(mode, trace, dataset_file) -> dict:
    def build():
        out, dt = _run(trace, dataset_file(PNG_NAME), "png")
        text = out["page_text"].get(1, "")
        return {"out": out, "text": text, "seconds": dt,
                "score": score("\n".join(PNG_ANCHORS), text)}
    return _cached(mode, "png", build)


def _scan_of(pdf_path: Path, dpi: int = 200) -> tuple[str, bytes]:
    """(page-1 text layer, image-only PDF of page 1 at its own page size).

    Local instead of _support.scanned_pdf because that one fixes the page at
    595x340, which would squash an A4 page and test the squash, not the OCR."""
    import fitz

    with fitz.open(pdf_path) as src:
        page = src[0]
        truth = page.get_text()
        png = page.get_pixmap(dpi=dpi).tobytes("png")
        rect = page.rect
    doc = fitz.open()
    new = doc.new_page(width=rect.width, height=rect.height)
    new.insert_image(new.rect, stream=png)
    data = doc.tobytes()
    doc.close()
    return truth, data


# --------------------------------------------------------------------------
# 1. Round trip: known text -> pixels -> defect -> real OCR -> exact compare
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("ocr_mode", ENGINE_MODES, indirect=True)
def test_round_trip(ocr_mode, kind, trace, tmp_path):
    r = _round_trip(ocr_mode, kind, trace, tmp_path)
    out, s = r["out"], r["score"]
    _metric(trace, engine=_engine(out, ocr_mode), ocr_seconds=round(r["seconds"], 2), **s.as_dict())

    assert not out.get("errors"), out.get("errors")
    rec = out["page_metadata"].get(1)
    assert rec is not None, "page 1 has no PageRecord"
    assert rec.ocr_applied and rec.status.value == "OCR", (rec.status, rec.notes)

    exact_floor, num_floor = FLOORS[kind]
    assert s.exact_word_recall >= exact_floor, (s.as_dict(), s.missing, r["text"])
    assert s.numbers_recall >= num_floor, (s.as_dict(), r["text"])
    if kind == "clean":
        # Already true on legacy; kept hard so they cannot regress.
        assert s.date_recall == 1.0, (s.as_dict(), r["text"])
        assert "RaghuRam" in r["text"].split(), r["text"]


TARGETS = ["clean-no_glued_words", "rotated_90-readable"]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("ocr_mode", ENGINE_MODES, indirect=True)
def test_round_trip_target(ocr_mode, target, trace, tmp_path, request):
    _gap(request, ocr_mode, target)
    kind = target.split("-")[0]
    r = _round_trip(ocr_mode, kind, trace, tmp_path)
    s = r["score"]
    if target == "clean-no_glued_words":
        assert s.glued_words == 0, s.glued_examples
    elif target == "rotated_90-readable":
        assert s.exact_word_recall >= 0.8, (s.as_dict(), r["text"])


# --------------------------------------------------------------------------
# 2. Real dataset (read in place, never copied)
# --------------------------------------------------------------------------


@pytest.mark.dataset
@pytest.mark.parametrize("ocr_mode", ENGINE_MODES, indirect=True)
def test_dataset_png(ocr_mode, trace, dataset_file):
    r = _png(ocr_mode, trace, dataset_file)
    out, s = r["out"], r["score"]
    _metric(trace, engine=_engine(out, ocr_mode), ocr_seconds=round(r["seconds"], 2), **s.as_dict())

    assert not out.get("errors"), out.get("errors")
    rec = out["page_metadata"][1]
    assert rec.ocr_applied and rec.status.value == "OCR", (rec.status, rec.notes)
    # Single-token anchors legacy already reads exactly.
    for anchor in ("0.70", "mg/dL", "239", "ALBUMIN"):
        assert anchor in r["text"], anchor
    assert s.exact_word_recall >= PNG_FLOOR, (s.as_dict(), s.missing)


@pytest.mark.dataset
@pytest.mark.parametrize("target", ["png-dates_exact", "png-heading_not_glued"])
@pytest.mark.parametrize("ocr_mode", ENGINE_MODES, indirect=True)
def test_dataset_png_target(ocr_mode, target, trace, dataset_file, request):
    _gap(request, ocr_mode, target)
    text = _png(ocr_mode, trace, dataset_file)["text"]
    if target == "png-dates_exact":
        assert [d for d in PNG_DATES if d not in text] == []
    else:
        assert "DIAGNOSTIC REPORT" in text and "DIAGNOSTICREPORT" not in text


@pytest.mark.dataset
@pytest.mark.parametrize("ocr_mode", ENGINE_MODES, indirect=True)
def test_dataset_digital_pdf_as_scan(ocr_mode, trace, dataset_file, tmp_path):
    with trace.step("text layer + rasterize page 1 @200dpi"):
        truth, data = _scan_of(dataset_file(DIGITAL_NAME))
        path = tmp_path / "scan.pdf"
        path.write_bytes(data)
    assert len(truth) > 500, "dataset PDF lost its text layer?"
    out, dt = _run(trace, path, "pdf")
    text = out["page_text"].get(1, "")
    s = score(truth, text)
    _metric(trace, engine=_engine(out, ocr_mode), ocr_seconds=round(dt, 2), **s.as_dict())

    rec = out["page_metadata"][1]
    assert rec.ocr_applied and rec.status.value == "OCR", (rec.status, rec.notes)
    assert s.exact_word_recall >= DIGITAL_FLOOR, (s.as_dict(), s.missing)
    assert s.date_recall >= 0.9, s.as_dict()


# --------------------------------------------------------------------------
# 3. Live Lambda (opt-in only: SUITE_LIVE_LAMBDA=1)
# --------------------------------------------------------------------------


@pytest.mark.live_lambda
def test_live_lambda_reports_model(trace):
    from app.documents.ocr_client import OcrClient

    png = rasterize(text_pdf([REPORT_LINES]), dpi=200)
    with trace.step("lambda extract (devanagari)"):
        r = OcrClient().extract(png, "devanagari")
    assert r.model_used, "engine did not report model_used"
    assert os.getenv("SUITE_LIVE_LAMBDA") == "1"
