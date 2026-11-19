"""Unit layer: component-level checks of the CURRENT code, no real OCR, no network.

Every test name reads as a requirement. Known gaps against the target
("saved text == page text") are strict xfails, so the day one is fixed it
turns into a visible XPASS failure and the marker must be removed.
"""
from __future__ import annotations

import io
import json

import pytest

from tests.suite._support import score, text_pdf

# --------------------------------------------------------------------------
# 1. OcrClient (Lambda adapter) with a stubbed boto3 client
# --------------------------------------------------------------------------

QUAD = [[10, 20], [110, 20], [110, 40], [10, 40]]
OCR_DATA = {
    "text": "HEMOGLOBIN 13.2 g/dL",
    "lines": ["HEMOGLOBIN", "13.2 g/dL"],
    "confidence": 0.91,
    "bounding_boxes": [QUAD, [[5, 50], [60, 50], [60, 70], [5, 70]]],
    "model_used": "devanagari-v2",
    "region_count": 2,
}


class _FakeLambda:
    def __init__(self, raw: bytes = b"", function_error: str | None = None, exc=None):
        self.raw, self.function_error, self.exc = raw, function_error, exc
        self.calls: list[dict] = []

    def invoke(self, **kw):
        self.calls.append(kw)
        if self.exc is not None:
            raise self.exc
        resp = {"Payload": io.BytesIO(self.raw), "StatusCode": 200}
        if self.function_error:
            resp["FunctionError"] = self.function_error
        return resp


def _client(monkeypatch, fake: _FakeLambda):
    from app.documents import ocr_client

    monkeypatch.setattr(ocr_client.boto3, "client", lambda *a, **k: fake)
    return ocr_client.OcrClient(function_name="unit-test-fn", region="ap-south-1", timeout=1)


def _envelope(body: dict, body_as_string: bool) -> bytes:
    return json.dumps({"statusCode": 200,
                       "body": json.dumps(body) if body_as_string else body}).encode()


@pytest.mark.parametrize("body_as_string", [True, False], ids=["body_json_string", "body_dict"])
def test_ocr_client_parses_success_envelope_and_sends_image_and_script(
        monkeypatch, trace, body_as_string):
    fake = _FakeLambda(_envelope({"success": True, "ocr_data": OCR_DATA}, body_as_string))
    client = _client(monkeypatch, fake)
    with trace.step("ocr_client.extract"):
        r = client.extract(b"\x89PNG-bytes", "devanagari")

    assert r.text == OCR_DATA["text"] and r.lines == OCR_DATA["lines"]
    assert r.model_used == "devanagari-v2" and r.region_count == 2
    assert len(r.boxes) == len(r.lines)
    assert (r.boxes[0].x, r.boxes[0].y, r.boxes[0].w, r.boxes[0].h) == (10, 20, 100, 20)
    assert len(fake.calls) == 1 and fake.calls[0]["FunctionName"] == "unit-test-fn"
    payload = json.loads(fake.calls[0]["Payload"])
    import base64
    assert base64.b64decode(payload["image_base64"]) == b"\x89PNG-bytes"
    assert payload["script"] == "devanagari"


@pytest.mark.parametrize("raw,function_error", [
    (_envelope({"success": False, "error": "model crashed"}, True), None),
    (_envelope({"success": False, "error": "model crashed"}, False), None),
    (b'{"errorMessage": "Task timed out"}', "Unhandled"),
], ids=["success_false_string_body", "success_false_dict_body", "function_error"])
def test_ocr_client_raises_ocr_error_when_lambda_reports_failure(monkeypatch, raw, function_error):
    from app.documents.ocr_client import OcrError

    client = _client(monkeypatch, _FakeLambda(raw, function_error))
    with pytest.raises(OcrError):
        client.extract(b"img", "arabic")


def _boto_errors():
    from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError

    return [
        NoCredentialsError(),
        EndpointConnectionError(endpoint_url="https://lambda.invalid"),
        ClientError({"Error": {"Code": "AccessDeniedException", "Message": "denied"}}, "Invoke"),
    ]


@pytest.mark.parametrize("idx", range(3), ids=["no_credentials", "endpoint_down", "access_denied"])
def test_ocr_client_wraps_aws_sdk_errors_in_ocr_error(monkeypatch, idx):
    from app.documents.ocr_client import OcrError

    client = _client(monkeypatch, _FakeLambda(exc=_boto_errors()[idx]))
    with pytest.raises(OcrError, match="call failed"):
        client.extract(b"img", "ta")


@pytest.mark.parametrize("script", ["english", "en", "", "DEVANAGARI", "hindi"])
def test_ocr_client_rejects_unsupported_script_without_invoking_lambda(monkeypatch, script):
    from app.documents.ocr_client import OcrError

    fake = _FakeLambda(_envelope({"success": True, "ocr_data": OCR_DATA}, True))
    client = _client(monkeypatch, fake)
    with pytest.raises(OcrError, match="no recognition model"):
        client.extract(b"img", script)
    assert fake.calls == []


# --------------------------------------------------------------------------
# 2. local_ocr.extract_english (RapidOCR path) with stubbed internals
# --------------------------------------------------------------------------


@pytest.mark.parametrize("exc", [ImportError("No module named 'deepreef_ocr'"),
                                 ValueError("cannot decode image"),
                                 RuntimeError("onnx session failed")],
                         ids=["import_error", "value_error", "runtime_error"])
def test_extract_english_wraps_any_failure_in_local_ocr_error(monkeypatch, exc):
    from app.documents import local_ocr

    def boom(_b):
        raise exc

    monkeypatch.setattr(local_ocr, "_decode_and_resize", boom)
    with pytest.raises(local_ocr.LocalOcrError, match="local OCR call failed") as ei:
        local_ocr.extract_english(b"bytes")
    assert ei.value.__cause__ is exc


def _stub_rapidocr(monkeypatch, result):
    from app.documents import local_ocr

    monkeypatch.setattr(local_ocr, "_decode_and_resize", lambda b: ("IMG", {}))
    monkeypatch.setattr(local_ocr, "_rapidocr_engine", lambda: (lambda img: (result, 0.01)))
    return local_ocr


@pytest.mark.parametrize("empty", [None, []], ids=["none", "empty_list"])
def test_extract_english_empty_result_gives_empty_text_and_zero_confidence(monkeypatch, empty):
    local_ocr = _stub_rapidocr(monkeypatch, empty)
    r = local_ocr.extract_english(b"bytes")
    assert (r.text, r.lines, r.confidence, r.region_count, r.boxes) == ("", [], 0.0, 0, [])


def test_extract_english_reports_mean_confidence_and_one_box_per_line(monkeypatch, trace):
    rows = [(QUAD, "PATIENT ID", 0.9), (QUAD, "25/08/2021 10:04", 0.8), (QUAD, "ALBUMIN", 0.55)]
    local_ocr = _stub_rapidocr(monkeypatch, rows)
    with trace.step("extract_english (stubbed engine)"):
        r = local_ocr.extract_english(b"bytes")
    assert r.lines == ["PATIENT ID", "25/08/2021 10:04", "ALBUMIN"]
    assert r.text == "PATIENT ID 25/08/2021 10:04 ALBUMIN"
    assert r.confidence == pytest.approx(round((0.9 + 0.8 + 0.55) / 3, 4))
    assert len(r.boxes) == r.region_count == 3
    assert r.model_used == "rapidocr-default-en"


# --------------------------------------------------------------------------
# 3. preprocess pure helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text,ok", [
    ("a" * 19, False),                       # one under the 20-char floor
    ("a" * 20, True),                        # exactly at the floor
    ("abcdefg" + "." * 13, True),            # 7/20 = 0.35 alnum: boundary passes
    ("abcdef" + "." * 14, False),            # 6/20 = 0.30 alnum: noise
    (" " * 50, False),                       # whitespace only
    ("\n\t \n" * 10, False),
    ("~!@#$%^&*()_+-=[]{}|;:,<>?/" * 2, False),  # punctuation noise
    ("a" * 10 + " " * 30 + "b" * 8, False),  # space run collapses to one: 19 chars
], ids=["19_chars", "20_chars", "ratio_0.35", "ratio_0.30", "spaces_only",
        "blank_lines_only", "punctuation_noise", "collapsed_spaces_count_once"])
def test_quality_ok_enforces_min_chars_and_alnum_ratio(text, ok):
    from app.graph.nodes.preprocess import _quality_ok

    assert _quality_ok(text, 20) is ok


def test_quality_ok_real_report_line_passes():
    from app.graph.nodes.preprocess import _quality_ok

    assert _quality_ok("Hemoglobin 13.2 g/dL 13.0 - 17.0", 20)


NATIVE = "LABORATORY TEST REPORT\nPatient Name: RaghuRam\nHemoglobin 13.2 g/dL"


@pytest.mark.parametrize("ocr,added", [
    ("", []),
    ("LABORATORY TEST REPORT\nPatient Name: RaghuRam", []),       # all already present
    ("laboratory  test report", []),                               # case/space variant
    ("City Diagnostics Pvt Ltd", ["City Diagnostics Pvt Ltd"]),    # letterhead image
    ("ab\nx1\n..", []),                                            # < 4 normalised chars
    ("NABL Accredited\nNABL Accredited", ["NABL Accredited"]),     # no self-duplicates
    ("Dr.  A  Kumar\nRaghuRam", ["Dr. A Kumar"]),                  # cleaned, known dropped
], ids=["empty_ocr", "all_known", "case_space_variant", "new_line", "short_noise",
        "dedup_within_ocr", "mixed"])
def test_merge_ocr_lines_appends_only_new_lines_under_label(ocr, added):
    from app.graph.nodes.preprocess import _merge_ocr_lines

    out = _merge_ocr_lines(NATIVE, ocr)
    if not added:
        assert out == NATIVE
    else:
        assert out == NATIVE + "\n[image region]\n" + "\n".join(added)
    for line in NATIVE.splitlines():
        assert out.count(line) == 1, f"text-layer line duplicated: {line!r}"


@pytest.mark.parametrize("ocr", ["City Diagnostics Pvt Ltd\nNABL MC-1234",
                                 "LABORATORY TEST REPORT\nSeal of Approval"])
def test_merge_ocr_lines_is_idempotent(ocr):
    from app.graph.nodes.preprocess import _merge_ocr_lines

    once = _merge_ocr_lines(NATIVE, ocr)
    assert _merge_ocr_lines(once, ocr) == once


@pytest.mark.parametrize("data", [b"", b"not an image", b"\x89PNG\r\n\x1a\n" + b"\x00" * 40],
                         ids=["empty", "garbage", "truncated_png"])
def test_binarised_returns_none_on_undecodable_bytes(data):
    from app.graph.nodes.preprocess import _binarised

    assert _binarised(data) is None


def test_binarised_returns_black_and_white_png():
    from PIL import Image

    from app.graph.nodes.preprocess import _binarised

    buf = io.BytesIO()
    im = Image.new("RGB", (40, 20), (200, 120, 40))
    im.paste((10, 10, 200), (0, 0, 20, 20))
    im.save(buf, format="PNG")
    out = _binarised(buf.getvalue())
    assert out is not None and out.startswith(b"\x89PNG")
    with Image.open(io.BytesIO(out)) as bw:
        hist = bw.convert("L").histogram()
        assert sum(hist[1:255]) == 0 and hist[0] and hist[255]


# --------------------------------------------------------------------------
# 4. clean_text / normalise_for_match exact-text probes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("s", [
    "RaghuRam", "McDonald", "O'Brien", "25/08/2021 10:04", "1,00,000", "< 0.01",
    "3.5 - 5.2", "NABL-TC-1234", "13.2 g/dL", "Page 4 Of 15",
])
def test_clean_text_preserves_exact_report_text(s):
    from app.graph.nodes.helpers import clean_text

    assert clean_text(s) == s


@pytest.mark.parametrize("raw,folded", [
    ("ﬁle", "file"),                  # fi ligature: same word on the page
    ("１２３", "123"),         # full-width digits
    ("3.5 – 5.2", "3.5 - 5.2"),       # en dash
    ("O’Brien", "O'Brien"),           # curly apostrophe
    ("a  b\t c", "a b c"),                 # whitespace runs
    ("x y", "x y"),                   # nbsp
], ids=["ligature", "fullwidth_digits", "en_dash", "curly_quote", "space_runs", "nbsp"])
def test_clean_text_folds_cosmetic_variants(raw, folded):
    from app.graph.nodes.helpers import clean_text

    assert clean_text(raw) == folded


@pytest.mark.parametrize("s", [
    pytest.param("mg/m²", marks=pytest.mark.xfail(
        strict=True, reason="known gap: exact-text fix (not Fix 1) — NFKC rewrites 'mg/m²' to 'mg/m2'")),
    pytest.param("µg/L", marks=pytest.mark.xfail(
        strict=True, reason="known gap: exact-text fix (not Fix 1) — NFKC rewrites micro sign U+00B5 to Greek mu U+03BC")),
    pytest.param("10³/µL", marks=pytest.mark.xfail(
        strict=True, reason="known gap: exact-text fix (not Fix 1) — NFKC rewrites '10³/µL' to '103/μL'")),
    pytest.param("½", marks=pytest.mark.xfail(
        strict=True, reason="known gap: exact-text fix (not Fix 1) — NFKC rewrites '½' to '1⁄2'")),
], ids=["superscript_2", "micro_sign", "superscript_3_micro", "vulgar_half"])
def test_clean_text_keeps_units_and_symbols_verbatim(s):
    from app.graph.nodes.helpers import clean_text

    assert clean_text(s) == s


@pytest.mark.parametrize("a,b", [("RaghuRam", "raghu ram"), ("40 - 129", "40-129"),
                                 ("IN 123", "IN123"), ("BILIRUBIN, TOTAL", "bilirubin total")])
def test_normalise_for_match_ignores_case_space_and_punctuation(a, b):
    from app.graph.nodes.helpers import normalise_for_match

    assert normalise_for_match(a) == normalise_for_match(b)


@pytest.mark.xfail(strict=True, reason="known gap: evidence check #5 treats 1.5 as found in 0.15")
def test_normalise_for_match_does_not_find_1_5_inside_0_15():
    from app.graph.nodes.helpers import normalise_for_match

    assert normalise_for_match("1.5") not in normalise_for_match("0.15")


@pytest.mark.xfail(strict=True, reason="known gap: normalise_for_match drops µ, so 'µg/L' matches 'g/L'")
def test_normalise_for_match_distinguishes_microgram_from_gram():
    from app.graph.nodes.helpers import normalise_for_match

    assert normalise_for_match("µg/L") != normalise_for_match("g/L")


# --------------------------------------------------------------------------
# 5. _support.score — the suite's own ruler
# --------------------------------------------------------------------------

EXPECTED = "Patient Name: RaghuRam\nCollected: 25/08/2021 10:04\nTSH < 0.01"


def test_score_identical_text_is_perfect():
    s = score(EXPECTED, EXPECTED)
    assert (s.exact_word_recall, s.date_recall, s.numbers_recall, s.glued_words) == (1.0, 1.0, 1.0, 0)
    assert s.missing == []


def test_score_counts_glued_tokens_and_missing_dates():
    got = "PatientName: RaghuRam\nCollected: 25/08/202110:04\nTSH < 0.01"
    s = score(EXPECTED, got)
    assert s.glued_words == 2 and set(s.glued_examples) == {"PatientName:", "25/08/202110:04"}
    assert s.dates_expected == 1 and s.dates_found == 0 and s.date_recall == 0.0
    assert {"Patient", "Name:", "25/08/2021", "10:04"} <= set(s.missing)


def test_score_is_case_sensitive():
    s = score("RaghuRam", "raghuram")
    assert s.exact_word_recall == 0.0 and s.missing == ["RaghuRam"]


def test_score_numbers_recall_counts_each_value_once():
    s = score("0.15 0.15 1.5", "0.15 1.5")
    assert s.numbers_recall == pytest.approx(2 / 3)


# --------------------------------------------------------------------------
# 6. validate_document adversarial inputs
# --------------------------------------------------------------------------


def _write(tmp_path, name, data: bytes) -> str:
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def _encrypted(tmp_path, name):
    import fitz

    path = str(tmp_path / name)
    doc = fitz.open()
    doc.new_page().insert_text((72, 100), "secret", fontsize=11)
    doc.save(path, encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    doc.close()
    return path


BAD_FILES = {
    "empty_pdf": (lambda t: _write(t, "e.pdf", b""), "EMPTY_DOCUMENT", "empty"),
    "no_pdf_header": (lambda t: _write(t, "c.pdf", b"hello this is not a pdf"),
                      "CORRUPTED_FILE", "no PDF header"),
    "corrupt_pdf_body": (lambda t: _write(t, "b.pdf", b"%PDF-1.7\n" + b"\x00\xff" * 64),
                         "CORRUPTED_FILE", "could not be opened"),
    "txt_file": (lambda t: _write(t, "notes.txt", b"plain text"), "UNSUPPORTED_FILE_TYPE",
                 "Unsupported file type"),
    "encrypted_pdf": (lambda t: _encrypted(t, "locked.pdf"), "PASSWORD_PROTECTED",
                      "password-protected"),
    "corrupt_png": (lambda t: _write(t, "x.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 30),
                    "CORRUPTED_FILE", "could not be decoded"),
    "fake_docx": (lambda t: _write(t, "x.docx", b"PK not really a zip"), "CORRUPTED_FILE",
                  "not a valid DOCX"),
    "missing_file": (lambda t: str(t / "nope.pdf"), "INVALID_FILE", "does not exist"),
    "no_path": (lambda t: "", "INVALID_FILE", "No file path"),
    "directory": (lambda t: str(t), "INVALID_FILE", "not a file"),
}


@pytest.mark.parametrize("case", list(BAD_FILES))
def test_validation_rejects_bad_input_with_clear_reason_and_never_raises(tmp_path, case):
    from app.graph.nodes.validation import validate_document
    from app.graph.schemas import FinalStatus
    from app.graph.state import GraphState, WorkflowStatus

    make, err_type, phrase = BAD_FILES[case]
    out = validate_document(GraphState(document_id="t", file_path=make(tmp_path)))
    assert out["final_status"] == FinalStatus.REJECTED
    assert out["current_status"] == WorkflowStatus.FAILED
    assert phrase.lower() in out["final_reasoning"].lower()
    assert str(getattr(out["errors"][0].error_type, "value", out["errors"][0].error_type)) == err_type


def test_validation_rejects_pdf_over_page_limit(tmp_path, monkeypatch, trace):
    from app.graph.config import get_graph_settings
    from app.graph.nodes.validation import validate_document
    from app.graph.schemas import FinalStatus
    from app.graph.state import GraphState

    monkeypatch.setattr(get_graph_settings(), "max_pages", 2, raising=False)
    path = _write(tmp_path, "big.pdf", text_pdf([["p1"], ["p2"], ["p3"]]))
    with trace.step("validate_document (3 pages, limit 2)"):
        out = validate_document(GraphState(document_id="t", file_path=path))
    assert out["final_status"] == FinalStatus.REJECTED
    assert "3 pages, over the 2 limit" in out["final_reasoning"]


def test_validation_accepts_born_digital_pdf(tmp_path):
    from app.graph.nodes.validation import validate_document
    from app.graph.state import GraphState

    path = _write(tmp_path, "ok.pdf", text_pdf([["LABORATORY TEST REPORT"], ["Page 2"]]))
    out = validate_document(GraphState(document_id="t", file_path=path))
    assert "final_status" not in out
    assert (out["file_type"], out["total_pages"]) == ("pdf", 2)
