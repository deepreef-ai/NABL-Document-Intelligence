"""Integration: how the pipeline FLOW handles OCR output, end to end.

OCR is replaced by `fake_ocr` and every LLM node by `recording_llm`, so these
tests are fast and deterministic. They test what the pipeline does WITH the
text an OCR engine hands it (page classification, page accounting, script
routing, failure handling, text reaching the extractor untouched) — not how
well any engine reads pixels.

Known gaps are strict xfails: when the gap is fixed the test XPASSes and the
suite fails until the marker is removed, so the gap list cannot go stale.
"""
from __future__ import annotations

import pytest

from tests.suite._support import REPORT_LINES, degrade, rasterize, scanned_pdf, text_pdf

# Long enough (>= 20 chars, mostly alphanumeric) to pass preprocess._quality_ok.
P1 = "LABORATORY TEST REPORT\nPatient Name: RaghuRam\nReferred By: Dr. McDonald"
P2 = "Collected: 25/08/2021 10:04\nReported: 26/08/2021 15:28\nHemoglobin 13.2 g/dL 13.0 - 17.0"
P3 = "Creatinine 0.15 mg/dL 0.60 - 1.20\nTotal Count 1,00,000 /cumm"
NOISE = "~~ :: ~~ :: ~~ :: ~~ :: ~~ ::"


# --------------------------------------------------------------------------
# Local helpers (shared helpers live in _support.py, which is not ours to edit)
# --------------------------------------------------------------------------


def _png(lines=REPORT_LINES[:4]) -> bytes:
    return rasterize(text_pdf([list(lines)]), dpi=100)


def _write(tmp_path, name: str, data: bytes):
    p = tmp_path / name
    p.write_bytes(data)
    return p


def _mixed_pdf() -> bytes:
    """[digital page, scanned page, blank page] in one file."""
    import fitz

    digital = fitz.open(stream=text_pdf([REPORT_LINES]), filetype="pdf")
    scanned = fitz.open(stream=scanned_pdf([_png(), None]), filetype="pdf")
    digital.insert_pdf(scanned)
    out = digital.tobytes()
    digital.close()
    scanned.close()
    return out


def _encrypted_pdf() -> bytes:
    import fitz

    doc = fitz.open(stream=text_pdf([REPORT_LINES]), filetype="pdf")
    out = doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    doc.close()
    return out


def _preprocess(path, file_type: str, script: str = "english") -> dict:
    from app.graph.nodes.preprocess import preprocess_document
    from app.graph.state import GraphState

    return preprocess_document(GraphState(
        document_id="t", file_path=str(path), file_type=file_type, script=script))


def _statuses(out: dict) -> dict[int, str]:
    return {n: r.status.value for n, r in out["page_metadata"].items()}


def _assert_every_page_accounted(out: dict, total: int) -> None:
    """The business rule: never silently skip a page."""
    assert out["total_pages"] == total
    assert sorted(out["page_metadata"]) == list(range(1, total + 1))
    assert sorted(out["page_text"]) == list(range(1, total + 1))
    assert sorted(out["processed_pages"] + out["unreadable_pages"]) == list(range(1, total + 1))
    assert set(out["ocr_pages"]) <= set(out["processed_pages"])


def _graph_page_text(document_id: str) -> dict[int, str]:
    """page_text as the graph checkpointed it (FinalResult does not carry it)."""
    from app.graph.graph import build_checkpointer, build_graph
    from app.graph.runner import _config

    snap = build_graph(build_checkpointer()).get_state(_config(document_id))
    return {int(k): v for k, v in snap.values["page_text"].items()}


@pytest.fixture
def graph_on(monkeypatch):
    from app.graph.config import get_graph_settings

    monkeypatch.setattr(get_graph_settings(), "graph_pipeline_enabled", True, raising=False)


# --------------------------------------------------------------------------
# 1. preprocess_document: digital, scanned, mixed, image uploads
# --------------------------------------------------------------------------


def test_digital_pdf_uses_text_layer_byte_exact_and_never_calls_ocr(tmp_path, fake_ocr, trace):
    fake = fake_ocr()
    path = _write(tmp_path, "digital.pdf", text_pdf([REPORT_LINES]))
    with trace.step("preprocess"):
        out = _preprocess(path, "pdf")
    with trace.step("assert"):
        assert fake.calls == []
        assert _statuses(out) == {1: "NATIVE_TEXT"}
        assert out["ocr_pages"] == []
        assert out["page_text"][1].splitlines() == REPORT_LINES
        assert "RaghuRam" in out["page_text"][1]
        _assert_every_page_accounted(out, 1)


@pytest.mark.parametrize("n_pages", [1, 3])
def test_scanned_pdf_ocr_called_once_per_page(tmp_path, fake_ocr, trace, n_pages):
    texts = [P1, P2, P3][:n_pages]
    fake = fake_ocr(pages=texts)
    path = _write(tmp_path, "scan.pdf", scanned_pdf([_png()] * n_pages))
    with trace.step("preprocess"):
        out = _preprocess(path, "pdf")
    with trace.step("assert"):
        assert [c["engine"] for c in fake.calls] == ["rapidocr"] * n_pages
        assert set(_statuses(out).values()) == {"OCR"}
        assert out["ocr_pages"] == list(range(1, n_pages + 1))
        # OCR lines reach page_text unchanged, rows kept, in page order.
        assert [out["page_text"][i + 1] for i in range(n_pages)] == texts
        assert all(r.ocr_applied and r.ocr_confidence == pytest.approx(0.93)
                   for r in out["page_metadata"].values())
        _assert_every_page_accounted(out, n_pages)


def test_mixed_pdf_every_page_has_a_record(tmp_path, fake_ocr, trace):
    # OCR is called for the scanned page AND the blank page (no text layer).
    fake = fake_ocr(pages=[P1, ""])
    path = _write(tmp_path, "mixed.pdf", _mixed_pdf())
    with trace.step("preprocess"):
        out = _preprocess(path, "pdf")
    with trace.step("assert"):
        assert len(fake.calls) == 2
        assert _statuses(out) == {1: "NATIVE_TEXT", 2: "OCR", 3: "EMPTY"}
        assert out["ocr_pages"] == [2]
        assert out["processed_pages"] == [1, 2, 3]
        assert out["unreadable_pages"] == []
        assert "RaghuRam" in out["page_text"][1] and out["page_text"][2] == P1
        assert out["page_text"][3] == "" and out["page_metadata"][3].notes
        _assert_every_page_accounted(out, 3)


@pytest.mark.parametrize("ext,kind", [("png", "clean"), ("jpg", "jpeg_q25")])
def test_image_upload_goes_through_single_ocr_call(tmp_path, fake_ocr, trace, ext, kind):
    fake = fake_ocr(pages=[P1])
    path = _write(tmp_path, f"upload.{ext}", degrade(_png(), kind))
    with trace.step("preprocess"):
        out = _preprocess(path, ext)
    with trace.step("assert"):
        assert fake.calls == [{"engine": "rapidocr", "script": "english"}]
        assert _statuses(out) == {1: "OCR"} and out["ocr_pages"] == [1]
        # Observed: the image branch uses OcrResult.text (lines joined by a
        # space), not the lines — so an image page loses its row breaks while
        # a scanned PDF page keeps them (preprocess.py image branch).
        assert out["page_text"][1] == " ".join(P1.splitlines())
        _assert_every_page_accounted(out, 1)


# --------------------------------------------------------------------------
# 2. Partial OCR failure
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fail_at", [0, 1, 2])
def test_one_page_ocr_failure_is_recorded_not_raised(tmp_path, fake_ocr, trace, fail_at):
    pages: list = [P1, P2, P3]
    pages[fail_at] = RuntimeError("boom")
    fake_ocr(pages=pages)
    path = _write(tmp_path, "scan3.pdf", scanned_pdf([_png()] * 3))
    with trace.step("preprocess"):
        out = _preprocess(path, "pdf")
    with trace.step("assert"):
        failed = fail_at + 1
        expected = {i: "OCR" for i in (1, 2, 3)}
        expected[failed] = "OCR_FAILED"
        assert _statuses(out) == expected
        assert out["unreadable_pages"] == [failed]
        assert out["page_text"][failed] == ""
        assert "boom" in out["page_metadata"][failed].notes
        page_errors = [e for e in out["errors"] if getattr(e, "page_number", None) == failed]
        assert page_errors and "boom" in str(page_errors[0])
        for i in {1, 2, 3} - {failed}:
            assert out["page_text"][i] == [P1, P2, P3][i - 1]
        _assert_every_page_accounted(out, 3)


# --------------------------------------------------------------------------
# 3. Script routing (current behaviour + known gap G10)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("script,engine", [
    ("english", "rapidocr"),
    ("devanagari", "lambda"),
    ("arabic", "lambda"),
    ("hindi", "rapidocr"),   # not a SUPPORTED_SCRIPTS key today: routed as Latin
])
def test_scanned_pdf_script_selects_engine(tmp_path, fake_ocr, trace, script, engine):
    fake = fake_ocr(pages=[P1])
    path = _write(tmp_path, "scan.pdf", scanned_pdf([_png()]))
    with trace.step("preprocess"):
        out = _preprocess(path, "pdf", script=script)
    with trace.step("assert"):
        assert [c["engine"] for c in fake.calls] == [engine]
        if engine == "lambda":
            assert fake.calls[0]["script"] == script
        assert _statuses(out) == {1: "OCR"}
        expected_engine = f"deepreef:{script}" if engine == "lambda" else "rapidocr"
        assert any(f"read by {expected_engine}" == a.detail for a in out["audit_log"]
                   if a.event == "page_ocr")


@pytest.mark.parametrize("lambda_result", [RuntimeError("lambda unreachable"), ""],
                         ids=["lambda_raises", "lambda_empty"])
def test_lambda_failure_falls_back_to_rapidocr(tmp_path, fake_ocr, trace, lambda_result):
    fake = fake_ocr(pages=[lambda_result, P1])
    path = _write(tmp_path, "scan.pdf", scanned_pdf([_png()]))
    with trace.step("preprocess"):
        out = _preprocess(path, "pdf", script="devanagari")
    with trace.step("assert"):
        assert [(c["engine"], c["script"]) for c in fake.calls] == [
            ("lambda", "devanagari"), ("rapidocr", "english")]
        assert _statuses(out) == {1: "OCR"} and out["page_text"][1] == P1
        details = [a.detail for a in out["audit_log"] if a.event == "page_ocr"]
        assert details == ["read by rapidocr(fallback)"]
        # Observed: WHY the Lambda failed is only logged, not kept on the page
        # record, once the fallback succeeds.
        assert out["page_metadata"][1].notes == ""


def test_lambda_and_fallback_both_fail_keeps_lambda_reason(tmp_path, fake_ocr, trace):
    fake_ocr(pages=[RuntimeError("lambda unreachable"), RuntimeError("rapid boom")])
    path = _write(tmp_path, "scan.pdf", scanned_pdf([_png()]))
    with trace.step("preprocess"):
        out = _preprocess(path, "pdf", script="devanagari")
    with trace.step("assert"):
        assert _statuses(out) == {1: "OCR_FAILED"}
        # The rapidocr error wins (error or note): the Lambda reason is lost.
        assert "rapid boom" in out["page_metadata"][1].notes


def test_image_upload_ignores_script_today(tmp_path, fake_ocr, trace):
    fake = fake_ocr(pages=[P1])
    path = _write(tmp_path, "upload.png", _png())
    with trace.step("preprocess"):
        _preprocess(path, "png", script="devanagari")
    with trace.step("assert"):
        assert fake.calls == [{"engine": "rapidocr", "script": "english"}]


@pytest.mark.xfail(strict=True, reason="known gap G10: image uploads ignore script")
def test_image_upload_honours_devanagari_script(tmp_path, fake_ocr, trace):
    fake = fake_ocr(pages=[P1])
    path = _write(tmp_path, "upload.png", _png())
    with trace.step("preprocess"):
        _preprocess(path, "png", script="devanagari")
    with trace.step("assert"):
        assert fake.calls and fake.calls[0]["engine"] in ("lambda", "deepreef_local")
        assert fake.calls[0]["script"] in ("devanagari", "hi")


# --------------------------------------------------------------------------
# 4. Low-quality OCR text
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ocr_text", [NOISE, "a b", ""], ids=["punctuation_noise", "too_short", "nothing"])
def test_low_quality_ocr_text_is_not_counted_as_text(tmp_path, fake_ocr, trace, ocr_text):
    fake_ocr(pages=[ocr_text])
    path = _write(tmp_path, "scan.pdf", scanned_pdf([_png()]))
    with trace.step("preprocess"):
        out = _preprocess(path, "pdf")
    with trace.step("assert"):
        # Junk is dropped (not passed to extraction) and the page is EMPTY,
        # not OCR_FAILED: OCR "ran and found nothing".
        assert _statuses(out) == {1: "EMPTY"}
        assert out["page_text"][1] == "" and out["ocr_pages"] == []
        assert any(str(e.error_type).endswith("EMPTY_DOCUMENT")
                   for e in out["errors"])


def test_image_branch_has_no_quality_gate_today(tmp_path, fake_ocr, trace):
    """Documented inconsistency: the same punctuation noise that makes a
    scanned-PDF page EMPTY is accepted as OCR text on an image upload."""
    fake_ocr(pages=[NOISE])
    path = _write(tmp_path, "upload.png", _png())
    with trace.step("preprocess"):
        out = _preprocess(path, "png")
    with trace.step("assert"):
        assert _statuses(out) == {1: "OCR"}
        assert out["page_text"][1] == NOISE


# --------------------------------------------------------------------------
# 5. Adversarial inputs through the whole graph
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name,data,reason_word", [
    ("empty.pdf", b"", "empty"),
    ("corrupt.pdf", b"%PDF-1.7\nthis is not really a pdf\x00\xff", "pdf"),
    ("notpdf.pdf", b"hello, I am text", "pdf header"),
    ("encrypted.pdf", None, "password"),
    ("text.png", b"just some text renamed to png\n" * 4, "decoded"),
    ("zero.png", b"", "empty"),
    ("macro.exe", b"MZ\x90\x00", "unsupported"),
])
def test_adversarial_file_is_rejected_with_reason(tmp_path, fake_ocr, recording_llm, trace,
                                                  name, data, reason_word):
    from app.graph.runner import run_document
    from app.graph.schemas import FinalStatus

    fake = fake_ocr()
    path = _write(tmp_path, name, _encrypted_pdf() if data is None else data)
    with trace.step("run_document"):
        result = run_document(str(path), document_id=f"adv-{path.stem}")
    with trace.step("assert"):
        assert result.overall_status == FinalStatus.REJECTED
        assert reason_word in result.final_reasoning.lower(), result.final_reasoning
        assert fake.calls == [] and recording_llm.calls == []


# --------------------------------------------------------------------------
# 6. End to end: OCR text reaches extraction untouched
# --------------------------------------------------------------------------


def test_run_document_ocr_text_reaches_extractor_verbatim(tmp_path, fake_ocr, recording_llm, trace):
    from app.graph.runner import run_document
    from app.graph.schemas import FinalStatus

    fake = fake_ocr(pages=[P1, P2])
    path = _write(tmp_path, "scan2.pdf", scanned_pdf([_png(), _png()]))
    with trace.step("run_document"):
        result = run_document(str(path), document_id="e2e-1")
    with trace.step("assert"):
        assert len(fake.calls) == 2
        seen = recording_llm.text_seen()
        assert "Patient Name: RaghuRam" in seen
        assert "Collected: 25/08/2021 10:04" in seen
        assert result.overall_status != FinalStatus.REJECTED
        assert result.document_summary.page_count == 2
        assert result.document_summary.ocr_pages == 2


def test_process_document_api_entry_completes(tmp_path, fake_ocr, recording_llm, graph_on, trace):
    from app.documents.grounding import PipelineResult
    from app.documents.pipeline import process_document

    fake_ocr(pages=[P1, P2])
    data = scanned_pdf([_png(), _png()])
    with trace.step("process_document"):
        result = process_document(data, "report.pdf", "application/pdf",
                                  script="english", document_id="api-1")
    with trace.step("assert"):
        assert isinstance(result, PipelineResult)
        assert result.page_count == 2
        assert result.extraction_source == "ocr_pdf"
        assert "Patient Name: RaghuRam" in recording_llm.text_seen()


# --------------------------------------------------------------------------
# 7. Idempotency
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["digital", "scanned", "mixed"])
def test_preprocess_twice_is_identical(tmp_path, fake_ocr, trace, kind):
    data = {"digital": lambda: text_pdf([REPORT_LINES]),
            "scanned": lambda: scanned_pdf([_png(), _png()]),
            "mixed": _mixed_pdf}[kind]()
    path = _write(tmp_path, f"{kind}.pdf", data)
    runs = []
    for i in range(2):
        fake_ocr(pages=[P1, P2])
        with trace.step(f"preprocess #{i + 1}"):
            runs.append(_preprocess(path, "pdf"))
    with trace.step("assert"):
        assert runs[0]["page_text"] == runs[1]["page_text"]
        assert _statuses(runs[0]) == _statuses(runs[1])
        assert runs[0]["ocr_pages"] == runs[1]["ocr_pages"]


def test_run_document_twice_gives_identical_page_text(tmp_path, fake_ocr, recording_llm, trace):
    from app.graph.runner import run_document

    path = _write(tmp_path, "scan2.pdf", scanned_pdf([_png(), _png()]))
    texts = []
    for doc_id in ("idem-a", "idem-b"):
        fake_ocr(pages=[P1, P2])
        with trace.step(f"run_document {doc_id}"):
            run_document(str(path), document_id=doc_id)
        texts.append(_graph_page_text(doc_id))
    with trace.step("assert"):
        assert texts[0] == texts[1] == {1: P1, 2: P2}


# --------------------------------------------------------------------------
# 8. Status honesty (known gap G2)
# --------------------------------------------------------------------------


def _all_pages_fail(tmp_path, fake_ocr):
    from app.documents.pipeline import process_document

    fake_ocr(pages=[RuntimeError("ocr down"), RuntimeError("ocr down")])
    return process_document(scanned_pdf([_png(), _png()]), "report.pdf", "application/pdf",
                            script="english", document_id="g2")


def test_all_pages_ocr_failed_today_returns_normally_with_warning(
        tmp_path, fake_ocr, recording_llm, graph_on, trace):
    with trace.step("process_document"):
        result = _all_pages_fail(tmp_path, fake_ocr)
    with trace.step("assert"):
        # Today: no exception, a normal-looking PipelineResult, and NOT even
        # the word "rejected": the graph's final_decision only treats
        # document_validation errors as fatal, so a document whose every page
        # failed OCR ends INCOMPLETE ("Partial read") although preprocessing
        # logged EMPTY_DOCUMENT "workflow will report REJECTED".
        assert result.fields == []
        assert result.doc_type == "unknown"   # document_summary.document_type, not a rejection marker
        assert result.doc_confidence == 0.5
        assert result.page_count == 2
        assert result.extraction_source == "born_digital_pdf"   # ocr_pages == 0
        assert result.extraction_warnings == [
            "Partial read: no field-value pairs were extracted from the document"]
        assert not any("reject" in w.lower() for w in result.extraction_warnings)
        assert recording_llm.calls == []


def test_all_pages_ocr_failed_run_document_status_today(tmp_path, fake_ocr, recording_llm, trace):
    from app.graph.runner import run_document
    from app.graph.schemas import FinalStatus

    fake_ocr(pages=[RuntimeError("ocr down"), RuntimeError("ocr down")])
    path = _write(tmp_path, "dead.pdf", scanned_pdf([_png(), _png()]))
    with trace.step("run_document"):
        result = run_document(str(path), document_id="g2-graph")
    with trace.step("assert"):
        assert result.overall_status == FinalStatus.INCOMPLETE
        assert result.document_summary.processed_pages == 0
        assert result.document_summary.ocr_pages == 0
        assert recording_llm.calls == []


@pytest.mark.xfail(strict=True, reason="known gap G2: rejected document is reported as extracted")
def test_all_pages_ocr_failed_run_document_is_rejected(tmp_path, fake_ocr, recording_llm, trace):
    """Target: preprocess.py promises 'workflow will report REJECTED' when no
    page is readable; the graph's final status should say so."""
    from app.graph.runner import run_document
    from app.graph.schemas import FinalStatus

    fake_ocr(pages=[RuntimeError("ocr down"), RuntimeError("ocr down")])
    path = _write(tmp_path, "dead.pdf", scanned_pdf([_png(), _png()]))
    with trace.step("run_document"):
        result = run_document(str(path), document_id="g2-graph-target")
    with trace.step("assert"):
        assert result.overall_status == FinalStatus.REJECTED


@pytest.mark.xfail(strict=True, reason="known gap G2: rejected document is reported as extracted")
def test_all_pages_ocr_failed_is_an_explicit_failure(tmp_path, fake_ocr, recording_llm, graph_on, trace):
    """Target: the API entry either raises, or returns a result whose
    doc_type is "rejected" with doc_confidence 0 — something a caller can
    branch on without parsing warning prose."""
    with trace.step("process_document"):
        try:
            result = _all_pages_fail(tmp_path, fake_ocr)
        except Exception:  # noqa: BLE001 — raising is an acceptable explicit signal
            return
    with trace.step("assert"):
        assert result.doc_type == "rejected"
        assert result.doc_confidence == 0.0
