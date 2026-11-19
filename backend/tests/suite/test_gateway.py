"""Fix 1: the OCR gateway (app/ocr_gateway), tested black-box against the spec.

A. Unit      - script aliases, engine order, fallback, Lambda envelope; fakes only.
B. Wiring    - preprocess_document routes its OCR through the gateway when (and
               only when) OCR_GATEWAY_ENABLED; fakes only.
C. Real OCR  - round-trip accuracy with the real deepreef checkout (realocr) and
               one real dataset page (dataset).

No test here may reach AWS: boto3.client is replaced for every test (autouse).
"""
from __future__ import annotations

import base64
import io
import json
import re
import threading
import time

import pytest

gw = pytest.importorskip("app.ocr_gateway")

from app.ocr_gateway import engines  # noqa: E402
from app.ocr_gateway import settings as gw_settings  # noqa: E402
from tests.suite import _support as sup  # noqa: E402

DEFAULT_LATIN = "deepreef_local,rapidocr"
DEFAULT_OTHER = "lambda,deepreef_local,rapidocr"


# --------------------------------------------------------------------------
# Helpers + fixtures (local to this file; conftest/_support are shared)
# --------------------------------------------------------------------------


def _res(engine: str, text_lines: str = "", lines=None, text=None, model="fake-model"):
    lines = list(lines if lines is not None else [ln for ln in text_lines.split("\n") if ln])
    return engines.EngineResult(
        text=" ".join(lines) if text is None else text, lines=lines,
        confidence=0.9 if lines else 0.0, boxes=[], model_used=model,
        region_count=len(lines), engine=engine, text_lines=text_lines)


class Engines:
    """Replaces every ENGINES entry; `plan[name]` is an EngineResult, an
    Exception (raised), or absent (-> AssertionError: must not be called)."""

    def __init__(self, monkeypatch, plan: dict):
        self.plan, self.calls = plan, []
        for name in ("deepreef_local", "lambda", "rapidocr"):
            monkeypatch.setitem(engines.ENGINES, name, self._make(name))

    def _make(self, name):
        def run(image_bytes, code, s):
            self.calls.append((name, code))
            item = self.plan.get(name, AssertionError(f"{name} must not be called"))
            if isinstance(item, BaseException):
                raise item
            return item
        return run


class _StubLambda:
    """boto3 'lambda' client stand-in. Never touches the network."""

    def __init__(self, response_body=None, function_error: bool = False):
        self.response_body, self.function_error = response_body, function_error
        self.payloads: list[dict] = []

    def invoke(self, FunctionName=None, Payload=None, **kw):  # noqa: N803 — boto3 API
        if self.response_body is None:
            raise AssertionError("the Lambda must not be invoked in this test")
        self.payloads.append(json.loads(Payload))
        raw = json.dumps({"statusCode": 200, "body": self.response_body}).encode()
        out = {"StatusCode": 200, "Payload": io.BytesIO(raw)}
        if self.function_error:
            out["FunctionError"] = "Unhandled"
        return out


@pytest.fixture(autouse=True)
def _gateway_isolation(monkeypatch):
    """Fresh settings per test, no inherited OCR_GATEWAY_* env, no AWS."""
    import os

    import boto3

    for key in [k for k in os.environ if k.upper().startswith("OCR_GATEWAY_")]:
        monkeypatch.delenv(key, raising=False)
    gw_settings.get_gateway_settings.cache_clear()
    stub = _StubLambda()
    monkeypatch.setattr(boto3, "client", lambda *a, **k: stub)
    yield stub
    gw_settings.get_gateway_settings.cache_clear()


@pytest.fixture
def gs(monkeypatch):
    """The cached gateway settings with the spec's engine lists pinned (a
    developer's backend/.env must not change what these tests mean)."""
    s = gw_settings.get_gateway_settings()
    monkeypatch.setattr(s, "latin_engines", DEFAULT_LATIN)
    monkeypatch.setattr(s, "other_engines", DEFAULT_OTHER)
    return s


@pytest.fixture
def gateway_on(gs, monkeypatch):
    from app.graph.config import get_graph_settings

    monkeypatch.setattr(gs, "enabled", True)
    monkeypatch.setattr(get_graph_settings(), "ocr_enabled", True, raising=False)
    return gs


@pytest.fixture
def legacy_calls(monkeypatch):
    """Records any call to the legacy OCR functions (they must stay unused
    while the gateway is on, including the chart second pass)."""
    return _legacy_tripwire(monkeypatch)


def _legacy_tripwire(monkeypatch) -> list[str]:
    from app.documents import local_ocr, ocr_client

    calls: list[str] = []

    def _rec(name):
        def f(*a, **k):
            calls.append(name)
            raise RuntimeError(f"legacy {name} called")
        return f

    monkeypatch.setattr(local_ocr, "extract_english", _rec("extract_english"))
    monkeypatch.setattr(ocr_client.OcrClient, "__init__", lambda self_, *a, **k: None)
    monkeypatch.setattr(ocr_client.OcrClient, "extract", _rec("OcrClient.extract"))
    return calls


_SMALL_PNG: bytes | None = None


def _small_png() -> bytes:
    global _SMALL_PNG
    if _SMALL_PNG is None:
        _SMALL_PNG = sup.rasterize(sup.text_pdf([sup.REPORT_LINES[:3]]), dpi=60)
    return _SMALL_PNG


def _doc(tmp_path, file_type: str, image: bytes | None = None) -> str:
    image = image or _small_png()
    if file_type == "pdf":
        p = tmp_path / "scan.pdf"
        p.write_bytes(sup.scanned_pdf([image]))
    else:
        p = tmp_path / f"upload.{file_type}"
        p.write_bytes(image)
    return str(p)


def _run(path: str, file_type: str, script: str) -> dict:
    from app.graph.nodes.preprocess import preprocess_document
    from app.graph.state import GraphState

    return preprocess_document(GraphState(document_id="t", file_path=path,
                                          file_type=file_type, script=script))


def _page_story(out: dict, page: int = 1) -> str:
    """Everything the node said about a page: notes, audit and error text."""
    parts = [out["page_metadata"][page].notes or ""]
    parts += [e.detail for e in out.get("audit_log", []) if e.page_number in (page, None)]
    parts += [e.error_message for e in out.get("errors", [])]
    return "\n".join(parts)


# ==========================================================================
# A. Unit
# ==========================================================================

ALIASES = {
    "en": [None, "", "english", "en", "latin"],
    "devanagari": ["hindi", "marathi", "nepali", "devanagari"],
    "arabic": ["urdu", "arabic"],
    "ta": ["tamil", "ta"], "te": ["telugu", "te"], "ka": ["kannada", "ka"],
    "pa": ["punjabi", "gurmukhi", "pa"], "or": ["odia", "oriya", "or"],
    "bn": ["bengali", "bangla", "bn"], "as": ["assamese", "as"],
    "gu": ["gujarati", "gu"], "ml": ["malayalam", "ml"],
}


def test_resolve_script_alias_table():
    wrong = {}
    for code, aliases in ALIASES.items():
        for alias in aliases:
            for v in ([alias] if not alias else [alias, alias.upper(), alias.title()]):
                got = gw.resolve_script(v)
                if got != code:
                    wrong[v] = (got, code)
    assert not wrong, f"mis-resolved (got, expected): {wrong}"


@pytest.mark.parametrize("raw,code", [
    ("klingon", "en"), ("hin di", "en"), ("  Hindi  ", "devanagari"), ("DEVANAGARI", "devanagari"),
], ids=["unknown", "space-inside-unknown", "padded", "upper"])
def test_resolve_script_unknown_and_variants(raw, code):
    assert gw.resolve_script(raw) == code


@pytest.mark.parametrize("code,latin,other,expected", [
    ("en", DEFAULT_LATIN, DEFAULT_OTHER, ["deepreef_local", "rapidocr"]),
    ("devanagari", DEFAULT_LATIN, DEFAULT_OTHER, ["lambda", "deepreef_local", "rapidocr"]),
    ("en", "rapidocr, bogus ,deepreef_local", DEFAULT_OTHER, ["rapidocr", "deepreef_local"]),
], ids=["en-default", "other-default", "en-custom-unknown-ignored"])
def test_engine_order(code, latin, other, expected):
    s = gw_settings.GatewaySettings(_env_file=None, latin_engines=latin, other_engines=other)
    assert gw.engine_order(code, s) == expected


def test_read_image_empty_bytes_calls_no_engine(gs, monkeypatch):
    eng = Engines(monkeypatch, {})
    with pytest.raises(gw.OcrGatewayError):
        gw.read_image(b"", "english")
    assert eng.calls == []


@pytest.mark.parametrize("failure", ["raises", "empty_text"])
def test_read_image_falls_back(gs, monkeypatch, failure):
    first = RuntimeError("deepreef exploded") if failure == "raises" else _res("deepreef_local", "")
    eng = Engines(monkeypatch, {"deepreef_local": first,
                                "rapidocr": _res("rapidocr", "Row one\nRow two")})
    r = gw.read_image(b"img", "english")
    assert r.engine == "rapidocr" and r.text
    assert eng.calls == [("deepreef_local", "en"), ("rapidocr", "en")]
    assert [(a["engine"], a["ok"]) for a in r.attempts] == [("deepreef_local", False),
                                                            ("rapidocr", True)]
    assert all({"engine", "ok", "reason", "seconds"} <= set(a) for a in r.attempts)
    assert r.attempts[0]["reason"], "a failed attempt must say why"
    if failure == "raises":
        assert "deepreef exploded" in r.attempts[0]["reason"]


@pytest.mark.parametrize("script,order", [
    ("english", ["deepreef_local", "rapidocr"]),
    ("hindi", ["lambda", "deepreef_local", "rapidocr"]),
])
def test_read_image_all_fail_lists_every_engine(gs, monkeypatch, script, order):
    eng = Engines(monkeypatch, {n: RuntimeError(f"down-{n}") for n in order})
    with pytest.raises(gw.OcrGatewayError) as ei:
        gw.read_image(b"img", script)
    assert [c[0] for c in eng.calls] == order
    assert [a["engine"] for a in ei.value.attempts] == order
    assert not any(a["ok"] for a in ei.value.attempts)
    for n in order:
        assert f"down-{n}" in str(ei.value), f"message must name the {n} failure"


@pytest.mark.parametrize("script,code,first", [
    ("hindi", "devanagari", "lambda"), ("Urdu", "arabic", "lambda"),
    (None, "en", "deepreef_local"),
])
def test_read_image_passes_resolved_code_and_stops_at_first_success(gs, monkeypatch,
                                                                    script, code, first):
    eng = Engines(monkeypatch, {first: _res(first, "ok text")})
    r = gw.read_image(b"img", script)
    assert eng.calls == [(first, code)]
    assert r.script_code == code
    assert [a["engine"] for a in r.attempts] == [first]


@pytest.mark.parametrize("text_lines,lines,expected", [
    ("A  B\nC D", ["x", "y"], "A  B\nC D"),
    ("", ["Row one", "Row two"], "Row one\nRow two"),
], ids=["prefers-text_lines", "falls-back-to-lines"])
def test_page_text(text_lines, lines, expected):
    assert gw.page_text(_res("rapidocr", text_lines, lines=lines)) == expected


def test_engine_result_from_legacy():
    from app.documents.ocr_client import OcrResult

    legacy = OcrResult(text="A B", lines=["A", "B"], confidence=0.8, boxes=[],
                       model_used="m1", region_count=2)
    r = engines.EngineResult.from_legacy(legacy, engine="rapidocr")
    assert (r.text, r.lines, r.confidence, r.model_used, r.region_count, r.engine) == \
        ("A B", ["A", "B"], 0.8, "m1", 2, "rapidocr")
    assert r.text_lines == "A\nB"


def test_self_check_bogus_path_never_raises(gs, monkeypatch, tmp_path, _gateway_isolation):
    bogus = tmp_path / "no_such_deepreef"
    monkeypatch.setattr(gs, "deepreef_path", str(bogus))
    report = gw.self_check()
    assert report["deepreef_local"]["ok"] is False
    assert "no_such_deepreef" in str(report["deepreef_local"]["path"])
    assert report["deepreef_local"]["detail"]
    assert {"configured", "function"} <= set(report["lambda"])
    assert "enabled" in report
    assert _gateway_isolation.payloads == [], "self_check must never invoke the Lambda"


def test_settings_defaults():
    s = gw_settings.GatewaySettings(_env_file=None)
    assert (s.enabled, s.max_dim, s.max_words) == (False, 2048, 1000)
    assert (s.latin_engines, s.other_engines) == (DEFAULT_LATIN, DEFAULT_OTHER)
    assert (s.deepreef_path, s.keep_dated_lines, s.lambda_payload_limit_bytes) == \
        ("", True, 5_500_000)


def test_settings_env_override_and_cache(monkeypatch):
    monkeypatch.setenv("OCR_GATEWAY_ENABLED", "true")
    monkeypatch.setenv("OCR_GATEWAY_MAX_DIM", "3072")
    gw_settings.get_gateway_settings.cache_clear()
    s = gw_settings.get_gateway_settings()
    assert s.enabled is True and s.max_dim == 3072
    assert gw_settings.get_gateway_settings() is s, "settings must be cached"


# --- Lambda engine, against a stubbed boto3 client -------------------------


def _lambda_body(model_used="deepreef-devanagari-v2", success=True, as_string=True):
    body = {"success": success, "error": None if success else "model crashed",
            "ocr_data": {"text": "नमस्ते दुनिया", "lines": ["नमस्ते", "दुनिया"],
                         "confidence": 0.91, "model_used": model_used, "region_count": 2,
                         "bounding_boxes": [[[0, 0], [10, 0], [10, 5], [0, 5]],
                                            [[0, 6], [10, 6], [10, 11], [0, 11]]]},
            "image": {}}
    return json.dumps(body) if as_string else body


@pytest.fixture
def lambda_stub(monkeypatch):
    import boto3

    def _make(**kw):
        stub = _StubLambda(**kw)
        monkeypatch.setattr(boto3, "client", lambda *a, **k: stub)
        return stub
    return _make


@pytest.mark.parametrize("as_string", [True, False], ids=["string-body", "dict-body"])
def test_lambda_engine_payload_and_parse(lambda_stub, as_string):
    stub = lambda_stub(response_body=_lambda_body(as_string=as_string))
    s = gw_settings.GatewaySettings(_env_file=None, max_dim=1536, max_words=700)
    img = _small_png()
    r = engines.ENGINES["lambda"](img, "devanagari", s)
    (payload,) = stub.payloads
    assert (payload["script"], payload["max_dim"], payload["max_words"]) == \
        ("devanagari", 1536, 700)
    assert base64.b64decode(payload["image_base64"]) == img
    assert r.engine == "lambda" and r.model_used == "deepreef-devanagari-v2"
    assert r.lines == ["नमस्ते", "दुनिया"] and len(r.boxes) == 2


@pytest.mark.parametrize("case", ["success-false", "function-error", "empty-model_used"])
def test_lambda_engine_errors(lambda_stub, case):
    body = {"success-false": _lambda_body(success=False),
            "function-error": _lambda_body(),
            "empty-model_used": _lambda_body(model_used="")}[case]
    lambda_stub(response_body=body, function_error=(case == "function-error"))
    s = gw_settings.GatewaySettings(_env_file=None)
    with pytest.raises(engines.EngineError) as ei:
        engines.ENGINES["lambda"](_small_png(), "devanagari", s)
    if case == "empty-model_used":
        assert "model_used" in str(ei.value)


def _noisy_png(side: int = 1400) -> bytes:
    import numpy as np
    from PIL import Image

    arr = np.random.default_rng(3).integers(0, 256, (side, side, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _call_with_deadline(fn, seconds: float = 60):
    """The 0.8 downscale loop must terminate; a hang is a failure, not a stall."""
    box: dict = {}

    def run():
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(seconds)
    assert not t.is_alive(), f"lambda engine did not return within {seconds}s (resize loop?)"
    return box


@pytest.mark.parametrize("limit", [200_000, 100], ids=["fits-after-reencode", "impossible"])
def test_lambda_engine_oversized_payload(lambda_stub, limit):
    from PIL import Image

    stub = lambda_stub(response_body=_lambda_body())
    s = gw_settings.GatewaySettings(_env_file=None, lambda_payload_limit_bytes=limit)
    big = _noisy_png()
    assert len(base64.b64encode(big)) > 200_000
    box = _call_with_deadline(lambda: engines.ENGINES["lambda"](big, "devanagari", s))
    if "error" in box:
        assert isinstance(box["error"], engines.EngineError), repr(box["error"])
        assert stub.payloads == [], "an oversized payload must never be sent"
    else:
        (payload,) = stub.payloads
        assert len(payload["image_base64"]) <= limit
        sent = base64.b64decode(payload["image_base64"])
        assert Image.open(io.BytesIO(sent)).format == "JPEG"
    if limit == 100:
        assert "error" in box, "a 100-byte limit cannot be met; expected EngineError"


# ==========================================================================
# B. Wiring: preprocess_document through the gateway (fakes only)
# ==========================================================================

ROWS = "Patient Name: RaghuRam\nCollected: 25/08/2021 10:04\nHemoglobin 13.2 g/dL 13.0 - 17.0"


def test_wiring_scanned_pdf_english_keeps_rows(tmp_path, gateway_on, monkeypatch, legacy_calls):
    eng = Engines(monkeypatch, {"deepreef_local": _res(
        "deepreef_local", ROWS, lines=["Patient", "Name:", "RaghuRam"], text="glued blob")})
    out = _run(_doc(tmp_path, "pdf"), "pdf", "english")
    assert eng.calls == [("deepreef_local", "en")]
    assert out["page_text"][1] == ROWS
    rec = out["page_metadata"][1]
    assert rec.status.value == "OCR" and rec.ocr_applied and out["ocr_pages"] == [1]
    assert legacy_calls == []


@pytest.mark.parametrize("file_type,script,first_call", [
    ("pdf", "hindi", ("lambda", "devanagari")),
    ("png", "devanagari", ("lambda", "devanagari")),
    ("png", "english", ("deepreef_local", "en")),
], ids=["pdf-hindi", "image-devanagari-G10", "image-english"])
def test_wiring_routes_script(tmp_path, gateway_on, fake_ocr, monkeypatch,
                              file_type, script, first_call):
    fake = fake_ocr(pages=["Gateway read this page fine 123"])
    legacy_calls = _legacy_tripwire(monkeypatch)  # after fake_ocr, which patches legacy too
    out = _run(_doc(tmp_path, file_type), file_type, script)
    assert (fake.calls[0]["engine"], fake.calls[0]["script"]) == first_call
    assert "Gateway read this page fine 123" in out["page_text"][1]
    assert legacy_calls == []


@pytest.mark.parametrize("file_type", ["pdf", "png"])
def test_wiring_deepreef_fails_rapidocr_reads(tmp_path, gateway_on, monkeypatch,
                                              legacy_calls, file_type):
    eng = Engines(monkeypatch, {
        "deepreef_local": RuntimeError("deepreef path missing"),
        "rapidocr": _res("rapidocr", "Fallback row one value 1\nFallback row two value 2",
                         model="rapidocr-default")})
    out = _run(_doc(tmp_path, file_type), file_type, "english")
    assert [c[0] for c in eng.calls] == ["deepreef_local", "rapidocr"]
    assert out["page_metadata"][1].status.value == "OCR"
    assert "Fallback row one value 1" in out["page_text"][1]
    assert "rapidocr" in _page_story(out), "the page record must name the engine that read it"
    assert legacy_calls == []


@pytest.mark.parametrize("file_type", ["pdf", "png"])
def test_wiring_all_engines_fail_is_recorded(tmp_path, gateway_on, monkeypatch,
                                             legacy_calls, file_type):
    Engines(monkeypatch, {"deepreef_local": RuntimeError("dr-down"),
                          "rapidocr": RuntimeError("rapid-down")})
    out = _run(_doc(tmp_path, file_type), file_type, "english")
    assert out["page_metadata"][1].status.value == "OCR_FAILED"
    assert out["unreadable_pages"] == [1]
    story = _page_story(out)
    assert "dr-down" in story and "rapid-down" in story, story
    assert legacy_calls == []


@pytest.mark.parametrize("file_type,script,expected_calls,expected_text", [
    ("pdf", "english", [("rapidocr", "english")],
     "Fake OCR line one\nFake OCR line two value 42"),
    ("pdf", "devanagari", [("lambda", "devanagari")],
     "Fake OCR line one Fake OCR line two value 42"),
    ("png", "devanagari", [("rapidocr", "english")],
     "Fake OCR line one Fake OCR line two value 42"),
], ids=["pdf-english", "pdf-devanagari", "image-devanagari-legacy"])
def test_wiring_disabled_is_legacy(tmp_path, gs, monkeypatch, fake_ocr,
                                   file_type, script, expected_calls, expected_text):
    from app.graph.config import get_graph_settings

    monkeypatch.setattr(gs, "enabled", False)
    monkeypatch.setattr(get_graph_settings(), "ocr_enabled", True, raising=False)
    fake = fake_ocr()
    tripwire = Engines(monkeypatch, {})  # any gateway engine call fails loudly
    out = _run(_doc(tmp_path, file_type), file_type, script)
    assert tripwire.calls == []
    assert [(c["engine"], c["script"]) for c in fake.calls] == expected_calls
    assert out["page_text"][1] == expected_text


# ==========================================================================
# C. Real OCR acceptance (deepreef read-only checkout; no Lambda: script en)
# ==========================================================================

GLUED_DATE = re.compile(r"\d{1,2}/\d{1,2}/\d{4}[0-9A-Za-z]")


@pytest.fixture
def real_gateway(deepreef_path, gateway_on, monkeypatch):
    monkeypatch.setenv("OCR_GATEWAY_DEEPREEF_PATH", str(deepreef_path))
    gw_settings.get_gateway_settings.cache_clear()
    s = gw_settings.get_gateway_settings()
    for attr, val in [("enabled", True), ("keep_dated_lines", True),
                      ("latin_engines", DEFAULT_LATIN), ("other_engines", DEFAULT_OTHER)]:
        monkeypatch.setattr(s, attr, val)
    assert s.deepreef_path == str(deepreef_path)

    seen: list[dict] = []   # spy around the REAL engines: which one answered
    for name, fn in list(engines.ENGINES.items()):
        def spy(image_bytes, code, st, _fn=fn, _name=name):
            t0 = time.perf_counter()
            r = _fn(image_bytes, code, st)
            seen.append({"engine": _name, "model_used": r.model_used,
                         "seconds": r.ocr_seconds or time.perf_counter() - t0})
            return r
        monkeypatch.setitem(engines.ENGINES, name, spy)
    return seen


@pytest.mark.realocr
@pytest.mark.parametrize("kind", ["clean", "skew_2deg", "low_dpi_100", "noise", "jpeg_q25"],
                         ids=lambda k: f"gateway-{k}")
def test_real_round_trip(tmp_path, real_gateway, trace, kind):
    expected = "\n".join(sup.REPORT_LINES)
    with trace.step("build scan"):
        png = sup.degrade(sup.rasterize(sup.text_pdf([sup.REPORT_LINES]), dpi=200), kind)
        path = _doc(tmp_path, "pdf", png)
    with trace.step("preprocess (gateway)"):
        out = _run(path, "pdf", "english")
    got = out["page_text"].get(1, "")
    sc = sup.score(expected, got)
    assert real_gateway, "no gateway engine answered"
    used = real_gateway[-1]
    trace.metric(engine=f"{used['engine']}:{used['model_used']}",
                 ocr_seconds=round(sum(x["seconds"] for x in real_gateway), 2), **sc.as_dict())
    assert used["model_used"], "engine did not report model_used"
    detail = f"missing={sc.missing} glued={sc.glued_examples}\n{got}"
    if kind == "clean":
        assert sc.glued_words == 0, detail
        assert sc.date_recall == 1.0, detail
        assert "RaghuRam" in got, detail
        assert sc.exact_word_recall >= 0.9, detail
    else:
        assert not GLUED_DATE.search(got), detail
        assert sc.exact_word_recall >= 0.75, detail


@pytest.mark.realocr
@pytest.mark.dataset
def test_real_dataset_lab_report_image(real_gateway, dataset_file, trace):
    path = dataset_file("001_Lab-report.png")
    with trace.step("preprocess image (gateway)"):
        out = _run(str(path), "png", "english")
    got = out["page_text"].get(1, "")
    assert real_gateway, "no gateway engine answered"
    used = real_gateway[-1]
    trace.metric(engine=f"{used['engine']}:{used['model_used']}",
                 ocr_seconds=round(sum(x["seconds"] for x in real_gateway), 2))
    assert used["model_used"]
    for needle in ["25/08/2021 10:04", "25/08/2021 10:07", "25/08/2021 15:28",
                   "DIAGNOSTIC REPORT", "PATIENT ID"]:
        assert needle in got, f"{needle!r} not found verbatim in:\n{got}"
