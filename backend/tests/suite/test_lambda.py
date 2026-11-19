"""Gateway -> deepreef-ocr Lambda.

Stubbed tests always run (no AWS call ever): they pin WHICH function is
called, with what payload, and the routing that sends a page to it.

Live tests run only with SUITE_LIVE_LAMBDA=1 and AWS credentials allowed to
invoke the function. They prove the deployed function answers, names the
model that ran, and whether it keeps dated lines (its camera-stamp filter,
which the backend cannot switch off per request, drops them today).
"""
from __future__ import annotations

import base64
import io
import json
import os

import pytest

gw = pytest.importorskip("app.ocr_gateway")

DATASET_PNG = "001_Lab-report.png"
DATES = ["25/08/2021 10:04", "25/08/2021 10:07", "25/08/2021 15:28"]


@pytest.fixture(autouse=True)
def _fresh_settings(monkeypatch):
    for k in list(os.environ):
        if k.startswith("OCR_GATEWAY_"):
            monkeypatch.delenv(k, raising=False)
    gw.get_gateway_settings.cache_clear()
    yield
    gw.get_gateway_settings.cache_clear()


class _Lambda:
    def __init__(self, body: dict):
        self.body, self.calls, self.region = body, [], None

    def invoke(self, FunctionName=None, Payload=None, **kw):  # noqa: N803 — boto3 API
        self.calls.append({"FunctionName": FunctionName, "payload": json.loads(Payload)})
        raw = json.dumps({"statusCode": 200, "body": json.dumps(self.body)}).encode()
        return {"Payload": io.BytesIO(raw)}


def _ok_body(text_lines="LINE ONE\nLINE TWO", model="devanagari_rec.onnx"):
    lines = text_lines.split("\n")
    return {"success": True, "ocr_data": {
        "text": " ".join(lines), "lines": lines, "confidence": 0.97, "script": "en",
        "model_used": model, "region_count": len(lines),
        "bounding_boxes": [[[0, 0], [10, 0], [10, 10], [0, 10]]] * len(lines),
        "text_lines": text_lines}, "image": {"quality_flag": "acceptable", "blur_score": 900.0}}


@pytest.fixture
def stub_lambda(monkeypatch):
    def _install(body):
        fake = _Lambda(body)

        def client(service, region_name=None, config=None):
            assert service == "lambda"
            fake.region = region_name
            return fake

        monkeypatch.setattr("boto3.client", client)
        return fake
    return _install


def _png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (40, 20), "white").save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- stubbed


def test_gateway_calls_the_deepreef_ocr_function_by_default(stub_lambda, trace):
    fake = stub_lambda(_ok_body())
    with trace.step("lambda engine (stubbed)"):
        r = gw.ENGINES["lambda"](_png(), "en", gw.get_gateway_settings())
    call = fake.calls[0]
    assert call["FunctionName"] == "deepreef-ocr"
    assert fake.region == "ap-south-1"
    assert call["payload"]["script"] == "en"
    assert call["payload"]["max_dim"] == 2048 and call["payload"]["max_words"] == 1000
    assert base64.b64decode(call["payload"]["image_base64"])
    assert r.engine == "lambda" and r.model_used == "devanagari_rec.onnx"
    assert r.text_lines == "LINE ONE\nLINE TWO"


def test_function_name_and_region_are_configurable(stub_lambda, monkeypatch):
    monkeypatch.setenv("OCR_GATEWAY_LAMBDA_FUNCTION", "deepreef-ocr-staging")
    monkeypatch.setenv("OCR_GATEWAY_LAMBDA_REGION", "us-east-1")
    gw.get_gateway_settings.cache_clear()
    fake = stub_lambda(_ok_body())
    gw.ENGINES["lambda"](_png(), "devanagari", gw.get_gateway_settings())
    assert fake.calls[0]["FunctionName"] == "deepreef-ocr-staging"
    assert fake.region == "us-east-1"


@pytest.mark.parametrize("script,code", [("hindi", "devanagari"), ("tamil", "ta"),
                                         ("bengali", "bn"), ("urdu", "arabic")])
def test_indian_language_pages_go_to_lambda_first(stub_lambda, script, code):
    fake = stub_lambda(_ok_body(model=f"{code}_model.onnx"))
    r = gw.read_image(_png(), script)
    assert r.engine == "lambda" and fake.calls[0]["payload"]["script"] == code
    assert r.attempts[0]["engine"] == "lambda" and r.attempts[0]["ok"]


def test_english_uses_lambda_when_configured_first(stub_lambda, monkeypatch):
    monkeypatch.setenv("OCR_GATEWAY_LATIN_ENGINES", "lambda,deepreef_local,rapidocr")
    gw.get_gateway_settings.cache_clear()
    fake = stub_lambda(_ok_body())
    r = gw.read_image(_png(), "english")
    assert r.engine == "lambda" and fake.calls[0]["payload"]["script"] == "en"


def test_lambda_access_denied_falls_back_and_says_why(monkeypatch):
    from botocore.exceptions import ClientError

    class Denied:
        def invoke(self, **kw):
            raise ClientError({"Error": {"Code": "AccessDeniedException",
                                         "Message": "not authorized"}}, "Invoke")

    monkeypatch.setattr("boto3.client", lambda *a, **k: Denied())
    monkeypatch.setitem(gw.ENGINES, "deepreef_local",
                        lambda b, c, s: gw.EngineResult(text="x y", lines=["x y"], confidence=0.9,
                                                        boxes=[], model_used="m", region_count=1,
                                                        engine="deepreef_local", text_lines="x y"))
    r = gw.read_image(_png(), "hindi")
    assert r.engine == "deepreef_local"
    assert r.attempts[0]["engine"] == "lambda" and not r.attempts[0]["ok"]
    assert "AccessDenied" in r.attempts[0]["reason"]


def test_self_check_reports_the_lambda_function_without_calling_it(monkeypatch):
    monkeypatch.setattr("boto3.client", lambda *a, **k: pytest.fail("self_check must not call AWS"))
    rep = gw.self_check()
    assert rep["lambda"]["function"] == "deepreef-ocr" and rep["lambda"]["configured"]


# ---------------------------------------------------------------- live (opt-in)


@pytest.mark.live_lambda
@pytest.mark.dataset
@pytest.mark.parametrize("script", ["en", "devanagari"])
def test_live_lambda_reads_a_real_report(dataset_file, trace, script):
    data = dataset_file(DATASET_PNG).read_bytes()
    with trace.step(f"invoke deepreef-ocr script={script}"):
        r = gw.ENGINES["lambda"](data, script, gw.get_gateway_settings())
    trace.metric(engine=f"lambda:{r.model_used}", regions=r.region_count,
                 quality=r.quality_flag)
    assert r.model_used and r.region_count > 20
    assert "DIAGNOSTIC REPORT" in r.text_lines


@pytest.mark.live_lambda
@pytest.mark.dataset
@pytest.mark.xfail(strict=True, reason="deployed Lambda's camera-stamp filter drops lines "
                   "with dd/mm/yyyy dates; needs a filter-off config on the OCR side")
def test_live_lambda_keeps_report_dates(dataset_file):
    data = dataset_file(DATASET_PNG).read_bytes()
    r = gw.ENGINES["lambda"](data, "en", gw.get_gateway_settings())
    assert all(d in r.text_lines for d in DATES)
