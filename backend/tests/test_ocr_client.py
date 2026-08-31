import io
import json

import pytest

from app.documents.ocr_client import OcrClient, OcrError

FAKE_OCR_DATA = {
    "text": "hello world",
    "lines": ["hello", "world"],
    "confidence": 0.91,
    "script": "devanagari",
    "model_used": "devanagari_rec.onnx",
    "region_count": 2,
    "bounding_boxes": [[[10, 12], [100, 12], [100, 30], [10, 30]], [[10, 40], [80, 40], [80, 60], [10, 60]]],
    "ocr_ms": 142.0,
}


class FakeLambdaClient:
    def __init__(self, response_body: dict, function_error: str | None = None):
        self._response_body = response_body
        self._function_error = function_error
        self.invoke_calls: list[dict] = []

    def invoke(self, FunctionName, Payload):  # noqa: N803 - matches boto3's signature
        self.invoke_calls.append({"FunctionName": FunctionName, "Payload": Payload})
        result = {"Payload": io.BytesIO(json.dumps(self._response_body).encode("utf-8"))}
        if self._function_error:
            result["FunctionError"] = self._function_error
        return result


def _lambda_envelope(payload: dict) -> dict:
    # Mirrors the real deepreef-ocr contract: the Lambda's own envelope has a
    # `body` key whose value is itself a JSON string.
    return {"statusCode": 200, "headers": {}, "body": json.dumps(payload)}


def _client_with(fake_lambda: FakeLambdaClient, monkeypatch) -> OcrClient:
    monkeypatch.setattr("app.documents.ocr_client.boto3.client", lambda *a, **k: fake_lambda)
    return OcrClient(function_name="akash-ocr", region="ap-south-1")


def test_extract_parses_the_real_contract_shape(monkeypatch):
    fake_lambda = FakeLambdaClient(
        _lambda_envelope({"success": True, "ocr_data": FAKE_OCR_DATA, "image": {}, "latency_ms": 1.0})
    )
    client = _client_with(fake_lambda, monkeypatch)

    result = client.extract(b"fake-image-bytes", script="devanagari")

    assert result.text == "hello world"
    assert result.lines == ["hello", "world"]
    assert result.model_used == "devanagari_rec.onnx"
    assert len(result.boxes) == 2
    # quad -> axis-aligned rect
    assert result.boxes[0].x == 10
    assert result.boxes[0].y == 12
    assert result.boxes[0].w == 90
    assert result.boxes[0].h == 18
    assert fake_lambda.invoke_calls[0]["FunctionName"] == "akash-ocr"


def test_unsupported_script_is_rejected_before_any_network_call(monkeypatch):
    fake_lambda = FakeLambdaClient(_lambda_envelope({}))
    client = _client_with(fake_lambda, monkeypatch)

    with pytest.raises(OcrError, match="no recognition model"):
        client.extract(b"bytes", script="english")

    assert fake_lambda.invoke_calls == []


def test_upstream_failure_response_raises_ocr_error(monkeypatch):
    fake_lambda = FakeLambdaClient(_lambda_envelope({"success": False, "error": "boom"}))
    client = _client_with(fake_lambda, monkeypatch)

    with pytest.raises(OcrError, match="boom"):
        client.extract(b"bytes", script="arabic")


def test_function_error_raises_ocr_error(monkeypatch):
    fake_lambda = FakeLambdaClient({"errorMessage": "boom"}, function_error="Unhandled")
    client = _client_with(fake_lambda, monkeypatch)

    with pytest.raises(OcrError, match="Lambda errored"):
        client.extract(b"bytes", script="devanagari")
