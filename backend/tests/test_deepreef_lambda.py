import io
import json
import threading
import time

from botocore.exceptions import ClientError

from app.documents import deepreef_lambda
from app.documents.deepreef_lambda import extract_pages_lambda


def _ok(ocr_data):
    body = json.dumps({"success": True, "ocr_data": ocr_data, "latency_ms": 5})
    return {"Payload": io.BytesIO(json.dumps({"statusCode": 200, "body": body}).encode())}


class StubClient:
    def __init__(self, delay=0.0, deny_page=None):
        self.delay = delay
        self.deny_page = deny_page
        self.payloads = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def invoke(self, FunctionName, Payload):
        req = json.loads(Payload)
        with self._lock:
            self.payloads.append(req)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            page = int(__import__("base64").b64decode(req["image_base64"]).decode())
            if page == self.deny_page:
                raise ClientError(
                    {"Error": {"Code": "AccessDeniedException", "Message": "denied"}}, "Invoke"
                )
            return _ok({"text": f"flat {page}", "text_lines": [f"row1 p{page}", f"row2 p{page}"],
                        "confidence": 0.9, "lines": [], "model_used": "en", "region_count": 2})
        finally:
            with self._lock:
                self.active -= 1


def _images(n):
    return {i: str(i).encode() for i in range(1, n + 1)}


def test_parallel_fanout_returns_all_pages_and_uses_boto_config(monkeypatch):
    stub = StubClient(delay=0.2)
    captured = {}

    def fake_client(service, region_name, config):
        captured.update(service=service, region=region_name, config=config)
        return stub

    monkeypatch.setattr(deepreef_lambda.boto3, "client", fake_client)
    start = time.monotonic()
    out = extract_pages_lambda(_images(8), function_name="akash-ocr", region="ap-south-1")
    elapsed = time.monotonic() - start

    assert sorted(out) == list(range(1, 9))
    assert all(err == "" for _, _, err in out.values())
    assert stub.max_active == 8 and elapsed < 1.0  # ~one page's latency, not 8x
    assert captured["service"] == "lambda" and captured["region"] == "ap-south-1"
    assert captured["config"].max_pool_connections == 16


def test_request_payload_has_script_en_and_max_words_zero():
    stub = StubClient()
    extract_pages_lambda(_images(2), function_name="f", region="r", client=stub)
    assert len(stub.payloads) == 2
    for req in stub.payloads:
        assert req["script"] == "en" and req["max_words"] == 0 and req["image_base64"]


def test_text_lines_preferred_over_text():
    out = extract_pages_lambda(_images(1), function_name="f", region="r", client=StubClient())
    text, conf, err = out[1]
    assert text == "row1 p1\nrow2 p1" and conf == 0.9 and err == ""


def test_access_denied_on_one_page_only_sets_that_pages_error():
    stub = StubClient(deny_page=2)
    out = extract_pages_lambda(_images(3), function_name="f", region="r", client=stub)
    assert out[2][0] == "" and out[2][1] is None and "AccessDenied" in out[2][2]
    assert out[1][2] == "" and out[3][2] == "" and out[3][0].startswith("row1")
