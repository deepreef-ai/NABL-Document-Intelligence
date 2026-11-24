"""Remote OCR service + client (2026-11-24). No real models are loaded."""
import base64

import httpx
import pytest
from fastapi.testclient import TestClient

from app.documents import deepreef_local, ocr_service_client
from app.documents.deepreef_local import PageOcr


@pytest.fixture
def client(monkeypatch):
    calls = []

    def fake_extract(image_bytes, *, rendered_pdf_page=False):
        calls.append((image_bytes, rendered_pdf_page))
        return PageOcr(text="Lead 0.5 mg/kg", lines=["Lead 0.5 mg/kg"],
                       confidence=0.91, model_used="fake", region_count=3)

    monkeypatch.setattr(deepreef_local, "extract_page", fake_extract)
    monkeypatch.setattr(deepreef_local, "model_report", lambda: {"fake": True})
    from ocr_service.app import app
    c = TestClient(app)
    c.calls = calls
    return c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model": {"fake": True}}


def test_ocr_page_round_trip(client):
    r = client.post("/ocr/page", json={"image_base64": base64.b64encode(b"PNG").decode(),
                                       "rendered_pdf_page": True})
    assert r.status_code == 200
    assert r.json() == {"text": "Lead 0.5 mg/kg", "lines": ["Lead 0.5 mg/kg"],
                        "confidence": 0.91, "model_used": "fake", "region_count": 3}
    assert client.calls == [(b"PNG", True)]


def test_client_returns_page_ocr_via_server(client, monkeypatch):
    # Route httpx.post into the TestClient so client and server are exercised together.
    def fake_post(url, json, timeout):
        assert url == "http://ocr:8100/ocr/page"
        return client.post("/ocr/page", json=json)

    monkeypatch.setattr(httpx, "post", fake_post)
    out = ocr_service_client.extract_pages_remote({1: b"a", 2: b"b"}, base_url="http://ocr:8100/")
    assert set(out) == {1, 2}
    assert all(isinstance(p, PageOcr) and p.text == "Lead 0.5 mg/kg" and p.region_count == 3
               for p in out.values())


def test_client_non_200_raises(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda url, json, timeout:
                        httpx.Response(500, json={"detail": "OCR failed: ValueError"}))
    with pytest.raises(RuntimeError, match="HTTP 500"):
        ocr_service_client.extract_page_remote(b"x", rendered_pdf_page=False, base_url="http://ocr")
