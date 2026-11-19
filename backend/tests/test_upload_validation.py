import asyncio
import io

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import upload_validation as uv
from app.config import get_settings


def _upload(name, data, content_type="application/octet-stream"):
    return UploadFile(file=io.BytesIO(data), filename=name,
                      headers={"content-type": content_type})


def _read(name, data, content_type="application/octet-stream"):
    return asyncio.run(uv.read_validated_upload(_upload(name, data, content_type)))


def _status(name, data, content_type="application/octet-stream"):
    with pytest.raises(HTTPException) as exc:
        _read(name, data, content_type)
    return exc.value.status_code


def test_valid_files_accepted():
    assert _read("a.pdf", b"%PDF-1.7 x") == b"%PDF-1.7 x"
    assert _read("a.PNG", b"\x89PNG\r\n\x1a\n123")
    assert _read("a.jpg", b"\xff\xd8\xff\xe0abc")
    assert _read("a.webp", b"RIFF\x00\x00\x00\x00WEBPxx")
    assert _read("a.docx", b"PK\x03\x04rest")


def test_no_extension_uses_content_type():
    assert _read("scan", b"%PDF-1.4", "application/pdf")
    assert _status("scan", b"abc") == 415


def test_unsupported_extension_is_415():
    assert _status("evil.exe", b"MZ") == 415
    assert _status("a.txt", b"hello") == 415


def test_empty_is_400():
    assert _status("a.pdf", b"") == 400


def test_magic_mismatch_is_400():
    assert _status("a.pdf", b"not a pdf") == 400
    assert _status("a.png", b"%PDF-1.4") == 400
    assert _status("a.webp", b"RIFF\x00\x00\x00\x00WAVExx") == 400


def test_too_large_is_413(monkeypatch):
    monkeypatch.setattr(get_settings(), "max_upload_mb", 1)
    assert _status("a.pdf", b"%PDF-" + b"0" * (1024 * 1024)) == 413
    assert _read("a.pdf", b"%PDF-" + b"0" * 1000)


def test_endpoint_rejections_via_testclient(monkeypatch):
    from app.db import Base, get_db
    from app.main import app
    from app.models import Application

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    with Session() as db:
        db.add(Application(id="app1", form_type="NABL_151", status="unlocked"))
        db.commit()

    def override():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override
    monkeypatch.setattr(get_settings(), "max_upload_mb", 1)
    try:
        client = TestClient(app)
        url = "/applications/app1/documents"
        r = client.post(url, files={"file": ("x.exe", b"MZ", "application/octet-stream")})
        assert r.status_code == 415 and "unsupported" in r.json()["detail"]
        r = client.post(url, files={"file": ("x.pdf", b"nope", "application/pdf")})
        assert r.status_code == 400
        big = b"%PDF-" + b"0" * (1024 * 1024)
        r = client.post(url, files={"file": ("x.pdf", big, "application/pdf")})
        assert r.status_code == 413
    finally:
        app.dependency_overrides.clear()
