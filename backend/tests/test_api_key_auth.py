import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import get_settings
from app.main import app
from app.db import Base, get_db


@pytest.fixture()
def client():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    def override():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_disabled_by_default(client):
    assert get_settings().api_key == ""
    assert client.get("/applications/nope/documents").status_code == 404  # reached the route


def test_enabled_requires_key(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "api_key", "s3cret")
    url = "/applications/nope/documents"
    r = client.get(url)
    assert r.status_code == 401 and r.json()["detail"]
    assert client.get(url, headers={"X-API-Key": "wrong"}).status_code == 401
    # right key gets through to the route (404: no such application)
    assert client.get(url, headers={"X-API-Key": "s3cret"}).status_code == 404


def test_health_and_docs_stay_open(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "api_key", "s3cret")
    assert client.get("/health").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200


def test_preflight_not_blocked(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "api_key", "s3cret")
    r = client.options("/applications/x/documents", headers={
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "x-api-key",
    })
    assert r.status_code == 200
