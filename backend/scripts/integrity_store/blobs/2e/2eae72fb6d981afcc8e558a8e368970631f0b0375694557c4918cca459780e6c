from types import SimpleNamespace

from app.documents import extractor
from app.documents.extractor import (
    MAX_OPEN_FIELDS,
    MAX_OPEN_VALUE_CHARS,
    _sanitize_open_fields,
    extract_open_fields,
    extract_open_fields_vision,
)


def _fake_chain(fields):
    return SimpleNamespace(generate_json=lambda **kwargs: {"fields": fields})


def test_extract_open_fields_returns_whatever_field_names_the_model_invents(monkeypatch):
    # Unlike extract_fields, there is no FIELD_SETS whitelist here — an
    # arbitrary model-invented field name must survive unfiltered.
    monkeypatch.setattr(
        extractor, "get_llm_chain",
        lambda: _fake_chain([{"field": "patient_name", "value": "Gunu", "confidence": 0.9}]),
    )

    result = extract_open_fields("some document text")

    assert result == [{"field": "patient_name", "value": "Gunu", "confidence": 0.9}]


def test_extract_open_fields_vision_calls_the_chain_with_the_image(monkeypatch):
    seen = {}

    def fake_generate_json(**kwargs):
        seen.update(kwargs)
        return {"fields": [{"field": "lab_name", "value": "Acme Labs", "confidence": 0.8}]}

    monkeypatch.setattr(extractor, "get_llm_chain", lambda: SimpleNamespace(generate_json=fake_generate_json))

    result = extract_open_fields_vision(b"imgbytes", "image/jpeg")

    assert result == [{"field": "lab_name", "value": "Acme Labs", "confidence": 0.8}]
    assert seen["image"] == b"imgbytes"
    assert seen["image_media_type"] == "image/jpeg"


def test_sanitize_open_fields_drops_duplicate_field_names_first_wins():
    fields = [
        {"field": "name", "value": "first", "confidence": 0.9},
        {"field": "name", "value": "second", "confidence": 0.9},
    ]

    result = _sanitize_open_fields(fields)

    assert result == [{"field": "name", "value": "first", "confidence": 0.9}]


def test_sanitize_open_fields_truncates_absurdly_long_values():
    long_value = "x" * (MAX_OPEN_VALUE_CHARS + 100)

    result = _sanitize_open_fields([{"field": "notes", "value": long_value, "confidence": 0.5}])

    assert len(result[0]["value"]) == MAX_OPEN_VALUE_CHARS


def test_sanitize_open_fields_caps_total_row_count():
    fields = [{"field": f"field_{i}", "value": "v", "confidence": 0.5} for i in range(MAX_OPEN_FIELDS + 20)]

    result = _sanitize_open_fields(fields)

    assert len(result) == MAX_OPEN_FIELDS


def test_sanitize_open_fields_skips_entries_with_no_field_name():
    result = _sanitize_open_fields([{"field": None, "value": "x", "confidence": 0.5}])

    assert result == []
