"""Truncated-reply salvage: a model that runs out of output tokens mid-table
must not cost us the whole document."""
import json

import pytest

from app.llm.json_utils import JsonParseError, parse_json_object


def test_complete_object_unaffected():
    assert parse_json_object('{"fields": {"a": "b"}, "tests": []}') == {"fields": {"a": "b"}, "tests": []}


def test_fenced_object_still_works():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_unterminated_fence_with_balanced_json():
    # An opening fence with no closing one — the brace scan handles it.
    assert parse_json_object('```json\n{"a": 1}') == {"a": 1}


def test_truncated_mid_row_keeps_the_completed_rows():
    raw = (
        '```json\n{\n "fields": {"report_no": "R-1"},\n "tests": [\n'
        '  {"test_name": "Acidity", "result": "0.10"},\n'
        '  {"test_name": "Moisture", "result": "3.4"},\n'
        '  {"test_name": "Prot'
    )
    out = parse_json_object(raw)
    assert out["fields"] == {"report_no": "R-1"}
    assert [t["test_name"] for t in out["tests"]] == ["Acidity", "Moisture"]


def test_truncated_inside_fields_keeps_what_arrived():
    out = parse_json_object('{"fields": {"a": "1", "b": "2", "c": "unfinis')
    assert out["fields"] == {"a": "1", "b": "2"}


def test_braces_and_commas_inside_strings_are_not_structure():
    raw = '{"fields": {"note": "a { b , c [ d"}, "tests": [{"test_name": "X, {Y}", "result": "1"}, {"test_name": "trunc'
    out = parse_json_object(raw)
    assert out["fields"]["note"] == "a { b , c [ d"
    assert [t["test_name"] for t in out["tests"]] == ["X, {Y}"]


def test_escaped_quote_inside_string_survives():
    out = parse_json_object('{"fields": {"q": "say \\"hi\\"", "b": "2", "c": "trunc')
    assert out["fields"]["q"] == 'say "hi"'
    assert out["fields"]["b"] == "2"


def test_genuine_garbage_still_raises():
    with pytest.raises(JsonParseError):
        parse_json_object("the model refused to answer")


def test_truncated_beyond_recovery_raises():
    # Nothing complete ever closed, so there is nothing to keep.
    with pytest.raises(JsonParseError):
        parse_json_object('{"fields": {"a": "unterminated')
