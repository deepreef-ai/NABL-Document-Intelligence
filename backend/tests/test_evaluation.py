import json
from pathlib import Path

import pytest

from app.evaluation import (character_error_rate, field_exact_match,
                            false_verification_rate, latency_summary,
                            levenshtein, review_rate, word_error_rate)
from app.evaluation.run import main

FIX = Path(__file__).parent / "fixtures" / "eval"


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein(["a", "b"], ["a"]) == 1


def test_cer_wer():
    assert character_error_rate("abcd", "abxd") == 0.25
    assert character_error_rate("", "") == 0.0
    assert character_error_rate("", "x") == 1.0
    assert word_error_rate("a b c d", "a x c") == 0.5


def test_field_exact_match():
    r = field_exact_match({"a": "X  Y", "b": "2", "c": "3"}, {"a": " x y", "b": {"value": "9"}})
    assert (r["matched"], r["total"], r["rate"]) == (1, 3, pytest.approx(1 / 3))
    assert field_exact_match({}, {})["rate"] is None


def test_false_verification_and_review():
    gold = {"a": "1", "b": "2", "c": "3"}
    pred = {"a": {"value": "1", "status": "verified"}, "b": {"value": "9", "status": "supported"},
            "c": {"value": "3", "status": "needs_review"}}
    r = false_verification_rate(gold, pred)
    assert (r["wrong_verified"], r["verified"], r["rate"]) == (1, 2, 0.5)
    assert review_rate(pred)["rate"] == pytest.approx(1 / 3)
    assert false_verification_rate(gold, {})["rate"] is None


def test_latency_nearest_rank():
    s = latency_summary({"x": [10, 20, 30, 40]})["x"]
    assert (s["p50"], s["p95"], s["mean"], s["count"]) == (20, 40, 25, 4)


def test_cli_on_fixtures(tmp_path, capsys):
    out = tmp_path / "r.json"
    assert main(["--gold", str(FIX / "gold"), "--pred", str(FIX / "pred"), "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    # hand-computed: edits 1+2 over 14 chars; word edits 1+2 over 4 words
    assert r["character_error_rate"] == pytest.approx(3 / 14)
    assert r["word_error_rate"] == 0.75
    # gold fields 5; matches: name, glucose
    assert r["field_exact_match"] == {"matched": 2, "total": 5, "rate": 0.4}
    # verified: name (ok), hb (wrong)
    assert r["false_verification_rate"] == {"wrong_verified": 1, "verified": 2, "rate": 0.5}
    # predicted 4 fields, 2 needs_review
    assert r["review_rate"] == {"flagged": 2, "total": 4, "rate": 0.5}
    assert r["latency_ms"]["ocr"]["p50"] == 100 and r["latency_ms"]["ocr"]["p95"] == 200
    assert r["latency_ms"]["extract"]["p95"] == 3000
    assert "field exact match: 2/5" in capsys.readouterr().out
