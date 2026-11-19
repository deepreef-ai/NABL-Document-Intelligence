"""strict_value_key: numbers keep decimal point / sign / comparator; formatting noise is ignored."""
import pytest

from app.graph.nodes.helpers import strict_value_key as key


@pytest.mark.parametrize("a,b", [
    ("1.5", "0.15"), ("1.5", "15"), ("0.15", "15"),
    ("5", "-5"), ("<0.01", "0.01"), (">5", "<5"), ("<=5", "<5"),
    ("1.5 mg/dL", "0.15 mg/dL"), ("1.5 mg/dL", "15 mg/dL"),
    ("Hb 13.2", "Hb 132"),
])
def test_different_numbers_get_different_keys(a, b):
    assert key(a) != key(b)


@pytest.mark.parametrize("a,b", [
    ("1,00,000", "100000"), ("007", "7"), ("1.50", "1.5"),
    ("13.2 G/DL", "13.2   g/dl"), ("40 - 129", "40-129"), ("40–129", "40-129"),
    ("≤5", "<=5"), ("Sunrise  Labs", "sunrise labs"),
])
def test_same_value_different_formatting_same_key(a, b):
    assert key(a) == key(b)


def test_none_and_empty():
    assert key(None) == "" and key("   ") == ""


def test_ranges_do_not_read_as_negative_numbers():
    assert key("40-129") != key("40--129")
    assert key("40-129") == key("40 - 129")
