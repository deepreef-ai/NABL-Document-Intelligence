import pytest

from app.documents.value_matching import normalize_for_search, value_in_source


def _check(value, source):
    return value_in_source(value, normalize_for_search(source))


@pytest.mark.parametrize(
    "value, source, expected",
    [
        # decimals must not match inside longer numbers
        (1.5, "0.15 mg/dL", False),
        ("1.5", "0.15 mg/dL", False),
        ("1.5", "hb 1.5 g/dl", True),
        (15, "0.15", False),
        ("15", "0.15", False),
        # sign / comparator must agree
        (-5, "5", False),
        (5, "-5", False),
        (-5, "-5", True),
        ("<0.01", "0.01", False),
        ("<0.01", "< 0.01", True),
        ("<0.01", "<0.01", True),
        ("0.01", "<0.01", False),
        # numeric normalisation
        (100000, "1,00,000", True),
        ("100000", "total 1,00,000 units", True),
        (7, "007", True),
        ("7", "id 007", True),
        # ranges
        ("40-129", "40 - 129", True),
        ("40-129", "range 40–129 u/l", True),
        ("40-129", "140-129", False),
        ("40-129", "40-1290", False),
        # mixed units / words
        ("13.2 g/dL", "Hb 13.2 g/dL", True),
        ("13.2 g/dL", "Hb 113.2 g/dL", False),
        ("RaghuRam", "Raghu Ram", True),
        ("Raghu Ram", "RAGHURAM", True),
        ("Raghu-Ram", "raghu–ram", True),
        # empty / absent
        ("", "something", False),
        (None, "something", False),
        ("5", "", False),
        ("abc", "", False),
        ("xyz", "nothing like it here 42", False),
        (99, "nothing like it here 42", False),
    ],
)
def test_value_in_source(value, source, expected):
    assert _check(value, source) is expected


def test_normalize_for_search_basics():
    assert normalize_for_search(None) == ""
    assert normalize_for_search("  Raghu \n  RAM ") == "raghu ram"
    assert normalize_for_search("40–29") == "40-29"
    assert normalize_for_search("≤ 5") == "<= 5"
