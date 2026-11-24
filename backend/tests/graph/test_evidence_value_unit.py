"""Table-driven unit tests for the evidence value matcher (no mocks)."""
import pytest

from app.documents.value_matching import numbers_supported
from app.graph.nodes.evidence import _value_in_text
from app.graph.nodes.helpers import normalise_for_match


@pytest.mark.parametrize(
    "value, text, expected",
    [
        ("1.5", "0.15 mg/dL", False),
        ("1.5", "Hb 1.5 g/dL", True),
        ("5", "Temp -5", False),
        ("-5", "Temp -5", True),
        ("<0.01", "0.01", False),
        ("100000", "Fees 1,00,000", True),
        ("13.2 g/dL", "Hemoglobin 13.2 g/dL", True),
        ("1.5 g/dL", "0.15 g/dL", False),
        ("12/03/2024", "Date 12-03-2024", True),
        ("CIN123", "CIN1234", False),
        ("CIN123", "CIN123 issued", True),
        ("Raghu Ram", "RaghuRam", True),
        ("Acme Labs", "ACME  LABS", True),
        ("", "anything 1", False),
    ],
)
def test_value_in_text(value, text, expected):
    assert _value_in_text(value, text, normalise_for_match(text)) is expected


@pytest.mark.parametrize(
    "value, source, expected",
    [
        ("CIN123", "cin123 issued", False),
        ("13.2 g/dL", "hemoglobin 13.2 g/dl", True),
        ("12/03/2024", "date 12-03-2024", True),
        ("12/03/2024", "date 12-03", False),
    ],
)
def test_numbers_supported(value, source, expected):
    assert numbers_supported(value, source) is expected
