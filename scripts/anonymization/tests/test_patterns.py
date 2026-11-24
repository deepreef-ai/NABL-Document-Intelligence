"""Pattern and validator tests.

The Aadhaar cases matter most. A lab report is dense with 12-digit specimen and
accession numbers, and without the Verhoeff check the detector would redact
them — turning a privacy tool into a data-destruction tool.
"""
from __future__ import annotations

import pytest

from anonymizer.patterns import (
    PATTERNS,
    is_aadhaar,
    is_ifsc,
    is_pan,
    looks_like_null,
    normalise_label,
    verhoeff_valid,
)


class TestVerhoeff:
    """The numbers in this file are not arbitrary: 234123456783 carries a
    correct Verhoeff check digit and 987654321011 does not. Changing a digit
    to make a test "look nicer" will silently invert its meaning.
    """

    @pytest.mark.parametrize("vector", ["2363", "123451", "1428570", "8473643095"])
    def test_matches_published_vectors(self, vector):
        # Guards the transcription of the D and P tables themselves.
        assert verhoeff_valid(vector)

    def test_known_valid_number(self):
        assert verhoeff_valid("234123456783")

    def test_single_digit_change_is_rejected(self):
        assert not verhoeff_valid("234123456782")

    def test_non_digits_are_rejected(self):
        assert not verhoeff_valid("2341234567AB")


class TestAadhaar:
    def test_valid_with_spaces(self):
        assert is_aadhaar("2341 2345 6783")

    def test_valid_with_hyphens(self):
        assert is_aadhaar("2341-2345-6783")

    def test_rejects_leading_zero_or_one(self):
        # UIDAI never issues a number starting 0 or 1.
        assert not is_aadhaar("0341 2345 6783")
        assert not is_aadhaar("1341 2345 6783")

    def test_rejects_wrong_length(self):
        assert not is_aadhaar("2341 2345 678")

    def test_rejects_repeated_digit_placeholder(self):
        assert not is_aadhaar("222222222222")

    def test_checksum_rejects_a_specimen_number(self):
        # The real point of the checksum: a random 12-digit lab number should
        # not be mistaken for somebody's Aadhaar.
        assert not is_aadhaar("987654321011")

    def test_checksum_can_be_disabled(self):
        assert is_aadhaar("987654321011", require_checksum=False)


class TestPan:
    @pytest.mark.parametrize("value", ["ABCPK1234F", "abcpk1234f", " ABCPK1234F "])
    def test_accepts_valid(self, value):
        assert is_pan(value)

    @pytest.mark.parametrize("value", ["ABCP1234F", "ABCPK12345", "ABCPK1234", "12345ABCDE"])
    def test_rejects_malformed(self, value):
        assert not is_pan(value)


class TestIfsc:
    def test_accepts_valid(self):
        assert is_ifsc("HDFC0001234")

    def test_requires_zero_in_fifth_position(self):
        assert not is_ifsc("HDFC1001234")


class TestShapePatterns:
    def test_email(self):
        m = PATTERNS["EMAIL"].search("contact ramesh.kumar@example.com today")
        assert m and m.group() == "ramesh.kumar@example.com"

    def test_indian_mobile_with_country_code(self):
        m = PATTERNS["PHONE"].search("Mobile: +91 9876543210")
        assert m and "9876543210" in m.group()

    def test_phone_does_not_match_a_landline_style_number(self):
        # Indian mobiles start 6-9; a number starting 2 must not match.
        assert not PATTERNS["PHONE"].search("2345678901")

    def test_phone_does_not_match_inside_a_longer_digit_run(self):
        assert not PATTERNS["PHONE"].search("129876543210345")

    def test_gstin_shape(self):
        assert PATTERNS["GSTIN"].search("27ABCPK1234F1Z5")


class TestHelpers:
    @pytest.mark.parametrize("value", ["-", "N/A", "nil", "NONE", " Unknown "])
    def test_null_values_recognised(self, value):
        assert looks_like_null(value)

    def test_real_value_is_not_null(self):
        assert not looks_like_null("Ramesh")

    def test_label_normalisation_strips_punctuation_and_case(self):
        assert normalise_label("  Patient   Name :  ") == "patient name"
