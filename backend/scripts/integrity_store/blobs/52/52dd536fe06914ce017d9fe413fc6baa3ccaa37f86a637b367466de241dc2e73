"""The evidence gate must survive a two-column page without going blind.

A quote that crosses a column break is not a contiguous substring of the
page's reading order, because the other column's text falls between the label
and its value. MEASURED on a real cytology report: nine correct values were
rejected as unsupported for exactly that reason — "Epithelial Fragments:" and
"Urothelial Fragment" sit one above the other in a right-hand sidebar, and
"pH 6.5 5-9" from the results table lands between them.

What must NOT happen is the fix turning the gate into a rubber stamp. Half
these cases are values that are plausible for this document and are simply not
on the page; every one of them still has to be refused.
"""
from app.graph.nodes.evidence import _gapped_match, _tokenise

# Reading order as the layout rebuild actually produces it for the report's
# two-column body: results table on the left, findings sidebar on the right.
PAGE = """\
Accession No.   Chart No. Sex D.O.B.   Page
UH0000000   1400000 M 01/01/1900   1 of 2
Patient Name   Lifetime ID: ***-**-0000 Collected
SAMPLE, PATIENT   01/10/13
DIANON SYSTEMS   Referring Physician   Reported
SAMPLE 840 RESEARCH PARKWAY
01/14/13
MACROSCOPIC DESCRIPTION
Specific Gravity: 1.019 (1.000 - 1.035)
URINE CHEMISTRY   Erythrocytes per /10 HPFs:
Qualitative   Abnormal   Normal   Reference Range Epithelial Fragments:
pH   6.5   5-9   Urothelial Fragment
Macroalbumin   25   < 30 mg/dL
Glucose   NEG   < 50 mg/dL
Ketone   NEG   NEGATIVE
Bilirubin   NEG   NEGATIVE   NONSPECIFIC FINDINGS
Blood   25   < 5 ery/uL   Concretions:
Nitrite   NEG   NEGATIVE   Calcified (Lithiasis)
Total Protein   270   0-150 mg/L
"""

TOKENS = _tokenise(PAGE)


def _match(quote: str) -> bool:
    return _gapped_match(_tokenise(quote), TOKENS)


class TestColumnSplitQuotesAreAccepted:
    """The pair a reader sees, interrupted only by the neighbouring column."""

    def test_sidebar_label_and_value_split_by_the_results_table(self):
        assert _match("Epithelial Fragments:\nUrothelial Fragment")

    def test_second_sidebar_pair_split_by_two_table_rows(self):
        assert _match("Concretions:\nCalcified (Lithiasis)")

    def test_header_pair_split_by_an_address_line(self):
        assert _match("Referring Physician Reported\nSAMPLE 01/14/13")

    def test_a_quote_that_is_already_contiguous_still_matches(self):
        assert _match("Specific Gravity: 1.019 (1.000 - 1.035)")


class TestFabricationsAreStillRefused:
    """Every one of these is plausible for this report and absent from it.

    These are the cases that make the relaxation safe rather than reckless: a
    gap budget large enough to admit them would have made the gate worthless.
    """

    def test_a_wrong_digit_in_an_identifier(self):
        # UH0000001 for UH0000000 — one character, and the whole point of the
        # gate is that it is the one character that matters.
        assert not _match("Accession No. Chart No. Sex D.O.B.\nUH0000001 1400000 M")

    def test_a_value_from_the_wrong_row(self):
        assert not _match("Ketone POSITIVE NEGATIVE")

    def test_an_invented_measurement(self):
        assert not _match("Bilirubin 42 mg/dL")

    def test_an_invented_result_for_a_real_analyte(self):
        assert not _match("Total Protein 999 0-150 mg/L")

    def test_a_name_that_is_nowhere_on_the_page(self):
        assert not _match("Referring Physician: Edward C. Poole, M.D.")

    def test_words_scattered_too_far_apart_to_be_one_reading(self):
        # Every word IS on the page — "Accession" at the top, "Total Protein"
        # at the bottom. Distance is what tells them apart from a column pair.
        assert not _match("Accession No. Total Protein 270")


class TestFloors:
    """Short quotes never take this path; they are too easy to hit by chance."""

    def test_two_words_is_not_enough(self):
        assert not _match("Nitrite NEGATIVE")

    def test_three_very_short_words_are_not_enough(self):
        assert not _match("pH 7 4")
