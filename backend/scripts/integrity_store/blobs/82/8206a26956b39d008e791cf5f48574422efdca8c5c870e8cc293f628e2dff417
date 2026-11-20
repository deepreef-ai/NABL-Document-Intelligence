"""A chart's axis is not a measurement.

MEASURED on `001_Lab-report.png`, an HbA1c trend chart on a scanned page. OCR
flattens the plot into loose tokens in reading order:

    Diabetics 7.52 Prediabetics 5.6 5.64- 4.8 5.0 3.76 Nondiabetic % 1.88-
    01-MAR-201525-MAR-2016 28-AUG-2016 18-JUL-2021 25-AUG-2021 Date-

and the extractor emitted `GLYCOSYLATED HEMOGLOBIN (HBA1C) = 7.52 %`. 7.52 is
the second tick on the y-axis. The patient's readings are 5.6, 5.4, 5.5, 4.8
and 5.0. A wrong measurement presented as a real one is worse than a gap,
because a gap is visible and this was not.

The coordinates below are the real ones RapidOCR returned for that page.
"""
from dataclasses import dataclass

from app.graph.charts import annotate, detect_chart


@dataclass
class Box:
    """Stands in for documents.geometry.Rect — same four attributes."""

    x: float
    y: float
    w: float
    h: float


def ocr(*tokens):
    """(text, x, y, w, h) tuples -> the (lines, boxes) pair local_ocr returns."""
    return [t[0] for t in tokens], [Box(t[1], t[2], t[3], t[4]) for t in tokens]


#: The HbA1c chart, verbatim from RapidOCR.
CHART = (
    ("GLYCUSYLAIEU HEMUGLOBIN(HBAIC)", 308.0, 513.0, 250.0, 20.0),
    ("Diabetics",              184.0, 600.0,  71.0, 23.0),
    ("7.52",                   144.0, 612.0,  40.0, 16.0),
    ("Prediabetics",           187.0, 653.0,  89.0, 17.0),
    ("5.6",                    277.0, 645.0,  26.0, 16.0),
    ("5.64-",                  144.0, 662.0,  44.0, 16.0),
    ("4.8",                    583.0, 664.0,  26.0, 16.0),
    ("5.0",                    670.0, 678.0,  26.0, 16.0),
    ("3.76",                   143.0, 710.0,  40.0, 16.0),
    ("Nondiabetic",            187.0, 733.0,  86.0, 20.0),
    ("%",                      114.0, 750.0,  12.0, 16.0),
    ("1.88-",                  145.0, 760.0,  44.0, 16.0),
    ("01-MAR-201525-MAR-2016", 243.0, 826.0, 198.0, 17.0),
    ("28-AUG-2016",            448.0, 826.0,  93.0, 17.0),
    ("18-JUL-2021",            558.0, 826.0,  83.0, 17.0),
    ("25-AUG-2021",            659.0, 826.0,  90.0, 17.0),
)

#: The liver-function results table from the SAME page. Its Result column is a
#: taller vertical run of numbers than the chart's axis, which is why "pick the
#: longest column" was the wrong rule.
TABLE = (
    ("BILIRUBIN,TOTAL", 900.0, 1200.0, 180.0, 16.0), ("0.70", 1300.0, 1200.0, 40.0, 16.0),
    ("BILIRUBIN,DIRECT", 900.0, 1230.0, 180.0, 16.0), ("0.35", 1300.0, 1230.0, 40.0, 16.0),
    ("TOTALPROTEIN", 900.0, 1260.0, 180.0, 16.0), ("7.5", 1300.0, 1260.0, 40.0, 16.0),
    ("ALBUMIN", 900.0, 1290.0, 180.0, 16.0), ("4.8", 1300.0, 1290.0, 40.0, 16.0),
    ("GLOBULIN", 900.0, 1320.0, 180.0, 16.0), ("2.7", 1300.0, 1320.0, 40.0, 16.0),
    ("AST", 900.0, 1350.0, 180.0, 16.0), ("20", 1300.0, 1350.0, 40.0, 16.0),
    ("ALT", 900.0, 1380.0, 180.0, 16.0), ("14", 1300.0, 1380.0, 40.0, 16.0),
    ("ALP", 900.0, 1410.0, 180.0, 16.0), ("61", 1300.0, 1410.0, 40.0, 16.0),
)


class TestTheRealChart:
    def test_the_y_axis_is_the_scale_not_the_readings(self):
        region = detect_chart(*ocr(*CHART))
        assert region.y_axis_scale == ["7.52", "5.64", "3.76", "1.88"]

    def test_every_x_axis_date_is_recovered_including_the_fused_pair(self):
        # OCR returns the first two dates run together as one token. Left
        # fused it matches no date pattern, so the axis read three labels long
        # instead of five and the chart failed detection outright.
        region = detect_chart(*ocr(*CHART))
        assert region.x_axis_labels == [
            "01-MAR-2015", "25-MAR-2016", "28-AUG-2016", "18-JUL-2021", "25-AUG-2021",
        ]

    def test_the_band_labels_come_through(self):
        region = detect_chart(*ocr(*CHART))
        assert region.zone_labels == ["Diabetics", "Prediabetics", "Nondiabetic"]

    def test_plotted_values_are_the_points_not_the_ticks(self):
        region = detect_chart(*ocr(*CHART))
        assert region.plotted_values == ["5.6", "4.8", "5.0"]
        assert "7.52" not in region.plotted_values


class TestItPicksTheChartNotTheTable:
    """The bug this class exists for: a results table looks like an axis."""

    def test_a_taller_table_column_does_not_win(self):
        region = detect_chart(*ocr(*CHART, *TABLE))
        assert region.y_axis_scale == ["7.52", "5.64", "3.76", "1.88"]
        assert "0.70" not in region.y_axis_scale

    def test_a_page_of_pure_table_is_not_a_chart(self):
        # No x-axis under it, so there is no corner and nothing to annotate.
        assert not detect_chart(*ocr(*TABLE)).is_chart()

    def test_an_empty_page_is_not_a_chart(self):
        assert not detect_chart([], []).is_chart()

    def test_mismatched_lines_and_boxes_do_not_raise(self):
        assert not detect_chart(["a", "b"], []).is_chart()


class TestTheSeries:
    """The reading a chart exists to communicate: a value AND its date.

    The gold record for this page stores the five plotted points as five
    ordinary test rows, each carrying a sample_date — not as a paragraph
    describing a picture. Refusing to pair them, as this module first did,
    threw away three rows it could have had.
    """

    def test_each_readable_point_is_paired_with_its_own_date(self):
        # Verified against the gold record: 5.6 belongs to 01-MAR-2015, 4.8 to
        # 18-JUL-2021, 5.0 to 25-AUG-2021.
        assert detect_chart(*ocr(*CHART)).series == [
            ("5.6", "01-MAR-2015"),
            ("4.8", "18-JUL-2021"),
            ("5.0", "25-AUG-2021"),
        ]

    def test_the_first_point_pairs_with_the_first_of_two_fused_dates(self):
        # "01-MAR-201525-MAR-2016" is one OCR token holding two labels. Split
        # onto a single x they would both sit half a slot from the truth and
        # 5.6 could pair with either; spread across the box, it pairs right.
        series = dict(detect_chart(*ocr(*CHART)).series)
        assert series["5.6"] == "01-MAR-2015"

    def test_the_chart_is_named_so_the_readings_have_an_analyte(self):
        assert detect_chart(*ocr(*CHART)).title == "GLYCUSYLAIEU HEMUGLOBIN(HBAIC)"

    def test_a_band_label_is_not_mistaken_for_the_title(self):
        # "Diabetics" sits ABOVE the highest tick OCR read, so searching up
        # from the tick found the band label instead of the heading.
        assert detect_chart(*ocr(*CHART)).title != "Diabetics"

    def test_the_axis_unit_is_carried(self):
        assert detect_chart(*ocr(*CHART)).unit == "%"

    def test_dates_with_no_readable_value_are_counted_not_invented(self):
        # OCR loses 5.4 and 5.5 at every scale. Two labels are left bare, and
        # that is reported rather than filled in with a guessed reading.
        assert detect_chart(*ocr(*CHART)).unread_points == 2


class TestAnnotation:
    def test_it_says_the_axis_is_a_scale(self):
        out = annotate("page text", detect_chart(*ocr(*CHART)))
        assert "page text" in out          # never replaces what OCR read
        assert "SCALE of the picture, NOT readings" in out
        assert "7.52" in out

    def test_each_reading_gets_its_own_line_to_become_a_row(self):
        out = annotate("", detect_chart(*ocr(*CHART)))
        assert "GLYCUSYLAIEU HEMUGLOBIN(HBAIC) | 5.6 % | 01-MAR-2015" in out
        assert "GLYCUSYLAIEU HEMUGLOBIN(HBAIC) | 4.8 % | 18-JUL-2021" in out
        assert "GLYCUSYLAIEU HEMUGLOBIN(HBAIC) | 5.0 % | 25-AUG-2021" in out

    def test_it_says_how_many_readings_are_missing(self):
        # Without this the list of three reads as the whole chart.
        out = annotate("", detect_chart(*ocr(*CHART)))
        assert "2 further x-axis label(s)" in out

    def test_a_page_with_no_chart_is_returned_untouched(self):
        assert annotate("page text", detect_chart(*ocr(*TABLE))) == "page text"


#: The SAME chart as read from a black-and-white threshold of the page. The
#: text detector is defeated by the coloured bands, so this rendering recovers
#: 5.4 and 5.5 and both top/bottom ticks — but mangles the heading and loses
#: the "%" beside the axis. Coordinates and tokens are the real ones.
CHART_BINARISED = (
    ("GLYLUSYLAIEUHEMUGLUBINI(HBAIL)", 308.0, 513.0, 250.0, 20.0),
    ("9.4",                    148.0, 560.0,  33.0, 16.0),
    ("Diabetics",              184.0, 600.0,  71.0, 23.0),
    ("7.52",                   144.0, 612.0,  40.0, 16.0),
    ("5.6",                    277.0, 645.0,  26.0, 16.0),
    ("Prediabetics",           187.0, 653.0,  89.0, 17.0),
    ("5.4",                    378.0, 650.0,  26.0, 16.0),
    ("5.64",                   144.0, 662.0,  44.0, 16.0),
    ("5.5",                    481.0, 648.0,  26.0, 16.0),
    ("4.8",                    583.0, 664.0,  26.0, 16.0),
    ("5.0",                    670.0, 678.0,  26.0, 16.0),
    ("3.76",                   143.0, 710.0,  40.0, 16.0),
    ("Nondiabetic",            187.0, 733.0,  86.0, 20.0),
    ("1.88",                   145.0, 760.0,  44.0, 16.0),
    ("0",                      152.0, 800.0,  16.0, 16.0),
    ("01-MAR-2015",            243.0, 826.0,  99.0, 17.0),
    ("25-MAR-2016",            342.0, 826.0,  99.0, 17.0),
    ("28-AUG-2016",            448.0, 826.0,  93.0, 17.0),
    ("18-JUL-2021",            558.0, 826.0,  83.0, 17.0),
    ("25-AUG-2021",            659.0, 826.0,  90.0, 17.0),
)


class TestTwoRenderingsOfOnePage:
    """Neither rendering reads the whole chart; together they do.

    Thresholding the page to black and white beats the coloured bands and
    recovers every plotted point — and costs the faint and the small, mangling
    the heading and erasing the "%". So the readings come from whichever pass
    found more, and the labelling from the pass that reads text well.
    """

    def test_every_reading_is_recovered_from_the_two_passes(self):
        from app.graph.charts import best_chart

        region = best_chart(ocr(*CHART), ocr(*CHART_BINARISED))
        assert region.series == [
            ("5.6", "01-MAR-2015"),
            ("5.4", "25-MAR-2016"),
            ("5.5", "28-AUG-2016"),
            ("4.8", "18-JUL-2021"),
            ("5.0", "25-AUG-2021"),
        ]

    def test_nothing_is_left_unread_once_both_passes_are_used(self):
        from app.graph.charts import best_chart

        assert best_chart(ocr(*CHART), ocr(*CHART_BINARISED)).unread_points == 0

    def test_the_heading_comes_from_the_pass_that_reads_text(self):
        from app.graph.charts import best_chart

        # The thresholded pass says "GLYLUSYLAIEUHEMUGLUBINI(HBAIL)".
        region = best_chart(ocr(*CHART), ocr(*CHART_BINARISED))
        assert region.title == "GLYCUSYLAIEU HEMUGLOBIN(HBAIC)"

    def test_the_unit_survives_even_though_thresholding_erased_it(self):
        from app.graph.charts import best_chart

        assert best_chart(ocr(*CHART), ocr(*CHART_BINARISED)).unit == "%"

    def test_one_rendering_alone_still_works(self):
        from app.graph.charts import best_chart

        assert len(best_chart(ocr(*CHART)).series) == 3

    def test_no_rendering_finds_a_chart(self):
        from app.graph.charts import best_chart

        assert not best_chart(ocr(*TABLE)).is_chart()
