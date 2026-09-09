"""Read the structure of a chart out of OCR token geometry.

Why this exists
---------------
A trend chart on a scanned report is a picture. OCR flattens it to a bag of
loose tokens in reading order, and everything that made it a chart — which
numbers are the scale, which are plotted, which labels sit under the x-axis —
is gone by the time an extractor sees the text.

MEASURED on `001_Lab-report.png`, an HbA1c trend chart. OCR produced, in
reading order:

    Diabetics 7.52 Prediabetics 5.6 5.64- 4.8 5.0 3.76 Nondiabetic % 1.88-
    01-MAR-201525-MAR-2016 28-AUG-2016 18-JUL-2021 25-AUG-2021 Date-

The extractor read the chart's title next to the first number it found and
emitted a test row: `GLYCOSYLATED HEMOGLOBIN (HBA1C) = 7.52 %`. But 7.52 is the
SECOND TICK ON THE Y-AXIS. The patient's actual readings are 5.6, 5.4, 5.5,
4.8 and 5.0. A wrong measurement presented as a real one is worse than a gap:
a gap is visible, and this was not.

What it does
------------
The tokens still carry their positions, and a chart's parts are defined by
position: a scale is a vertical run of numbers sharing an x, an axis is a
horizontal run sharing a y. That is recoverable, and it is enough to say "these
numbers are the scale" — which is what stops a tick being read as a result.

What it deliberately does NOT do
--------------------------------
It does not pair plotted values with their dates. On this same chart RapidOCR
finds only three of the five plotted values (5.4 and 5.5 are lost at 1x, 2x and
3x) and fuses two x-axis dates into one token
(`01-MAR-201525-MAR-2016`). Pairing what survives would produce a series that
is not merely incomplete but MISALIGNED — 5.6 attributed to the wrong date —
and a confidently wrong reading is the failure this module exists to prevent.
Reading a full series off a scan of this quality needs a vision model, not
geometry over lossy tokens.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: A tick is a bare number: "7.52", "5.64-", "0", "100". The trailing dash is
#: the tick mark itself, which OCR frequently glues to the label.
_NUMERIC = re.compile(r"^[-+]?\d+(?:[.,]\d+)?\s*[-–—]?$")

#: A date in any of the forms this corpus prints on an x-axis.
_DATEISH = re.compile(
    r"^\d{1,2}[-/][A-Za-z]{3}[-/]\d{2,4}$"      # 01-MAR-2015
    r"|^[A-Za-z]{3}\s*\d{1,2}\s*\d{4}$"          # Aug 15 2012
    r"|^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$"        # 12/11/12
)

#: Two tokens are in the same column (or row) when their centres are within
#: this many pixels. Generous because OCR box edges wobble by a few pixels on
#: a 200dpi scan, and too tight a value splits one axis into two runs of two.
_ALIGN_TOLERANCE = 14.0

#: Below this a "run" is a coincidence. Three ticks is the smallest thing that
#: reads as a scale rather than as two numbers that happen to line up.
_MIN_RUN = 3

#: How far below the scale the x-axis may sit, beyond the scale's own height,
#: and still be the same picture. Without it a column near the top of a page
#: pairs with an unrelated row of dates near the bottom.
_CORNER_SLACK = 60.0

#: How far above the topmost tick to look for the chart's heading.
_TITLE_BAND = 90.0

#: OCR runs adjacent x-axis labels together when the plot is narrow:
#: "01-MAR-201525-MAR-2016" is two dates, not one. Split them back apart so
#: the axis is not silently one label short.
_FUSED_DATE = re.compile(r"\d{1,2}[-/][A-Za-z]{3}[-/]\d{2,4}")


@dataclass
class ChartRegion:
    """One chart's recoverable structure."""

    y_axis_scale: list[str] = field(default_factory=list)
    x_axis_labels: list[str] = field(default_factory=list)
    #: Numbers inside the plot area that are not on either axis. Reported for
    #: context only — see the module docstring on why they are not paired.
    plotted_values: list[str] = field(default_factory=list)
    #: Text labels inside the plot area: "Diabetics", "Nondiabetic".
    zone_labels: list[str] = field(default_factory=list)
    #: (value, x-axis label) for each plotted point that could be matched to a
    #: label. This is the reading the chart exists to communicate — the gold
    #: records store these as ordinary test rows carrying a sample_date.
    series: list[tuple[str, str]] = field(default_factory=list)
    #: The chart's own heading, e.g. "GLYCOSYLATED HEMOGLOBIN (HBA1C)".
    title: str = ""
    #: The y-axis unit, where the axis is labelled with one ("%").
    unit: str = ""
    #: x-axis labels left with no reading. On a trend chart each label carries
    #: a point, so a label with nothing paired to it means OCR could not read
    #: that point's value — stated as "labels without a value" rather than as a
    #: count of dots, because dots are not what was measured.
    unread_points: int = 0

    def is_chart(self) -> bool:
        """Both axes present. One alone is a table column or a row of dates."""
        return len(self.y_axis_scale) >= _MIN_RUN and len(self.x_axis_labels) >= _MIN_RUN


def _centre(box: Any) -> tuple[float, float]:
    return box.x + box.w / 2.0, box.y + box.h / 2.0


@dataclass(frozen=True)
class _Tok:
    index: int
    text: str
    cx: float
    cy: float
    width: float = 0.0


def _runs(items: list[_Tok], axis: int) -> list[list[_Tok]]:
    """Group tokens whose coordinate agrees within tolerance. axis 0=x, 1=y."""
    key = (lambda t: t.cx) if axis == 0 else (lambda t: t.cy)
    groups: list[list[_Tok]] = []
    for item in sorted(items, key=key):
        for g in groups:
            if abs(key(g[0]) - key(item)) <= _ALIGN_TOLERANCE:
                g.append(item)
                break
        else:
            groups.append([item])
    return groups


def _split_fused_dates(tokens: list[_Tok]) -> list[_Tok]:
    """Expand a token holding several run-together dates into one token each.

    MEASURED: RapidOCR returns "01-MAR-201525-MAR-2016" for two adjacent
    x-axis labels. Left fused it matches no date pattern at all, so the axis
    looked three labels long instead of five and the chart failed detection.
    The split copies share the original's position, which is close enough:
    they are used to establish that a row exists, not to place a point.
    """
    out: list[_Tok] = []
    for t in tokens:
        if _DATEISH.match(t.text):
            out.append(t)
            continue
        found = _FUSED_DATE.findall(t.text)
        if len(found) >= 2 and len("".join(found)) >= len(t.text) - 2:
            # Spread them across the original box rather than stacking them on
            # its centre. The x of a label is what a plotted point is matched
            # against, so two labels sharing one x would both sit half a slot
            # away from the truth and the nearer point could pair with either.
            span = t.width or 0.0
            left = t.cx - span / 2.0
            step = span / len(found) if span else 0.0
            out.extend(
                _Tok(t.index, d, left + step * (i + 0.5), t.cy)
                for i, d in enumerate(found)
            )
    return out


def detect_chart(lines: list[str], boxes: list[Any]) -> ChartRegion:
    """Recover a chart's axes from OCR tokens and their boxes.

    `lines` and `boxes` are what local_ocr.extract_english returns, positionally
    paired. A page with no chart comes back with `is_chart()` false and is not
    worth annotating.
    """
    tokens: list[_Tok] = []
    for i, (text, box) in enumerate(zip(lines, boxes)):
        text = (text or "").strip()
        if text and box is not None:
            cx, cy = _centre(box)
            tokens.append(_Tok(i, text, cx, cy, float(getattr(box, "w", 0.0) or 0.0)))
    if len(tokens) < _MIN_RUN * 2:
        return ChartRegion()

    numeric = [t for t in tokens if _NUMERIC.match(t.text)]
    dateish = [t for t in _split_fused_dates(tokens)]

    # A chart's axes form a CORNER: the scale runs down the left, the labels
    # run along the bottom, and the scale sits above and to the left of them.
    #
    # Picking the tallest column instead — the obvious heuristic, and the one
    # tried first — chose the RESULT COLUMN of the liver-function table on
    # 001_Lab-report.png, which has twelve figures to the chart axis's five.
    # Length says which run is longest, not which run is an axis. Only the
    # corner does that.
    columns = [g for g in _runs(numeric, axis=0) if len(g) >= _MIN_RUN]
    rows = [g for g in _runs(dateish, axis=1) if len(g) >= 2]
    if not rows:
        rows = [g for g in _runs(numeric, axis=1) if len(g) >= _MIN_RUN]

    best: tuple[int, list[_Tok], list[_Tok]] | None = None
    for col in columns:
        col_x = min(t.cx for t in col)
        col_bottom = max(t.cy for t in col)
        col_top = min(t.cy for t in col)
        for row in rows:
            row_y = min(t.cy for t in row)
            row_left = min(t.cx for t in row)
            # scale to the LEFT of the labels, and ABOVE them
            if col_x >= row_left:
                continue
            if not (col_bottom - _ALIGN_TOLERANCE <= row_y):
                continue
            # and the two must belong to the same picture rather than being a
            # column near the top of the page and a row near the bottom.
            if row_y - col_bottom > (col_bottom - col_top) + _CORNER_SLACK:
                continue
            score = len(col) + len(row)
            if best is None or score > best[0]:
                best = (score, col, row)

    if best is None:
        return ChartRegion()
    _, y_axis, x_axis = best

    region = ChartRegion(
        y_axis_scale=[t.text.rstrip(" -–—") for t in sorted(y_axis, key=lambda t: t.cy)],
        x_axis_labels=[t.text for t in sorted(x_axis, key=lambda t: t.cx)],
    )
    if not region.is_chart():
        return ChartRegion()

    # The plot area: right of the scale, above the x-axis, no higher than the
    # topmost tick. Anything outside that box belongs to the rest of the page.
    left = min(t.cx for t in y_axis)
    bottom = min(t.cy for t in x_axis)
    top = min(t.cy for t in y_axis)
    on_axis = {t.index for t in y_axis} | {t.index for t in x_axis}
    plotted: list[_Tok] = []

    for t in tokens:
        if t.index in on_axis:
            continue
        if t.cx <= left + _ALIGN_TOLERANCE:
            continue
        if not (top - _ALIGN_TOLERANCE <= t.cy <= bottom - _ALIGN_TOLERANCE):
            continue
        if _NUMERIC.match(t.text):
            region.plotted_values.append(t.text.rstrip(" -–—"))
            plotted.append(t)
        elif len(t.text) > 2 and any(c.isalpha() for c in t.text):
            region.zone_labels.append(t.text)

    region.series = _pair_points(plotted, x_axis)
    # The ceiling for the title search is the topmost thing INSIDE the plot,
    # not the topmost tick: a band label ("Diabetics") routinely sits above the
    # highest tick OCR managed to read, and searching from the tick picked the
    # band label up as the chart's name.
    inside = plotted + [t for t in tokens if t.text in set(region.zone_labels)]
    ceiling = min([t.cy for t in y_axis] + [t.cy for t in inside])
    region.title = _find_title(tokens, ceiling, y_axis, x_axis)
    if region.series:
        region.unread_points = max(0, len(region.x_axis_labels) - len(region.series))
    region.unit = _find_unit(tokens, y_axis)
    return region


def _pair_points(plotted: list[_Tok], x_axis: list[_Tok]) -> list[tuple[str, str]]:
    """Match each plotted value to the x-axis label it sits above.

    A point belongs to the label nearest it horizontally — that is what the
    picture means, and MEASURED on the HbA1c chart it is exact: three readable
    points matched their gold dates with at most 21px of error against a slot
    roughly 100px wide.

    The guard is half a slot. Beyond that the nearest label is not evidence of
    anything, and a reading attributed to the wrong date is worse than a
    reading left unreported — so it is dropped rather than guessed.
    """
    if not plotted or len(x_axis) < 2:
        return []

    labels = sorted(x_axis, key=lambda t: t.cx)
    gaps = [b.cx - a.cx for a, b in zip(labels, labels[1:]) if b.cx > a.cx]
    if not gaps:
        return []
    slot = sorted(gaps)[len(gaps) // 2]
    limit = slot / 2.0

    out: list[tuple[str, str]] = []
    for point in sorted(plotted, key=lambda t: t.cx):
        label, distance = min(
            ((lab, abs(point.cx - lab.cx)) for lab in labels), key=lambda pair: pair[1]
        )
        if distance <= limit:
            out.append((point.text.rstrip(" -–—"), label.text))
    return out


def _find_title(tokens: list[_Tok], ceiling: float, y_axis: list[_Tok], x_axis: list[_Tok]) -> str:
    """The words sitting just above the plot, across its width.

    A trend chart names what it plots, and without that name the readings are
    numbers with dates and no analyte — unusable in a results table.
    """
    left = min(t.cx for t in y_axis)
    right = max(t.cx for t in x_axis)
    band = [
        t for t in tokens
        if left <= t.cx <= right
        and ceiling - _TITLE_BAND <= t.cy < ceiling
        and any(c.isalpha() for c in t.text)
    ]
    if not band:
        return ""
    # Words on the same line, left to right.
    line_y = min(t.cy for t in band)
    line = [t for t in band if abs(t.cy - line_y) <= _ALIGN_TOLERANCE]
    return " ".join(t.text for t in sorted(line, key=lambda t: t.cx)).strip()


def _find_unit(tokens: list[_Tok], y_axis: list[_Tok]) -> str:
    """A short symbol beside the scale — "%", "mg/dL" — is the axis's unit."""
    left = min(t.cx for t in y_axis)
    top, bottom = min(t.cy for t in y_axis), max(t.cy for t in y_axis)
    for t in tokens:
        if t.cx < left and top <= t.cy <= bottom and 0 < len(t.text) <= 6:
            if not _NUMERIC.match(t.text):
                return t.text
    return ""


def best_chart(*token_sets: tuple[list[str], list[Any]]) -> ChartRegion:
    """Detect over several renderings of the same page and combine the reads.

    A text detector is defeated by the coloured bands behind a trend chart.
    MEASURED on `001_Lab-report.png`: reading the page as-is finds three of the
    five plotted values, while reading a black-and-white threshold of the SAME
    page finds all five, recovers the top and bottom axis ticks, and un-fuses
    the two x-axis dates that OCR otherwise runs together.

    But the threshold is not better at everything. It read the chart's heading
    as "GLYLUSYLAIEUHEMUGLUBINI(HBAIL)" where the natural page gave
    "GLYCUSYLAIEU HEMUGLOBIN(HBAIC)", and it lost the "%" beside the axis
    entirely — thresholding costs the faint and the small. So the readings come
    from whichever rendering recovered more of them, while the LABELLING comes
    from the first rendering wherever it has any, because that is the one that
    reads text well.
    """
    regions: list[ChartRegion] = []
    for lines, boxes in token_sets:
        if not lines or len(boxes) != len(lines):
            continue
        region = detect_chart(lines, boxes)
        if region.is_chart():
            regions.append(region)
    if not regions:
        return ChartRegion()

    # Most readings paired with a date, then the longest x-axis: a rendering
    # that recovered more points has read more of the picture.
    best = max(regions, key=lambda r: (len(r.series), len(r.x_axis_labels)))

    for other in regions:
        if not best.title and other.title:
            best.title = other.title
        if not best.unit and other.unit:
            best.unit = other.unit
    # The first rendering is the un-thresholded one; prefer its wording.
    first = regions[0]
    if first.title and " " in first.title and " " not in best.title:
        best.title = first.title
    if first.unit:
        best.unit = first.unit

    best.unread_points = max(0, len(best.x_axis_labels) - len(best.series))
    return best


def annotate(text: str, region: ChartRegion) -> str:
    """Append the chart, rewritten as a small table, to a page's OCR text.

    Appended rather than substituted: the loose tokens stay, because an
    evidence quote may legitimately point at one of them. What this adds is
    what reading order destroyed — which numbers are the scale, and which
    reading belongs to which date.

    The readings are laid out one per line, because that is what they are: the
    gold records store a trend chart's points as ordinary test rows carrying a
    sample_date, not as a paragraph about a picture.
    """
    if not region.is_chart():
        return text

    name = region.title or "chart"
    unit = f" {region.unit}" if region.unit else ""

    parts = [
        f"[chart] {name}",
        "The numbers on the axes below are the SCALE of the picture, NOT readings:",
        f"  y-axis scale{unit}: " + " ".join(region.y_axis_scale),
        "  x-axis labels: " + " ".join(region.x_axis_labels),
    ]
    if region.zone_labels:
        parts.append("  plot bands: " + " ".join(region.zone_labels))

    if region.series:
        parts.append(
            "Readings plotted on this chart — record EACH LINE as its own test "
            f"row, with the date as sample_date and {name!r} as the test name:"
        )
        for value, label in region.series:
            parts.append(f"  {name} | {value}{unit} | {label}")

    if region.unread_points:
        # Said out loud so a reviewer knows the series is short, rather than
        # believing the chart held only the points that happen to be listed.
        parts.append(
            f"  ({region.unread_points} further x-axis label(s) on this chart have "
            f"no readable value — the chart holds more readings than are listed above)"
        )

    return (text + "\n" + "\n".join(parts)).strip()
