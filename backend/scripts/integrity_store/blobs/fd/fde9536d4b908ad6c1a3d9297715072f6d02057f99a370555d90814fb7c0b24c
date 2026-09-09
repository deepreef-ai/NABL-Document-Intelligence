"""Reads the sectioned ground-truth records in labelled_dataset/<name>.json
and flattens them into the shape compare.py scores against.

The labelled dataset was re-cut into a sectioned layout (lab_info /
client_info / sample_info / report_info / patient_info / signatories /
notes / tests, plus a `reports` list for the 18 source PDFs that carry more
than one report), stored FLAT — one JSON per document, no per-document
subfolder. compare.py's contract is unchanged and deliberately not touched:
it scores a flat {"fields": {...}} dict plus a {"tests": [...]} list whose
rows carry test_name/result/unit/reference_range/method/sample_id, exactly
as documents/lab_report.py's SYSTEM_PROMPT asks the model to produce. This
module is the only thing that knows about the sectioned layout, so the
scorer and the extractor keep agreeing on one wire format.

Two mappings do real work here:

1. SECTIONS -> FLAT FIELDS. A section is only a grouping; the leaf name is
   the field name the extractor is asked for (`lab_info.lab_name` ->
   `lab_name`). Leaf collisions across sections are rare but real (MEASURED
   on the 53-document set: 5 distinct leaf names, each in at most 2
   documents, all inside narrative/notes sections — e.g. `note` under both
   `radiology_test` and `recommendations.*`). The FIRST occurrence in
   document order wins and the rest are reported via `dropped_duplicate_fields`
   rather than vanishing, because a silent drop would make gold look like it
   never asked for the field.

2. TEST-ROW COLUMNS -> THE CANONICAL SIX. The corpus prints the same three
   scored columns under many different headings, because the source
   documents do: a food-testing report's spec bound is "LIMITS*" or
   "REQUIREMENTS*" or "As per FSSR 2011 Specifications", a clinical
   report's is "Biological Reference Interval" or "Normal Range". They are
   the same column for scoring purposes, so they fold onto
   `reference_range`. Ditto `parameter`/`test_parameter`/`analyte` ->
   `test_name` and `result_of_analysis`/`observed_value` -> `result`.
   Columns with no counterpart in the extractor's schema (`s_no`, `loq`,
   `mou`, `instrument_used`, `discipline`, ...) are dropped: gold cannot
   score a value the extractor is never asked to produce.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Sections whose value is a list of result rows rather than scalar fields.
# Collected into `tests`; never flattened into `fields`.
_TEST_LIST_KEYS = ("tests", "medical_examination", "stone_types")

# Nested containers that hold their own `tests` lists (multi-report bundles,
# the Apollo panel layout). Walked for rows, not flattened into fields.
_NESTED_REPORT_KEYS = ("reports", "lab_panel_results")

# Not scoreable against the extractor's schema, and enormous: the 295-entry
# "list of screened molecules and not detected" in a516f67...  is a printed
# appendix of names the extractor is never asked to enumerate.
_SKIP_SECTIONS = frozenset({"screened_molecules_not_detected"})

# First match wins, so order is the precedence order. `panel_name` is LAST:
# it is a grouping heading, right only for a table that carries no per-row
# name of its own, and it must never win over a real test_name sitting beside
# it (every medical_examination row has both).
_TEST_NAME_KEYS = ("test_name", "parameter", "test_parameter", "analyte", "test",
                   "stone_type", "panel_name")
_RESULT_KEYS = ("result", "result_of_analysis", "results_of_analysis", "test_result",
                "observed_value", "a1_a2_typing", "saturation_ratio")
_UNIT_KEYS = ("unit", "unit_of_measurement", "unit_of_measurements", "units")
_REFERENCE_KEYS = ("reference_range", "bio_ref_interval", "reference_interval", "normal_range",
                   "limit", "requirement", "specification", "fssai_limit",
                   "as_per_fssr_2011_specifications", "as_per_fssr_2011_rules")
_METHOD_KEYS = ("method", "test_method", "method_of_testing", "test_protocol")
# The extractor is told to put a trend table's column heading in sample_id
# ("Put the column's date/sample heading in 'sample_id'"), which is what
# result_datetime/sample_date hold on this side.
_SAMPLE_ID_KEYS = ("sample_id", "sample_date", "result_datetime")


def _first(row: dict, keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def _row_result(row: dict) -> Any:
    """AH3500627/994-Hematuria print results in an Abnormal | Normal pair of
    columns — the value sits in whichever one applies, so fold them into the
    single `result` the extractor produces."""
    direct = _first(row, _RESULT_KEYS)
    if direct is not None:
        return direct
    for k in ("abnormal", "normal"):
        if row.get(k) not in (None, ""):
            return row[k]
    return None


def normalize_test_row(row: dict) -> dict | None:
    """One gold result row -> the canonical scored shape. None when the row
    carries no test name at all (a bare sub-heading row such as
    high-protein-paneer's "Banned Pesticides"), which compare.py's
    _test_row_key would reject anyway."""
    name = _first(row, _TEST_NAME_KEYS)
    if name in (None, ""):
        return None
    result = _row_result(row)
    # A printed out-of-range flag is part of the result cell per
    # SYSTEM_PROMPT ("0.35 High"); gold splits it into its own column, so
    # put it back for comparison. compare.py's _split_result_flag handles
    # either form, but re-joining keeps the raw values comparable too.
    flag = row.get("flag")
    if flag and result not in (None, ""):
        result = f"{result} {flag}"
    return {
        "test_name": name,
        "result": result,
        "unit": _first(row, _UNIT_KEYS),
        "reference_range": _first(row, _REFERENCE_KEYS),
        "method": _first(row, _METHOD_KEYS),
        "sample_id": _first(row, _SAMPLE_ID_KEYS),
    }


def _is_si_row(row: dict) -> bool:
    """SYSTEM_PROMPT: "If a report prints BOTH a conventional and an S.I.
    column for the same analyte, use the conventional (first) one." Gold
    records both (003_Lab-report prints both), so the S.I. half must be
    dropped here or every one of those rows scores as a guaranteed miss
    against an extractor that was told not to produce it."""
    return str(row.get("unit_system") or "").strip().upper() in {"S.I.", "SI"}


def _collect_rows(node: Any, out: list[dict]) -> None:
    """Depth-first walk gathering every result row, wherever it is nested —
    top-level `tests`, each entry of `reports`, each panel of
    `lab_panel_results`."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _SKIP_SECTIONS:
                continue
            if key in _TEST_LIST_KEYS and isinstance(value, list):
                out.extend(r for r in value if isinstance(r, dict))
            elif key in _NESTED_REPORT_KEYS and isinstance(value, list):
                for entry in value:
                    _collect_rows(entry, out)
            elif isinstance(value, (dict, list)):
                _collect_rows(value, out)
    elif isinstance(node, list):
        for item in node:
            _collect_rows(item, out)


def _flatten_fields(doc: dict) -> tuple[dict, list[str]]:
    """Scalar leaves of every non-result section, keyed by leaf name.

    Lists of scalars (an `impressions` array, a checkbox `options` list) are
    joined with "; " — the extractor emits one string per field, so a list
    has no comparable representation otherwise.
    """
    fields: dict[str, Any] = {}
    dropped: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in _SKIP_SECTIONS or key in _TEST_LIST_KEYS:
                    continue
                # A `reports` entry carries its OWN report_info/client_info/
                # sample_info, and those are real fields — the extractor
                # produces one flat dict per document, so they flatten into
                # the same namespace (first wins). Skipping them entirely
                # cost hp-lab-report all but one of its fields.
                if key in _NESTED_REPORT_KEYS:
                    walk(value, path)
                    continue
                walk(value, f"{path}.{key}" if path else key)
            return
        if isinstance(node, list):
            if node and all(not isinstance(x, (dict, list)) for x in node):
                _put(path, "; ".join("" if x is None else str(x) for x in node))
            else:
                for item in node:
                    walk(item, path)
            return
        _put(path, node)

    def _put(path: str, value: Any) -> None:
        leaf = path.rsplit(".", 1)[-1]
        if leaf in fields:
            # First occurrence wins; see this module's docstring.
            dropped.append(path)
            return
        fields[leaf] = value

    for section, value in doc.items():
        if section in _SKIP_SECTIONS or section in _TEST_LIST_KEYS:
            continue
        if section in _NESTED_REPORT_KEYS:
            walk(value, "")
            continue
        walk(value, section)
    # `original_filename` lives in document_info and is dataset bookkeeping,
    # not a field the extractor is asked to read off the page.
    fields.pop("original_filename", None)
    return fields, dropped


def flatten_fields(doc: dict) -> dict:
    """Every section's scalar leaves as one {name: value} map — the shape both
    benchmark drivers compare against. Public because
    scripts/benchmark_api.py scores the RUNNING API and must read gold
    through exactly this reader, not a second private copy of it."""
    return _flatten_fields(doc)[0]


def flatten_tests(doc: dict) -> list[dict]:
    """Every result row in the document, wherever nested, folded onto the
    canonical scored columns. See flatten_fields for why this is public."""
    rows: list[dict] = []
    _collect_rows(doc, rows)
    out: list[dict] = []
    for raw in rows:
        if _is_si_row(raw):
            continue
        row = normalize_test_row(raw)
        if row is not None:
            out.append(row)
    return out


def load_gold(path: str | Path) -> dict:
    """Returns {"original_filename", "fields", "tests"} plus the two
    bookkeeping keys the CLI reports (`dropped_duplicate_fields`,
    `unscoreable_duplicate_rows`) — compare.py ignores extra keys."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    fields, dropped = _flatten_fields(doc)

    raw_rows: list[dict] = []
    _collect_rows(doc, raw_rows)

    tests: list[dict] = []
    seen: set[tuple[str, str]] = set()
    duplicates = 0
    for raw in raw_rows:
        if _is_si_row(raw):
            continue
        row = normalize_test_row(raw)
        if row is None:
            continue
        # compare.py keys rows on (test_name, sample_id). Where gold
        # distinguishes two rows by a column the extractor has no slot for
        # (hp-lab-report prints "Crude protein" twice, under AOAC 991.20 and
        # AOAC 920.87), both sides collapse onto one key and the second is
        # unscoreable either way. Counted and reported, never silently lost.
        key = (str(row["test_name"]).strip().lower(), str(row["sample_id"] or "").strip().lower())
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        tests.append(row)

    return {
        "original_filename": (doc.get("document_info") or {}).get("original_filename"),
        "fields": fields,
        "tests": tests,
        "dropped_duplicate_fields": dropped,
        "unscoreable_duplicate_rows": duplicates,
    }


def iter_gold(labelled_dir: str | Path):
    """Every gold record in labelled_dir, flat layout, sorted by stem.

    Flat only: the dataset was re-cut from labelled_dataset/<name>/<name>.json
    to labelled_dataset/<name>.json. A leftover per-document subfolder is
    ignored rather than double-counted.
    """
    for p in sorted(Path(labelled_dir).glob("*.json")):
        yield p.stem, load_gold(p)
