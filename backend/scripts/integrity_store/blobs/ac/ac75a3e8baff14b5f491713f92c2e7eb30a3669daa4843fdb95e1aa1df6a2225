#!/usr/bin/env python3
"""Scores the RUNNING system (http://localhost:8000) against the golden
labelled dataset, and reports accuracy / precision / recall / F1.

Why over HTTP rather than importing the extractor
-------------------------------------------------
scripts/generate_predictions_and_score.py calls the extraction functions
in-process. That has repeatedly drifted from what the application actually
does — MEASURED: the app path once scored 52.0% while the in-process script
reported 65.1%, because the two had silently grown different prompts. Driving
the real endpoints makes that class of divergence impossible: whatever the API
returns IS the product.

The cost is that this needs the server up, and every upload spends real LLM
calls (see app/documents/call_budget.py for the per-document ceiling).

Scoring
-------
A prediction counts as correct only when BOTH the field name and its value
match the golden record — a right value under the wrong key is not a hit.

  TP  a golden (key, value) pair the system produced
  FP  a pair the system produced that is not in the golden record
  FN  a golden pair the system did not produce

  precision = TP / (TP + FP)     of what it output, how much was right
  recall    = TP / (TP + FN)     of what it should have found, how much it found
  f1        = harmonic mean
  accuracy  = TP / (TP + FP + FN)

`accuracy` is over the UNION of golden and predicted pairs. Extraction has no
true-negative class — there is no enumerable set of "fields correctly not
extracted" — so the usual (TP+TN)/total is undefined here. Defining accuracy
as TP/|golden| instead would just restate recall under a second name, which
hides that the two are the same number.

Values are compared by READING, not by string. app/benchmark/compare.py's
normalisation runs first (case, all whitespace, separator punctuation, unicode
dashes, trailing punctuation, "N/A"/"-" as absent), then
app/benchmark/equivalence.py folds:

    numeric      3.5 == 3.50 == 3.5000,  5 == 5.0,  0.35 == .35
    range flags  "0.35 High" == "0.35"
    synonyms     Male == M,  Positive == +ve,  Absent == Not Detected
    dates        02 Nov 2020 == 2020-11-02 == 02/11/2020  (day-first)

Still NOT equal, because these differ in reading rather than spelling:
6.5 vs 65, 40-129 vs 40129, "0.35 High" vs "0.35 Low", 3.5% vs 3.5,
09:55:20 vs "03/02/2019 09:55:20", Absent vs Negative.

Field names are folded through compare.py's measured alias table by default;
pass --strict-keys to require them verbatim.

Test-table rows are matched by TEST NAME, never by position: the golden record
and the system order rows independently, so comparing tests[3] to tests[3]
would score a correctly-read table as entirely wrong.

Usage
-----
    # 1. start the API first
    uvicorn app.main:app --port 8000

    # 2. then, in another shell
    python scripts/benchmark_api.py
    python scripts/benchmark_api.py --limit 5          # smoke-test the harness
    python scripts/benchmark_api.py --strict-keys
    python scripts/benchmark_api.py --application-id <existing-unlocked-id>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/ on sys.path

import httpx  # noqa: E402

from app.benchmark.compare import (  # noqa: E402
    _canonicalize_keys,
    _normalize_field_value,
    _normalize_test_name,
    _normalize_value,
)
from app.benchmark.equivalence import values_equal  # noqa: E402
from app.benchmark.gold_loader import flatten_fields, flatten_tests  # noqa: E402

DEFAULT_ROOT = r"G:\Shared drives\Product & Engineering\Projects\NABL Document Intelligence"
DEFAULT_API = "http://localhost:8000"

# One results row's comparable columns. test_name is the row's identity, so it
# is matched on rather than scored; method/sample_id are transcription details
# the golden set records inconsistently, so they are not scored either.
_SCORED_TEST_COLUMNS = ("result", "unit", "reference_range")

_INDEXED = re.compile(r"^(\w+)\[(\d+)\]\.(.+)$")


# --------------------------------------------------------------------------- driving the API

def create_unlocked_application(client: httpx.Client, form_type: str) -> str:
    """One application, reused for every document.

    The upload endpoint rejects an application whose eligibility questions
    aren't answered (409). Those questions are an LLM conversation and are NOT
    what this benchmark measures, so the wizard is satisfied directly in the
    database rather than spending calls — and a stale answer cannot affect
    extraction, which never reads it.
    """
    response = client.post("/applications", json={"form_type": form_type})
    response.raise_for_status()
    application_id = response.json()["application"]["id"]

    from app.db import SessionLocal
    from app.models import Application
    from app.wizard.prerequisites import PREREQUISITES
    from app.schemas.forms import NablFormType

    db = SessionLocal()
    try:
        application = db.get(Application, application_id)
        application.prerequisite_answers = {
            q.id: {"question_id": q.id, "satisfied": True, "detail": "benchmark harness"}
            for q in PREREQUISITES[NablFormType(form_type)]
        }
        application.status = "unlocked"
        db.add(application)
        db.commit()
    finally:
        db.close()
    return application_id


_CONTENT_TYPES = {
    ".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg", ".tif": "image/tiff", ".tiff": "image/tiff",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def upload(client: httpx.Client, application_id: str, path: Path) -> dict:
    # Derived from the suffix, never hard-coded: documents/pipeline.py routes on
    # the content type, so sending a PNG as application/pdf makes PyMuPDF try to
    # parse it as a PDF and the document fails. 9 of the 53 sources are PNGs.
    content_type = _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
    with path.open("rb") as fh:
        response = client.post(
            f"/applications/{application_id}/documents",
            files={"file": (path.name, fh, content_type)},
            data={"script": "english"},
        )
    response.raise_for_status()
    return response.json()


def find_source(dataset_dir: Path, filename: str) -> Path | None:
    candidate = dataset_dir / filename
    if candidate.is_file():
        return candidate
    # dataset/ is organised into subfolders (food/, medical/, ...); the golden
    # record only stores the bare filename.
    matches = [p for p in dataset_dir.rglob(filename) if p.is_file()]
    return matches[0] if matches else None


# --------------------------------------------------------------------------- shaping the response

def split_response(document: dict) -> tuple[dict, list[dict]]:
    """Rebuild the API response into the golden record's shape: a {name: value}
    map plus a list of test rows.

    Test rows come from the response's own `tests` array, which is already the
    labelled dataset's shape — one dict per analyte with result, unit and
    reference_range kept as separate columns.

    The indexed-path fallback below is for the LEGACY extraction path, which
    had no `tests` array and flattened a results table into field paths like
    "tests[0].result". Reading only that shape is what made this harness score
    a document whose table WAS read correctly as 80 misses: the rows were in
    the response the whole time, in the field the harness never looked at.
    """
    flat: dict[str, str] = {}
    rows: dict[int, dict] = defaultdict(dict)

    for f in document.get("fields") or []:
        path, value = f.get("field_path") or "", f.get("value")
        if value is None:
            continue
        match = _INDEXED.match(path)
        if match and match.group(1) == "tests":
            rows[int(match.group(2))][match.group(3)] = value
        elif not match:
            flat[path] = value

    tests = [t for t in (document.get("tests") or []) if t]
    if not tests:
        tests = [rows[i] for i in sorted(rows)]
    return flat, tests


# --------------------------------------------------------------------------- reading the golden record

# ONE flattener, shared with the rest of the benchmark rather than re-implemented
# here. The private copy this script used to carry was one level deep and read
# only the top-level `tests` list, so it silently dropped real golden data:
# `reports` (a LIST — 18 of the 53 documents put their report/client/sample
# fields AND all their result rows inside it), sections two or three levels
# down, `lab_panel_results` (75 rows) and `stone_types`, and the food-testing
# spelling `unit_of_measurement` (singular — 267 rows). Gold the scorer cannot
# see turns every correct extraction of it into a false positive.
gold_fields = flatten_fields
gold_tests = flatten_tests


def gold_source_name(record: dict) -> str | None:
    return (record.get("document_info") or {}).get("original_filename") or record.get("original_filename")


# --------------------------------------------------------------------------- scoring

# A HIT is a golden value the system reproduced. Everything else carries a
# CATEGORY and a plain-English REASON, because "MISS" alone never says whether
# to fix the extractor or the label.
FAILURE_CATEGORIES = {
    "correct": "The system reproduced the golden value.",
    "not_extracted": "The value is in the golden record but the system returned nothing for it.",
    "wrong_key": "The system found the golden value but filed it under a different field name.",
    "wrong_value_truncated": "Right field, but the value is cut short or carries extra text.",
    "wrong_value_different": "Right field, but the system read a genuinely different value.",
    "unlabelled_extra": "The system returned a field the golden record has no entry for — often a "
                        "correct reading the label set does not cover. Excluded from gold-covered metrics.",
    "test_not_extracted": "The whole results row is in the golden record but was not returned.",
    "test_wrong_result": "Row matched, but its result value differs.",
    "test_wrong_unit": "Row matched, but its unit differs.",
    "test_wrong_reference_range": "Row matched, but its reference range differs.",
    "test_extra_row": "The system returned a results row with no counterpart in the golden record — "
                      "usually a duplicate under a variant name. Excluded from gold-covered metrics.",
}

# Categories that measure the LABEL SET's coverage rather than the extractor's
# accuracy. Counted in the raw metrics, excluded from the gold-covered ones.
_UNLABELLED = {"unlabelled_extra", "test_extra_row"}


def _row(document, kind, key, gold, predicted, category, detail=""):
    hit = category == "correct"
    return {
        "document": document, "kind": kind, "key": key,
        "gold": gold, "predicted": predicted,
        "verdict": "HIT" if hit else "MISS",
        "category": category,
        "reason": "" if hit else FAILURE_CATEGORIES.get(category, category) + (f" {detail}" if detail else ""),
        "in_gold_covered": category not in _UNLABELLED,
    }


def _difference_kind(gold, predicted) -> str:
    """Why two values at the SAME key disagree. Truncation is usually a
    labelling-convention clash; different text is a real misread."""
    g, p = str(gold), str(predicted)
    return "wrong_value_truncated" if (p in g or g in p) else "wrong_value_different"


def score_fields(document: str, gold: dict, predicted: dict, strict_keys: bool) -> list[dict]:
    if not strict_keys:
        gold, predicted = _canonicalize_keys(gold), _canonicalize_keys(predicted)

    gold_n = {k: v for k, v in ((k, _normalize_field_value(k, v)) for k, v in gold.items()) if v is not None}
    pred_n = {k: v for k, v in ((k, _normalize_field_value(k, v)) for k, v in predicted.items()) if v is not None}
    unmatched = {k: v for k, v in pred_n.items() if k not in gold_n}

    rows = []
    for key, want in gold_n.items():
        got = pred_n.get(key)
        # values_equal, not string equality: 3.5 == 3.50, Male == M,
        # 02 Nov 2020 == 2020-11-02, "0.35 High" == "0.35".
        if got is not None and values_equal(gold[key], predicted.get(key), key):
            rows.append(_row(document, "field", key, gold[key], predicted.get(key), "correct"))
        elif got is not None:
            rows.append(_row(document, "field", key, gold[key], predicted[key],
                             _difference_kind(gold[key], predicted[key])))
        else:
            # Is the golden value sitting under a different predicted key? Only
            # claimed when exactly one candidate carries it — a boilerplate
            # value repeats across unrelated fields, so one coincidence is not
            # evidence of shared identity.
            elsewhere = [k for k, v in unmatched.items() if v == want]
            if len(elsewhere) == 1:
                other = elsewhere.pop()
                del unmatched[other]
                rows.append(_row(document, "field", key, gold[key], predicted[other],
                                 "wrong_key", f"The system called it {other!r}."))
            else:
                rows.append(_row(document, "field", key, gold[key], None, "not_extracted"))

    for key in unmatched:
        rows.append(_row(document, "field", key, None, predicted[key], "unlabelled_extra"))
    return rows


def score_tests(document: str, gold: list[dict], predicted: list[dict], _strict: bool) -> list[dict]:
    """Matched by test name, never by position — the two sides order rows
    independently. Each scored column of a matched row is its own comparison,
    so a row with a right result and a wrong unit is one HIT and one MISS."""
    gold_by_name = {n: r for r in gold if (n := _normalize_test_name(r.get("test_name") or ""))}
    pred_by_name = {n: r for r in predicted if (n := _normalize_test_name(r.get("test_name") or ""))}

    rows = []
    for name, gold_row in gold_by_name.items():
        pred_row = pred_by_name.get(name)
        label = gold_row.get("test_name")
        if pred_row is None:
            # ONE comparison per value the row actually carried, not one for
            # the whole row. Collapsing it made dropping a row CHEAPER than
            # matching it and getting the columns wrong (1 FN vs 3), and it
            # shrank the denominator, so a run that lost rows scored better
            # recall for losing them. MEASURED on 002_Lab-report: the same
            # 8 fields + 14 rows scored 45 golden values in one run and 17 in
            # another purely because of how many rows the system matched.
            carried = [c for c in _SCORED_TEST_COLUMNS if _normalize_value(gold_row.get(c)) is not None]
            for column in carried or ["result"]:
                rows.append(_row(document, "test", f"tests.{label}.{column}",
                                 gold_row.get(column), None, "test_not_extracted"))
            continue
        for column in _SCORED_TEST_COLUMNS:
            want, got = _normalize_value(gold_row.get(column)), _normalize_value(pred_row.get(column))
            if want is None and got is None:
                continue
            key = f"tests.{label}.{column}"
            if values_equal(gold_row.get(column), pred_row.get(column)):
                rows.append(_row(document, "test", key, gold_row.get(column), pred_row.get(column), "correct"))
            elif want is None:
                rows.append(_row(document, "test", key, None, pred_row.get(column), "unlabelled_extra"))
            else:
                rows.append(_row(document, "test", key, gold_row.get(column), pred_row.get(column),
                                 f"test_wrong_{column}"))

    for name, pred_row in pred_by_name.items():
        if name not in gold_by_name:
            rows.append(_row(document, "test", f"tests.{pred_row.get('test_name')}",
                             None, pred_row.get("result"), "test_extra_row"))
    return rows


def tally(rows: list[dict], gold_covered: bool = False) -> dict:
    """TP/FP/FN and the four metrics over a set of comparison rows.

    gold_covered=True drops rows whose category measures the LABEL SET's
    coverage rather than the extractor: a field read correctly off the page
    that the golden record has no slot for is not an extraction error.
    """
    if gold_covered:
        rows = [r for r in rows if r["in_gold_covered"]]
    tp = sum(1 for r in rows if r["verdict"] == "HIT")
    # Set-based: a golden pair the system failed to produce is an FN, and a
    # pair it produced that is not in the golden record is an FP. A WRONG
    # VALUE at a known key is both — the golden pair is missing and a bogus
    # one was emitted. Counting it only as FN made gold-covered precision
    # structurally 1.0 (the only FP-able rows are the ones gold-covered mode
    # drops), which collapsed accuracy, recall and F1 into one number.
    fn = sum(1 for r in rows if r["verdict"] == "MISS" and r["gold"] is not None)
    fp = sum(1 for r in rows if r["verdict"] == "MISS"
             and (r["gold"] is None or r["predicted"] is not None))

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else (
        0.0 if precision is not None and recall is not None else None)
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) else None

    def r4(v):
        return round(v, 4) if v is not None else None

    return {"accuracy": r4(accuracy), "precision": r4(precision), "recall": r4(recall),
            "f1": r4(f1), "tp": tp, "fp": fp, "fn": fn}


def per_field_accuracy(rows: list[dict], minimum: int = 3) -> list[dict]:
    """Accuracy for each golden field name, commonest first. Field-level
    numbers are what tell you WHICH field to work on; an overall figure never
    does."""
    stats: dict[str, list[int]] = {}
    for r in rows:
        if not r["in_gold_covered"] or r["gold"] is None:
            continue
        key = r["key"].split(".")[-1] if r["kind"] == "test" else r["key"]
        seen = stats.setdefault(key, [0, 0])
        seen[0] += 1
        seen[1] += r["verdict"] == "HIT"
    table = [{"field": k, "documents": n, "correct": c, "accuracy": round(c / n, 4)}
             for k, (n, c) in stats.items() if n >= minimum]
    return sorted(table, key=lambda r: (-r["documents"], r["field"]))


# --------------------------------------------------------------------------- excel

_FILL = {
    "HIT": "C6EFCE", "not_extracted": "FFEB9C", "wrong_key": "FFD9B3",
    "wrong_value_truncated": "FFE0E0", "wrong_value_different": "FFC7CE",
    "test_not_extracted": "FFEB9C", "test_wrong_result": "FFC7CE",
    "test_wrong_unit": "FFD7DC", "test_wrong_reference_range": "FFD7DC",
    "unlabelled_extra": "E4C7FF", "test_extra_row": "E4C7FF",
}


def write_excel(path: Path, summary: dict, per_document: list[dict], rows: list[dict]) -> Path:
    """One reviewable workbook: the numbers, a row per document, per-field
    accuracy, every comparison with HIT/MISS + category + reason, and a
    failure breakdown."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()

    def sheet(ws, table, colour=False):
        if not table:
            ws.append(["(no rows)"])
            return
        columns = list(table[0].keys())
        ws.append([c.replace("_", " ").title() for c in columns])
        for cell in ws[1]:
            cell.font = Font(bold=True)
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(columns))}1"
        for record in table:
            ws.append([record.get(c) for c in columns])
            if colour:
                key = "HIT" if record.get("verdict") == "HIT" else record.get("category")
                fill = _FILL.get(key)
                if fill:
                    for i in range(1, len(columns) + 1):
                        ws.cell(ws.max_row, i).fill = PatternFill("solid", fgColor=fill)
        for i, column in enumerate(columns, start=1):
            width = max(len(str(column)), *(len(str(r.get(column, ""))) for r in table[:300]))
            ws.column_dimensions[get_column_letter(i)].width = min(max(width + 2, 10), 60)

    ws = wb.active
    ws.title = "Summary"
    ws.append(["Metric", "Gold-covered", "Raw (all output)"])
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for label, name in (("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"),
                        ("F1", "f1"), ("True positives", "tp"), ("False positives", "fp"),
                        ("False negatives", "fn")):
        ws.append([label, summary["gold_covered"][name], summary["raw"][name]])
    ws.append([])
    for label, name in (("Documents scored", "documents_scored"), ("Documents failed", "documents_failed"),
                        ("Table row recall", "table_row_recall"), ("Hallucination rate", "hallucination_rate")):
        ws.append([label, summary.get(name)])
    ws.append([])
    ws.append(["Gold-covered excludes fields and rows the golden record has no entry for "
               "(unlabelled_extra / test_extra_row) — often correct readings the label set does "
               "not cover. Raw counts everything the system output."])
    ws.cell(ws.max_row, 1).alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells(start_row=ws.max_row, start_column=1, end_row=ws.max_row, end_column=3)
    ws.row_dimensions[ws.max_row].height = 46
    for col, width in (("A", 24), ("B", 16), ("C", 18)):
        ws.column_dimensions[col].width = width

    sheet(wb.create_sheet("Per document"), per_document)
    sheet(wb.create_sheet("Per field"), per_field_accuracy(rows))
    sheet(wb.create_sheet("Comparisons"), rows, colour=True)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    total = sum(counts.values()) or 1
    sheet(wb.create_sheet("Failure summary"), [
        {"category": c, "count": n, "share": round(n / total, 4),
         "counts_toward_gold_covered": c not in _UNLABELLED,
         "what_it_means": FAILURE_CATEGORIES.get(c, "")}
        for c, n in sorted(counts.items(), key=lambda kv: -kv[1])
    ])

    wb.save(path)
    return path


# --------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default=DEFAULT_API, help=f"Base URL of the running system (default: {DEFAULT_API}).")
    parser.add_argument("--labelled", default=os.path.join(DEFAULT_ROOT, "labelled_dataset"),
                        help="Golden dataset directory.")
    parser.add_argument("--dataset", default=os.path.join(DEFAULT_ROOT, "dataset"), help="Raw source files.")
    parser.add_argument("--output", default=None, help="Where to write results (default: ./benchmark_api_results).")
    parser.add_argument("--form-type", default="NABL_151")
    parser.add_argument("--application-id", default=None,
                        help="Reuse an existing UNLOCKED application instead of creating one.")
    parser.add_argument("--limit", type=int, default=None, help="Only score the first N documents.")
    parser.add_argument("--strict-keys", action="store_true",
                        help="Require field names verbatim, without compare.py's measured alias folding.")
    parser.add_argument("--timeout", type=float, default=600.0, help="Per-upload timeout in seconds (default: 600).")
    args = parser.parse_args()

    labelled_dir, dataset_dir = Path(args.labelled), Path(args.dataset)
    if not labelled_dir.is_dir():
        print(f"Golden dataset not found: {labelled_dir}", file=sys.stderr)
        return 1
    output_dir = Path(args.output) if args.output else Path("benchmark_api_results")

    client = httpx.Client(base_url=args.api, timeout=args.timeout)
    try:
        health = client.get("/health")
        health.raise_for_status()
    except Exception as exc:  # noqa: BLE001 — a clear message beats a traceback
        print(f"Cannot reach the API at {args.api}: {exc}\nStart it with:  uvicorn app.main:app --port 8000",
              file=sys.stderr)
        return 1

    application_id = args.application_id or create_unlocked_application(client, args.form_type)
    print(f"API          : {args.api}")
    print(f"Golden set   : {labelled_dir}")
    print(f"Application  : {application_id}")
    print(f"Key matching : {'strict (verbatim)' if args.strict_keys else 'alias-folded (compare.py)'}\n")

    # Both layouts: labelled_dataset/001.json (current) and the older
    # labelled_dataset/001/001.json.
    label_paths = sorted(labelled_dir.glob("*.json")) or sorted(labelled_dir.glob("*/*.json"))
    label_paths = label_paths[: args.limit]
    per_document, all_rows, failures = [], [], []

    for i, label_path in enumerate(label_paths, 1):
        gold = json.loads(label_path.read_text(encoding="utf-8"))
        stem = label_path.stem
        source_name = gold_source_name(gold)
        source = find_source(dataset_dir, source_name) if source_name else None
        print(f"[{i}/{len(label_paths)}] {stem}", end=" ", flush=True)

        if source is None:
            print("SKIPPED (source file not found)")
            failures.append((stem, "source file not found"))
            continue

        started = time.perf_counter()
        try:
            document = upload(client, application_id, source)
        except Exception as exc:  # noqa: BLE001 — one bad document must not stop the run
            print(f"FAILED ({type(exc).__name__})")
            failures.append((stem, f"{type(exc).__name__}: {exc}"))
            continue

        pred_fields, pred_tests = split_response(document)
        rows = (score_fields(stem, gold_fields(gold), pred_fields, args.strict_keys)
                + score_tests(stem, gold_tests(gold), pred_tests, args.strict_keys))
        all_rows.extend(rows)

        covered, raw = tally(rows, gold_covered=True), tally(rows)
        per_document.append({
            "document": stem, "original_filename": source_name,
            "source_kind": "scanned/image" if Path(source).suffix.lower() != ".pdf" else "pdf",
            "doc_type": document.get("doc_type"), "status": document.get("status"),
            "seconds": round(time.perf_counter() - started, 1),
            "accuracy": covered["accuracy"], "precision": covered["precision"],
            "recall": covered["recall"], "f1": covered["f1"],
            "tp": covered["tp"], "fp": covered["fp"], "fn": covered["fn"],
            "raw_accuracy": raw["accuracy"], "raw_precision": raw["precision"],
            "unlabelled_extras": sum(1 for r in rows if not r["in_gold_covered"]),
            "error": document.get("error"),
        })
        print(f"acc={covered['accuracy']} P={covered['precision']} R={covered['recall']} "
              f"F1={covered['f1']}  ({covered['tp']}/{covered['tp'] + covered['fn']} golden)")

    covered, raw = tally(all_rows, gold_covered=True), tally(all_rows)

    # Table-row recall and hallucination rate are reported separately because
    # neither is visible in a micro-averaged total: a run can look healthy
    # while losing whole tables, or while inventing rows that never existed.
    gold_rows = sum(1 for r in all_rows if r["kind"] == "test" and r["category"] == "test_not_extracted")
    matched_rows = len({(r["document"], r["key"].rsplit(".", 1)[0])
                        for r in all_rows if r["kind"] == "test" and r["in_gold_covered"] and r["gold"] is not None
                        and r["category"] != "test_not_extracted"})
    invented = sum(1 for r in all_rows if r["category"] == "test_extra_row")
    total_predicted = sum(1 for r in all_rows if r["predicted"] is not None)

    summary = {
        "api": args.api,
        "documents_scored": len(per_document),
        "documents_failed": len(failures),
        "strict_keys": args.strict_keys,
        "table_row_recall": round(matched_rows / (matched_rows + gold_rows), 4) if (matched_rows + gold_rows) else None,
        "hallucination_rate": round(invented / total_predicted, 4) if total_predicted else None,
        "gold_covered": covered,
        "raw": raw,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for name, table in (("per_document.csv", per_document), ("comparisons.csv", all_rows),
                        ("per_field.csv", per_field_accuracy(all_rows))):
        if table:
            with (output_dir / name).open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
                writer.writeheader()
                writer.writerows(table)

    workbook = None
    try:
        workbook = write_excel(output_dir / "benchmark.xlsx", summary, per_document, all_rows)
    except PermissionError:
        # Reviewing the workbook in Excel holds a Windows write lock, and this
        # runs LAST — after every paid LLM call — so a lock must not throw away
        # a completed run's reporting.
        from datetime import datetime
        workbook = write_excel(output_dir / f"benchmark-{datetime.now():%Y%m%d-%H%M%S}.xlsx",
                               summary, per_document, all_rows)
        print("  NOTE: benchmark.xlsx was locked; wrote a timestamped copy.", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 — the CSVs are already written
        print(f"  NOTE: could not write the workbook ({type(exc).__name__}: {exc})", file=sys.stderr)

    print()
    print("=== Overall (micro-averaged over every field and test value) ===")
    print(f"  {'':<12}{'gold-covered':>14}{'raw':>12}")
    for label, key in (("Accuracy", "accuracy"), ("Precision", "precision"),
                       ("Recall", "recall"), ("F1", "f1")):
        print(f"  {label:<12}{str(covered[key]):>14}{str(raw[key]):>12}")
    print(f"\n  gold-covered  TP {covered['tp']}  FP {covered['fp']}  FN {covered['fn']}")
    print(f"  raw           TP {raw['tp']}  FP {raw['fp']}  FN {raw['fn']}")
    print(f"\n  table row recall   {summary['table_row_recall']}")
    print(f"  hallucination rate {summary['hallucination_rate']}")

    counts: dict[str, int] = {}
    for r in all_rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    print("\n  === failure categories ===")
    for category, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        if category != "correct":
            print(f"    {n:>5}  {category}")

    worst = per_field_accuracy(all_rows, minimum=5)[:10]
    if worst:
        print("\n  === weakest fields (>=5 documents) ===")
        for row in sorted(worst, key=lambda r: r["accuracy"])[:8]:
            print(f"    {row['accuracy']:.0%}  {row['field'][:34]:<34} ({row['correct']}/{row['documents']})")

    print(f"\n  documents scored {len(per_document)}, failed {len(failures)}")
    for stem, reason in failures:
        print(f"    {stem}: {reason}")
    print(f"\n  Written to {output_dir.resolve()}")
    if workbook:
        print(f"  Workbook   {workbook.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
