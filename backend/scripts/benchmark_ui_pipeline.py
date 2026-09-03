#!/usr/bin/env python3
"""Benchmark the LIVE UPLOAD PIPELINE — the same code path the review UI
drives — against labelled_dataset's hand-written ground truth.

generate_predictions_and_score.py measures a deliberately minimal path: one
Nova call per document, text-or-image, no classification, no schema, no
grounding. That is the right instrument for "how well can the model read a
lab report", but it is NOT what the product runs. This script closes that
gap by calling documents/pipeline.py's process_document() — byte for byte
the function routers/documents.py's upload endpoint calls — so the number it
produces describes the shipped pipeline: per-page OCR fallback, doc-type
classification, schema extraction, open-ended extraction, retrieval, and
grounding all included.

WHAT IS COMPARED, AND WHY
labelled_dataset's ground truth is open key/value pairs invented from each
document's own labels ("sample_name", "report_no", ...). The upload pipeline
produces two kinds of field in one run:
  * schema fields  — NABL form paths ("organisation.gst_number"), which can
    never match a lab report's own vocabulary, and
  * open_extraction fields — the model's own snake_case names, which are
    exactly the same shape as the ground truth.
So scoring uses the open_extraction fields. Schema fields are counted and
reported separately rather than folded in, because scoring a NABL form path
against a lab-report key would manufacture both a false positive and a false
negative from one correct read.

Ground-truth `tests` rows (test_name/result/unit/reference_range) have no
counterpart in the upload pipeline at all — it has no table-row concept — so
they are reported as an explicit, separate coverage gap instead of being
silently counted as misses that make the headline number look worse for a
reason that has nothing to do with reading accuracy.

Usage:
    python scripts/benchmark_ui_pipeline.py
    python scripts/benchmark_ui_pipeline.py --limit 5
    python scripts/benchmark_ui_pipeline.py --force
    python scripts/benchmark_ui_pipeline.py --score-only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/ on sys.path

from app.benchmark.accumulator import MetricAccumulator  # noqa: E402
from app.benchmark.compare import compare_fields, compare_tests  # noqa: E402

DEFAULT_ROOT = r"G:\Shared drives\Product & Engineering\Projects\NABL Document Intelligence"
CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def predict_one(source_path: Path, form_type: str, script: str) -> dict:
    """Runs the real upload pipeline once and reshapes its result into the
    same {fields, tests, pipeline_error} contract the Nova benchmark's cache
    uses, so both benchmarks' scoring code and output files stay identical
    in shape and can be diffed against each other."""
    from app.documents.pipeline import process_document

    try:
        data = source_path.read_bytes()
        result = process_document(
            data,
            source_path.name,
            CONTENT_TYPES.get(source_path.suffix.lower(), "application/octet-stream"),
            script=script,
            form_type=form_type,
            document_id=source_path.stem,
        )
    except Exception as exc:  # noqa: BLE001 — a pipeline failure is a result, not a crash
        return {"fields": {}, "schema_fields": {}, "tests": [], "page_count": None,
                "doc_type": None, "extraction_source": None,
                "pipeline_error": f"{type(exc).__name__}: {exc}"}

    open_fields, schema_fields = {}, {}
    for f in result.fields:
        if f.value is None or not str(f.value).strip():
            continue
        (open_fields if f.source == "open_extraction" else schema_fields)[f.field] = f.value

    return {
        "fields": open_fields,
        "schema_fields": schema_fields,
        "tests": [],  # the upload pipeline has no table-row concept — see module docstring
        "page_count": result.page_count,
        "doc_type": result.doc_type,
        "extraction_source": result.extraction_source,
        "pipeline_error": None,
        "extraction_warnings": result.extraction_warnings,
    }


def generate_predictions(
    labelled_dir: Path, dataset_dir: Path, predictions_dir: Path,
    form_type: str, script: str, force: bool, limit: int | None,
) -> dict:
    predictions_dir.mkdir(parents=True, exist_ok=True)
    stats = {"total": 0, "predicted": 0, "skipped": 0, "failed": 0, "failures": []}

    label_paths = sorted(labelled_dir.glob("*/*.json"))
    if limit:
        label_paths = label_paths[:limit]

    for label_path in label_paths:
        stem = label_path.stem
        out_path = predictions_dir / f"{stem}.json"
        stats["total"] += 1

        if out_path.exists() and not force:
            # A cached FAILURE is not "already done" — same resume rule as
            # generate_predictions_and_score.py, so a transient provider
            # outage retries automatically on the next run.
            if json.loads(out_path.read_text(encoding="utf-8")).get("pipeline_error") is None:
                stats["skipped"] += 1
                continue

        ground_truth = json.loads(label_path.read_text(encoding="utf-8"))
        source_path = dataset_dir / ground_truth["original_filename"]
        if not source_path.is_file():
            prediction = {"fields": {}, "schema_fields": {}, "tests": [], "page_count": None,
                          "doc_type": None, "extraction_source": None,
                          "pipeline_error": f"source file not found: {ground_truth['original_filename']}"}
        else:
            print(f"  {stem} … ", end="", flush=True)
            prediction = predict_one(source_path, form_type, script)
            print(
                f"{prediction.get('doc_type')} | {prediction.get('page_count')}p | "
                f"{len(prediction['fields'])} open + {len(prediction['schema_fields'])} schema fields"
                if prediction["pipeline_error"] is None else f"FAILED: {prediction['pipeline_error'][:80]}"
            )

        out_path.write_text(json.dumps(prediction, indent=2, ensure_ascii=False), encoding="utf-8")
        if prediction["pipeline_error"]:
            stats["failed"] += 1
            stats["failures"].append((stem, prediction["pipeline_error"]))
        else:
            stats["predicted"] += 1

    return stats


def _value_coverage(ground_truth_fields: dict, predicted_fields: dict) -> dict:
    """Naming-INDEPENDENT scoring: was the information found at all, under
    any key?

    The key-level metrics above compare field NAMES, which punishes a
    correct read that simply chose different words. Measured on
    000_Prescription: ground truth stores one `medicines` list; the pipeline
    returned the same four medicines as twelve flat fields
    (medicine_1_name, medicine_1_timing, ...). That is 1 "missing" + 12
    "hallucinated" = 13 counted errors for information extracted perfectly.
    Ground truth's `date` vs the pipeline's `prescription_date` costs another
    two. So the key-level number measures naming agreement, not reading.

    This measures reading instead: for every ground-truth value, does that
    text appear among the predicted values, whatever the key is called
    (substring either direction, so one side splitting a value across
    several fields still counts). Reported ALONGSIDE the key-level metrics,
    never replacing them — naming consistency genuinely matters for a
    product that has to fill named form fields, so both numbers are real,
    they just answer different questions.
    """
    def norm(v) -> str:
        text = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, sort_keys=True)
        return re.sub(r"[^a-z0-9]", "", str(text).lower())

    gt_values = [norm(v) for v in ground_truth_fields.values() if v is not None and norm(v)]
    pred_values = [norm(v) for v in predicted_fields.values() if v is not None and norm(v)]

    found = sum(1 for g in gt_values if any(g in p or p in g for p in pred_values))
    used = sum(1 for p in pred_values if any(p in g or g in p for g in gt_values))
    return {
        "gt_values": len(gt_values),
        "gt_values_found": found,
        "pred_values": len(pred_values),
        "pred_values_supported": used,
    }


def _test_row_coverage(ground_truth_tests: list[dict], predicted_fields: dict) -> dict:
    """Ground-truth TEST ROWS, checked the same naming-independent way.

    The upload pipeline has no table-row extraction, so these can never
    match a predicted `tests` list — but open extraction sometimes picks a
    table's contents up as ordinary flat fields ("acid_value": "3.57"), and
    that genuinely IS the datum captured. Excluding test rows from the
    headline number entirely would flatter it (they are ~44% of everything
    the ground truth records); counting them as automatic misses would
    understate it. So each row's result is checked against the predicted
    values, and whatever is really there counts.
    """
    def norm(v) -> str:
        return re.sub(r"[^a-z0-9]", "", str(v).lower())

    pred_values = [norm(v) for v in predicted_fields.values() if v is not None and norm(v)]
    rows = [r for r in ground_truth_tests if r.get("result") is not None and norm(r.get("result"))]
    found = sum(1 for r in rows if any(norm(r["result"]) in p or p in norm(r["result"]) for p in pred_values))
    return {"gt_test_results": len(rows), "gt_test_results_found": found}


def _stringify_values(fields: dict) -> dict:
    """Non-scalar values (lists/dicts) rendered one canonical way, so the
    ground-truth and prediction sides are always comparable."""
    return {
        k: (v if v is None or isinstance(v, (str, int, float, bool)) else json.dumps(v, ensure_ascii=False, sort_keys=True))
        for k, v in fields.items()
    }


def score(labelled_dir: Path, predictions_dir: Path, limit: int | None) -> tuple[dict, list[dict], dict]:
    overall = MetricAccumulator()
    rows: list[dict] = []
    coverage = {"documents": 0, "ground_truth_tests": 0, "schema_fields_produced": 0}

    label_paths = sorted(labelled_dir.glob("*/*.json"))
    if limit:
        label_paths = label_paths[:limit]

    for label_path in label_paths:
        stem = label_path.stem
        ground_truth = json.loads(label_path.read_text(encoding="utf-8"))
        pred_path = predictions_dir / f"{stem}.json"
        prediction = (
            json.loads(pred_path.read_text(encoding="utf-8")) if pred_path.exists()
            else {"fields": {}, "schema_fields": {}, "tests": [], "pipeline_error": "no cached prediction found"}
        )

        extraction_ok = prediction.get("pipeline_error") is None
        # Values are sometimes lists/dicts (a prescription's medication
        # table, say). compare.py only normalizes scalars, so both sides get
        # stringified HERE, identically — stringifying only the ground-truth
        # side would compare json.dumps' '["A", "B"]' against compare.py's
        # str() of "['A', 'B']" and never match, quietly understating
        # accuracy on every document that has a list-valued field.
        gt_fields = _stringify_values(ground_truth.get("fields", {}))
        pred_fields = _stringify_values(prediction.get("fields", {}))

        field_failures, field_counters = compare_fields(stem, gt_fields, pred_fields)
        test_failures, test_counters = compare_tests(stem, [], prediction.get("tests", []))
        exact_match = extraction_ok and not field_failures and not test_failures

        doc_acc = MetricAccumulator()
        for acc in (doc_acc, overall):
            acc.add(domain_match=True, extraction_ok=extraction_ok, exact_match=exact_match,
                    field_counters=field_counters, test_counters=test_counters)

        coverage["documents"] += 1
        coverage["ground_truth_tests"] += len(ground_truth.get("tests", []))
        coverage["schema_fields_produced"] += len(prediction.get("schema_fields", {}))

        vc = _value_coverage(gt_fields, pred_fields)
        tc = _test_row_coverage(ground_truth.get("tests", []), pred_fields)
        for k, v in {**vc, **tc}.items():
            coverage[k] = coverage.get(k, 0) + v

        rows.append({
            "document": stem,
            "original_filename": ground_truth.get("original_filename"),
            "doc_type": prediction.get("doc_type"),
            "extraction_source": prediction.get("extraction_source"),
            "page_count": prediction.get("page_count"),
            "open_fields": len(prediction.get("fields", {})),
            "schema_fields": len(prediction.get("schema_fields", {})),
            "ground_truth_fields": len(gt_fields),
            "ground_truth_tests": len(ground_truth.get("tests", [])),
            "extraction_status": "ok" if extraction_ok else "failed",
            "pipeline_error": prediction.get("pipeline_error"),
            "gt_values_found": vc["gt_values_found"],
            "gt_values": vc["gt_values"],
            **doc_acc.finalize(),
        })

    return overall.finalize(), rows, coverage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labelled", default=os.path.join(DEFAULT_ROOT, "labelled_dataset"))
    parser.add_argument("--dataset", default=os.path.join(DEFAULT_ROOT, "dataset"))
    parser.add_argument("--output", default=None,
                        help="Defaults to a 'predictions_ui_pipeline' folder next to --labelled. Deliberately "
                             "separate from the Nova benchmark's output so the two numbers never overwrite "
                             "each other.")
    parser.add_argument("--form-type", default="NABL_151",
                        help="Only affects the schema half of extraction (reported, not scored); open extraction "
                             "is form-agnostic.")
    parser.add_argument("--script", default="english", help="OCR script for pages with no text layer.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N documents.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    labelled_dir = Path(args.labelled)
    if not labelled_dir.is_dir():
        print(f"Labelled dataset not found: {labelled_dir}", file=sys.stderr)
        return 1

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output) if args.output else labelled_dir.parent / "predictions_ui_pipeline"
    predictions_dir = output_dir / "predictions"

    print(f"Labelled dataset : {labelled_dir}")
    print(f"Dataset (raw)    : {dataset_dir}")
    print(f"Output           : {output_dir}")
    print(f"Pipeline         : documents/pipeline.py process_document() — the live upload path")

    if not args.score_only:
        print()
        print("=== Running the upload pipeline per document ===")
        stats = generate_predictions(
            labelled_dir, dataset_dir, predictions_dir, args.form_type, args.script, args.force, args.limit,
        )
        print()
        print(f"Total: {stats['total']} | predicted: {stats['predicted']} | "
              f"skipped: {stats['skipped']} | failed: {stats['failed']}")
        for name, reason in stats["failures"]:
            print(f"  {name}: {reason[:160]}")

    if not args.predict_only:
        summary, rows, coverage = score(labelled_dir, predictions_dir, args.limit)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(
            json.dumps({"summary": summary, "coverage": coverage}, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        if rows:
            with (output_dir / "per_document.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        print()
        print("=== Score Summary (upload pipeline, open-extraction fields) ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
        print()
        print("--- coverage caveats ---")
        print(f"documents scored                  : {coverage['documents']}")
        print(f"ground-truth test rows NOT scored : {coverage['ground_truth_tests']}"
              " (upload pipeline has no table-row extraction)")
        print(f"schema fields produced but unscored: {coverage['schema_fields_produced']}"
              " (NABL form paths can't match lab-report keys)")

        gv, gf = coverage.get("gt_values", 0), coverage.get("gt_values_found", 0)
        pv, ps = coverage.get("pred_values", 0), coverage.get("pred_values_supported", 0)
        print()
        print("=== Naming-INDEPENDENT (did it find the information, under any key?) ===")
        print(f"ground-truth values found   : {gf}/{gv}"
              f"  ({gf / gv:.1%})" if gv else "ground-truth values found   : n/a")
        print(f"predicted values supported  : {ps}/{pv}"
              f"  ({ps / pv:.1%})" if pv else "predicted values supported  : n/a")
        print("(Compare these to key_recall/key_precision above: the gap between them"
              " IS the naming/nesting divergence, not reading error.)")

        tr, trf = coverage.get("gt_test_results", 0), coverage.get("gt_test_results_found", 0)
        if tr:
            print(f"test-table results found    : {trf}/{tr}  ({trf / tr:.1%})")

        # THE single overall number: every distinct datum the ground truth
        # records — document fields AND test-table rows — against how much
        # the pipeline actually produced, regardless of what it named it.
        total, hit = gv + tr, gf + trf
        if total:
            print()
            print(f"OVERALL ACCURACY            : {hit}/{total} = {hit / total:.1%}")
            print("  (share of everything the ground truth records that the pipeline extracted,"
                  " fields + test rows, naming-independent)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
