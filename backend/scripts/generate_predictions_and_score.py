#!/usr/bin/env python3
"""Step 2 of the 2-script pipeline: for every document already labeled in
labelled_dataset/ (see create_ground_truth.py), ask an LLM — Amazon Nova on
Bedrock by default, see app/llm/providers.py's NovaProvider — to extract the
same document fresh, cache the result to predictions/<name>.json (resumable:
already-cached documents are skipped unless --force), then score every
cached prediction against its ground truth and write one summary.

No separate "normalize" stage or cached text folder: extraction happens
per-document, in-process, through app/documents/unified_extraction.py — the
SAME module, prompt and page handling the live upload pipeline
(documents/pipeline.py) uses, so these metrics describe the product rather
than a benchmark-only code path. Every page contributes BOTH text (PyMuPDF's
text layer where there is one, OCR where there isn't) and an image, so the
model can cross-check the transcription against the page itself.

The one thing deliberately NOT shared with the live app is the provider
chain: this script runs Nova alone (no fallback), because its job is to
measure one model, while the app runs the full fallback chain because its
job is to answer the request. Because OCR is now part of the shared path,
this script does depend on a working OCR backend (local RapidOCR for
English) — a page whose OCR fails is still sent as an image, with a warning.

Usage:
    python scripts/generate_predictions_and_score.py
    python scripts/generate_predictions_and_score.py --force
    python scripts/generate_predictions_and_score.py --score-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/ on sys.path

from app.benchmark.accumulator import MetricAccumulator  # noqa: E402
from app.benchmark.compare import compare_fields, compare_tests  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.documents import unified_extraction  # noqa: E402
from app.llm.chain import LlmChain  # noqa: E402
from app.llm.providers import NovaProvider  # noqa: E402

DEFAULT_ROOT = r"G:\Shared drives\Product & Engineering\Projects\NABL Document Intelligence"



def _find_source_file(dataset_dir: Path, filename: str) -> Path | None:
    candidate = dataset_dir / filename
    return candidate if candidate.is_file() else None


def _predict_one(chain: LlmChain, source_path: Path) -> dict:
    """Delegates to documents/unified_extraction.py — the SAME prompt, page
    handling and output shape the live upload pipeline uses, which is what
    makes these metrics describe the product rather than a benchmark-only
    code path."""
    payload = unified_extraction.build_payload(
        source_path.read_bytes(), source_path.suffix, script="english",
    )
    result = unified_extraction.extract(chain, payload)
    return {"fields": result["fields"], "tests": result["tests"], "pipeline_error": None}


def generate_predictions(labelled_dir: Path, dataset_dir: Path, predictions_dir: Path, chain: LlmChain, force: bool = False) -> dict:
    predictions_dir.mkdir(parents=True, exist_ok=True)
    stats = {"total": 0, "predicted": 0, "skipped": 0, "failed": 0, "failures": []}

    for label_path in sorted(labelled_dir.glob("*/*.json")):
        stem = label_path.stem
        out_path = predictions_dir / f"{stem}.json"
        stats["total"] += 1
        if out_path.exists() and not force:
            # A cached FAILURE is not "already done" — a timeout or a
            # transient chain cooldown is worth retrying automatically on
            # the next run; only a genuine success is skipped. --force
            # still means "redo everything, successes included".
            cached = json.loads(out_path.read_text(encoding="utf-8"))
            if cached.get("pipeline_error") is None:
                stats["skipped"] += 1
                continue

        ground_truth = json.loads(label_path.read_text(encoding="utf-8"))
        source_path = _find_source_file(dataset_dir, ground_truth["original_filename"])
        if source_path is None:
            prediction = {"fields": {}, "tests": [], "pipeline_error": f"source file not found: {ground_truth['original_filename']}"}
        else:
            try:
                prediction = _predict_one(chain, source_path)
            except Exception as exc:  # noqa: BLE001 — one bad document must never stop the batch
                prediction = {"fields": {}, "tests": [], "pipeline_error": f"{type(exc).__name__}: {exc}"}

        out_path.write_text(json.dumps(prediction, indent=2, ensure_ascii=False), encoding="utf-8")
        if prediction["pipeline_error"]:
            stats["failed"] += 1
            stats["failures"].append((stem, prediction["pipeline_error"]))
        else:
            stats["predicted"] += 1

    return stats


def score(labelled_dir: Path, predictions_dir: Path) -> tuple[dict, list[dict]]:
    overall = MetricAccumulator()
    rows: list[dict] = []

    for label_path in sorted(labelled_dir.glob("*/*.json")):
        stem = label_path.stem
        ground_truth = json.loads(label_path.read_text(encoding="utf-8"))
        pred_path = predictions_dir / f"{stem}.json"

        if pred_path.exists():
            prediction = json.loads(pred_path.read_text(encoding="utf-8"))
        else:
            prediction = {"fields": {}, "tests": [], "pipeline_error": "no cached prediction found"}

        extraction_ok = prediction.get("pipeline_error") is None
        field_failures, field_counters = compare_fields(stem, ground_truth.get("fields", {}), prediction.get("fields", {}))
        test_failures, test_counters = compare_tests(stem, ground_truth.get("tests", []), prediction.get("tests", []))
        exact_match = extraction_ok and not field_failures and not test_failures

        doc_acc = MetricAccumulator()
        doc_acc.add(domain_match=True, extraction_ok=extraction_ok, exact_match=exact_match,
                    field_counters=field_counters, test_counters=test_counters)
        overall.add(domain_match=True, extraction_ok=extraction_ok, exact_match=exact_match,
                    field_counters=field_counters, test_counters=test_counters)

        rows.append({
            "document": stem,
            "original_filename": ground_truth.get("original_filename"),
            "extraction_status": "ok" if extraction_ok else "failed",
            "pipeline_error": prediction.get("pipeline_error"),
            **doc_acc.finalize(),
        })

    return overall.finalize(), rows


def write_results(summary: dict, rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    import csv
    if rows:
        with (output_dir / "per_document.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labelled", default=os.path.join(DEFAULT_ROOT, "labelled_dataset"), help="Ground truth directory (create_ground_truth.py's output).")
    parser.add_argument("--dataset", default=os.path.join(DEFAULT_ROOT, "dataset"), help="Raw source files directory.")
    parser.add_argument("--output", default=None, help="Output directory for predictions/ and summary. Defaults to a 'predictions' folder next to --labelled.")
    parser.add_argument("--force", action="store_true", help="Re-run the LLM call even for already-predicted documents.")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument(
        "--timeout", type=float, default=180.0,
        help="Per-call timeout in seconds (default: 180). The app's usual LLM_TIMEOUT_SECONDS (30s, tuned for "
             "short chat-style calls elsewhere) is too short here — a document with a genuinely large results "
             "table (see high-protein-paneer.pdf's 269 test rows) needs real headroom to both generate and "
             "transfer a long JSON reply.",
    )
    args = parser.parse_args()

    labelled_dir = Path(args.labelled)
    if not labelled_dir.is_dir():
        print(f"Labelled dataset not found: {labelled_dir}", file=sys.stderr)
        return 1

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output) if args.output else labelled_dir.parent / "predictions"
    predictions_dir = output_dir / "predictions"

    settings = get_settings()
    if not settings.nova_model:
        print("NOVA_MODEL is not configured in .env — see NovaProvider in app/llm/providers.py.", file=sys.stderr)
        return 1
    chain = LlmChain([NovaProvider(settings.nova_model, settings.nova_region, args.timeout, max_tokens=settings.nova_max_tokens)])

    print(f"Labelled dataset: {labelled_dir}")
    print(f"Dataset (raw):    {dataset_dir}")
    print(f"Output:           {output_dir}")

    if not args.score_only:
        stats = generate_predictions(labelled_dir, dataset_dir, predictions_dir, chain, force=args.force)
        print()
        print("=== Prediction Summary ===")
        print(f"Total documents: {stats['total']}")
        print(f"Predicted (new): {stats['predicted']}")
        print(f"Skipped (cached): {stats['skipped']}")
        print(f"Failed:          {stats['failed']}")
        for name, reason in stats["failures"]:
            print(f"  {name}: {reason}")

    if not args.predict_only:
        summary, rows = score(labelled_dir, predictions_dir)
        write_results(summary, rows, output_dir)
        print()
        print("=== Score Summary ===")
        for key, value in summary.items():
            print(f"{key}: {value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
