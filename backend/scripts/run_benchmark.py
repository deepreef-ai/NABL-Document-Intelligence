#!/usr/bin/env python3
"""CLI entrypoint for the benchmark stage (Step 9): re-runs the SAME
production PDF/OCR extraction and domain classification + field extraction
functions Steps 2/5 use, fresh, against every approved document's ORIGINAL
raw file, and scores the result against final_labeled_dataset.jsonl.

Does not modify the production pipeline, any label, or the dataset. No
train/validation/test split — every approved document is scored.

Two phases, run in sequence by default:
  1. predict — one real OCR call + one real LLM call per document, cached
     immediately to predictions/<document_id>.json. RESUMABLE: rerun the
     same command and already-cached predictions are skipped, so a crash
     partway through only costs the documents not yet reached.
  2. score  — cheap, reads only the prediction cache; safe to re-run any
     number of times (e.g. after a scoring-logic fix) without repeating a
     single OCR/LLM call.

Use --predict-only or --score-only to run just one phase.

For a fast run, override the LLM provider order to skip Ollama, e.g.
(PowerShell):
    $env:LLM_PROVIDER_ORDER = "gemini,groq,cerebras,mistral,openrouter,sambanova,nvidia,github"
    python scripts/run_benchmark.py

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --predict-only
    python scripts/run_benchmark.py --score-only
    python scripts/run_benchmark.py --final "G:\\...\\final_dataset" --normalized "G:\\...\\normalized_dataset" --master-schema "G:\\...\\master_schema" --output "G:\\...\\benchmark"
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/ on sys.path

from app.benchmark.pipeline import print_summary, score, write_results  # noqa: E402
from app.benchmark.predict import generate_predictions, print_predict_summary  # noqa: E402

DEFAULT_ROOT = r"G:\Shared drives\Product & Engineering\Projects\NABL Document Intelligence"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--final", default=os.path.join(DEFAULT_ROOT, "final_dataset"),
                         help="Step 7's final_dataset directory (read-only; final_labeled_dataset.jsonl is the ground truth).")
    parser.add_argument("--normalized", default=os.path.join(DEFAULT_ROOT, "normalized_dataset"),
                         help="Step 2's normalized_dataset output directory (read-only; used only to look up each document's source format bucket).")
    parser.add_argument("--master-schema", default=os.path.join(DEFAULT_ROOT, "master_schema"),
                         help="Step 4's master_schema output directory (read-only; supplies the SAME hints Step 5's labeling used).")
    parser.add_argument("--output", default=None,
                         help="Output directory for predictions/ (cache) and benchmark_results.json/.csv. "
                              "Defaults to a 'benchmark' folder next to --final.")
    parser.add_argument("--force", action="store_true", help="Re-run the pipeline even for already-cached predictions.")
    parser.add_argument("--predict-only", action="store_true", help="Only run phase 1 (generate/resume the prediction cache).")
    parser.add_argument("--score-only", action="store_true", help="Only run phase 2 (score whatever is already cached).")
    args = parser.parse_args()

    final_dataset_dir = Path(args.final)
    if not (final_dataset_dir / "final_labeled_dataset.jsonl").exists():
        print(f"final_labeled_dataset.jsonl not found under: {final_dataset_dir}", file=sys.stderr)
        return 1

    normalized_dir = Path(args.normalized)
    master_schema_dir = Path(args.master_schema)
    output_dir = Path(args.output) if args.output else final_dataset_dir.parent / "benchmark"
    predictions_dir = output_dir / "predictions"

    print(f"Final dataset: {final_dataset_dir}")
    print(f"Normalized:    {normalized_dir}")
    print(f"Master schema: {master_schema_dir}")
    print(f"Output:        {output_dir}")

    if not args.score_only:
        stats = generate_predictions(final_dataset_dir, predictions_dir, master_schema_dir, force=args.force)
        print_predict_summary(stats)

    if not args.predict_only:
        results, _, per_document_rows = score(final_dataset_dir, normalized_dir, predictions_dir)
        write_results(results, per_document_rows, output_dir)
        print_summary(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
