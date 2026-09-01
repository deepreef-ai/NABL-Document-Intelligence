#!/usr/bin/env python3
"""CLI entrypoint for the dataset normalization stage.

Usage:
    python scripts/normalize_dataset.py
    python scripts/normalize_dataset.py --input "G:\\...\\dataset" --output "G:\\...\\normalized_dataset"
    python scripts/normalize_dataset.py --force
    python scripts/normalize_dataset.py --min-chars 50 --min-alnum-ratio 0.6

Never writes into --input — raw files are only ever read, never touched.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/ on sys.path

from app.dataset_normalization.pipeline import print_summary, run  # noqa: E402
from app.dataset_normalization.text_quality import TextQualityThresholds  # noqa: E402

DEFAULT_INPUT = r"G:\Shared drives\Product & Engineering\Projects\NABL Document Intelligence\dataset"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Root directory of the raw dataset (read-only).")
    parser.add_argument(
        "--output", default=None,
        help="Output directory for normalized/ and processing_index.jsonl. "
             "Defaults to a 'normalized_dataset' folder next to --input.",
    )
    parser.add_argument("--force", action="store_true", help="Reprocess documents even if already marked processed.")
    parser.add_argument("--min-chars", type=int, default=TextQualityThresholds.min_chars,
                         help="Minimum raw character count for a PDF page's text to be considered meaningful.")
    parser.add_argument("--min-alnum-ratio", type=float, default=TextQualityThresholds.min_alnum_ratio,
                         help="Minimum fraction of a page's text that must be alphanumeric (rejects symbol/whitespace noise).")
    parser.add_argument("--min-word-count", type=int, default=TextQualityThresholds.min_word_count,
                         help="Minimum number of word-shaped tokens for a page's text to be considered meaningful.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    output_dir = Path(args.output) if args.output else input_dir.parent / "normalized_dataset"

    thresholds = TextQualityThresholds(
        min_chars=args.min_chars,
        min_alnum_ratio=args.min_alnum_ratio,
        min_word_count=args.min_word_count,
    )

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Force:  {args.force}")
    print(f"Text-quality thresholds: {thresholds}")

    stats = run(input_dir, output_dir, force=args.force, thresholds=thresholds)
    print_summary(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
