#!/usr/bin/env python3
"""CLI entrypoint for the dataset-finalization stage (Step 7): compiles ONE
combined dataset (every domain together, no train/validation/test split)
from ONLY approved labels, producing final_labeled_dataset.jsonl,
final_labeled_dataset.csv, and dataset_summary.json.

Usage:
    python scripts/finalize_dataset.py
    python scripts/finalize_dataset.py --labeled "G:\\...\\labeled_dataset" --normalized "G:\\...\\normalized_dataset" --output "G:\\...\\final_dataset"
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/ on sys.path

from app.final_dataset.pipeline import build_final_dataset, print_summary, write_final_dataset  # noqa: E402

DEFAULT_ROOT = r"G:\Shared drives\Product & Engineering\Projects\NABL Document Intelligence"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labeled", default=os.path.join(DEFAULT_ROOT, "labeled_dataset"),
                         help="Step 5/6's labeled_dataset directory (read-only; labels must already carry annotation_status).")
    parser.add_argument("--normalized", default=os.path.join(DEFAULT_ROOT, "normalized_dataset"),
                         help="Step 2's normalized_dataset output directory (read-only).")
    parser.add_argument("--output", default=None,
                         help="Output directory for the final dataset files. Defaults to a 'final_dataset' "
                              "folder next to --labeled.")
    args = parser.parse_args()

    labeled_dir = Path(args.labeled)
    if not (labeled_dir / "labels").is_dir():
        print(f"Labeled dataset not found under: {labeled_dir}", file=sys.stderr)
        return 1

    normalized_dir = Path(args.normalized)
    output_dir = Path(args.output) if args.output else labeled_dir.parent / "final_dataset"

    print(f"Labeled:    {labeled_dir}")
    print(f"Normalized: {normalized_dir}")
    print(f"Output:     {output_dir}")

    records, summary = build_final_dataset(labeled_dir, normalized_dir)
    write_final_dataset(records, summary, output_dir)
    print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
