#!/usr/bin/env python3
"""CLI entrypoint for the final-dataset validation stage: an independent,
read-only integrity check over the COMPILED final_labeled_dataset.jsonl —
never modifies any label or annotation_status, only writes
final_dataset_qc.json.

Usage:
    python scripts/validate_final_dataset.py
    python scripts/validate_final_dataset.py --final "G:\\...\\final_dataset" --normalized "G:\\...\\normalized_dataset" --master-schema "G:\\...\\master_schema"
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/ on sys.path

from app.final_dataset_qc.pipeline import print_summary, run  # noqa: E402

DEFAULT_ROOT = r"G:\Shared drives\Product & Engineering\Projects\NABL Document Intelligence"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--final", default=os.path.join(DEFAULT_ROOT, "final_dataset"),
                         help="Step 7's final_dataset directory (final_dataset_qc.json is written here).")
    parser.add_argument("--normalized", default=os.path.join(DEFAULT_ROOT, "normalized_dataset"),
                         help="Step 2's normalized_dataset output directory (read-only).")
    parser.add_argument("--master-schema", default=os.path.join(DEFAULT_ROOT, "master_schema"),
                         help="Step 4's master_schema output directory (read-only).")
    args = parser.parse_args()

    final_dataset_dir = Path(args.final)
    if not (final_dataset_dir / "final_labeled_dataset.jsonl").exists():
        print(f"final_labeled_dataset.jsonl not found under: {final_dataset_dir}", file=sys.stderr)
        return 1

    print(f"Final dataset: {final_dataset_dir}")
    print(f"Normalized:    {args.normalized}")
    print(f"Master schema: {args.master_schema}")

    report = run(final_dataset_dir, Path(args.normalized), Path(args.master_schema))
    print_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
