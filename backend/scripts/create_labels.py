#!/usr/bin/env python3
"""CLI entrypoint for the labeling stage (Step 5): for every normalized
document, classifies its domain and extracts fields/tests guided by Step 4's
master schema, writing one label JSON per document plus a dataset index.
Every label starts with annotation_status="pending" — see
scripts/run_quality_control.py (Step 6) for the gate before "approved".

Usage:
    python scripts/create_labels.py
    python scripts/create_labels.py --normalized "G:\\...\\normalized_dataset" --master-schema "G:\\...\\master_schema" --output "G:\\...\\labeled_dataset"
    python scripts/create_labels.py --force
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/ on sys.path

from app.labeling.pipeline import print_summary, run  # noqa: E402

DEFAULT_ROOT = r"G:\Shared drives\Product & Engineering\Projects\NABL Document Intelligence"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--normalized", default=os.path.join(DEFAULT_ROOT, "normalized_dataset"),
                         help="Step 2's normalized_dataset output directory (read-only).")
    parser.add_argument("--master-schema", default=os.path.join(DEFAULT_ROOT, "master_schema"),
                         help="Step 4's master_schema output directory (read-only).")
    parser.add_argument("--output", default=None,
                         help="Output directory for labels/ and label_index.jsonl. "
                              "Defaults to a 'labeled_dataset' folder next to --normalized's parent.")
    parser.add_argument("--force", action="store_true", help="Re-run the LLM call even for already-labeled documents.")
    args = parser.parse_args()

    normalized_dir = Path(args.normalized)
    if not (normalized_dir / "normalized").is_dir():
        print(f"Normalized dataset not found under: {normalized_dir}", file=sys.stderr)
        return 1

    master_schema_dir = Path(args.master_schema)
    output_dir = Path(args.output) if args.output else normalized_dir.parent / "labeled_dataset"

    print(f"Normalized:    {normalized_dir}")
    print(f"Master schema: {master_schema_dir}")
    print(f"Output:        {output_dir}")
    print(f"Force:         {args.force}")

    stats = run(normalized_dir, master_schema_dir, output_dir, force=args.force)
    print_summary(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
