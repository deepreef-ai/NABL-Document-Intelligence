#!/usr/bin/env python3
"""CLI entrypoint for the (rule-based, no-LLM) key-value extraction stage.
Runs AFTER normalize_dataset.py — reads its normalized_dataset output,
writes normalized/<document_id>/table_rows.json alongside each document.json.

Usage:
    python scripts/extract_key_values.py
    python scripts/extract_key_values.py --input "G:\\...\\normalized_dataset"
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.kv_extraction.pipeline import print_summary, run  # noqa: E402

DEFAULT_INPUT = r"G:\Shared drives\Product & Engineering\Projects\NABL Document Intelligence\normalized_dataset"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="dataset_normalization's output directory.")
    args = parser.parse_args()

    normalized_dir = Path(args.input)
    if not (normalized_dir / "normalized").is_dir():
        print(f"No normalized/ folder found under: {normalized_dir}", file=sys.stderr)
        return 1

    stats = run(normalized_dir)
    print_summary(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
