#!/usr/bin/env python3
"""CLI entrypoint for the master-schema stage (Step 4): reads Step 3's
domain_keys.json and produces master_schema.json + key_mapping.json. Makes
no LLM calls — pure deterministic clustering over already-discovered keys.

Usage:
    python scripts/build_master_schema.py
    python scripts/build_master_schema.py --input "G:\\...\\schema_discovery" --output "G:\\...\\master_schema"
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/ on sys.path

from app.master_schema.pipeline import print_summary, write_master_schema  # noqa: E402

DEFAULT_INPUT = r"G:\Shared drives\Product & Engineering\Projects\NABL Document Intelligence\schema_discovery"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Step 3's schema_discovery output directory (read-only).")
    parser.add_argument(
        "--output", default=None,
        help="Output directory for master_schema.json and key_mapping.json. "
             "Defaults to a 'master_schema' folder next to --input.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not (input_dir / "domain_keys.json").exists():
        print(f"domain_keys.json not found under: {input_dir}", file=sys.stderr)
        return 1

    output_dir = Path(args.output) if args.output else input_dir.parent / "master_schema"

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    master_schema, mapping_entries = write_master_schema(input_dir, output_dir)
    print_summary(master_schema, mapping_entries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
