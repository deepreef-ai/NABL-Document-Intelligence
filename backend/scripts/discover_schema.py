#!/usr/bin/env python3
"""CLI entrypoint for the schema-discovery stage (Step 3): reads the
normalized documents Step 2 (scripts/normalize_dataset.py) produced, sends a
small representative sample of them to the LLM, and writes each sampled
document's {document_id, candidate_domain, keys} plus a domain -> keys
aggregate. Never extracts values, never sends the whole dataset to the LLM.

Usage:
    python scripts/discover_schema.py
    python scripts/discover_schema.py --input "G:\\...\\normalized_dataset" --output "G:\\...\\schema_discovery"
    python scripts/discover_schema.py --max-per-domain 8 --force
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/ on sys.path

from app.schema_discovery.pipeline import print_summary, run  # noqa: E402

DEFAULT_INPUT = r"G:\Shared drives\Product & Engineering\Projects\NABL Document Intelligence\normalized_dataset"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Step 2's normalized_dataset output directory (read-only).")
    parser.add_argument(
        "--output", default=None,
        help="Output directory for samples/, discovery_index.jsonl, and domain_keys.json. "
             "Defaults to a 'schema_discovery' folder next to --input.",
    )
    parser.add_argument(
        "--max-per-domain", type=int, default=5,
        help="Max documents sampled per heuristic domain bucket — bounds total LLM calls (default: 5).",
    )
    parser.add_argument("--force", action="store_true", help="Re-run the LLM call even for already-discovered documents.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    output_dir = Path(args.output) if args.output else input_dir.parent / "schema_discovery"

    print(f"Input:          {input_dir}")
    print(f"Output:         {output_dir}")
    print(f"Max per domain: {args.max_per_domain}")
    print(f"Force:          {args.force}")

    stats = run(input_dir, output_dir, max_per_domain=args.max_per_domain, force=args.force)
    print_summary(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
