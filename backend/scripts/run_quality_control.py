#!/usr/bin/env python3
"""CLI entrypoint for the quality-control stage (Step 6): checks every label
Step 5 produced, writes back only annotation_status + a qc block (never the
field/test values themselves), and produces qc_report.json.

A label that fails is always forced to "rejected", even overriding a prior
"approved" — see app/quality_control/pipeline.py's docstring. Approving a
label is a separate, explicit action — use --approve, which re-checks the
label fresh and refuses if it currently fails QC.

Usage:
    python scripts/run_quality_control.py
    python scripts/run_quality_control.py --labeled "G:\\...\\labeled_dataset" --normalized "G:\\...\\normalized_dataset" --master-schema "G:\\...\\master_schema"
    python scripts/run_quality_control.py --approve LR_000001
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/ on sys.path

from app.quality_control.approve import ApprovalRefused, approve_all_passing, approve_document  # noqa: E402
from app.quality_control.pipeline import DEFAULT_OCR_CONFIDENCE_THRESHOLD, print_summary, run  # noqa: E402

DEFAULT_ROOT = r"G:\Shared drives\Product & Engineering\Projects\NABL Document Intelligence"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labeled", default=os.path.join(DEFAULT_ROOT, "labeled_dataset"),
                         help="Step 5's labeled_dataset output directory (labels are updated in place).")
    parser.add_argument("--normalized", default=os.path.join(DEFAULT_ROOT, "normalized_dataset"),
                         help="Step 2's normalized_dataset output directory (read-only).")
    parser.add_argument("--master-schema", default=os.path.join(DEFAULT_ROOT, "master_schema"),
                         help="Step 4's master_schema output directory (read-only).")
    parser.add_argument("--ocr-confidence-threshold", type=float, default=DEFAULT_OCR_CONFIDENCE_THRESHOLD,
                         help="OCR confidence below this is flagged as an OCR failure.")
    parser.add_argument("--approve", metavar="DOCUMENT_ID", default=None,
                         help="Instead of running the full QC pass, try to approve one document_id "
                              "(re-checks it fresh; refuses if it currently fails QC).")
    parser.add_argument("--approve-all-passing", action="store_true",
                         help="Instead of running the full QC pass, try to approve every not-yet-approved "
                              "label that currently passes hard QC checks (each one is still individually "
                              "re-checked and can be refused — this is not a bulk bypass).")
    args = parser.parse_args()

    labeled_dir = Path(args.labeled)
    normalized_dir = Path(args.normalized)
    master_schema_dir = Path(args.master_schema)

    if not (labeled_dir / "labels").is_dir():
        print(f"Labeled dataset not found under: {labeled_dir}", file=sys.stderr)
        return 1

    if args.approve:
        try:
            approve_document(args.approve, labeled_dir, normalized_dir, master_schema_dir, args.ocr_confidence_threshold)
        except ApprovalRefused as exc:
            print(f"Refused: {exc}", file=sys.stderr)
            return 1
        print(f"{args.approve} approved.")
        return 0

    if args.approve_all_passing:
        approved_ids, refused = approve_all_passing(labeled_dir, normalized_dir, master_schema_dir, args.ocr_confidence_threshold)
        print(f"Approved: {len(approved_ids)}")
        print(f"Refused:  {len(refused)}")
        for document_id, reason in refused:
            print(f"  {document_id}: {reason}")
        return 0

    print(f"Labeled:       {labeled_dir}")
    print(f"Normalized:    {normalized_dir}")
    print(f"Master schema: {master_schema_dir}")

    report, _ = run(labeled_dir, normalized_dir, master_schema_dir, ocr_confidence_threshold=args.ocr_confidence_threshold)
    print_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
