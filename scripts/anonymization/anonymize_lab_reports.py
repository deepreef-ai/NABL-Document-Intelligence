#!/usr/bin/env python3
"""Anonymise laboratory reports in a folder — standalone utility.

NOT part of the NABL Document Intelligence pipeline. This script does not
import the application's extraction graph, OCR pipeline, Docling adapter,
classifier or form-filling code, and none of them import this. It exists to
produce de-identified copies of documents sitting in a Google Drive Desktop
folder, and it writes every result to a separate output tree.

Originals are opened read-only and are never modified.

Examples
--------
Dry run — see what would be detected, write nothing:

    python scripts/anonymization/anonymize_lab_reports.py \\
        --input  "G:/Shared drives/.../lab_reports" \\
        --output "D:/anonymised/lab_reports" \\
        --dry-run

Full run with a config file and stamped placeholders:

    python scripts/anonymization/anonymize_lab_reports.py \\
        --input  "G:/Shared drives/.../lab_reports" \\
        --output "D:/anonymised/lab_reports" \\
        --config scripts/anonymization/anonymizer.example.yaml \\
        --mode label

Fast pass over born-digital PDFs only, no OCR models loaded:

    python scripts/anonymization/anonymize_lab_reports.py \\
        -i IN -o OUT --no-ocr

Exit codes
----------
0  every document processed and verified clean
1  at least one document is FAILED_REVIEW or ERROR
2  the run could not start (bad paths, bad config)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the sibling package importable when this file is run directly by path,
# which is how an operator will invoke it.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from anonymizer.config import VALID_REDACTION_MODES, AnonymizerConfig  # noqa: E402
from anonymizer.report import (  # noqa: E402
    STATUS_ERROR,
    STATUS_FAILED_REVIEW,
    STATUS_OK,
    STATUS_SKIPPED,
)
from anonymizer.runner import UnsafePathError, run  # noqa: E402

log = logging.getLogger("anonymize")


def setup_logging(verbosity: int, log_file: Path | None) -> None:
    level = logging.WARNING if verbosity < 0 else logging.INFO if verbosity == 0 else logging.DEBUG
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    # These are chatty at DEBUG and say nothing useful about anonymisation.
    for noisy in ("PIL", "urllib3", "onnxruntime"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="anonymize_lab_reports",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-i", "--input", required=True, help="Folder to scan (recursively). Never modified.")
    p.add_argument("-o", "--output", required=True,
                   help="Folder for anonymised copies. Must be outside --input.")
    p.add_argument("-c", "--config", default=None, help="YAML or JSON config file.")
    p.add_argument("--mode", choices=VALID_REDACTION_MODES, default=None,
                   help="Redaction style: black_box, label (default), or pseudonym.")
    p.add_argument("--reports", default=None,
                   help="Where per-document reports go. Default: <output>/_anonymisation_reports")
    p.add_argument("--no-ocr", action="store_true",
                   help="Skip OCR entirely. Scanned pages are reported as unreadable, never "
                        "silently passed through unredacted.")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip the post-anonymisation verification pass (not recommended).")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace existing files in the output folder.")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N files.")
    p.add_argument("--dry-run", action="store_true",
                   help="Detect and report, but write no anonymised files.")
    p.add_argument("--include-pii-samples", action="store_true",
                   help="Put truncated PII samples in reports for rule tuning. This makes the "
                        "reports themselves sensitive.")
    p.add_argument("--pseudonym-salt", default=None,
                   help="Secret that makes pseudonyms stable across runs. Without it, the same "
                        "person gets a different pseudonym each run.")
    p.add_argument("--dpi", type=int, default=None, help="Raster DPI for scanned pages.")
    p.add_argument("--log-file", default=None, help="Also write logs to this file.")
    p.add_argument("-q", "--quiet", action="store_true", help="Warnings and errors only.")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return p


def apply_overrides(cfg: AnonymizerConfig, args: argparse.Namespace) -> AnonymizerConfig:
    """CLI flags win over the config file, which wins over defaults."""
    if args.mode:
        cfg.redaction.mode = args.mode
    if args.dpi:
        cfg.redaction.raster_dpi = args.dpi
        cfg.ocr.dpi = args.dpi
    if args.pseudonym_salt is not None:
        cfg.redaction.pseudonym_salt = args.pseudonym_salt
    if args.no_ocr:
        cfg.ocr.enabled = False
        cfg.verification.check_ocr = False
    if args.no_verify:
        cfg.verification.enabled = False
    if args.include_pii_samples:
        cfg.include_pii_samples = True
    if cfg.redaction.mode not in VALID_REDACTION_MODES:
        raise ValueError(
            f"invalid redaction mode {cfg.redaction.mode!r}; "
            f"choose one of {', '.join(VALID_REDACTION_MODES)}")
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(-1 if args.quiet else (1 if args.verbose else 0),
                  Path(args.log_file) if args.log_file else None)

    try:
        cfg = apply_overrides(AnonymizerConfig.load(args.config), args)
    except (ValueError, FileNotFoundError, OSError) as exc:
        log.error("configuration problem: %s", exc)
        return 2

    input_root = Path(args.input).expanduser()
    output_root = Path(args.output).expanduser()

    if args.dry_run:
        # A dry run must be unable to write anything, so it is pointed at a
        # throwaway location and every file is reported as SKIPPED-by-design.
        import tempfile

        temp_dir = Path(tempfile.mkdtemp(prefix="anonymise-dryrun-"))
        log.warning("DRY RUN — detecting only; anonymised files go to a temporary folder "
                    "and are discarded")
        output_root = temp_dir

    try:
        report = run(
            input_root=input_root, output_root=output_root, config=cfg,
            use_ocr=not args.no_ocr, overwrite=args.overwrite or args.dry_run,
            limit=args.limit,
            reports_dir=Path(args.reports).expanduser() if args.reports else None,
        )
    except UnsafePathError as exc:
        log.error("refusing to run: %s", exc)
        return 2
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        log.warning("interrupted; files already written are complete and verified")
        return 1

    reports_dir = (Path(args.reports).expanduser() if args.reports
                   else output_root / "_anonymisation_reports")
    try:
        report.write(reports_dir / "run_report.json")
        report.write_csv(reports_dir / "run_report.csv")
    except OSError as exc:
        log.warning("could not write the run report: %s", exc)

    counts = report.counts
    total = len(report.documents)
    print()
    print("=" * 62)
    print(f"  documents processed : {total}")
    for status in (STATUS_OK, STATUS_FAILED_REVIEW, STATUS_ERROR, STATUS_SKIPPED):
        if counts.get(status):
            print(f"  {status:<20}: {counts[status]}")
    print(f"  detections          : {sum(d.total_detections for d in report.documents)}")
    print(f"  redactions applied  : {sum(d.total_redactions for d in report.documents)}")
    print(f"  reports             : {reports_dir}")
    if not args.dry_run:
        print(f"  output              : {output_root}")
    print("=" * 62)

    flagged = [d for d in report.documents if d.status == STATUS_FAILED_REVIEW]
    if flagged:
        print("\nFAILED_REVIEW — check these by hand before release:")
        for d in flagged[:20]:
            print(f"  - {d.relative_path}: {d.verification_summary}")
        if len(flagged) > 20:
            print(f"  ... and {len(flagged) - 20} more (see run_report.csv)")

    errored = [d for d in report.documents if d.status == STATUS_ERROR]
    if errored:
        print("\nERROR — not anonymised, no output written:")
        for d in errored[:20]:
            print(f"  - {d.relative_path}: {d.error}")

    return 1 if (flagged or errored) else 0


if __name__ == "__main__":
    raise SystemExit(main())
