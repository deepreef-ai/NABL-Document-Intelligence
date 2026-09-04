#!/usr/bin/env python3
"""Runs the budgeted extraction path (documents/orchestrator.py) over real
documents and reports, per document, exactly what the optimisation spec asks
for: extraction method per page, relevant pages, calls by category, fields
extracted, fields recovered, and the validation outcome.

This makes REAL Nova calls — one run is a handful per document by design,
which is the whole point being verified. Use --dry-run to see the scenario
selection and per-page inspection (no LLM calls at all).

Usage:
    python scripts/verify_optimized_pipeline.py --dry-run
    python scripts/verify_optimized_pipeline.py --limit 4
    python scripts/verify_optimized_pipeline.py --only 1-page,mixed-pdf
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/ on sys.path

from app.documents import orchestrator, page_inspection  # noqa: E402

DEFAULT_DATASET = r"G:\Shared drives\Product & Engineering\Projects\NABL Document Intelligence\dataset"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _inspect_only(path: Path) -> dict:
    """Page-level inspection with no LLM call — the basis for scenario
    selection, and useful on its own as the spec's section 1 output."""
    data = path.read_bytes()
    inspected = page_inspection.inspect(data, path.suffix, path.stem, path.name)
    methods = [p.page.extraction_method for p in inspected.pages]
    return {
        "pages": inspected.page_count,
        "source_type": inspected.document.source_type,
        "methods": methods,
        "pymupdf_pages": methods.count("pymupdf"),
        "ocr_pages": methods.count("ocr"),
        "table_pages": sum(1 for p in inspected.pages if p.has_table),
        "low_confidence_pages": sum(1 for p in inspected.pages if p.needs_visual_check),
        "chars": sum(len(p.text or "") for p in inspected.pages),
    }


def pick_scenarios(dataset: Path) -> dict[str, Path]:
    """Selects one real document per spec scenario, by inspecting them."""
    files = [f for f in sorted(dataset.iterdir()) if f.is_file()]
    profiled: list[tuple[Path, dict]] = []
    for f in files:
        if f.suffix.lower() != ".pdf" and f.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            profiled.append((f, _inspect_only(f)))
        except Exception as exc:  # noqa: BLE001 — an unreadable file is not a scenario
            print(f"  (skipping {f.name}: {exc})")

    scenarios: dict[str, Path] = {}

    def first(name: str, predicate, key=None):
        matches = [(f, p) for f, p in profiled if predicate(p)]
        if not matches:
            return
        if key:
            matches.sort(key=lambda m: key(m[1]))
        scenarios[name] = matches[0][0]

    first("1-page-born-digital", lambda p: p["pages"] == 1 and p["ocr_pages"] == 0)
    first("long-born-digital", lambda p: p["pages"] >= 5 and p["ocr_pages"] == 0,
          key=lambda p: -p["pages"])
    first("scanned-pdf", lambda p: p["pages"] >= 1 and p["pymupdf_pages"] == 0 and p["source_type"] == "scanned_pdf",
          key=lambda p: -p["pages"])
    first("mixed-pdf", lambda p: p["source_type"] == "mixed_pdf", key=lambda p: -p["pages"])
    first("standalone-image", lambda p: p["source_type"] == "image")
    first("low-quality-scan", lambda p: p["low_confidence_pages"] > 0, key=lambda p: -p["low_confidence_pages"])
    first("with-tables", lambda p: p["table_pages"] >= 2, key=lambda p: -p["table_pages"])
    first("text-heavy", lambda p: p["chars"] > 20000, key=lambda p: -p["chars"])
    return scenarios


def run_one(name: str, path: Path, form_type: str) -> dict:
    profile = _inspect_only(path)
    started = time.time()
    result = orchestrator.run(
        path.read_bytes(), path.suffix, filename=path.name,
        form_type=form_type, document_id=path.stem,
    )
    elapsed = time.time() - started
    log = result.call_log
    validation = result.validation or {}
    return {
        "scenario": name,
        "document": path.name,
        "doc_type": result.doc_type,
        "classification": f"{result.classification_method} ({result.doc_confidence:.2f})",
        "pages": profile["pages"],
        "methods": f"{profile['pymupdf_pages']} pymupdf / {profile['ocr_pages']} ocr",
        "table_pages": profile["table_pages"],
        "relevant_pages": len(result.relevant_pages),
        "chunks": ",".join(result.chunk_labels),
        "classification_calls": log.get("classification_calls"),
        "extraction_calls": log.get("extraction_calls"),
        "recovery_calls": log.get("recovery_calls"),
        "vision_calls": log.get("vision_calls"),
        "total_llm_calls": log.get("total_llm_calls"),
        "tokens": log.get("total_tokens"),
        "fields": len(result.fields),
        "recovered": len(validation.get("recovered_fields") or []),
        "validation_ok": validation.get("ok"),
        "missing": len(validation.get("missing") or []),
        "suspicious": len(validation.get("suspicious") or []),
        "stop_reason": log.get("stop_reason"),
        "seconds": round(elapsed, 1),
    }


print = functools.partial(__builtins__["print"] if isinstance(__builtins__, dict) else __builtins__.print, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--form-type", default="NABL_151")
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N scenarios.")
    parser.add_argument("--only", default="", help="Comma-separated scenario names.")
    parser.add_argument("--dry-run", action="store_true", help="Inspection only; no LLM calls.")
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--files", default="",
        help="Comma-separated name=filename pairs, skipping scenario auto-selection "
             "(which has to inspect every document in the dataset to choose).",
    )
    args = parser.parse_args()

    dataset = Path(args.dataset)
    if not dataset.is_dir():
        print(f"dataset not found: {dataset}", file=sys.stderr)
        return 1

    if args.files:
        scenarios = {}
        for pair in args.files.split(","):
            name, _, fname = pair.partition("=")
            path = dataset / fname.strip()
            if path.is_file():
                scenarios[name.strip()] = path
            else:
                print(f"  (missing: {fname.strip()})")
    else:
        print(f"Selecting scenarios from {dataset} ...")
        scenarios = pick_scenarios(dataset)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        scenarios = {k: v for k, v in scenarios.items() if k in wanted}
    if args.limit:
        scenarios = dict(list(scenarios.items())[: args.limit])

    print(f"\n{len(scenarios)} scenarios:")
    for name, path in scenarios.items():
        p = _inspect_only(path)
        print(f"  {name:22s} {path.name[:44]:46s} {p['pages']:>3}p  "
              f"{p['pymupdf_pages']} pymupdf/{p['ocr_pages']} ocr  "
              f"{p['table_pages']} table  {p['low_confidence_pages']} low-conf")

    if args.dry_run:
        print("\n--dry-run: no LLM calls made.")
        return 0

    rows = []
    for name, path in scenarios.items():
        print(f"\n=== {name}: {path.name} ===")
        try:
            row = run_one(name, path, args.form_type)
        except Exception as exc:  # noqa: BLE001 — one scenario failing must not stop the matrix
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            rows.append({"scenario": name, "document": path.name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        rows.append(row)
        print(f"  {row['doc_type']} via {row['classification']}  |  "
              f"{row['pages']}p ({row['methods']}), {row['relevant_pages']} relevant, chunks {row['chunks']}")
        print(f"  CALLS classify={row['classification_calls']} extract={row['extraction_calls']} "
              f"recover={row['recovery_calls']} vision={row['vision_calls']} "
              f"TOTAL={row['total_llm_calls']}  tokens={row['tokens']}  {row['seconds']}s")
        print(f"  fields={row['fields']} recovered={row['recovered']} "
              f"validation_ok={row['validation_ok']} missing={row['missing']} suspicious={row['suspicious']}")
        if row.get("stop_reason"):
            print(f"  stopped: {row['stop_reason']}")

    totals = [r["total_llm_calls"] for r in rows if r.get("total_llm_calls") is not None]
    print("\n=== Summary ===")
    print(f"scenarios run     : {len(totals)}/{len(rows)}")
    if totals:
        print(f"total LLM calls   : {sum(totals)}")
        print(f"avg calls/document: {sum(totals) / len(totals):.2f}")
        print(f"max calls/document: {max(totals)}")
        over = [r['scenario'] for r in rows if (r.get('total_llm_calls') or 0) > 6]
        print(f"over budget (>6)  : {over or 'none'}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
