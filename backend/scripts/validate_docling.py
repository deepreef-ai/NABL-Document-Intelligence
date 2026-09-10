#!/usr/bin/env python3
"""Run one document through preprocessing twice — Docling off, then on.

Preprocessing ONLY. The structure pass lives in `preprocess_document`, so
everything this integration can affect is visible by the time that node
returns, and stopping there means a validation sweep costs no LLM calls and no
provider quota.

The comparison that matters is not "did Docling find more". It is whether the
flag changes anything a downstream consumer depends on:

    page count          must be identical — PyMuPDF is the authority
    page status         must be identical — Docling does not read text
    ocr_applied         must be identical
    ocr_confidence      must be identical — RapidOCR's number, untouched
    page_text           may only GROW, and only by an appended structure block

Anything else differing is a regression, not an improvement.

    python scripts/validate_docling.py                       # the default set
    python scripts/validate_docling.py path/to/doc.pdf ...
    python scripts/validate_docling.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.graph.config import get_graph_settings  # noqa: E402
from app.graph.docling_adapter import LAYOUT_MARKER, docling_available  # noqa: E402
from app.graph.nodes.preprocess import preprocess_document  # noqa: E402
from app.graph.state import GraphState  # noqa: E402

STORAGE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage")


def _kind(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return "pdf"
    if ext == ".docx":
        return "docx"
    return "image"


def run_once(path: str, *, use_docling: bool) -> dict:
    settings = get_graph_settings()
    previous = settings.use_docling
    settings.use_docling = use_docling
    started = time.monotonic()
    try:
        out = preprocess_document(
            GraphState(document_id="validate", file_path=path, file_type=_kind(path))
        )
    finally:
        settings.use_docling = previous

    pages = out.get("page_metadata", {})
    audits = [a for a in out.get("audit_log", []) if "docling" in str(getattr(a, "event", ""))]
    return {
        "seconds": round(time.monotonic() - started, 2),
        "total_pages": out.get("total_pages", 0),
        "page_text": out.get("page_text", {}),
        "pages": {
            n: {
                "chars": m.char_count,
                "status": m.status.value,
                "has_table": m.has_table,
                "ocr_applied": m.ocr_applied,
                "ocr_confidence": m.ocr_confidence,
                "has_layout": m.layout is not None,
                "tables": len(getattr(m.layout, "tables", []) or []) if m.layout else 0,
                "blocks": len(getattr(m.layout, "blocks", []) or []) if m.layout else 0,
            }
            for n, m in pages.items()
        },
        "errors": [str(getattr(e, "error_message", e))[:120] for e in out.get("errors", [])],
        "docling_events": [
            f"{getattr(a, 'event', '?')}: {getattr(a, 'detail', '')[:140]}" for a in audits
        ],
    }


#: Fields that MUST NOT change when the flag flips. has_table is excluded on
#: purpose: replacing a ruled-line heuristic with a table model is the point.
INVARIANT = ("status", "ocr_applied", "ocr_confidence")


def compare(off: dict, on: dict) -> list[str]:
    problems: list[str] = []
    if off["total_pages"] != on["total_pages"]:
        problems.append(f"page count changed {off['total_pages']} -> {on['total_pages']}")

    for n, a in off["pages"].items():
        b = on["pages"].get(n)
        if b is None:
            problems.append(f"page {n} vanished with docling on")
            continue
        for f in INVARIANT:
            if a[f] != b[f]:
                problems.append(f"page {n}: {f} changed {a[f]!r} -> {b[f]!r}")

    for n, before in off["page_text"].items():
        after = on["page_text"].get(n, "")
        if after == before:
            continue
        if not after.startswith(before):
            problems.append(f"page {n}: docling REWROTE the OCR text rather than appending")
        elif LAYOUT_MARKER not in after:
            problems.append(f"page {n}: text grew without a {LAYOUT_MARKER} block")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("documents", nargs="*", help="paths; defaults to the bundled storage set")
    ap.add_argument("--json", dest="json_out", default=None)
    a = ap.parse_args()

    docs = a.documents or [
        os.path.join(STORAGE, f)
        for f in sorted(os.listdir(STORAGE))[:6]
        if f.lower().endswith((".pdf", ".png", ".jpg"))
    ] if os.path.isdir(STORAGE) else []

    if not docs:
        print("no documents given and none found in storage/", file=sys.stderr)
        return 2

    print(f"docling importable: {docling_available()}")
    print(f"{len(docs)} document(s)\n")

    report, failures = [], 0
    for path in docs:
        name = os.path.basename(path)
        if not os.path.exists(path):
            print(f"  SKIP {name}: not found")
            continue
        off = run_once(path, use_docling=False)
        on = run_once(path, use_docling=True)
        problems = compare(off, on)
        failures += bool(problems)

        grew = sum(1 for n, t in on["page_text"].items() if t != off["page_text"].get(n))
        tables = sum(p["tables"] for p in on["pages"].values())
        print(f"  {name[:46]:46s} pages={off['total_pages']:<3d} "
              f"off={off['seconds']:>6.2f}s on={on['seconds']:>6.2f}s "
              f"tables={tables:<3d} pages_enriched={grew}")
        for event in on["docling_events"]:
            print(f"       {event}")
        for p in problems:
            print(f"       !! {p}")

        report.append({
            "document": name, "problems": problems,
            "off": {k: v for k, v in off.items() if k != "page_text"},
            "on": {k: v for k, v in on.items() if k != "page_text"},
        })

    print(f"\n{len(report) - failures}/{len(report)} document(s) preserved every invariant")
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"written to {a.json_out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
