"""Manual check for Fix 1: read one file with the legacy OCR path and the OCR
gateway, side by side, through the real preprocessing node.

    python scripts/ocr_compare.py "G:\\...\\dataset\\001_Lab-report.png" [--script english]

Read-only: the input file is never modified or copied into the repo.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}(?: \d{1,2}:\d{2})?")
GLUED = re.compile(r"\b[A-Z]{14,}\b|\d{4}\d{1,2}:\d{2}")


def run(path: Path, script: str, enabled: bool) -> tuple[str, list[str], float]:
    from app.graph.nodes.preprocess import preprocess_document
    from app.graph.state import GraphState
    from app.ocr_gateway import get_gateway_settings

    os.environ["OCR_GATEWAY_ENABLED"] = "true" if enabled else "false"
    get_gateway_settings.cache_clear()
    ext = path.suffix.lower().lstrip(".")
    state = GraphState(document_id="manual", file_path=str(path), script=script,
                       file_type="pdf" if ext == "pdf" else ("docx" if ext == "docx" else ext))
    t0 = time.perf_counter()
    out = preprocess_document(state)
    secs = time.perf_counter() - t0
    text = "\n".join(out["page_text"][p] for p in sorted(out["page_text"]))
    how = [e.detail for e in out["audit_log"] if e.event == "page_ocr"]
    return text, how, secs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--script", default="english")
    a = ap.parse_args()
    path = Path(a.file)
    for label, enabled in (("LEGACY (before Fix 1)", False), ("GATEWAY (Fix 1)", True)):
        text, how, secs = run(path, a.script, enabled)
        print(f"\n===== {label}  {secs:.1f}s  {'; '.join(how) or 'no OCR call'}")
        print(f"dates found : {sorted(set(DATE.findall(text)))}")
        print(f"glued tokens: {sorted(set(GLUED.findall(text)))[:10]}")
        print("first 15 lines:")
        for ln in text.splitlines()[:15]:
            print("   ", ln)
    return 0


if __name__ == "__main__":
    sys.exit(main())
