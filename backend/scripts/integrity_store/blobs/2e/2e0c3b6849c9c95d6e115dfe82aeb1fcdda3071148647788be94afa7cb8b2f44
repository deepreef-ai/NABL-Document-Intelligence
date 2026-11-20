#!/usr/bin/env python3
"""Upload a document and see its extraction WITHOUT the frontend.

Hits the same running backend the UI hits (POST /applications/{id}/documents),
so this exercises the real pipeline — classification, per-page OCR fallback,
whole-form section extraction, grounding — not a separate code path. Only the
eligibility wizard is bypassed (see --skip-wizard below).

The upload endpoint rejects an application that hasn't cleared the wizard's
prerequisite questions, and each of those answers costs an LLM call. For
testing extraction that gate is pure overhead, so by default this marks a
freshly-created application "unlocked" directly in the database. That is a
LOCAL TESTING shortcut and the only thing here that doesn't go through the
API; pass --answer-wizard to go through the real questions instead.

Usage:
    python scripts/upload_document.py path/to/report.pdf
    python scripts/upload_document.py report.pdf --form-type NABL_153
    python scripts/upload_document.py report.pdf --application-id <existing-id>
    python scripts/upload_document.py scan.pdf --script devanagari
    python scripts/upload_document.py report.pdf --show-fields 40 --save-form out.json

Requires the backend to be running:
    uvicorn app.main:app --reload --port 8010
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/ on sys.path

import httpx  # noqa: E402

DEFAULT_BASE_URL = "http://localhost:8010"


def _unlock_directly(application_id: str) -> None:
    """Mark the application unlocked in the DB, skipping the wizard's LLM
    calls. Imports the app's own session/model rather than opening the
    SQLite file by hand, so it honours DATABASE_URL and the real schema."""
    from app.db import SessionLocal
    from app.models import Application

    db = SessionLocal()
    try:
        application = db.get(Application, application_id)
        if application is None:
            raise SystemExit(f"application {application_id} not found in the database")
        application.status = "unlocked"
        db.add(application)
        db.commit()
    finally:
        db.close()


def _answer_wizard(client: httpx.Client, base_url: str, application_id: str) -> None:
    """Walk the real prerequisite questions, answering each affirmatively.
    One LLM call per question — slower and rate-limit-exposed, which is why
    it isn't the default."""
    while True:
        state = client.get(f"{base_url}/applications/{application_id}/wizard").raise_for_status().json()
        question = state["state"].get("next_question")
        if not question:
            print("  wizard: all prerequisites satisfied")
            return
        print(f"  wizard: answering {question['id']!r}")
        client.post(
            f"{base_url}/applications/{application_id}/wizard/answer",
            json={"message": "Yes, this is in place and documented."},
        ).raise_for_status()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", type=Path, help="Document to upload (PDF, DOCX, PNG/JPG/TIFF).")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"Backend base URL (default: {DEFAULT_BASE_URL}).")
    parser.add_argument("--form-type", default="NABL_151", help="NABL form type for a NEW application (default: NABL_151).")
    parser.add_argument("--application-id", default=None, help="Upload into an existing application instead of creating one.")
    parser.add_argument("--script", default="english", help="OCR script for pages with no text layer (default: english).")
    parser.add_argument("--answer-wizard", action="store_true",
                        help="Clear the prerequisites through the real wizard (LLM calls) instead of unlocking directly.")
    parser.add_argument("--show-fields", type=int, default=15, help="How many extracted fields to print (default: 15; 0 for none).")
    parser.add_argument("--save-form", type=Path, default=None, help="Write the compiled form JSON to this path.")
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="Upload timeout in seconds (default: 900). Extraction runs inside the request, and a "
                             "long scanned document means per-page OCR plus one LLM call per schema section.")
    args = parser.parse_args()

    if not args.file.is_file():
        print(f"File not found: {args.file}", file=sys.stderr)
        return 1

    with httpx.Client(timeout=args.timeout) as client:
        try:
            client.get(f"{args.base_url}/health", timeout=10).raise_for_status()
        except Exception as exc:  # noqa: BLE001 — a clear message beats a stack trace here
            print(f"Backend not reachable at {args.base_url} ({exc}).\n"
                  f"Start it with:  uvicorn app.main:app --reload --port 8010", file=sys.stderr)
            return 1

        application_id = args.application_id
        if application_id is None:
            created = client.post(
                f"{args.base_url}/applications", json={"form_type": args.form_type},
            ).raise_for_status().json()
            application_id = created["application"]["id"]
            print(f"Created application {application_id} ({args.form_type})")

            if args.answer_wizard:
                _answer_wizard(client, args.base_url, application_id)
            else:
                _unlock_directly(application_id)
                print("  unlocked directly (skipped the eligibility wizard)")
        else:
            print(f"Using existing application {application_id}")

        print(f"Uploading {args.file.name} … (extraction runs server-side; this can take a while)")
        with args.file.open("rb") as fh:
            response = client.post(
                f"{args.base_url}/applications/{application_id}/documents",
                files={"file": (args.file.name, fh, "application/octet-stream")},
                data={"script": args.script},
            )
        if response.status_code >= 400:
            print(f"Upload failed [{response.status_code}]: {response.text[:800]}", file=sys.stderr)
            return 1
        document = response.json()

    print()
    print("=== Extraction result ===")
    print(f"document_id       : {document['id']}")
    print(f"status            : {document['status']}")
    print(f"doc_type          : {document['doc_type']}")
    print(f"extraction_source : {document['extraction_source']}")
    print(f"pages read        : {document.get('page_count')}")
    print(f"fields extracted  : {len(document['fields'])}")
    if document.get("error"):
        print(f"notes             : {document['error'][:500]}")

    by_page: dict[object, int] = {}
    for f in document["fields"]:
        by_page[f["source_page"]] = by_page.get(f["source_page"], 0) + 1
    if by_page:
        ordered = sorted(by_page.items(), key=lambda kv: (kv[0] is None, kv[0]))
        print("fields per page   : " + ", ".join(
            f"{'ungrounded' if p is None else f'p{p + 1}'}={n}" for p, n in ordered
        ))

    if args.show_fields:
        print()
        print(f"--- first {min(args.show_fields, len(document['fields']))} fields ---")
        for f in document["fields"][:args.show_fields]:
            page = "?" if f["source_page"] is None else f["source_page"] + 1
            print(f"  [p{page}] {f['field_path']} = {str(f['value'])[:70]!r}  ({f['confidence']:.2f}, {f['source']})")

    if args.save_form:
        with httpx.Client(timeout=120) as client:
            form = client.get(f"{args.base_url}/applications/{application_id}/form").raise_for_status().json()
        args.save_form.write_text(json.dumps(form, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nCompiled form written to {args.save_form}")

    print(f"\nReview in the UI (optional): {args.base_url.replace('8010', '5173')}/applications/{application_id}/review")
    return 0


if __name__ == "__main__":
    sys.exit(main())
