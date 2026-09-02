#!/usr/bin/env python3
"""Step 2 of the 2-script pipeline: for every document already labeled in
labelled_dataset/ (see create_ground_truth.py), ask an LLM — Amazon Nova on
Bedrock by default, see app/llm/providers.py's NovaProvider — to extract the
same document fresh, cache the result to predictions/<name>.json (resumable:
already-cached documents are skipped unless --force), then score every
cached prediction against its ground truth and write one summary.

No separate "normalize" stage or cached text folder: text/image extraction
happens per-document, in-process, using app/documents/pdf_utils.py directly
(pure PyMuPDF, no OCR dependency) — a born-digital page's real text is sent
as text; a page with no text layer (scanned PDF or a raw image file) is
rasterized and sent to the model as an image instead of running local OCR
first. Nova is multimodal, so this hands it the actual page rather than a
possibly-degraded OCR transcription, and it means this script never depends
on the deepreef-ocr sibling checkout at all.

Usage:
    python scripts/generate_predictions_and_score.py
    python scripts/generate_predictions_and_score.py --force
    python scripts/generate_predictions_and_score.py --score-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/ on sys.path

from app.benchmark.accumulator import MetricAccumulator  # noqa: E402
from app.benchmark.compare import compare_fields, compare_tests  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.documents import pdf_utils  # noqa: E402
from app.llm.chain import LlmChain  # noqa: E402
from app.llm.providers import NovaProvider  # noqa: E402

DEFAULT_ROOT = r"G:\Shared drives\Product & Engineering\Projects\NABL Document Intelligence"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

_SYSTEM_PROMPT = (
    "You are an information-extraction system for laboratory/testing reports "
    "(milk, food, chemical, medical, and similar documents). Extract EVERY "
    "key-value pair actually present in the document — never invent, guess, "
    "or infer a value that isn't really there; if unsure, leave it out.\n\n"
    "Respond with ONLY a JSON object of this exact shape:\n"
    '{"fields": {"<snake_case_key>": "<value exactly as written>", ...}, '
    '"tests": [{"test_name": "<name>", "result": "<value as written>", '
    '"unit": "<unit or null>", "reference_range": "<range or null>"}, ...]}\n\n'
    '"fields" holds every header/metadata value (names, dates, addresses, '
    "report numbers, sample details, etc.) using natural snake_case keys "
    'derived from the document\'s own labels. "tests" holds every row of a '
    "results table. Use an empty list for \"tests\" if the document has no "
    "tabular results."
)


def _find_source_file(dataset_dir: Path, filename: str) -> Path | None:
    candidate = dataset_dir / filename
    return candidate if candidate.is_file() else None


def _build_message_content(source_path: Path) -> tuple[list[dict], bytes | None, str | None]:
    """Returns (text_content_blocks, single_image_bytes, media_type) — Nova's
    content list mixes text and image blocks freely, so a page with a real
    text layer contributes text, one without contributes its rasterized
    image, in document order."""
    suffix = source_path.suffix.lower()
    data = source_path.read_bytes()

    if suffix in IMAGE_EXTENSIONS:
        media_type = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
        return [], data, media_type

    if suffix != ".pdf":
        raise ValueError(f"unsupported file type: {suffix}")

    pages = pdf_utils.extract_text_and_boxes(data)
    count = pdf_utils.page_count(data)
    text_blocks: list[dict] = []
    image_bytes: bytes | None = None
    for i in range(count):
        page = next((p for p in pages if p.page_number == i), None)
        text = page.text if page else ""
        if len(text.strip()) >= 20:
            text_blocks.append({"text": f"--- Page {i + 1} ---\n{text}"})
        else:
            # No usable text layer on this page — rasterize and attach as an
            # image instead of running local OCR. Bedrock's Converse API
            # accepts multiple image blocks in one message; kept simple here
            # by taking only the first such page per document (a document
            # mixing several image-only pages is rare in this dataset and
            # multi-image requests cost more without much extra signal).
            if image_bytes is None:
                image_bytes = pdf_utils.rasterize_page(data, i)
    return text_blocks, image_bytes, "image/png" if image_bytes else None


def _predict_one(chain: LlmChain, source_path: Path) -> dict:
    text_blocks, image_bytes, media_type = _build_message_content(source_path)
    user_text = "\n\n".join(b["text"] for b in text_blocks) if text_blocks else (
        "(This document has no extractable text layer — read the attached image.)"
    )
    result = chain.generate_json(_SYSTEM_PROMPT, user_text, image=image_bytes, image_media_type=media_type)
    return {
        "fields": result.get("fields", {}) if isinstance(result, dict) else {},
        "tests": result.get("tests", []) if isinstance(result, dict) else [],
        "pipeline_error": None,
    }


def generate_predictions(labelled_dir: Path, dataset_dir: Path, predictions_dir: Path, chain: LlmChain, force: bool = False) -> dict:
    predictions_dir.mkdir(parents=True, exist_ok=True)
    stats = {"total": 0, "predicted": 0, "skipped": 0, "failed": 0, "failures": []}

    for label_path in sorted(labelled_dir.glob("*/*.json")):
        stem = label_path.stem
        out_path = predictions_dir / f"{stem}.json"
        stats["total"] += 1
        if out_path.exists() and not force:
            # A cached FAILURE is not "already done" — a timeout or a
            # transient chain cooldown is worth retrying automatically on
            # the next run; only a genuine success is skipped. --force
            # still means "redo everything, successes included".
            cached = json.loads(out_path.read_text(encoding="utf-8"))
            if cached.get("pipeline_error") is None:
                stats["skipped"] += 1
                continue

        ground_truth = json.loads(label_path.read_text(encoding="utf-8"))
        source_path = _find_source_file(dataset_dir, ground_truth["original_filename"])
        if source_path is None:
            prediction = {"fields": {}, "tests": [], "pipeline_error": f"source file not found: {ground_truth['original_filename']}"}
        else:
            try:
                prediction = _predict_one(chain, source_path)
            except Exception as exc:  # noqa: BLE001 — one bad document must never stop the batch
                prediction = {"fields": {}, "tests": [], "pipeline_error": f"{type(exc).__name__}: {exc}"}

        out_path.write_text(json.dumps(prediction, indent=2, ensure_ascii=False), encoding="utf-8")
        if prediction["pipeline_error"]:
            stats["failed"] += 1
            stats["failures"].append((stem, prediction["pipeline_error"]))
        else:
            stats["predicted"] += 1

    return stats


def score(labelled_dir: Path, predictions_dir: Path) -> tuple[dict, list[dict]]:
    overall = MetricAccumulator()
    rows: list[dict] = []

    for label_path in sorted(labelled_dir.glob("*/*.json")):
        stem = label_path.stem
        ground_truth = json.loads(label_path.read_text(encoding="utf-8"))
        pred_path = predictions_dir / f"{stem}.json"

        if pred_path.exists():
            prediction = json.loads(pred_path.read_text(encoding="utf-8"))
        else:
            prediction = {"fields": {}, "tests": [], "pipeline_error": "no cached prediction found"}

        extraction_ok = prediction.get("pipeline_error") is None
        field_failures, field_counters = compare_fields(stem, ground_truth.get("fields", {}), prediction.get("fields", {}))
        test_failures, test_counters = compare_tests(stem, ground_truth.get("tests", []), prediction.get("tests", []))
        exact_match = extraction_ok and not field_failures and not test_failures

        doc_acc = MetricAccumulator()
        doc_acc.add(domain_match=True, extraction_ok=extraction_ok, exact_match=exact_match,
                    field_counters=field_counters, test_counters=test_counters)
        overall.add(domain_match=True, extraction_ok=extraction_ok, exact_match=exact_match,
                    field_counters=field_counters, test_counters=test_counters)

        rows.append({
            "document": stem,
            "original_filename": ground_truth.get("original_filename"),
            "extraction_status": "ok" if extraction_ok else "failed",
            "pipeline_error": prediction.get("pipeline_error"),
            **doc_acc.finalize(),
        })

    return overall.finalize(), rows


def write_results(summary: dict, rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    import csv
    if rows:
        with (output_dir / "per_document.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labelled", default=os.path.join(DEFAULT_ROOT, "labelled_dataset"), help="Ground truth directory (create_ground_truth.py's output).")
    parser.add_argument("--dataset", default=os.path.join(DEFAULT_ROOT, "dataset"), help="Raw source files directory.")
    parser.add_argument("--output", default=None, help="Output directory for predictions/ and summary. Defaults to a 'predictions' folder next to --labelled.")
    parser.add_argument("--force", action="store_true", help="Re-run the LLM call even for already-predicted documents.")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument(
        "--timeout", type=float, default=180.0,
        help="Per-call timeout in seconds (default: 180). The app's usual LLM_TIMEOUT_SECONDS (30s, tuned for "
             "short chat-style calls elsewhere) is too short here — a document with a genuinely large results "
             "table (see high-protein-paneer.pdf's 269 test rows) needs real headroom to both generate and "
             "transfer a long JSON reply.",
    )
    args = parser.parse_args()

    labelled_dir = Path(args.labelled)
    if not labelled_dir.is_dir():
        print(f"Labelled dataset not found: {labelled_dir}", file=sys.stderr)
        return 1

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output) if args.output else labelled_dir.parent / "predictions"
    predictions_dir = output_dir / "predictions"

    settings = get_settings()
    if not settings.nova_model:
        print("NOVA_MODEL is not configured in .env — see NovaProvider in app/llm/providers.py.", file=sys.stderr)
        return 1
    chain = LlmChain([NovaProvider(settings.nova_model, settings.nova_region, args.timeout, max_tokens=settings.nova_max_tokens)])

    print(f"Labelled dataset: {labelled_dir}")
    print(f"Dataset (raw):    {dataset_dir}")
    print(f"Output:           {output_dir}")

    if not args.score_only:
        stats = generate_predictions(labelled_dir, dataset_dir, predictions_dir, chain, force=args.force)
        print()
        print("=== Prediction Summary ===")
        print(f"Total documents: {stats['total']}")
        print(f"Predicted (new): {stats['predicted']}")
        print(f"Skipped (cached): {stats['skipped']}")
        print(f"Failed:          {stats['failed']}")
        for name, reason in stats["failures"]:
            print(f"  {name}: {reason}")

    if not args.predict_only:
        summary, rows = score(labelled_dir, predictions_dir)
        write_results(summary, rows, output_dir)
        print()
        print("=== Score Summary ===")
        for key, value in summary.items():
            print(f"{key}: {value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
