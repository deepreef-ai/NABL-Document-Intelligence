#!/usr/bin/env python3
"""Step 1 of the 2-script pipeline: prepares raw documents in dataset/ for
DIRECT (LLM-free) labeling — no Gemini/Groq/Ollama/Nova call anywhere in
this script. Ground truth here means a human (or Claude, reading each
document directly) writes down what's actually on the page; nothing about
that judgment call can be automated, so this script does only the
mechanical part:

  1. Finds every raw file in dataset/ that has no corresponding
     labelled_dataset/<name>/<name>.json yet.
  2. For each one, extracts its real text (PyMuPDF, if it has a text layer)
     or rasterizes its pages to PNGs (if not) into a staging folder, so
     whoever labels it next has the material ready to read without
     re-deriving it by hand.
  3. Prints a to-do list of exactly what's missing and where its staged
     material landed.

The actual labeling step is: read the staged text/images for each listed
document, and write labelled_dataset/<name>/<name>.json shaped like:
    {"original_filename": "...", "fields": {...}, "tests": [...]}
by hand (or have Claude do it, the same way this project's existing
labelled_dataset/ entries were produced).

Usage:
    python scripts/create_ground_truth.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/ on sys.path

from app.documents import pdf_utils  # noqa: E402

DEFAULT_ROOT = r"G:\Shared drives\Product & Engineering\Projects\NABL Document Intelligence"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _stage_pdf(source_path: Path, staging_dir: Path) -> tuple[str, list[str]]:
    data = source_path.read_bytes()
    pages = pdf_utils.extract_text_and_boxes(data)
    count = pdf_utils.page_count(data)

    full_text_parts = []
    image_paths: list[str] = []
    for i in range(count):
        page = next((p for p in pages if p.page_number == i), None)
        text = page.text if page else ""
        if len(text.strip()) >= 20:
            full_text_parts.append(f"--- Page {i + 1} ---\n{text}")
        else:
            png_bytes = pdf_utils.rasterize_page(data, i)
            image_path = staging_dir / f"{source_path.stem}_p{i}.png"
            image_path.write_bytes(png_bytes)
            image_paths.append(str(image_path))

    full_text = "\n\n".join(full_text_parts)
    if full_text:
        text_path = staging_dir / f"{source_path.stem}.txt"
        text_path.write_text(full_text, encoding="utf-8")
    return full_text, image_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=os.path.join(DEFAULT_ROOT, "dataset"), help="Raw source files directory.")
    parser.add_argument("--labelled", default=os.path.join(DEFAULT_ROOT, "labelled_dataset"), help="Ground truth directory — existing entries here are skipped.")
    parser.add_argument("--staging", default=None, help="Where to write extracted text/rendered page images for review. Defaults to a 'labelling_staging' folder next to --dataset.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    labelled_dir = Path(args.labelled)
    staging_dir = Path(args.staging) if args.staging else dataset_dir.parent / "labelling_staging"

    if not dataset_dir.is_dir():
        print(f"Dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 1

    already_labeled = {p.parent.name for p in labelled_dir.glob("*/*.json")} if labelled_dir.is_dir() else set()
    staging_dir.mkdir(parents=True, exist_ok=True)

    todo: list[tuple[str, str]] = []  # (filename, staged material description)
    for source_path in sorted(p for p in dataset_dir.rglob("*") if p.is_file()):
        stem = source_path.stem
        if stem in already_labeled:
            continue

        suffix = source_path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            # No extraction needed — the raw image itself is the material.
            todo.append((source_path.name, f"read directly: {source_path}"))
        elif suffix == ".pdf":
            text, image_paths = _stage_pdf(source_path, staging_dir)
            if text and not image_paths:
                todo.append((source_path.name, f"text staged at {staging_dir / (stem + '.txt')}"))
            elif image_paths and not text:
                todo.append((source_path.name, f"{len(image_paths)} page image(s) staged in {staging_dir}"))
            else:
                todo.append((source_path.name, f"text + {len(image_paths)} page image(s) staged in {staging_dir}"))
        else:
            todo.append((source_path.name, f"SKIPPED — unsupported extension {suffix!r}"))

    print(f"Dataset:  {dataset_dir}")
    print(f"Labelled: {labelled_dir}")
    print(f"Staging:  {staging_dir}")
    print()
    print(f"Already labeled: {len(already_labeled)}")
    print(f"Needs labeling:  {len(todo)}")
    if todo:
        print()
        print("--- To label ---")
        for filename, note in todo:
            print(f"  {filename}: {note}")
        print()
        print(f"Next step: read each file's staged material above and write "
              f"{labelled_dir}\\<name>\\<name>.json by hand — see this script's own docstring for the exact shape.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
