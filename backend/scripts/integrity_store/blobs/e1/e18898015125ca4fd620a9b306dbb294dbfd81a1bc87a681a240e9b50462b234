"""Offline benchmark stage (Step 9) — evaluates the SAME production
components used earlier in this pipeline (app/dataset_normalization's
PDF/OCR extraction, app/labeling's domain classification + field/key
extraction) against final_dataset/final_labeled_dataset.jsonl as ground
truth. For every approved document, the ORIGINAL raw file is re-run through
those exact functions fresh — nothing here is a second, separate
implementation of extraction or classification, and nothing here modifies
either the production functions or any label/dataset file.

No train/validation/test split — every approved document is scored, and
results are reported both overall and broken down by domain and by source
format.

Run via scripts/run_benchmark.py.
"""
