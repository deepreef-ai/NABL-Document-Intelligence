"""Offline dataset normalization stage — PDF/scanned-PDF/JPG/JPEG/PNG/TIFF in,
one common NormalizedDocument JSON structure out, for the Lab Report Document
Intelligence project's raw multi-domain dataset.

Deliberately independent of the live FastAPI app (routers/, models.py's DB
models, documents/pipeline.py's per-upload flow) — this is a standalone batch
tool over an external dataset directory, run via scripts/normalize_dataset.py.
It reuses documents/pdf_utils.py and documents/local_ocr.py exactly as they
already exist; nothing in either module was changed for this.
"""
