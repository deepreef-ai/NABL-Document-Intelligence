"""Orchestrates a full dataset run: discover files -> assign/resolve
document_id -> skip if already processed (unless --force) -> normalize ->
write normalized/<document_id>/document.json -> append to
processing_index.jsonl -> accumulate summary stats.

One bad document is caught and recorded as status="failed"; the loop always
continues to the next file — the batch as a whole never aborts because of
one corrupted/unsupported/failed input.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.dataset_normalization.discovery import IdRegistry, content_hash, discover_files
from app.dataset_normalization.models import IndexEntry, NormalizedDocument
from app.dataset_normalization.normalize import UnsupportedOrInvalidFile, normalize_file
from app.dataset_normalization.text_quality import TextQualityThresholds


@dataclass
class RunStats:
    total_discovered: int = 0
    processed: int = 0
    failed: int = 0
    skipped: int = 0
    pdf_files: int = 0
    image_files: int = 0
    born_digital_pdf_pages: int = 0
    scanned_pdf_pages: int = 0  # PDF pages only, not counting standalone images
    ocr_pages: int = 0  # every page that went through OCR: scanned PDF pages + image files
    pymupdf_pages: int = 0
    mixed_pdfs: int = 0
    _total_pages_for_avg: int = 0
    _total_duration_for_avg: float = 0.0
    _docs_for_avg: int = 0
    failures: list[tuple[str, str, str]] = field(default_factory=list)  # (document_id, filename, reason)

    @property
    def average_pages_per_document(self) -> float:
        return self._total_pages_for_avg / self._docs_for_avg if self._docs_for_avg else 0.0

    @property
    def average_processing_time_seconds(self) -> float:
        return self._total_duration_for_avg / self._docs_for_avg if self._docs_for_avg else 0.0


def _doc_dir(output_dir: Path, document_id: str) -> Path:
    return output_dir / "normalized" / document_id


def _existing_status(doc_dir: Path) -> str | None:
    doc_json = doc_dir / "document.json"
    if not doc_json.exists():
        return None
    try:
        return json.loads(doc_json.read_text(encoding="utf-8")).get("status")
    except Exception:  # noqa: BLE001 — a corrupt prior-run artifact just means "reprocess it"
        return None


def _failed_document(document_id: str, filename: str, source_format: str, reason: str) -> NormalizedDocument:
    return NormalizedDocument(
        document_id=document_id,
        original_filename=filename,
        source_path="",
        source_format=source_format,
        source_type="image" if source_format != "pdf" else "born_digital_pdf",  # placeholder; status=failed is what matters
        page_count=0,
        status="failed",
        pages=[],
        error=reason,
    )


def run(input_dir: Path, output_dir: Path, force: bool = False, thresholds: TextQualityThresholds | None = None) -> RunStats:
    thresholds = thresholds or TextQualityThresholds()
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = IdRegistry(output_dir / "id_registry.json")
    index_path = output_dir / "processing_index.jsonl"

    files = discover_files(input_dir)
    stats = RunStats(total_discovered=len(files))

    with index_path.open("a", encoding="utf-8") as index_file:
        for path in files:
            data = path.read_bytes()
            document_id = registry.get_or_assign(content_hash(data))
            doc_dir = _doc_dir(output_dir, document_id)

            if _existing_status(doc_dir) == "processed" and not force:
                stats.skipped += 1
                continue

            suffix = path.suffix.lstrip(".").lower()
            is_pdf = suffix == "pdf"
            if is_pdf:
                stats.pdf_files += 1
            else:
                stats.image_files += 1

            try:
                doc = normalize_file(document_id, path, data, thresholds)
                doc.source_path = str(path)
            except UnsupportedOrInvalidFile as exc:
                doc = _failed_document(document_id, path.name, suffix, str(exc))
                doc.source_path = str(path)
            except Exception as exc:  # noqa: BLE001 — one bad document must never stop the batch
                doc = _failed_document(document_id, path.name, suffix, f"{type(exc).__name__}: {exc}")
                doc.source_path = str(path)

            doc_dir.mkdir(parents=True, exist_ok=True)
            (doc_dir / "document.json").write_text(
                json.dumps(doc.to_json_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
            )

            extraction_method_summary: str | None = None
            ocr_used_summary = False

            if doc.status == "failed":
                stats.failed += 1
                stats.failures.append((document_id, path.name, doc.error or "unknown error"))
            else:
                stats.processed += 1
                methods = {p.extraction_method for p in doc.pages}
                extraction_method_summary = "mixed" if len(methods) > 1 else (next(iter(methods), None))
                ocr_used_summary = any(p.ocr_used for p in doc.pages)

                if doc.source_type == "mixed_pdf":
                    stats.mixed_pdfs += 1

                if is_pdf:
                    for p in doc.pages:
                        if p.extraction_method == "pymupdf":
                            stats.pymupdf_pages += 1
                            stats.born_digital_pdf_pages += 1
                        else:
                            stats.scanned_pdf_pages += 1
                            stats.ocr_pages += 1
                else:
                    stats.ocr_pages += doc.page_count  # every image page is an OCR page

                stats._total_pages_for_avg += doc.page_count
                stats._total_duration_for_avg += doc.processing_duration_seconds or 0.0
                stats._docs_for_avg += 1

            entry = IndexEntry(
                document_id=document_id,
                original_filename=path.name,
                source_format=doc.source_format,
                source_type=doc.source_type,
                page_count=doc.page_count,
                status=doc.status,
                extraction_method=extraction_method_summary,
                ocr_used=ocr_used_summary,
                error=doc.error,
                processing_duration_seconds=doc.processing_duration_seconds,
            )
            index_file.write(json.dumps(entry.to_json_dict(), ensure_ascii=False) + "\n")
            index_file.flush()

    return stats


def print_summary(stats: RunStats) -> None:
    print()
    print("=== Dataset Normalization Summary ===")
    print(f"Total files discovered:   {stats.total_discovered}")
    print(f"Successfully processed:   {stats.processed}")
    print(f"Failed:                   {stats.failed}")
    print(f"Skipped (already done):   {stats.skipped}")
    print(f"PDF files:                {stats.pdf_files}")
    print(f"Image files:              {stats.image_files}")
    print(f"Born-digital PDF pages:   {stats.born_digital_pdf_pages}")
    print(f"Scanned PDF pages:        {stats.scanned_pdf_pages}")
    print(f"OCR pages (all sources):  {stats.ocr_pages}")
    print(f"PyMuPDF pages:            {stats.pymupdf_pages}")
    print(f"Mixed PDFs:               {stats.mixed_pdfs}")
    print(f"Average pages/document:   {stats.average_pages_per_document:.2f}")
    print(f"Average processing time:  {stats.average_processing_time_seconds:.2f}s")
    if stats.failures:
        print()
        print("--- Failures ---")
        for document_id, filename, reason in stats.failures:
            print(f"  {document_id}  {filename!r}: {reason}")
