"""Orchestration: walk a folder, anonymise each file, verify, report.

Safety rules enforced here rather than trusted to the caller:

* The output root may not sit inside the input root, and vice versa. A nested
  output would be re-scanned on the next run, so anonymised copies would be
  anonymised again and — worse — a mistake in path handling could write into
  the source tree.
* Input files are opened read-only and never written to. The output is a new
  file at a mirrored relative path.
* An existing output is not overwritten unless `--overwrite` is passed, so a
  half-finished run cannot silently destroy a reviewed batch.
* Any per-file exception is caught, recorded as ERROR, and the walk continues.
  One corrupt scan must not abandon the other four hundred.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np

from .config import IMAGE_SUFFIXES, PDF_SUFFIXES, AnonymizerConfig
from .detect import Detection, detect_page
from .ocr_engine import NullOcrEngine, OcrEngine, OcrUnavailable
from .protect import ProtectedVocabulary
from .redact import (
    build_redacted_pdf,
    image_to_array,
    load_image,
    paint_boxes_on_image,
    plan_from_detections,
)
from .report import (
    STATUS_ERROR,
    STATUS_FAILED_REVIEW,
    STATUS_OK,
    STATUS_SKIPPED,
    DetectionRecord,
    DocumentReport,
    PageSummary,
    RunReport,
)
from .textsource import words_from_ocr, words_from_pdf_page
from .verify import verify_image, verify_pdf

log = logging.getLogger(__name__)


class UnsafePathError(ValueError):
    """Raised when input and output roots overlap."""


def ensure_safe_roots(input_root: Path, output_root: Path) -> None:
    src = input_root.resolve()
    dst = output_root.resolve()
    if src == dst:
        raise UnsafePathError("input and output folders are the same; originals must never be overwritten")
    if dst.is_relative_to(src):
        raise UnsafePathError(
            f"output folder {dst} is inside the input folder {src}; choose a location outside it"
        )
    if src.is_relative_to(dst):
        raise UnsafePathError(
            f"input folder {src} is inside the output folder {dst}; choose separate locations"
        )


def discover(input_root: Path) -> list[Path]:
    """Every processable file under the root, in a stable order."""
    allowed = set(IMAGE_SUFFIXES) | set(PDF_SUFFIXES)
    files = [
        p for p in sorted(input_root.rglob("*"))
        if p.is_file() and p.suffix.lower() in allowed and not p.name.startswith(("~$", "."))
    ]
    return files


def file_type_of(path: Path) -> str:
    return "pdf" if path.suffix.lower() in PDF_SUFFIXES else "image"


@dataclass
class Anonymizer:
    config: AnonymizerConfig
    ocr: OcrEngine

    def __post_init__(self) -> None:
        self.vocab = ProtectedVocabulary(self.config.extra_protected_terms)

    # ---------------------------------------------------------------- PDF
    def _image_coverage(self, page) -> float:
        """Fraction of the page covered by embedded images."""
        try:
            page_area = abs(page.rect.get_area())
            if not page_area:
                return 0.0
            covered = 0.0
            for info in page.get_images(full=True):
                for rect in page.get_image_rects(info[0]):
                    covered += abs(rect.get_area())
            return min(covered / page_area, 1.0)
        except Exception:  # noqa: BLE001 — a probe must never fail the page
            return 0.0

    def _ocr_words_outside_text(self, page, index: int, text_words):
        """OCR a text-layer page and return only what the text layer lacks.

        Boxes come back in PDF points so they can be redacted by the same
        annotation path as the text layer — which also blanks the underlying
        image pixels, so the letterhead raster is genuinely cleared rather than
        merely covered.
        """
        cfg = self.config
        zoom = cfg.ocr.dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        from PIL import Image

        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        lines = self.ocr.read(np.array(image))
        ocr_page = words_from_ocr(lines, index, pix.width, pix.height)

        existing = [w.bbox for w in text_words]

        def already_in_text(bbox) -> bool:
            cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            return any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in existing)

        kept = []
        for w in ocr_page.words:
            points = tuple(v / zoom for v in w.bbox)
            if already_in_text(points):
                continue
            kept.append(replace(w, bbox=points, line=w.line + 500_000))
        return kept, ocr_page.ocr_mean_confidence

    def _process_pdf(self, src: Path, dst: Path, report: DocumentReport) -> None:
        cfg = self.config
        doc = fitz.open(src)
        plans: dict[int, object] = {}
        try:
            report.page_count = doc.page_count
            for index in range(doc.page_count):
                page = doc[index]
                words = words_from_pdf_page(page, index)
                summary = PageSummary(page=index, source=words.source)

                usable = len(words.text.strip()) >= cfg.ocr.min_page_text_chars
                if usable and cfg.ocr.enabled and cfg.ocr.augment_text_pages:
                    coverage = self._image_coverage(page)
                    if coverage >= cfg.ocr.image_area_threshold:
                        try:
                            extra, conf = self._ocr_words_outside_text(page, index, words.words)
                            if extra:
                                words.words.extend(extra)
                                summary.note = (
                                    f"{len(extra)} word(s) read from embedded images "
                                    f"({coverage:.0%} of the page)")
                                summary.ocr_mean_confidence = round(conf, 4)
                        except OcrUnavailable as exc:
                            report.warnings.append(
                                f"page {index + 1}: images cover {coverage:.0%} of the page but "
                                f"could not be read ({exc}) — PII inside them would not be found")
                if not usable:
                    # No real text layer: this page has to be read from pixels.
                    try:
                        zoom = cfg.ocr.dpi / 72.0
                        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                        from PIL import Image

                        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                        lines = self.ocr.read(np.array(image))
                        words = words_from_ocr(lines, index, pix.width, pix.height)
                        summary.source = "ocr"
                        summary.ocr_mean_confidence = round(words.ocr_mean_confidence, 4)
                        summary.low_confidence = words.ocr_mean_confidence < cfg.ocr.low_confidence
                        if summary.low_confidence:
                            msg = (f"page {index + 1}: OCR confidence "
                                   f"{words.ocr_mean_confidence:.2f} is low — a missed identifier "
                                   f"is more likely on this page")
                            report.warnings.append(msg)
                    except OcrUnavailable as exc:
                        summary.source = "unreadable"
                        summary.note = str(exc)
                        report.warnings.append(f"page {index + 1}: {exc}")
                        report.pages.append(summary)
                        continue

                detections = detect_page(words, cfg.detection, self.vocab,
                                         cfg.extra_patient_labels)
                summary.detections = len(detections)
                self._record(report, detections)

                # Text-layer boxes are PDF points (72 dpi); OCR boxes are pixels
                # at the DPI the page was recognised at. Passing the real unit
                # is what stops the rebuild step scaling them a second time.
                source_dpi = float(cfg.ocr.dpi) if summary.source == "ocr" else 72.0
                if detections:
                    plans[index] = plan_from_detections(
                        detections, index, summary.source, source_dpi)
                elif summary.source == "ocr":
                    # Rebuild scanned pages even when nothing was detected, so
                    # the original scan image never carries over to the output.
                    plans[index] = plan_from_detections([], index, "ocr", source_dpi)
                report.pages.append(summary)

            pdf_bytes, applied = build_redacted_pdf(str(src), plans, cfg.redaction)  # type: ignore[arg-type]
            report.total_redactions = applied
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(pdf_bytes)
        finally:
            doc.close()

        for summary in report.pages:
            plan = plans.get(summary.page)
            if plan is not None:
                summary.redactions_applied = len(getattr(plan, "boxes", []))

    # -------------------------------------------------------------- image
    def _process_image(self, src: Path, dst: Path, report: DocumentReport) -> None:
        cfg = self.config
        image = load_image(src)
        report.page_count = 1
        summary = PageSummary(page=0, source="ocr")
        try:
            lines = self.ocr.read(image_to_array(image))
        except OcrUnavailable as exc:
            summary.source = "unreadable"
            summary.note = str(exc)
            report.pages.append(summary)
            raise

        words = words_from_ocr(lines, 0, image.width, image.height)
        summary.ocr_mean_confidence = round(words.ocr_mean_confidence, 4)
        summary.low_confidence = words.ocr_mean_confidence < cfg.ocr.low_confidence
        if summary.low_confidence:
            report.warnings.append(
                f"OCR confidence {words.ocr_mean_confidence:.2f} is low — a missed "
                f"identifier is more likely on this image")

        detections = detect_page(words, cfg.detection, self.vocab, cfg.extra_patient_labels)
        summary.detections = len(detections)
        self._record(report, detections)

        boxes = [(b, d.kind) for d in detections for b in d.bboxes]
        redacted, painted = paint_boxes_on_image(image, boxes, cfg.redaction)
        summary.redactions_applied = painted
        report.total_redactions = painted
        report.pages.append(summary)

        dst.parent.mkdir(parents=True, exist_ok=True)
        # PNG for anything that was not already JPEG: re-encoding a redacted
        # JPEG repeatedly would degrade the report image for no benefit.
        if dst.suffix.lower() in (".jpg", ".jpeg"):
            redacted.save(dst, format="JPEG", quality=95, subsampling=0)
        elif dst.suffix.lower() in (".tif", ".tiff"):
            redacted.save(dst, format="TIFF")
        else:
            redacted.save(dst, format="PNG", optimize=True)

    # ------------------------------------------------------------- shared
    def _record(self, report: DocumentReport, detections: list[Detection]) -> None:
        report.total_detections += len(detections)
        for d in detections:
            report.detections_by_kind[d.kind] = report.detections_by_kind.get(d.kind, 0) + 1
            sample = ""
            if self.config.include_pii_samples:
                sample = f"{d.text[:2]}… ({len(d.text)} chars)"
            for bbox in d.bboxes[:1]:  # first box is enough to locate the finding
                report.detections.append(DetectionRecord(
                    kind=d.kind, page=d.page, rule=d.rule, source=d.source,
                    confidence=round(d.confidence, 4),
                    bbox=tuple(round(v, 2) for v in bbox),  # type: ignore[arg-type]
                    sample=sample,
                ))

    def process_file(self, src: Path, input_root: Path, output_root: Path,
                     overwrite: bool = False) -> DocumentReport:
        rel = src.relative_to(input_root)
        dst = output_root / rel
        report = DocumentReport(
            source_path=str(src), output_path=str(dst), relative_path=str(rel),
            file_type=file_type_of(src),
            started_at=datetime.now(timezone.utc).isoformat(),
            contains_pii_samples=self.config.include_pii_samples,
        )
        started = time.monotonic()

        if dst.exists() and not overwrite:
            report.status = STATUS_SKIPPED
            report.error = "output already exists; pass --overwrite to replace it"
            report.finished_at = datetime.now(timezone.utc).isoformat()
            return report

        try:
            if report.file_type == "pdf":
                self._process_pdf(src, dst, report)
            else:
                self._process_image(src, dst, report)
        except Exception as exc:  # noqa: BLE001 — one bad file must not stop the run
            log.exception("failed to anonymise %s", src)
            report.status = STATUS_ERROR
            report.error = f"{type(exc).__name__}: {exc}"
            report.finished_at = datetime.now(timezone.utc).isoformat()
            report.duration_seconds = round(time.monotonic() - started, 3)
            # A partial output is worse than none: it looks anonymised.
            if dst.exists():
                try:
                    dst.unlink()
                    report.warnings.append("partial output removed")
                except OSError:
                    report.warnings.append("PARTIAL OUTPUT COULD NOT BE REMOVED — delete it by hand")
            return report

        # ---- verification, on the file that will actually be handed over
        ocr_for_verify = None if isinstance(self.ocr, NullOcrEngine) else self.ocr
        if report.file_type == "pdf":
            result = verify_pdf(dst, self.config, self.vocab, ocr_for_verify)
        else:
            result = verify_image(dst, self.config, self.vocab, ocr_for_verify)

        if result.checked:
            report.verification_passed = result.passed
            report.verification_summary = result.summary
            report.verification_findings = [
                {"kind": f.kind, "channel": f.channel, "page": f.page, "sample": f.sample}
                for f in result.findings
            ]
            if not result.passed:
                report.status = STATUS_FAILED_REVIEW
        else:
            report.verification_summary = result.summary

        report.finished_at = datetime.now(timezone.utc).isoformat()
        report.duration_seconds = round(time.monotonic() - started, 3)
        return report


def run(
    input_root: Path, output_root: Path, config: AnonymizerConfig,
    *, use_ocr: bool = True, overwrite: bool = False, limit: int | None = None,
    reports_dir: Path | None = None,
) -> RunReport:
    ensure_safe_roots(input_root, output_root)
    if not input_root.is_dir():
        raise FileNotFoundError(f"input folder not found: {input_root}")

    files = discover(input_root)
    if limit:
        files = files[:limit]
    log.info("found %d file(s) under %s", len(files), input_root)

    engine: OcrEngine = OcrEngine() if (use_ocr and config.ocr.enabled) else NullOcrEngine()
    anonymiser = Anonymizer(config=config, ocr=engine)
    run_report = RunReport(
        input_root=str(input_root.resolve()), output_root=str(output_root.resolve()),
        config=config.to_dict(),
    )
    reports_dir = reports_dir or (output_root / "_anonymisation_reports")

    for i, src in enumerate(files, start=1):
        log.info("[%d/%d] %s", i, len(files), src.relative_to(input_root))
        doc_report = anonymiser.process_file(src, input_root, output_root, overwrite)
        run_report.documents.append(doc_report)

        report_path = reports_dir / Path(doc_report.relative_path).with_suffix(".report.json")
        try:
            doc_report.write(report_path)
        except OSError as exc:
            log.warning("could not write report for %s: %s", src, exc)

        level = logging.INFO if doc_report.status == STATUS_OK else logging.WARNING
        log.log(level, "    %s — %d detection(s), %d redaction(s)%s",
                doc_report.status, doc_report.total_detections,
                doc_report.total_redactions,
                f" — {doc_report.error}" if doc_report.error else "")

    run_report.finished_at = datetime.now(timezone.utc).isoformat()
    return run_report
