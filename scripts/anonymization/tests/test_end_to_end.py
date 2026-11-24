"""End-to-end tests over real PDF files.

These carry the weight of the suite. The unit tests prove the detectors agree
with their own rules; these prove the thing an operator actually cares about:
the identifiers are gone from the delivered file, the laboratory values are
still in it, and the original on the Drive is untouched.

The text-removal assertion is the important one. A redaction that draws a black
rectangle and leaves the glyphs underneath looks perfect to the eye and fails
completely — so the test extracts text from the output and asserts absence,
rather than inspecting pixels.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import fitz
import pytest

from anonymizer.config import AnonymizerConfig
from anonymizer.ocr_engine import NullOcrEngine
from anonymizer.report import STATUS_ERROR, STATUS_OK, STATUS_SKIPPED
from anonymizer.runner import Anonymizer, UnsafePathError, discover, ensure_safe_roots, run
from conftest import (
    PII_AADHAAR,
    PII_EMAIL,
    PII_NAME,
    PII_PAN,
    PII_PHONE,
    make_text_pdf,
)


def text_of(pdf: Path) -> str:
    doc = fitz.open(pdf)
    try:
        return "\n".join(doc[i].get_text("text") for i in range(doc.page_count))
    finally:
        doc.close()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def no_ocr_config() -> AnonymizerConfig:
    """OCR off: these fixtures are born-digital, and loading ONNX models would
    make the suite slow and environment-dependent."""
    cfg = AnonymizerConfig()
    cfg.ocr.enabled = False
    cfg.verification.check_ocr = False
    return cfg


class TestSafety:
    def test_output_inside_input_is_refused(self, tmp_path):
        src = tmp_path / "in"
        src.mkdir()
        with pytest.raises(UnsafePathError):
            ensure_safe_roots(src, src / "out")

    def test_input_inside_output_is_refused(self, tmp_path):
        dst = tmp_path / "out"
        dst.mkdir()
        with pytest.raises(UnsafePathError):
            ensure_safe_roots(dst / "in", dst)

    def test_identical_roots_are_refused(self, tmp_path):
        with pytest.raises(UnsafePathError):
            ensure_safe_roots(tmp_path, tmp_path)

    def test_separate_roots_are_allowed(self, tmp_path):
        ensure_safe_roots(tmp_path / "in", tmp_path / "out")


class TestDiscovery:
    def test_finds_pdfs_recursively_and_ignores_other_files(self, input_root):
        (input_root / "notes.txt").write_text("ignore me", encoding="utf-8")
        (input_root / "~$temp.pdf").write_bytes(b"%PDF-fake")
        found = {p.name for p in discover(input_root)}
        assert found == {"report.pdf", "nested.pdf"}


class TestTextLayerAnonymisation:
    def test_pii_is_removed_and_lab_values_survive(self, text_pdf, tmp_path, no_ocr_config):
        out_root = tmp_path / "out"
        anonymiser = Anonymizer(config=no_ocr_config, ocr=NullOcrEngine())
        report = anonymiser.process_file(text_pdf, text_pdf.parent, out_root)

        assert report.status == STATUS_OK, report.error
        out_text = text_of(Path(report.output_path))

        # The identifiers must be gone from the TEXT, not merely covered.
        for secret in (PII_AADHAAR, PII_AADHAAR.replace(" ", ""), PII_PAN,
                       PII_EMAIL, PII_PHONE, PII_NAME):
            assert secret not in out_text, f"{secret!r} survived in the output"

        # The report must remain a usable laboratory document.
        assert "Haemoglobin" in out_text
        assert "12.4" in out_text
        assert "13.0-17.0" in out_text
        assert "Complete Blood Count" in out_text
        assert "ACME DIAGNOSTICS LABORATORY" in out_text

    def test_original_file_is_never_modified(self, text_pdf, tmp_path, no_ocr_config):
        before = digest(text_pdf)
        anonymiser = Anonymizer(config=no_ocr_config, ocr=NullOcrEngine())
        anonymiser.process_file(text_pdf, text_pdf.parent, tmp_path / "out")
        assert digest(text_pdf) == before

    def test_label_mode_stamps_a_placeholder(self, text_pdf, tmp_path, no_ocr_config):
        no_ocr_config.redaction.mode = "label"
        anonymiser = Anonymizer(config=no_ocr_config, ocr=NullOcrEngine())
        report = anonymiser.process_file(text_pdf, text_pdf.parent, tmp_path / "out")
        out_text = text_of(Path(report.output_path))
        assert "[" in out_text and "]" in out_text

    def test_black_box_mode_leaves_no_placeholder_text(self, text_pdf, tmp_path, no_ocr_config):
        no_ocr_config.redaction.mode = "black_box"
        anonymiser = Anonymizer(config=no_ocr_config, ocr=NullOcrEngine())
        report = anonymiser.process_file(text_pdf, text_pdf.parent, tmp_path / "out")
        out_text = text_of(Path(report.output_path))
        assert "[PATIENT_NAME]" not in out_text
        assert PII_NAME not in out_text

    def test_document_with_no_pii_is_still_copied(self, tmp_path, no_ocr_config):
        src_root = tmp_path / "in"
        src_root.mkdir()
        clean = make_text_pdf(src_root / "clean.pdf", with_pii=False)
        anonymiser = Anonymizer(config=no_ocr_config, ocr=NullOcrEngine())
        report = anonymiser.process_file(clean, src_root, tmp_path / "out")
        assert report.status == STATUS_OK
        assert Path(report.output_path).is_file()
        assert "Haemoglobin" in text_of(Path(report.output_path))


class TestReportingAndStatus:
    def test_report_does_not_contain_pii_by_default(self, text_pdf, tmp_path, no_ocr_config):
        anonymiser = Anonymizer(config=no_ocr_config, ocr=NullOcrEngine())
        report = anonymiser.process_file(text_pdf, text_pdf.parent, tmp_path / "out")
        blob = json.dumps(report.to_dict())
        for secret in (PII_AADHAAR, PII_PAN, PII_EMAIL, PII_PHONE, PII_NAME):
            assert secret not in blob, "the report itself leaked PII"

    def test_samples_are_truncated_when_explicitly_enabled(self, text_pdf, tmp_path, no_ocr_config):
        no_ocr_config.include_pii_samples = True
        anonymiser = Anonymizer(config=no_ocr_config, ocr=NullOcrEngine())
        report = anonymiser.process_file(text_pdf, text_pdf.parent, tmp_path / "out")
        blob = json.dumps(report.to_dict())
        assert PII_EMAIL not in blob          # never the whole value
        assert "chars)" in blob                # but a length fingerprint is present

    def test_detections_are_counted_by_kind(self, text_pdf, tmp_path, no_ocr_config):
        anonymiser = Anonymizer(config=no_ocr_config, ocr=NullOcrEngine())
        report = anonymiser.process_file(text_pdf, text_pdf.parent, tmp_path / "out")
        assert report.total_detections > 0
        assert set(report.detections_by_kind) & {"AADHAAR", "PAN", "EMAIL", "PHONE"}

    def test_existing_output_is_skipped_unless_overwrite(self, text_pdf, tmp_path, no_ocr_config):
        out_root = tmp_path / "out"
        anonymiser = Anonymizer(config=no_ocr_config, ocr=NullOcrEngine())
        first = anonymiser.process_file(text_pdf, text_pdf.parent, out_root)
        assert first.status == STATUS_OK

        second = anonymiser.process_file(text_pdf, text_pdf.parent, out_root)
        assert second.status == STATUS_SKIPPED

        third = anonymiser.process_file(text_pdf, text_pdf.parent, out_root, overwrite=True)
        assert third.status == STATUS_OK

    def test_corrupt_file_is_reported_not_raised(self, tmp_path, no_ocr_config):
        src_root = tmp_path / "in"
        src_root.mkdir()
        broken = src_root / "broken.pdf"
        broken.write_bytes(b"%PDF-1.4 this is not a pdf")
        anonymiser = Anonymizer(config=no_ocr_config, ocr=NullOcrEngine())
        report = anonymiser.process_file(broken, src_root, tmp_path / "out")
        assert report.status == STATUS_ERROR
        assert report.error
        # No half-anonymised artefact may be left behind.
        assert not (tmp_path / "out" / "broken.pdf").exists()


class TestRunOrchestration:
    def test_folder_structure_is_preserved(self, input_root, tmp_path, no_ocr_config):
        out_root = tmp_path / "out"
        report = run(input_root, out_root, no_ocr_config, use_ocr=False)
        assert (out_root / "report.pdf").is_file()
        assert (out_root / "sub" / "nested.pdf").is_file()
        assert len(report.documents) == 2

    def test_run_writes_per_document_and_summary_reports(self, input_root, tmp_path, no_ocr_config):
        out_root = tmp_path / "out"
        report = run(input_root, out_root, no_ocr_config, use_ocr=False)
        reports_dir = out_root / "_anonymisation_reports"
        assert (reports_dir / "report.report.json").is_file()
        assert (reports_dir / "sub" / "nested.report.json").is_file()

        report.write(reports_dir / "run_report.json")
        payload = json.loads((reports_dir / "run_report.json").read_text(encoding="utf-8"))
        assert payload["totals"]["documents"] == 2

    def test_originals_untouched_after_a_full_run(self, input_root, tmp_path, no_ocr_config):
        before = {p: digest(p) for p in discover(input_root)}
        run(input_root, tmp_path / "out", no_ocr_config, use_ocr=False)
        assert {p: digest(p) for p in discover(input_root)} == before
