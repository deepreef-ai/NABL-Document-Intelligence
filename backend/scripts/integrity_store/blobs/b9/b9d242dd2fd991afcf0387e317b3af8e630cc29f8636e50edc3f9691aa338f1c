import io
import json

import fitz
import pytest
from PIL import Image

from app.dataset_normalization import normalize, pipeline
from app.dataset_normalization.discovery import IdRegistry, content_hash, discover_files
from app.dataset_normalization.text_quality import TextQualityThresholds, is_meaningful_page_text
from app.documents import local_ocr
from app.documents.geometry import Rect
from app.documents.ocr_client import OcrResult


# --------------------------------------------------------------------------- helpers

def _make_pdf(pages_text: list[str | None]) -> bytes:
    """pages_text[i] is the page's real text (None = a blank/scanned page —
    nothing drawn, so PyMuPDF extracts no text and it needs OCR)."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((50, 72), text, fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


def _make_image_bytes(fmt: str = "PNG") -> bytes:
    img = Image.new("RGB", (100, 60), "white")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


_FAKE_OCR_RESULT = OcrResult(
    text="Hemoglobin 13.5",
    lines=["Hemoglobin", "13.5"],
    confidence=0.93,
    boxes=[Rect(x=10, y=10, w=80, h=15), Rect(x=10, y=30, w=40, h=15)],
    model_used="rapidocr-default-en",
    region_count=2,
)


# --------------------------------------------------------------------------- text_quality

def test_meaningful_text_requires_enough_characters():
    assert not is_meaningful_page_text("Page 3", TextQualityThresholds())
    assert not is_meaningful_page_text("", TextQualityThresholds())


def test_meaningful_text_rejects_symbol_noise_despite_length():
    noise = "-" * 40  # 40 chars, clears min_chars, but zero alphanumeric content
    assert not is_meaningful_page_text(noise, TextQualityThresholds())


def test_meaningful_text_rejects_too_few_words_despite_length_and_alnum_ratio():
    # Long enough and alnum-heavy, but only one "word".
    text = "1234567890123456789012345678901234567890"
    assert not is_meaningful_page_text(text, TextQualityThresholds())


def test_meaningful_text_accepts_real_sentence():
    text = "Patient Name: John Doe. Hemoglobin result: 13.5 g/dL, within normal range."
    assert is_meaningful_page_text(text, TextQualityThresholds())


def test_meaningful_text_thresholds_are_configurable():
    short_text = "GST Number is 27ABCDE1234F1Z5"
    strict = TextQualityThresholds(min_chars=50)
    lenient = TextQualityThresholds(min_chars=10)
    assert not is_meaningful_page_text(short_text, strict)
    assert is_meaningful_page_text(short_text, lenient)


# --------------------------------------------------------------------------- discovery

def test_discover_files_finds_only_supported_extensions_recursively(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.pdf").write_bytes(b"x")
    (tmp_path / "b.JPG").write_bytes(b"x")  # case-insensitive
    (tmp_path / "sub" / "c.png").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")  # unsupported -> ignored
    (tmp_path / "archive.zip").write_bytes(b"x")  # unsupported -> ignored

    found = discover_files(tmp_path)
    names = sorted(p.name for p in found)
    assert names == ["a.pdf", "b.JPG", "c.png"]


def test_id_registry_gives_same_id_for_same_hash_and_persists_across_instances(tmp_path):
    registry_path = tmp_path / "id_registry.json"
    registry1 = IdRegistry(registry_path)
    id_a = registry1.get_or_assign("hash-a")
    id_b = registry1.get_or_assign("hash-b")
    assert id_a == "LR_000001"
    assert id_b == "LR_000002"
    assert registry1.get_or_assign("hash-a") == id_a  # stable within one instance

    # A fresh instance loading the same file must agree, and a genuinely new
    # hash must continue the sequence rather than restart it.
    registry2 = IdRegistry(registry_path)
    assert registry2.get_or_assign("hash-a") == id_a
    assert registry2.get_or_assign("hash-c") == "LR_000003"


def test_same_file_content_gets_same_document_id_even_if_renamed(tmp_path):
    data = b"identical bytes"
    (tmp_path / "original_name.pdf").write_bytes(data)
    (tmp_path / "renamed_copy.pdf").write_bytes(data)

    registry = IdRegistry(tmp_path / "id_registry.json")
    ids = {registry.get_or_assign(content_hash(p.read_bytes())) for p in discover_files(tmp_path)}
    assert len(ids) == 1  # same content -> same id, regardless of filename


# --------------------------------------------------------------------------- normalize: PDFs

def test_born_digital_single_page_pdf_uses_pymupdf_not_ocr(monkeypatch):
    def fail_ocr(*a, **k):
        raise AssertionError("OCR must not be called for a born-digital page")

    monkeypatch.setattr(local_ocr, "extract_english", fail_ocr)

    data = _make_pdf(["Patient Name: John Doe. Hemoglobin result: 13.5 g/dL, normal range noted."])
    doc = normalize.normalize_pdf("LR_000001", "report.pdf", "/x/report.pdf", data, TextQualityThresholds())

    assert doc.status == "processed"
    assert doc.source_type == "born_digital_pdf"
    assert doc.page_count == 1
    assert doc.pages[0].extraction_method == "pymupdf"
    assert doc.pages[0].ocr_used is False
    assert doc.pages[0].elements == []
    assert "Hemoglobin" in doc.pages[0].text


def test_scanned_pdf_page_falls_back_to_existing_ocr(monkeypatch):
    monkeypatch.setattr(local_ocr, "extract_english", lambda image_bytes: _FAKE_OCR_RESULT)

    data = _make_pdf([None])  # blank page -> no extractable text -> must OCR
    doc = normalize.normalize_pdf("LR_000002", "scan.pdf", "/x/scan.pdf", data, TextQualityThresholds())

    assert doc.status == "processed"
    assert doc.source_type == "scanned_pdf"
    assert doc.pages[0].extraction_method == "ocr"
    assert doc.pages[0].ocr_used is True
    assert doc.pages[0].text == "Hemoglobin 13.5"
    assert len(doc.pages[0].elements) == 2
    assert doc.pages[0].elements[0].text == "Hemoglobin"
    assert doc.pages[0].elements[0].bbox == [10, 10, 90, 25]  # [x1,y1,x2,y2] from Rect(x=10,y=10,w=80,h=15)
    assert doc.pages[0].elements[0].confidence == 0.93
    assert doc.pages[0].ocr_confidence == 0.93


def test_multi_page_pdf_preserves_one_entry_per_page(monkeypatch):
    text = "Patient Name: John Doe. Hemoglobin result: 13.5 g/dL, normal range noted."
    data = _make_pdf([text, text, text])
    doc = normalize.normalize_pdf("LR_000003", "multi.pdf", "/x/multi.pdf", data, TextQualityThresholds())

    assert doc.page_count == 3
    assert [p.page_number for p in doc.pages] == [1, 2, 3]
    assert doc.source_type == "born_digital_pdf"


def test_mixed_pdf_is_not_classified_as_fully_scanned(monkeypatch):
    monkeypatch.setattr(local_ocr, "extract_english", lambda image_bytes: _FAKE_OCR_RESULT)

    real_text = "Patient Name: John Doe. Hemoglobin result: 13.5 g/dL, normal range noted."
    data = _make_pdf([real_text, None, real_text, None])  # digital, scanned, digital, scanned
    doc = normalize.normalize_pdf("LR_000004", "mixed.pdf", "/x/mixed.pdf", data, TextQualityThresholds())

    assert doc.source_type == "mixed_pdf"
    methods = [p.extraction_method for p in doc.pages]
    assert methods == ["pymupdf", "ocr", "pymupdf", "ocr"]


def test_password_protected_pdf_raises_unsupported_or_invalid_file():
    doc = fitz.open()
    doc.new_page().insert_text((50, 72), "secret")
    data = doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")
    doc.close()

    with pytest.raises(normalize.UnsupportedOrInvalidFile, match="password"):
        normalize.normalize_pdf("LR_000005", "locked.pdf", "/x/locked.pdf", data, TextQualityThresholds())


def test_zero_page_pdf_raises_unsupported_or_invalid_file(monkeypatch):
    # PyMuPDF itself refuses to serialize a genuinely zero-page document
    # (`tobytes()` raises "cannot save with zero pages"), so there's no real
    # PDF file to construct for this case — fake just the probe fitz.open()
    # call in normalize_pdf to return a zero-page document instead.
    class _FakeZeroPageDoc:
        needs_pass = False
        page_count = 0

        def close(self):
            pass

    monkeypatch.setattr(normalize.fitz, "open", lambda **kwargs: _FakeZeroPageDoc())

    with pytest.raises(normalize.UnsupportedOrInvalidFile, match="zero pages"):
        normalize.normalize_pdf("LR_000006", "empty.pdf", "/x/empty.pdf", b"irrelevant", TextQualityThresholds())


# --------------------------------------------------------------------------- normalize: images

def test_image_is_treated_as_a_single_page_document(monkeypatch):
    monkeypatch.setattr(local_ocr, "extract_english", lambda image_bytes: _FAKE_OCR_RESULT)

    data = _make_image_bytes("PNG")
    doc = normalize.normalize_image("LR_000007", "scan.png", "/x/scan.png", "png", data)

    assert doc.status == "processed"
    assert doc.source_type == "image"
    assert doc.page_count == 1
    assert doc.pages[0].page_number == 1
    assert doc.pages[0].extraction_method == "ocr"
    assert doc.pages[0].ocr_used is True
    assert len(doc.pages[0].elements) == 2


def test_corrupted_image_raises_unsupported_or_invalid_file():
    with pytest.raises(normalize.UnsupportedOrInvalidFile):
        normalize.normalize_image("LR_000008", "bad.png", "/x/bad.png", "png", b"not an image at all")


def test_pdf_and_image_produce_structurally_equivalent_page_shape(monkeypatch):
    """The actual point of this whole stage: downstream code must be able to
    read document.pages[*].text the same way regardless of source format."""
    monkeypatch.setattr(local_ocr, "extract_english", lambda image_bytes: _FAKE_OCR_RESULT)

    pdf_doc = normalize.normalize_pdf("LR_000009", "a.pdf", "/x/a.pdf", _make_pdf([None]), TextQualityThresholds())
    img_doc = normalize.normalize_image("LR_000010", "a.png", "/x/a.png", "png", _make_image_bytes("PNG"))

    pdf_page_keys = set(vars(pdf_doc.pages[0]).keys())
    img_page_keys = set(vars(img_doc.pages[0]).keys())
    assert pdf_page_keys == img_page_keys
    assert pdf_doc.pages[0].text == img_doc.pages[0].text == "Hemoglobin 13.5"


# --------------------------------------------------------------------------- pipeline (end-to-end over a tmp dataset)

def test_pipeline_run_processes_files_writes_index_and_handles_one_bad_file(tmp_path, monkeypatch):
    monkeypatch.setattr(local_ocr, "extract_english", lambda image_bytes: _FAKE_OCR_RESULT)

    input_dir = tmp_path / "dataset"
    input_dir.mkdir()
    good_text = "Patient Name: John Doe. Hemoglobin result: 13.5 g/dL, normal range noted."
    good_pdf_bytes = _make_pdf([good_text])
    (input_dir / "good.pdf").write_bytes(good_pdf_bytes)
    (input_dir / "scan.png").write_bytes(_make_image_bytes("PNG"))
    (input_dir / "corrupt.jpg").write_bytes(b"definitely not a jpeg")

    output_dir = tmp_path / "normalized_dataset"
    stats = pipeline.run(input_dir, output_dir)

    assert stats.total_discovered == 3
    assert stats.processed == 2
    assert stats.failed == 1
    assert stats.failures[0][1] == "corrupt.jpg"

    index_lines = (output_dir / "processing_index.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(index_lines) == 3
    statuses = {json.loads(line)["status"] for line in index_lines}
    assert statuses == {"processed", "failed"}

    # Original input files must be completely untouched — compare against
    # the exact bytes written, not a freshly regenerated PDF (PyMuPDF's
    # output isn't byte-identical across separate save calls even for
    # identical content, so that comparison would be flaky rather than meaningful).
    assert (input_dir / "good.pdf").read_bytes() == good_pdf_bytes


def test_pipeline_is_resumable_and_force_reprocesses(tmp_path, monkeypatch):
    monkeypatch.setattr(local_ocr, "extract_english", lambda image_bytes: _FAKE_OCR_RESULT)

    input_dir = tmp_path / "dataset"
    input_dir.mkdir()
    (input_dir / "a.png").write_bytes(_make_image_bytes("PNG"))
    output_dir = tmp_path / "normalized_dataset"

    first = pipeline.run(input_dir, output_dir)
    assert first.processed == 1
    assert first.skipped == 0

    second = pipeline.run(input_dir, output_dir)
    assert second.processed == 0
    assert second.skipped == 1  # already processed -> skipped without re-running OCR

    third = pipeline.run(input_dir, output_dir, force=True)
    assert third.processed == 1
    assert third.skipped == 0


def test_pipeline_gives_the_same_document_id_across_separate_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(local_ocr, "extract_english", lambda image_bytes: _FAKE_OCR_RESULT)

    input_dir = tmp_path / "dataset"
    input_dir.mkdir()
    (input_dir / "a.png").write_bytes(_make_image_bytes("PNG"))
    output_dir = tmp_path / "normalized_dataset"

    pipeline.run(input_dir, output_dir)
    first_id = next((output_dir / "normalized").iterdir()).name

    pipeline.run(input_dir, output_dir, force=True)
    second_id = next((output_dir / "normalized").iterdir()).name

    assert first_id == second_id
