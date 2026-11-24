import numpy as np

from app.documents import local_ocr


def _fake_rapidocr_result():
    # Mirrors RapidOCR's own raw call contract: a list of (quad, text, conf)
    # tuples plus an elapsed-time value, as returned by calling a RapidOCR
    # instance directly (see deepreef-ocr's engine.py's own extract() for
    # the same unpacking pattern this project's local_ocr.py now mirrors).
    result = [
        ([[10, 5], [130, 5], [130, 25], [10, 25]], "GST Number:", 0.93),
        ([[10, 30], [170, 30], [170, 50], [10, 50]], "27ABCDE1234F1Z5", 0.62),
    ]
    return result, 12.3


class _FakeEngine:
    def __call__(self, image):
        return _fake_rapidocr_result()


def _fake_decode_and_resize(image_bytes):
    return np.zeros((10, 10, 3)), {"format_detected": "png"}


def test_extract_english_maps_rapidocr_result_to_ocr_result(monkeypatch):
    monkeypatch.setattr(local_ocr, "_rapidocr_engine", lambda: _FakeEngine())
    monkeypatch.setattr(local_ocr, "_decode_and_resize", _fake_decode_and_resize)

    result = local_ocr.extract_english(b"fake-png-bytes")

    assert result.lines == ["GST Number:", "27ABCDE1234F1Z5"]
    assert result.model_used == "rapidocr-default-en"
    assert result.region_count == 2
    first = result.boxes[0]
    assert (first.x, first.y, first.w, first.h) == (10, 5, 120, 20)
    assert result.confidence == round((0.93 + 0.62) / 2, 4)
    assert result.text == "GST Number: 27ABCDE1234F1Z5"


def test_extract_english_wraps_a_preprocessing_failure(monkeypatch):
    def boom(image_bytes):
        raise ValueError("not an image")

    monkeypatch.setattr(local_ocr, "_decode_and_resize", boom)

    try:
        local_ocr.extract_english(b"not-an-image")
        assert False, "expected LocalOcrError"
    except local_ocr.LocalOcrError:
        pass


def test_extract_english_with_no_detected_lines_returns_zero_confidence(monkeypatch):
    class _EmptyEngine:
        def __call__(self, image):
            return [], 1.0

    monkeypatch.setattr(local_ocr, "_rapidocr_engine", lambda: _EmptyEngine())
    monkeypatch.setattr(local_ocr, "_decode_and_resize", _fake_decode_and_resize)

    result = local_ocr.extract_english(b"fake-png-bytes")

    assert result.lines == []
    assert result.confidence == 0.0
    assert result.region_count == 0
