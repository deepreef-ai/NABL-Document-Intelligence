"""Image uploads (PNG/JPG) must be read line by line, like scanned PDF pages.

MEASURED 2026-11-24 on 006_Lab-report.png: the image path returned the whole
page as one line, the compact extractor could not verify any cited line, and
all 20 rows were refused — the document "VALIDATED" with 0 results."""
import cv2
import numpy as np

from app.documents import deepreef_local, local_ocr
from app.graph.config import get_graph_settings
from app.graph.nodes import preprocess
from app.graph.state import GraphState


def _png(tmp_path):
    img = np.full((60, 200, 3), 255, np.uint8)
    path = tmp_path / "report.png"
    cv2.imwrite(str(path), img)
    return str(path)


def _run(path):
    return preprocess.preprocess_document(GraphState(document_id="img", file_path=path, file_type="png"))


def test_image_is_read_by_deepreef_with_one_row_per_line(tmp_path, monkeypatch):
    monkeypatch.setattr(get_graph_settings(), "ocr_engine", "deepreef_local", raising=False)
    seen = {}

    def fake(image, **kw):
        seen.update(kw)
        return deepreef_local.PageOcr(text="", lines=["HEAMOGLOBIN 13.6 mg/dl", "SUGAR ABSENT"],
                                      confidence=0.9, model_used="m")

    monkeypatch.setattr(deepreef_local, "extract_page", fake)
    out = _run(_png(tmp_path))
    assert out["page_text"][1].splitlines() == ["HEAMOGLOBIN 13.6 mg/dl", "SUGAR ABSENT"]
    assert seen["rendered_pdf_page"] is False, "a photo keeps the full clean-up profile"


def test_rapidocr_fallback_also_keeps_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(get_graph_settings(), "ocr_engine", "deepreef_local", raising=False)

    def broken(image, **kw):
        raise deepreef_local.DeepreefLocalError("model missing")

    class R:
        text = "HEAMOGLOBIN 13.6 mg/dl SUGAR ABSENT"
        lines = ["HEAMOGLOBIN 13.6 mg/dl", "SUGAR ABSENT"]
        boxes = []
        confidence = 0.8

    monkeypatch.setattr(deepreef_local, "extract_page", broken)
    monkeypatch.setattr(local_ocr, "extract_english", lambda data: R())
    out = _run(_png(tmp_path))
    assert out["page_text"][1].splitlines() == ["HEAMOGLOBIN 13.6 mg/dl", "SUGAR ABSENT"]
