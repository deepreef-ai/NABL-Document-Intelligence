from app.documents.combined_extraction import attach_confidence
from app.documents.lab_report import UNVERIFIED_CONFIDENCE_CAP, extract_lab_report, flatten_for_review

OCR = "Hemoglobin 13.2 g/dL  Creatinine 0.15 mg/dL  Temp -5\nLab Name: Acme Labs"


class FakeChain:
    def __init__(self, reply):
        self.reply = reply

    def generate_json(self, system, user_text, image=None, image_media_type=None, **kw):
        return self.reply


def _run(fields, tests):
    return extract_lab_report(FakeChain({"fields": fields, "tests": tests}), OCR)


def test_correct_values_are_verified():
    r = _run({"lab_name": "Acme Labs"}, [{"test_name": "Hemoglobin", "result": "13.2", "confidence": 1.0}])
    assert r["field_verified"]["lab_name"] is True
    assert r["tests"][0]["result_verified"] is True


def test_wrong_result_is_not_verified_even_if_substring_of_a_longer_number():
    r = _run({}, [{"test_name": "Creatinine", "result": "1.5"}, {"test_name": "Creatinine", "result": "0.15"}])
    assert r["tests"][0]["result_verified"] is False
    assert r["tests"][1]["result_verified"] is True


def test_wrong_field_is_not_verified():
    r = _run({"lab_name": "Acme Labs", "client_name": "Nobody Inc"}, [])
    assert r["field_verified"] == {"lab_name": True, "client_name": False}


def test_sign_mismatch_is_not_verified():
    r = _run({}, [{"test_name": "Temp", "result": "5"}, {"test_name": "Temp", "result": "-5"}])
    assert r["tests"][0]["result_verified"] is False
    assert r["tests"][1]["result_verified"] is True


def test_unverified_confidence_is_capped_in_flatten_for_review():
    r = _run({}, [{"test_name": "Creatinine", "result": "1.5", "confidence": 1.0}])
    flat = {f["field"]: f["confidence"] for f in flatten_for_review(r)}
    assert flat["tests[0].result"] == UNVERIFIED_CONFIDENCE_CAP
    r = _run({}, [{"test_name": "Hemoglobin", "result": "13.2", "confidence": 1.0}])
    flat = {f["field"]: f["confidence"] for f in flatten_for_review(r)}
    assert flat["tests[0].result"] == 1.0


def test_combined_extraction_attach_confidence_caps_unverified():
    result = {
        "fields": {"lab_name": "Acme Labs", "client_name": "Nobody Inc"},
        "tests": [{"test_name": "Creatinine", "result": "1.5"}, {"test_name": "Hemoglobin", "result": "13.2"}],
    }
    out = attach_confidence(result, OCR)
    assert out["field_verified"] == {"lab_name": True, "client_name": False}
    assert out["field_confidence"] == {"lab_name": 1.0, "client_name": UNVERIFIED_CONFIDENCE_CAP}
    assert out["tests"][0]["result_verified"] is False
    assert out["tests"][0]["confidence"] == UNVERIFIED_CONFIDENCE_CAP
    assert out["tests"][1]["result_verified"] is True
    assert out["tests"][1]["confidence"] == 1.0
