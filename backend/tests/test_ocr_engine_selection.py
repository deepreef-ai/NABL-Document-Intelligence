"""Which OCR engine reads an English page, and which pages ignore the setting.

dev  -> documents/local_ocr.py (RapidOCR, in-process)
prod -> the deepreef-ocr Lambda (documents/ocr_client.py)

Non-Latin scripts are NOT part of the switch: RapidOCR's bundled English model
garbles Devanagari digits and table layout badly enough to misattribute whole
rows, so routing them locally in dev would break them quietly. These pin that
asymmetry, because it is the kind of thing a later "simplification" removes.
"""
import pytest

from app.config import Settings, get_settings
from app.documents import local_ocr, pipeline
from app.documents.ocr_client import OcrError, OcrResult


def _ocr_result(text: str) -> OcrResult:
    return OcrResult(text=text, lines=[text], confidence=0.99, boxes=[],
                     model_used="stub", region_count=1)


class _RecordingClient:
    """Stands in for the Lambda: records the script it was asked for."""

    def __init__(self, result=None, error: Exception | None = None):
        self.calls: list[str] = []
        self._result = result if result is not None else _ocr_result("from the lambda")
        self._error = error

    def extract(self, image_bytes: bytes, script: str) -> OcrResult:
        self.calls.append(script)
        if self._error:
            raise self._error
        return self._result


@pytest.fixture
def _env(monkeypatch):
    """Set app_env (and friends) on the cached Settings the pipeline reads."""
    def apply(**values):
        settings = get_settings()
        for key, value in values.items():
            monkeypatch.setattr(settings, key, value)
        return settings
    return apply


@pytest.fixture(autouse=True)
def _no_live_classification(monkeypatch):
    """_process_ocr_result classifies the OCR text, which is a real LLM call.
    These tests are about ENGINE SELECTION, so stub it — otherwise each one
    spends provider quota and the file takes two minutes."""
    from app.documents import classifier
    monkeypatch.setattr(classifier, "classify",
                        lambda *a, **k: ("other", 0.9))


@pytest.fixture
def _no_llm(monkeypatch):
    """The vision-LLM fallback must never be reached by these tests silently —
    if it is, say so loudly rather than passing on an empty result."""
    from app.documents import classifier
    monkeypatch.setattr(classifier, "classify_image",
                        lambda *a, **k: pytest.fail("fell through to the vision LLM"))


# --------------------------------------------------------------- the switch

def test_dev_reads_english_locally(monkeypatch, _env, _no_llm):
    _env(app_env="dev")
    client = _RecordingClient()
    monkeypatch.setattr(local_ocr, "extract_english", lambda data: _ocr_result("from rapidocr"))

    result = pipeline._process_image(b"png", "image/png", "english", client)

    assert result.extraction_source == "rapidocr:english"
    assert client.calls == [], "dev must not call the Lambda for English"


def test_prod_reads_english_through_the_lambda(monkeypatch, _env, _no_llm):
    _env(app_env="prod", ocr_lambda_english_script="english")
    client = _RecordingClient()
    monkeypatch.setattr(local_ocr, "extract_english",
                        lambda data: pytest.fail("prod must not use local OCR"))

    result = pipeline._process_image(b"png", "image/png", "english", client)

    assert result.extraction_source == "deepreef:english"
    assert client.calls == ["english"]


def test_the_english_script_code_sent_to_the_lambda_is_configurable(monkeypatch, _env, _no_llm):
    """The function selects its model from `script`. We have only ever invoked
    it with the non-Latin codes, so the Latin one must not be hard-coded."""
    _env(app_env="prod", ocr_lambda_english_script="latin")
    client = _RecordingClient()

    pipeline._process_image(b"png", "image/png", "english", client)

    assert client.calls == ["latin"]


# --------------------------------------------------------------- what the switch must NOT touch

@pytest.mark.parametrize("script", ["devanagari", "arabic", "ta", "te", "ka"])
def test_non_latin_scripts_use_the_lambda_in_dev_too(monkeypatch, _env, _no_llm, script):
    """RapidOCR's English model garbles these. dev is not a reason to route
    them locally."""
    _env(app_env="dev")
    client = _RecordingClient()
    monkeypatch.setattr(local_ocr, "extract_english",
                        lambda data: pytest.fail(f"{script} must never go to local OCR"))

    result = pipeline._process_image(b"png", "image/png", script, client)

    assert result.extraction_source == f"ocr:{script}"
    assert client.calls == [script]


# --------------------------------------------------------------- failure behaviour

def test_prod_does_not_silently_fall_back_to_local_ocr(monkeypatch, _env):
    """Swapping engines on failure would make a production OCR outage look
    like a quality drop. It falls through to the vision LLM instead, exactly
    as an unsupported script already does."""
    _env(app_env="prod")
    client = _RecordingClient(error=OcrError("lambda is down"))
    monkeypatch.setattr(local_ocr, "extract_english",
                        lambda data: pytest.fail("prod must not fall back to local OCR"))

    from app.documents import classifier
    reached = []
    monkeypatch.setattr(classifier, "classify_image",
                        lambda *a, **k: (reached.append(True), ("other", 0.0))[1])

    pipeline._process_image(b"png", "image/png", "english", client)

    assert reached, "a Lambda failure should reach the vision-LLM path"


def test_dev_still_falls_through_when_local_ocr_is_unusable(monkeypatch, _env):
    """Unchanged behaviour: RapidOCR not installed or a corrupt image reads
    the page with a vision call rather than failing the document."""
    _env(app_env="dev")
    client = _RecordingClient()

    def boom(data):
        raise local_ocr.LocalOcrError("rapidocr not installed")
    monkeypatch.setattr(local_ocr, "extract_english", boom)

    from app.documents import classifier
    reached = []
    monkeypatch.setattr(classifier, "classify_image",
                        lambda *a, **k: (reached.append(True), ("other", 0.0))[1])

    pipeline._process_image(b"png", "image/png", "english", client)

    assert reached


# --------------------------------------------------------------- the setting itself

@pytest.mark.parametrize("value,expected", [("dev", "dev"), ("prod", "prod"),
                                            ("DEV", "dev"), ("  prod ", "prod")])
def test_app_env_is_normalised(value, expected):
    assert Settings(app_env=value).app_env == expected


@pytest.mark.parametrize("value", ["production", "development", "staging", "true", ""])
def test_an_unknown_env_fails_at_startup(value):
    """A typo must not read as "not prod" and quietly send production traffic
    to the local engine."""
    with pytest.raises(ValueError):
        Settings(app_env=value)
