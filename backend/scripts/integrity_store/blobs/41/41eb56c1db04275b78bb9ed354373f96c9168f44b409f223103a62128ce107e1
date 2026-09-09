import pytest

from app.documents.classifier import _parse_classification, classify_locally
from app.llm.base import LlmProviderError


def test_short_document_never_classified_locally_regardless_of_content():
    # Fewer pages than _MIN_PAGES_FOR_LOCAL_CLASSIFICATION — even text packed
    # with section keywords must defer to the real LLM classifier, since a
    # single certificate could coincidentally mention a couple of these terms.
    text = "organisation equipment staff scope pt_ilc authorized_signatories"
    assert classify_locally(text, page_count=1, form_type="NABL_151") is None


def test_long_document_with_few_section_keywords_defers_to_llm():
    text = "Just a long staff CV with lots of unrelated filler text. " * 50
    assert classify_locally(text, page_count=10, form_type="NABL_151") is None


def test_long_document_mentioning_several_real_sections_is_classified_locally():
    text = "Details on organisation, equipment list, staff qualifications, and scope of testing follow."
    result = classify_locally(text, page_count=10, form_type="NABL_151")
    assert result is not None
    doc_type, confidence = result
    assert doc_type == "completed_application_form"
    assert confidence > 0.5


def test_parse_classification_defaults_confidence_when_a_local_model_omits_it():
    """A capable cloud model always includes both keys; a small local model
    under Ollama's generic JSON grammar sometimes doesn't — observed in
    practice with qwen2.5:3b dropping "confidence" entirely."""
    doc_type, confidence = _parse_classification({"doc_type": "staff_cv_certificate"})
    assert doc_type == "staff_cv_certificate"
    assert confidence == 0.5


def test_parse_classification_raises_when_doc_type_itself_is_missing():
    with pytest.raises(LlmProviderError, match="no doc_type"):
        _parse_classification({"confidence": 0.9})
