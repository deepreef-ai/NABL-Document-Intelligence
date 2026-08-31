from app.llm.base import LlmProviderError
from app.llm.factory import get_llm_chain

# One global list works across all 9 NABL form types without threading
# form_type through classification: names are distinct enough (accreditation
# vs. recognition-scheme variants) that the LLM can pick correctly from
# content alone, and documents/compiler.py's LIST_TARGET/FLAT_MERGE_TARGETS
# silently ignore a doc_type that isn't relevant to the current form_type.
DOC_TYPES = [
    "legal_proof",
    "lab_contact_info",
    "equipment_calibration_certificate",
    "recognition_equipment_certificate",
    "reference_material_certificate",
    "recognition_reference_material_certificate",
    "staff_cv_certificate",
    "authorized_signatory_declaration",
    "pt_ilc_result",
    "recognition_pt_report",
    "project_reference_document",
    "regulatory_recognition_certificate",
    "product_list_declaration",
    "shareholder_declaration",
    "completed_application_form",
    "quality_manual_sop",
    "other",
]

_SYSTEM = (
    "You classify supporting documents uploaded for a NABL lab accreditation or "
    "recognition-scheme application into exactly one entity type:\n"
    "- legal_proof: GST/CIN/registration proof for a full accreditation application "
    "(NABL 151/152/153/153A/154/158)\n"
    "- lab_contact_info: lab identity/contact/registration proof for a lightweight "
    "recognition-scheme application (NABL 155/157/159)\n"
    "- equipment_calibration_certificate / recognition_equipment_certificate: an "
    "equipment calibration certificate (accreditation vs. recognition-scheme forms)\n"
    "- reference_material_certificate / recognition_reference_material_certificate: "
    "a reference material/standard certificate (accreditation vs. recognition-scheme forms)\n"
    "- staff_cv_certificate: a staff member's CV/qualification certificate\n"
    "- authorized_signatory_declaration: a declaration naming personnel authorized to "
    "sign/report/review results\n"
    "- pt_ilc_result / recognition_pt_report: a proficiency testing or inter-lab "
    "comparison report (accreditation vs. recognition-scheme forms)\n"
    "- project_reference_document: a tender/contract tying a lab to a specific "
    "construction project (NABL 159 only)\n"
    "- regulatory_recognition_certificate: a prior FSSAI/APEDA/EIC/etc. recognition "
    "certificate (NABL 154 only)\n"
    "- product_list_declaration: a list of products a lab tests, with the product standard "
    "(NABL 158 only)\n"
    "- shareholder_declaration: a shareholder/director ownership disclosure (NABL 158 only)\n"
    "- completed_application_form: the document IS a filled-out copy of the NABL "
    "application form itself (or covers many of its sections at once — organisation "
    "details, senior management, equipment, staff, scope, PT/ILC — in one file), "
    "rather than one narrow supporting certificate\n"
    "- quality_manual_sop: a quality manual or SOP excerpt\n"
    "- other: anything else"
)

_JSON_INSTRUCTION = (
    'Respond with ONLY a JSON object, no other text: {"doc_type": "<one of: '
    f"{', '.join(DOC_TYPES)}>\", \"confidence\": <0.0-1.0>}}"
)


def classify_text(text: str) -> tuple[str, float]:
    prompt = f"Document text (may be truncated):\n\n{text[:6000]}\n\n{_JSON_INSTRUCTION}"
    result = get_llm_chain().generate_json(system=_SYSTEM, user_text=prompt)
    return _parse_classification(result)


def classify_image(image_bytes: bytes, media_type: str) -> tuple[str, float]:
    prompt = f"Classify this document image.\n\n{_JSON_INSTRUCTION}"
    result = get_llm_chain().generate_json(system=_SYSTEM, user_text=prompt, image=image_bytes, image_media_type=media_type)
    return _parse_classification(result)


def _parse_classification(result: dict) -> tuple[str, float]:
    """A capable cloud model reliably includes both keys; a small local
    model under Ollama's generic `format: json` grammar (which only enforces
    "this is valid JSON", not a specific schema) can drop one — observed in
    practice with qwen2.5:3b omitting "confidence". A missing doc_type is a
    genuine failure (nothing to classify with); a missing confidence just
    means "uncertain," not "the call must be retried elsewhere.\""""
    doc_type = result.get("doc_type")
    if not doc_type:
        raise LlmProviderError(f"classifier: response had no doc_type: {result}")
    confidence = result.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = 0.5
    return doc_type, confidence


# Minimum number of the current form's own top-level schema sections
# (organisation, equipment, staff, scope, pt_ilc, ...) that must show up as
# section-like headings in the text before classify_locally() calls it a
# whole filled form — well above what any single supporting certificate
# (which covers exactly one section) would ever mention on its own.
_MIN_SECTION_HITS = 3
# A single certificate is rarely this long; below this, always defer to the
# real LLM classifier rather than risk a heuristic false positive.
_MIN_PAGES_FOR_LOCAL_CLASSIFICATION = 5


def _section_names(form_type: str) -> set[str]:
    from app.documents.extractor import form_field_templates

    templates = form_field_templates(form_type)
    return {t.split(".", 1)[0].split("[", 1)[0] for t in templates}


def classify_locally(text: str, page_count: int, form_type: str) -> tuple[str, float] | None:
    """Cheap, no-LLM-call check for exactly the one doc_type worth
    short-circuiting: completed_application_form. It's both the doc_type
    that drives the most expensive part of the pipeline (the RAG whole-form
    extraction) and, because classify_text() would otherwise spend one more
    call from the same rate-limited free-tier quota on every single upload,
    the one most worth skipping the LLM for. Returns None whenever it isn't
    confident, so callers fall back to the real LLM classifier for every
    other doc_type or ambiguous case rather than risk a wrong local guess."""
    if page_count < _MIN_PAGES_FOR_LOCAL_CLASSIFICATION:
        return None
    text_lower = text.lower()
    hits = sum(1 for section in _section_names(form_type) if section.replace("_", " ") in text_lower)
    if hits >= _MIN_SECTION_HITS:
        return "completed_application_form", 0.9
    return None
