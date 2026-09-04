import re

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


# --- Confidence-scored local classification ---------------------------------
# classify_locally() above answers exactly one question ("is this the whole
# filled form?") and refuses below 5 pages, so EVERY single-page upload — the
# common case — paid for an LLM classification call. This adds a scored
# classifier covering the doc types we can recognise deterministically, with
# no page-count gate: a lab report that prints SPECIMEN / TEST / RESULT /
# REFERENCE RANGE is not ambiguous, and asking a model about it is waste.
#
# Confidence is a bounded sum of independent signals, NOT a probability. It
# is calibrated only against the threshold it is compared to
# (local_classification_min_confidence); a value of 0.8 means "several
# independent signals agree", not "80% likely correct".

# Terminology that only appears on an actual laboratory/test report.
_LAB_TERMS = (
    "reference range", "reference interval", "bio ref", "biological reference",
    "specimen", "sample name", "sample id", "test report", "test result",
    "result of analysis", "results of analysis", "parameters", "test parameter",
    "method of analysis", "protocol", "ulr", "analysis report", "discipline",
)
# Terminology specific to the supporting-certificate doc types.
_DOC_TYPE_TERMS: dict[str, tuple[str, ...]] = {
    "equipment_calibration_certificate": (
        "calibration certificate", "calibrated on", "calibration due", "traceability",
        "least count", "uncertainty of measurement",
    ),
    "staff_cv_certificate": (
        "curriculum vitae", "resume", "qualification", "work experience", "designation",
    ),
    "pt_ilc_result": (
        "proficiency testing", "inter laboratory", "interlaboratory", "z-score", "z score",
        "en value", "pt provider",
    ),
    "legal_proof": (
        "gstin", "gst number", "permanent account number", "pan no", "tan no",
        "certificate of incorporation", "registrar of companies",
    ),
    "reference_material_certificate": (
        "certified reference material", "reference material", "crm", "certificate of analysis",
    ),
}
# Bare column headers. Individually far too common to mean anything, which
# is why they only ever count ALONGSIDE >=3 detected result rows: a real lab
# report prints "Test | Result | Unit | Reference Range" as a table header,
# and the phrase list above misses that because it looks for prose phrases.
_LAB_HEADER_TOKENS = (
    "test", "result", "unit", "patient", "sample", "method", "report", "range", "limit",
)
# A results-table row: name, number, optional unit/range.
_RESULT_ROW_PATTERN = re.compile(r"[A-Za-z][A-Za-z ()/.-]{2,}\s+[<>]?\d+(?:\.\d+)?", re.MULTILINE)


def _count_terms(haystack: str, terms: tuple[str, ...]) -> int:
    return sum(1 for t in terms if t in haystack)


def classify_locally_scored(
    text: str, page_count: int, form_type: str, filename: str = ""
) -> tuple[str, float]:
    """(doc_type, confidence) with NO LLM call. Confidence 0.0 means "no
    idea, ask the model"; the caller compares it against
    local_classification_min_confidence.

    Deliberately conservative in one direction: it will happily return 0.0
    (costing a classification call), but it should not confidently return the
    WRONG type, because that silently routes the document to the wrong
    FIELD_SETS and the error never surfaces.
    """
    low = (text or "").lower()
    name_low = (filename or "").lower()

    # 1. The whole filled form — reuse the existing conservative check first,
    #    then allow the same evidence to count below 5 pages too.
    existing = classify_locally(text, page_count, form_type)
    if existing is not None:
        return existing
    section_hits = sum(1 for s in _section_names(form_type) if s.replace("_", " ") in low)
    if section_hits >= _MIN_SECTION_HITS and page_count >= 2:
        # Same evidence classify_locally wants, minus the page-count gate;
        # scored rather than asserted, so a borderline case still defers.
        return "completed_application_form", min(0.6 + 0.1 * (section_hits - _MIN_SECTION_HITS), 0.9)

    # 2. A specific supporting-certificate type, when its own vocabulary and
    #    the filename agree.
    best_type, best_score = "", 0.0
    for doc_type, terms in _DOC_TYPE_TERMS.items():
        hits = _count_terms(low, terms)
        if not hits:
            continue
        score = 0.45 + 0.15 * min(hits, 3)
        if any(t.split()[0] in name_low for t in terms):
            score += 0.1
        if score > best_score:
            best_type, best_score = doc_type, score
    if best_type and best_score >= 0.7:
        return best_type, min(best_score, 0.95)

    # 3. A laboratory/test report that isn't one of the NABL supporting types.
    #    These are the documents FIELD_SETS has no entry for, so the model was
    #    being asked to classify something that routes to "other" anyway —
    #    the single most wasteful classification call in the pipeline.
    lab_hits = _count_terms(low, _LAB_TERMS)
    header_hits = _count_terms(low, _LAB_HEADER_TOKENS)
    result_rows = len(_RESULT_ROW_PATTERN.findall(text or ""))
    # Structure is the load-bearing signal: vocabulary alone is not enough
    # (a covering letter can say "test report"), and rows alone are not
    # either (an invoice has numeric rows). Both must hold.
    if result_rows >= 3 and (lab_hits + header_hits) >= 4:
        return "other", min(0.7 + 0.04 * min(lab_hits + header_hits - 4, 5), 0.9)

    if best_type:
        return best_type, best_score  # below threshold; caller will ask the model
    return "other", 0.0
