from app.schemas.eligibility import PrerequisiteQuestion
from app.schemas.forms import FORM_CATEGORY, FORM_STANDARD, FormCategory, NablFormType


def _accreditation_questions(form_type: NablFormType) -> list[PrerequisiteQuestion]:
    standard = FORM_STANDARD[form_type]
    return [
        PrerequisiteQuestion(
            id="standard_implemented",
            text=(
                f"Has your lab implemented {standard} for at least 3 to 6 months?"
            ),
            help_text="NABL requires a documented, operating management system, not just a written manual.",
        ),
        PrerequisiteQuestion(
            id="internal_audit_mrm",
            text="Have you conducted an Internal Audit and a Management Review Meeting (MRM)?",
            help_text="Both must be complete, with records, before you apply.",
        ),
        PrerequisiteQuestion(
            id="pt_ilc",
            text="Have you participated in a valid Proficiency Testing (PT) / Inter-Laboratory Comparison (ILC) scheme?",
            help_text="Needed for every parameter/scope you intend to seek accreditation for.",
        ),
        PrerequisiteQuestion(
            id="quality_manager_staff",
            text="Do you have a dedicated Quality Manager and trained technical staff in place?",
            help_text="Roles must be formally assigned, not informal.",
        ),
    ]


# 155/157/159 are lightweight, PT-performance-based *recognition* schemes, not
# full accreditation — their real eligibility gate is form-specific (who
# qualifies for the scheme at all) rather than the 4 generic ISO-readiness
# questions above. These 2 questions per form are inferred from each scheme's
# documented scope/exclusions, not a verbatim checklist transcription (the
# official forms' "Checklist Section" for these three is an
# assessment-readiness checklist, not a line-itemized eligibility gate).
_RECOGNITION_QUESTIONS: dict[NablFormType, list[PrerequisiteQuestion]] = {
    NablFormType.NABL_155: [
        PrerequisiteQuestion(
            id="eligible_lab_category",
            text=(
                "Is your lab an entry-level medical testing lab — NOT a medical college lab, "
                "a corporate hospital-group lab, or a hospital lab with 50 or more beds?"
            ),
            help_text="NABL M(EL)T recognition is scoped to small entry-level labs only.",
        ),
        PrerequisiteQuestion(
            id="pt_participation",
            text="Have you participated in a PT/EQAS scheme with a satisfactory result for every parameter you're applying for?",
            help_text="This scheme is performance-based, not a full ISO 15189 audit.",
        ),
    ],
    NablFormType.NABL_157: [
        PrerequisiteQuestion(
            id="government_block_level_lab",
            text="Is this a government-run drinking water testing laboratory at the block or sub-divisional level?",
            help_text="This scheme (G-LAP) is scoped to that tier of government laboratory.",
        ),
        PrerequisiteQuestion(
            id="pt_participation",
            text="Have you participated in a Proficiency Testing scheme for the parameters you're applying for?",
            help_text=None,
        ),
    ],
    NablFormType.NABL_159: [
        PrerequisiteQuestion(
            id="active_project_reference",
            text="Do you have an active project reference (tender/contract) requiring on-site aggregate/concrete testing?",
            help_text="This recognition is tied to a specific, temporary building project, not a permanent lab.",
        ),
        PrerequisiteQuestion(
            id="pt_participation",
            text="Have you participated in a Proficiency Testing scheme for the fixed test parameters this scheme covers?",
            help_text=None,
        ),
    ],
}


def _questions(form_type: NablFormType) -> list[PrerequisiteQuestion]:
    if FORM_CATEGORY[form_type] == FormCategory.RECOGNITION:
        return _RECOGNITION_QUESTIONS[form_type]
    return _accreditation_questions(form_type)


PREREQUISITES: dict[NablFormType, list[PrerequisiteQuestion]] = {
    form_type: _questions(form_type) for form_type in NablFormType
}
