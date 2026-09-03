from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.common import (
    ApplicationFees,
    AuthorizedSignatoryRecord,
    ComplaintRecord,
    EquipmentRecord,
    FssaiScopeStatement,
    InternalAuditReview,
    MobileLaboratoryRecord,
    OrgLegalInfo,
    ProductRecord,
    ProjectStaffRecord,
    PTILCRecord,
    RadiologicalSafetyOfficerRecord,
    RecognitionEquipmentRecord,
    RecognitionLabDetails,
    RecognitionPTRecord,
    RecognitionReferenceMaterialRecord,
    RecognitionScopeItem,
    RecognitionScopeStatement,
    ReferenceMaterialRecord,
    AffiliationRecord,
    FamilyMemberRecord,
    RelatedLabRecord,
    ContractRecord,
    SamplingScopeStatement,
    ScopeStatement,
    SeniorManagement,
    ShareholderRecord,
    StaffRecord,
)


class NablFormType(str, Enum):
    NABL_151 = "NABL_151"  # Testing laboratories — ISO/IEC 17025
    NABL_152 = "NABL_152"  # Calibration laboratories — ISO/IEC 17025
    NABL_153 = "NABL_153"  # Medical laboratories — ISO 15189
    NABL_153A = "NABL_153A"  # Medical laboratories under an Operational/Supporting Entities arrangement
    NABL_154 = "NABL_154"  # Testing laboratories — integrated assessment w/ regulatory recognition
    NABL_155 = "NABL_155"  # Medical (Entry Level) Testing Labs — PT-performance recognition
    NABL_157 = "NABL_157"  # Government drinking water testing lab (block/sub-divisional) — recognition
    NABL_158 = "NABL_158"  # Product-based testing laboratories
    NABL_159 = "NABL_159"  # Temporary site labs testing aggregates/concrete — recognition


class FormCategory(str, Enum):
    """Which shape a form's schema and eligibility questions follow — see
    `BaseNablForm` vs `BaseRecognitionForm` below, and `wizard/prerequisites.py`."""

    ACCREDITATION = "accreditation"
    RECOGNITION = "recognition"


FORM_CATEGORY: dict[NablFormType, FormCategory] = {
    NablFormType.NABL_151: FormCategory.ACCREDITATION,
    NablFormType.NABL_152: FormCategory.ACCREDITATION,
    NablFormType.NABL_153: FormCategory.ACCREDITATION,
    NablFormType.NABL_153A: FormCategory.ACCREDITATION,
    NablFormType.NABL_154: FormCategory.ACCREDITATION,
    NablFormType.NABL_158: FormCategory.ACCREDITATION,
    NablFormType.NABL_155: FormCategory.RECOGNITION,
    NablFormType.NABL_157: FormCategory.RECOGNITION,
    NablFormType.NABL_159: FormCategory.RECOGNITION,
}

# Only meaningful for ACCREDITATION-category forms — recognition schemes are
# performance/PT-based, not accreditation to one of these standards.
FORM_STANDARD = {
    NablFormType.NABL_151: "ISO/IEC 17025",
    NablFormType.NABL_152: "ISO/IEC 17025",
    NablFormType.NABL_153: "ISO 15189",
    NablFormType.NABL_153A: "ISO 15189",
    NablFormType.NABL_154: "ISO/IEC 17025",
    NablFormType.NABL_158: "ISO/IEC 17025",
}


class BaseNablForm(BaseModel):
    organisation: OrgLegalInfo = Field(default_factory=OrgLegalInfo)
    senior_management: SeniorManagement = Field(default_factory=SeniorManagement)
    internal_audit_review: InternalAuditReview = Field(default_factory=InternalAuditReview)
    application_fees: ApplicationFees = Field(default_factory=ApplicationFees)
    disciplines: list[str] = Field(default_factory=list)
    equipment: list[EquipmentRecord] = Field(default_factory=list)
    reference_materials: list[ReferenceMaterialRecord] = Field(default_factory=list)
    staff: list[StaffRecord] = Field(default_factory=list)
    authorized_signatories: list[AuthorizedSignatoryRecord] = Field(default_factory=list)
    scope: list[ScopeStatement] = Field(default_factory=list)
    pt_ilc: list[PTILCRecord] = Field(default_factory=list)
    # Open-ended extraction (documents/extractor.py's extract_open_fields*)
    # finds key/value pairs with no fixed schema slot — every doc_type, not
    # just unrecognized ones — and documents/compiler.py routes them here by
    # their own invented field name rather than dropping them. Keyed by
    # field name, not field_path, since these never had a dotted schema path
    # to begin with.
    extra_fields: dict[str, str] = Field(default_factory=dict)


class Nabl151Form(BaseNablForm):
    """Testing laboratory accreditation (ISO/IEC 17025)."""


class Nabl152Form(BaseNablForm):
    """Calibration laboratory accreditation (ISO/IEC 17025). Scope rows use
    `cmc`/`frequency`, not `measurement_uncertainty` — see ScopeStatement."""


class Nabl153Form(BaseNablForm):
    """Medical laboratory accreditation (ISO 15189). Also used for NABL 153A
    (Operational/Supporting Entities arrangement) under its own `NablFormType`
    key in `FORM_MODEL` — 153A is this same shape, distinguished only by
    `associated_entity` on personnel/equipment/reference-material records,
    not a separate schema."""

    radiological_safety_officers: list[RadiologicalSafetyOfficerRecord] = Field(default_factory=list)


class Nabl154Form(BaseNablForm):
    """Testing laboratory accreditation with a simultaneous regulatory-
    recognition overlay (FSSAI, APEDA, EIC, Spices Board, etc.) — a superset
    of Nabl151Form. `regulatory_bodies`/`last_recognition_*` are populated via
    the `regulatory_recognition_certificate` doc_type (flat merge onto the
    form root, see documents/compiler.py); the rest of the recognition-scope
    variants below are schema-only for now (see common.py's module docstring)."""

    regulatory_bodies: list[str] = Field(default_factory=list)
    last_recognition_certificate_number: str | None = None
    last_recognition_issuing_authority: str | None = None
    last_recognition_validity: str | None = None
    contact_for_accounts: str | None = None
    mobile_laboratories: list[MobileLaboratoryRecord] = Field(default_factory=list)
    recognition_scope: list[RecognitionScopeStatement] = Field(default_factory=list)
    fssai_scope: list[FssaiScopeStatement] = Field(default_factory=list)
    sampling_scope: list[SamplingScopeStatement] = Field(default_factory=list)
    complaints: list[ComplaintRecord] = Field(default_factory=list)


class Nabl158Form(BaseNablForm):
    """Product-based testing laboratory accreditation — a superset of
    Nabl151Form keyed on Product rather than a generic material/parameter,
    plus a shareholder/director ownership-disclosure annexure. `products`
    and `shareholders` are wired to doc_types (see documents/extractor.py);
    the affiliation/family-member/related-lab/contract sub-tables are
    schema-only for now (see common.py's module docstring)."""

    products: list[ProductRecord] = Field(default_factory=list)
    shareholders: list[ShareholderRecord] = Field(default_factory=list)
    affiliations: list[AffiliationRecord] = Field(default_factory=list)
    family_members: list[FamilyMemberRecord] = Field(default_factory=list)
    related_labs: list[RelatedLabRecord] = Field(default_factory=list)
    contracts: list[ContractRecord] = Field(default_factory=list)


class BaseRecognitionForm(BaseModel):
    """155/157/159 — lightweight PT-performance-based recognition schemes,
    not full accreditation. Deliberately not a subclass of BaseNablForm: no
    senior management, internal audit/review, or application-fees section on
    any of these forms, and scope/equipment/PT are narrower, mostly
    picklist-driven tables rather than open ones."""

    lab_details: RecognitionLabDetails = Field(default_factory=RecognitionLabDetails)
    scope: list[RecognitionScopeItem] = Field(default_factory=list)
    equipment: list[RecognitionEquipmentRecord] = Field(default_factory=list)
    pt_participation: list[RecognitionPTRecord] = Field(default_factory=list)
    # See BaseNablForm.extra_fields — same role, for the recognition-scheme forms.
    extra_fields: dict[str, str] = Field(default_factory=dict)


class Nabl155Form(BaseRecognitionForm):
    """NABL Medical (Entry Level) Testing Labs recognition program."""


class Nabl157Form(BaseRecognitionForm):
    """Government Drinking Water Testing Laboratory (block/sub-divisional
    level) recognition — the only recognition scheme with its own Reference
    Material/Reference Cultures table."""

    reference_materials: list[RecognitionReferenceMaterialRecord] = Field(default_factory=list)


class Nabl159Form(BaseRecognitionForm):
    """Temporary Site Laboratories testing aggregates/concrete, tied to a
    specific building project. `project_*` fields live on `lab_details`
    (RecognitionLabDetails) and are populated via the `project_reference_document`
    doc_type (flat merge — see documents/compiler.py)."""

    technical_staff: list[ProjectStaffRecord] = Field(default_factory=list)


FORM_MODEL: dict[NablFormType, type[BaseModel]] = {
    NablFormType.NABL_151: Nabl151Form,
    NablFormType.NABL_152: Nabl152Form,
    NablFormType.NABL_153: Nabl153Form,
    NablFormType.NABL_153A: Nabl153Form,
    NablFormType.NABL_154: Nabl154Form,
    NablFormType.NABL_155: Nabl155Form,
    NablFormType.NABL_157: Nabl157Form,
    NablFormType.NABL_158: Nabl158Form,
    NablFormType.NABL_159: Nabl159Form,
}
