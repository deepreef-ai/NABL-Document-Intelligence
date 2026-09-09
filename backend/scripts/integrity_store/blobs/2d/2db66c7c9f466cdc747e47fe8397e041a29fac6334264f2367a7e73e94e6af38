"""Shared entities extracted from uploaded documents.

Field names and structure here are taken directly from the official NABL
application form templates (NABL 151, 152, 153 — the .doc/.docx originals,
read in full on 2026-08-29), not a guess. Where the testing (151), calibration
(152), and medical (153) forms use different terms for the same concept, both
are kept as separate optional fields rather than collapsed into one generic
name, because NABL 100B (the accreditation procedure document) treats them as
distinct regulatory concepts:
  - testing/medical: "Measurement Uncertainty (MU)" and PT/ILC performance in
    "Z-score"
  - calibration: "Calibration and Measurement Capability (CMC)" and PT/ILC
    performance in "En value"
154 and 158 are legitimate supersets of this shape (regulatory-recognition
overlay / product-based accreditation with an ownership-disclosure annexure)
— see the extra entities near the bottom of this file and `forms.py`'s
`Nabl154Form`/`Nabl158Form`. 155/157/159 are structurally much thinner
*recognition* schemes (fixed picklists, no open scope/uncertainty table, no
senior-management/internal-audit/fees sections) that would misrepresent the
form if forced into `BaseNablForm` — see the `Recognition*` entities below and
`forms.py`'s `BaseRecognitionForm`. 100B is a pure procedure document with no
applicant-fillable fields at all, not modeled.

Some real sub-annexures (154's Mobile Laboratory/Complaints/FSSAI-scope/
Sampling-scope tables; 158's Affiliations/Family-members/Related-labs/
Contracts tables) are modeled here for fidelity to the actual forms but are
not yet wired to a document doc_type in `documents/extractor.py` — same
treatment as `disciplines: list[str]` already gets. They're real, present in
the compiled-form JSON, and reachable via manual field edits in the review
UI; just not auto-extracted from an uploaded document yet.
"""
from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class FacilityType(str, Enum):
    PERMANENT = "permanent"
    SITE = "site"
    MOBILE = "mobile"
    PERMANENT_SITE = "permanent_site"


class AccreditationApplicationType(str, Enum):
    INITIAL = "initial"
    RENEWAL = "renewal"
    EXTENSION_OF_SCOPE = "extension_of_scope"


class ServiceType(str, Enum):
    OPEN_TO_OTHERS = "open_to_others"
    PARTLY_OPEN_TO_OTHERS = "partly_open_to_others"
    IN_HOUSE = "in_house"


class AssociatedEntity(str, Enum):
    """NABL 153A (Medical Testing under an Operational/Supporting Entities
    arrangement — e.g. franchiser/franchisee, hub-and-spoke) tags every
    personnel/equipment/reference-material row with which entity it belongs
    to. It's the same form as 153 plus this one attribute, not a separate
    schema — left null for 151/152/153 where the question doesn't apply."""

    OPERATIONAL = "operational"
    SUPPORTING = "supporting"


class ContactInfo(BaseModel):
    name: str | None = None
    designation: str | None = None
    telephone: str | None = None
    fax: str | None = None
    email: str | None = None


class OrgLegalInfo(BaseModel):
    accreditation_application_type: AccreditationApplicationType | None = None
    previous_certificate_number: str | None = None
    previous_lab_id: str | None = None

    laboratory_name: str | None = None  # "Name/Identification of the Laboratory" — as it should appear on the certificate
    address: str | None = None
    address_ownership: str | None = None  # owned / long-term lease
    telephone: str | None = None
    fax: str | None = None
    email: str | None = None
    facility_types: list[FacilityType] = Field(default_factory=list)

    head_office_name: str | None = None
    head_office_telephone: str | None = None
    head_office_fax: str | None = None
    head_office_email: str | None = None

    legal_entity_registration_number: str | None = None
    legal_entity_registering_authority: str | None = None

    adverse_action_taken: bool | None = None
    adverse_action_details: str | None = None

    gst_number: str | None = None
    pan_number: str | None = None
    tan_number: str | None = None

    type_of_service: ServiceType | None = None
    other_accreditations: str | None = None
    certificate_name_and_address: str | None = None

    consultant_name: str | None = None


class SeniorManagement(BaseModel):
    """The 4 named roles every form asks for by title, each with their own
    contact details."""

    head_of_laboratory: ContactInfo = Field(default_factory=ContactInfo)
    management_system_representative: ContactInfo = Field(default_factory=ContactInfo)
    technical_operations_representative: ContactInfo = Field(default_factory=ContactInfo)
    nabl_contact_person: ContactInfo = Field(default_factory=ContactInfo)


class InternalAuditReview(BaseModel):
    last_internal_audit_date: date | None = None
    full_standard_coverage_last_year: bool | None = None
    last_management_review_date: date | None = None


class ApplicationFees(BaseModel):
    number_of_groups: int | None = None
    fee_amount: float | None = None


class EquipmentRecord(BaseModel):
    name: str | None = None
    make_model: str | None = None
    serial_number: str | None = None
    ownership: str | None = None  # owned / long-term lease
    range_and_accuracy: str | None = None
    calibration_date: date | None = None
    calibration_due_date: date | None = None
    calibrated_by: str | None = None  # agency name, or "in-house"
    associated_entity: AssociatedEntity | None = None


class ReferenceMaterialRecord(BaseModel):
    """Reference Material/Reference Standards — its own repeating table on
    every form, distinct from Equipment."""

    name: str | None = None
    source: str | None = None
    expiry_date: date | None = None
    traceability: str | None = None
    associated_entity: AssociatedEntity | None = None


class StaffRecord(BaseModel):
    """The general "Details of staff" table. Kept separate from
    AuthorizedSignatoryRecord below — NABL forms carry these as two tables
    with different columns, not one table with a role flag."""

    name: str | None = None
    designation: str | None = None
    qualification: str | None = None
    relevant_experience_years: float | None = None
    trainings: list[str] = Field(default_factory=list)
    associated_entity: AssociatedEntity | None = None


class AuthorizedSignatoryRecord(BaseModel):
    """"Proposed personnel declared to report, review and authorization of
    results" table."""

    department_or_section: str | None = None
    name: str | None = None
    designation: str | None = None
    qualification_with_specialization: str | None = None
    relevant_experience_years: float | None = None
    trainings: list[str] = Field(default_factory=list)
    authorized_area: str | None = None
    specimen_signature_on_file: bool = False
    associated_entity: AssociatedEntity | None = None


class RadiologicalSafetyOfficerRecord(BaseModel):
    """Medical Imaging discipline only (NABL 153/153A)."""

    name: str | None = None
    qualifications: str | None = None
    radiological_safety_training: str | None = None
    experience_years: float | None = None
    aerb_registration_number: str | None = None
    associated_entity: AssociatedEntity | None = None


class ScopeStatement(BaseModel):
    materials_or_products: str | None = None  # testing/medical only ("Materials or Products tested")
    parameter_or_measurand: str | None = None  # component/parameter tested (testing) OR measurand/instrument (calibration)
    test_or_calibration_method: str | None = None
    range: str | None = None
    frequency: str | None = None  # calibration only, alongside range
    measurement_uncertainty: str | None = None  # MU — testing/medical
    cmc: str | None = None  # Calibration and Measurement Capability — calibration only
    facility_type: FacilityType | None = None


class PTILCRecordType(str, Enum):
    PT = "pt"
    ILC = "ilc"


class PerformanceMetric(str, Enum):
    Z_SCORE = "z_score"  # testing/medical
    EN_VALUE = "en_value"  # calibration
    OTHER = "other"


class PTILCRecord(BaseModel):
    record_type: PTILCRecordType | None = None
    materials_or_products: str | None = None
    parameter_or_measurand: str | None = None
    test_or_calibration_method: str | None = None
    participation_date: date | None = None
    provider: str | None = None  # "Nodal laboratory/PT Provider (Accreditation Body/Country)"
    provider_accreditation_body_or_country: str | None = None
    performance_metric: PerformanceMetric | None = None
    performance_value: str | None = None
    corrective_action_taken: str | None = None


# --- NABL 154 (regulatory-recognition overlay on 151) -----------------------


class MobileLaboratoryRecord(BaseModel):
    laboratory_type: str | None = None  # "Type I Screening MFTL" / "Type II Advanced MFTL"
    vehicle_registration_number: str | None = None
    make_and_model: str | None = None
    chassis_number: str | None = None
    engine_number: str | None = None
    insurance_policy_number: str | None = None
    puc_certificate_validity: date | None = None


class RecognitionScopeStatement(BaseModel):
    """"Scope for Recognition/Approval (Format-2)" — the commodity-board
    regulator variant of the scope table, distinct from the accreditation
    ScopeStatement above."""

    category_or_group: str | None = None
    specific_test: str | None = None
    therapeutic_classification: str | None = None
    target_product_or_matrix: str | None = None


class FssaiScopeStatement(BaseModel):
    food_category: str | None = None
    food_sub_category: str | None = None
    specific_food_articles: str | None = None
    testing_parameters: str | None = None
    test_method_or_standard: str | None = None
    range: str | None = None


class SamplingScopeStatement(BaseModel):
    commodity_or_matrix: str | None = None
    sampling_procedure: str | None = None


class ComplaintRecord(BaseModel):
    """Complaints/Disputes (last 3 years) — unique to NABL 154."""

    client_name: str | None = None
    nature_of_complaint: str | None = None
    resolved_in_favor_of: str | None = None
    action_taken: str | None = None
    latest_status: str | None = None


# --- NABL 158 (product-based accreditation) ---------------------------------


class ProductRecord(BaseModel):
    product_name: str | None = None
    product_standard: str | None = None


class ShareholderRecord(BaseModel):
    name: str | None = None
    shareholding_percent: float | None = None
    relations_with_other_directors: str | None = None
    remarks: str | None = None


class AffiliationRecord(BaseModel):
    """Shareholder/Director's outside affiliations, last 10 years."""

    shareholder_or_director_name: str | None = None
    organization: str | None = None
    nature_of_activities: str | None = None
    roles_and_responsibilities: str | None = None
    period: str | None = None


class FamilyMemberRecord(BaseModel):
    """Shareholder/Director's family members' affiliations, last 5 years."""

    shareholder_or_director_name: str | None = None
    family_member_name: str | None = None
    relationship: str | None = None
    affiliation_or_activities: str | None = None


class RelatedLabRecord(BaseModel):
    name: str | None = None
    location: str | None = None
    relationship: str | None = None  # same legal entity / group company / common owners
    accreditation_status: str | None = None


class ContractRecord(BaseModel):
    organization_name: str | None = None
    nature_of_contract: str | None = None
    remarks: str | None = None


# --- NABL 155 / 157 / 159 (lightweight recognition schemes) -----------------
# Structurally distinct from the accreditation forms above: a flat lab
# details section instead of organisation + senior management, and fixed- or
# narrow-picklist scope/equipment/PT tables instead of an open scope table.
# See `forms.py`'s `BaseRecognitionForm`.


class RecognitionLabDetails(BaseModel):
    laboratory_name: str | None = None
    country: str | None = None
    state: str | None = None
    district: str | None = None
    address: str | None = None
    pincode: str | None = None
    mobile: str | None = None
    email: str | None = None
    technical_head_or_lab_manager: str | None = None
    accredited_pt_program: str | None = None
    adverse_action_taken: bool | None = None
    adverse_action_details: str | None = None
    # NABL 159 only:
    project_name: str | None = None
    project_size: str | None = None
    project_duration: str | None = None
    project_reference: str | None = None  # tender/contract reference
    # NABL 155 only:
    naco_ictc_laboratory: bool | None = None
    ever_applied_for_iso15189: bool | None = None


class RecognitionScopeItem(BaseModel):
    discipline_group_subgroup: str | None = None
    test_parameter: str | None = None
    range: str | None = None
    measurement_technique: str | None = None
    reference_material_or_crm: str | None = None


class RecognitionEquipmentRecord(BaseModel):
    discipline: str | None = None
    name: str | None = None
    in_house_or_traceability: str | None = None
    external_lab_name: str | None = None
    calibration_certificate_uploaded: bool = False
    image_uploaded: bool = False


class RecognitionReferenceMaterialRecord(BaseModel):
    discipline: str | None = None
    name: str | None = None
    traceability: str | None = None
    source: str | None = None
    expiry_date: date | None = None
    certificate_uploaded: bool = False


class RecognitionPTRecord(BaseModel):
    discipline_group_subgroup: str | None = None
    test_parameter: str | None = None
    provider: str | None = None
    performance: str | None = None  # satisfactory/unsatisfactory, or a z-score/value as reported
    date_of_report: date | None = None
    report_uploaded: bool = False


class ProjectStaffFunction(str, Enum):
    """NABL 159's technical-staff table has fixed roles, unlike the general
    StaffRecord table on the accreditation forms."""

    HEAD_OF_LAB = "head_of_lab"
    TECHNICAL_MANAGER = "technical_manager"
    TEST_ENGINEER = "test_engineer"


class ProjectStaffRecord(BaseModel):
    function: ProjectStaffFunction | None = None
    name: str | None = None
    qualification: str | None = None
    experience: str | None = None
    mobile: str | None = None
    email: str | None = None
