"""Renders the compiled NABL form JSON (documents/compiler.py's compile_form
output) into a filled PDF. Deterministic layout only — the compiled JSON is
already final by the time this runs, so no LLM is involved in producing the
PDF, just formatting. Not a binary edit of the official .doc/.docx template
(not practical to do reliably from code); a clean new document using the
real section/field labels captured from those templates earlier this
session, laid out with reportlab.
"""
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# Real labels from the official NABL 151/152/153 application form templates
# (read in full, antiword/python-docx, earlier this session). Anything not
# listed here (154/158-only fields, recognition-scheme fields, etc.) falls
# back to a prettified attribute name — see _label().
_LABELS: dict[str, str] = {
    "laboratory_name": "Name/Identification of the Laboratory",
    "address": "Address",
    "address_ownership": "Ownership of Premises",
    "telephone": "Telephone No.",
    "fax": "Fax No.",
    "email": "E-mail",
    "gst_number": "GST Number",
    "pan_number": "PAN Number",
    "tan_number": "TAN Number",
    "legal_entity_registration_number": "Legal Entity Registration No.",
    "legal_entity_registering_authority": "Registering Authority",
    "accreditation_application_type": "Type of Application",
    "previous_certificate_number": "Previous Accreditation Certificate No.",
    "previous_lab_id": "Previous Laboratory ID",
    "head_office_name": "Name of Head Office/Parent Organization",
    "head_office_telephone": "Head Office Telephone No.",
    "head_office_fax": "Head Office Fax No.",
    "head_office_email": "Head Office E-mail",
    "adverse_action_taken": "Adverse Action Initiated in the Past?",
    "adverse_action_details": "Adverse Action Details",
    "type_of_service": "Type of Service",
    "other_accreditations": "Other Accreditations",
    "certificate_name_and_address": "Name & Address as it Should Appear on Certificate",
    "consultant_name": "Consultant Name (if any)",
    "designation": "Designation",
    "make_model": "Model/Type/Year of Make",
    "serial_number": "Serial Number",
    "range_and_accuracy": "Range and Accuracy",
    "calibration_date": "Date of Last Calibration",
    "calibration_due_date": "Calibration Due On",
    "calibrated_by": "Calibrated By",
    "ownership": "Owned / Long-term Lease",
    "associated_entity": "Associated Entity (Operational/Supporting)",
    "qualification": "Academic and Professional Qualifications",
    "relevant_experience_years": "Relevant Experience (years)",
    "trainings": "Relevant Training",
    "materials_or_products": "Materials or Products Tested",
    "parameter_or_measurand": "Component/Parameter/Measurand",
    "test_or_calibration_method": "Test/Calibration Method",
    "range": "Range",
    "frequency": "Frequency",
    "measurement_uncertainty": "Measurement Uncertainty (MU)",
    "cmc": "Calibration and Measurement Capability (CMC)",
    "facility_type": "Type of Facility",
    "record_type": "PT / ILC",
    "provider": "PT Provider / Nodal Laboratory",
    "provider_accreditation_body_or_country": "Accreditation Body / Country",
    "participation_date": "Date of Testing",
    "performance_metric": "Performance Metric",
    "performance_value": "Performance Value",
    "corrective_action_taken": "Corrective Action Taken",
    "department_or_section": "Laboratory/Department/Section",
    "qualification_with_specialization": "Qualification with Specialization",
    "authorized_area": "Authorized Area",
    "specimen_signature_on_file": "Specimen Signature on File",
    "source": "Source",
    "expiry_date": "Date of Expiry/Validity",
    "traceability": "Traceability",
    "aerb_registration_number": "AERB Registration No.",
    "radiological_safety_training": "Radiological Safety Training Details",
    "experience_years": "Experience (years)",
    "number_of_groups": "Number of Groups Applied For",
    "fee_amount": "Application Fees (Rs.)",
    "last_internal_audit_date": "Date of Last Internal Audit",
    "full_standard_coverage_last_year": "All Requirements Audited in Last Year?",
    "last_management_review_date": "Date of Last Management Review",
    "country": "Country",
    "state": "State/Province",
    "district": "District",
    "pincode": "Pincode",
    "mobile": "Mobile No.",
    "technical_head_or_lab_manager": "Technical Head/Lab Manager",
    "accredited_pt_program": "Accredited PT Program",
    "project_name": "Project Name",
    "project_size": "Project Size",
    "project_duration": "Project Duration",
    "project_reference": "Project Reference",
    "naco_ictc_laboratory": "NACO ICTC Laboratory?",
    "ever_applied_for_iso15189": "Ever Applied for ISO 15189 Accreditation?",
    "discipline": "Discipline",
    "discipline_group_subgroup": "Discipline / Group / Sub-group",
    "test_parameter": "Test Parameter",
    "measurement_technique": "Measurement Technique",
    "reference_material_or_crm": "Reference Material / CRM",
    "in_house_or_traceability": "In-house / Traceability",
    "external_lab_name": "External Laboratory Name",
    "calibration_certificate_uploaded": "Calibration Certificate Uploaded",
    "image_uploaded": "Image Uploaded",
    "date_of_report": "Date of Issue of PT Report",
    "report_uploaded": "Report Uploaded",
    "function": "Function",
    "product_name": "Name of the Product",
    "product_standard": "Product Standard",
    "shareholding_percent": "% of Shareholding",
    "relations_with_other_directors": "Relations with Other Directors/Shareholders",
    "remarks": "Remarks",
    "regulatory_bodies": "Regulatory Bodies",
    "last_recognition_certificate_number": "Last Recognition/Approval Certificate No.",
    "last_recognition_issuing_authority": "Issuing Authority",
    "last_recognition_validity": "Validity",
    "contact_for_accounts": "Contact Person for Accounts",
}


def _label(leaf: str) -> str:
    return _LABELS.get(leaf, leaf.replace("_", " ").title())


def _cell(value) -> str:
    if value in (None, "", []):
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def render_form_pdf(form_type: str, compiled_form: dict) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(f"NABL Application — {form_type}", styles["Title"]),
        Spacer(1, 0.4 * cm),
    ]

    for section, value in compiled_form.items():
        if isinstance(value, dict):
            story.append(Paragraph(section.replace("_", " ").title(), styles["Heading2"]))
            story.extend(_render_flat_section(value, styles))
            story.append(Spacer(1, 0.3 * cm))
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            story.append(Paragraph(section.replace("_", " ").title(), styles["Heading2"]))
            story.append(_render_table_section(value))
            story.append(Spacer(1, 0.3 * cm))
        elif value not in (None, "", []):
            story.append(Paragraph(f"<b>{_label(section)}:</b> {_cell(value)}", styles["Normal"]))

    doc.build(story)
    return buf.getvalue()


def _render_flat_section(value: dict, styles) -> list:
    rows = []
    for k, v in value.items():
        if isinstance(v, dict):
            rows.append(Paragraph(f"<b>{_label(k)}</b>", styles["Normal"]))
            for kk, vv in v.items():
                if vv not in (None, "", []):
                    rows.append(Paragraph(f"&nbsp;&nbsp;{_label(kk)}: {_cell(vv)}", styles["Normal"]))
            continue
        if v in (None, "", []):
            continue
        rows.append(Paragraph(f"<b>{_label(k)}:</b> {_cell(v)}", styles["Normal"]))
    return rows or [Paragraph("<i>No data</i>", styles["Normal"])]


def _render_table_section(records: list[dict]) -> Table:
    keys = list(records[0].keys())
    header = [_label(k) for k in keys]
    data = [header] + [[_cell(r.get(k)) for k in keys] for r in records]
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b5fd1")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table
