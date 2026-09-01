"""Best-effort classification of a canonical key's role in an eventual
labeled document (Step 5 consumes this): a flat document-level attribute
("document_field"), a generic table-column header shared by many different
tests ("table_column"), or a specific measured parameter/test name that
becomes a row in that table ("parameter"). Heuristic and reviewable, same as
everything else in this stage — not a claim of ground truth.
"""
from __future__ import annotations

TABLE_COLUMN_KEYS = {
    "test_name", "parameter", "test_parameter", "parameters", "analyte",
    "result", "results", "results_of_analysis", "observed_value",
    "observed_values", "value", "reading", "unit", "units",
    "reference_range", "reference_interval", "reference_value",
    "normal_range", "biological_reference_interval", "bio_ref_interval",
    "permissible_limit", "limit", "limits", "specification", "standard",
    "method", "methods", "remark", "remarks",
}

# Any of these tokens appearing in a canonical key's snake_case tokens marks
# it as document-level metadata rather than a measured parameter.
_DOCUMENT_FIELD_HINT_TOKENS = {
    "name", "id", "no", "number", "date", "address", "type", "gender", "sex",
    "age", "batch", "product", "applicant", "organisation", "organization",
    "phone", "fax", "email", "report", "lab", "registration", "license",
    "signature", "doctor", "physician", "hospital", "client", "customer",
    "sample", "collected", "received", "reported", "registered", "printed",
    "released", "authorized", "authorised", "verified", "code", "tel",
    "cin", "gstin", "location", "discipline", "group", "page", "centre",
    "center", "department", "contact", "person", "state", "country", "city",
    "postal", "website", "reference", "po", "issue", "expiry",
    "manufacturing", "mfg", "exp", "specimen", "accession", "uhid", "corp",
    "format",
}


def classify_field_role(canonical_key: str) -> str:
    if canonical_key in TABLE_COLUMN_KEYS:
        return "table_column"
    if set(canonical_key.split("_")) & _DOCUMENT_FIELD_HINT_TOKENS:
        return "document_field"
    return "parameter"
