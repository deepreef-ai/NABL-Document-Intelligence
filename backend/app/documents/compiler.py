import re
from collections import defaultdict
from types import SimpleNamespace

from pydantic import BaseModel, ValidationError

from app.documents.extraction_report import ExtractionReport, FieldConflict, ValidationFailure
from app.models import Document
from app.schemas.forms import FORM_MODEL, NablFormType

# Sentinel strings a weaker free-tier model sometimes returns instead of a
# real null when it can't find a value — these are truthy in Python, so
# without this filter they'd get written into the form as visible garbage.
_SENTINEL_VALUES = {"null", "none", "n/a", "na", "not mentioned", "not available", "not found", "unknown", ""}


def _is_meaningful(value) -> bool:
    return bool(value) and str(value).strip().lower() not in _SENTINEL_VALUES

# Which compiled-form list a doc_type's fields append a record to.
LIST_TARGET = {
    "equipment_calibration_certificate": "equipment",
    "recognition_equipment_certificate": "equipment",
    "reference_material_certificate": "reference_materials",
    "recognition_reference_material_certificate": "reference_materials",
    "staff_cv_certificate": "staff",
    "authorized_signatory_declaration": "authorized_signatories",
    "pt_ilc_result": "pt_ilc",
    "recognition_pt_report": "pt_participation",
    "product_list_declaration": "products",
    "shareholder_declaration": "shareholders",
}

# doc_type -> the object its fields merge onto by matching each field_path's
# last dotted segment against an attribute name, rather than appending a new
# list record. `None` means "the form's own root object" (e.g. NABL 154's
# last_recognition_* fields live directly on Nabl154Form).
FLAT_MERGE_TARGETS: dict[str, str | None] = {
    "legal_proof": "organisation",  # 151/152/153/153A/154/158 — OrgLegalInfo
    "lab_contact_info": "lab_details",  # 155/157/159 — RecognitionLabDetails
    "project_reference_document": "lab_details",  # 159 only — project_* fields live on lab_details
    "regulatory_recognition_certificate": None,  # 154 only — merges onto the form root
}


def compile_form(
    form_type: str,
    documents: list[Document],
    accepted_only: bool = False,
    report: ExtractionReport | None = None,
) -> dict:
    """Builds the compiled form JSON (unchanged shape/behavior). Pass `report`
    (documents/extraction_report.py) to also collect conflicting values and
    Pydantic validation failures as a side effect — see
    routers/review.py's extraction-report endpoint; callers that don't need
    the report (e.g. the plain /form endpoint) are unaffected."""
    model_cls = FORM_MODEL[NablFormType(form_type)]
    form = model_cls()
    all_fields_seen = []

    for doc in documents:
        fields = [f for f in doc.fields if not accepted_only or f.accepted]
        if not fields:
            continue
        all_fields_seen.extend(fields)

        if doc.doc_type == "completed_application_form":
            _merge_full_form_fields(form, fields, report)
            continue

        if doc.doc_type in FLAT_MERGE_TARGETS:
            target_attr = FLAT_MERGE_TARGETS[doc.doc_type]
            target_obj = getattr(form, target_attr) if target_attr and hasattr(form, target_attr) else form
            _merge_flat_fields(target_obj, fields)
            continue

        target = LIST_TARGET.get(doc.doc_type)
        if not target or not hasattr(form, target):
            continue
        record, failures = _build_record(target, form, fields)
        if report is not None:
            report.validation_failures.extend(failures)
        if record is not None:
            getattr(form, target).append(record)

    if report is not None:
        report.conflicts.extend(detect_conflicts(all_fields_seen))

    return form.model_dump(mode="json")


def detect_conflicts(fields: list) -> list[FieldConflict]:
    """Same exact field_path (including any [index]) reported with more than
    one distinct non-null value — by rule-based extraction vs. the LLM, or by
    two different sources/pages disagreeing — is a conflict, not something to
    silently resolve by "first/last wins"."""
    by_path = defaultdict(list)
    for f in fields:
        if _is_meaningful(f.value):
            by_path[f.field_path].append(f)

    conflicts = []
    for path, entries in by_path.items():
        distinct_values = {e.value for e in entries}
        if len(distinct_values) > 1:
            conflicts.append(
                FieldConflict(
                    field_path=path,
                    values=[e.value for e in entries],
                    sources=[getattr(e, "source", "llm") for e in entries],
                )
            )
    return conflicts


def _merge_flat_fields(target_obj, fields) -> None:
    for f in fields:
        if not _is_meaningful(f.value):
            continue
        attr = f.field_path.rsplit(".", 1)[-1]
        if hasattr(target_obj, attr) and getattr(target_obj, attr) is None:
            setattr(target_obj, attr, f.value)


_INDEXED_PATH = re.compile(r"^(\w+)\[(\d+)\]\.(.+)$")


def _merge_full_form_fields(form, fields, report: ExtractionReport | None = None) -> None:
    """`completed_application_form` documents extract into MANY different
    sections/lists at once — "equipment[0].name", "organisation.gst_number",
    etc. — unlike every other doc_type here, which targets exactly one list
    or one flat object. Group indexed paths by (list attr, index) to build
    each record in order; flat-merge anything else onto its named nested
    object."""
    grouped: dict[tuple[str, int], list] = {}
    for f in fields:
        if not _is_meaningful(f.value):
            continue
        match = _INDEXED_PATH.match(f.field_path)
        if match:
            attr, index, sub_path = match.group(1), int(match.group(2)), match.group(3)
            if hasattr(form, attr):
                grouped.setdefault((attr, index), []).append(SimpleNamespace(field_path=sub_path, value=f.value))
            continue

        attr = f.field_path.split(".", 1)[0]
        if hasattr(form, attr) and isinstance(getattr(form, attr), BaseModel):
            _merge_flat_fields(getattr(form, attr), [f])

    for (attr, _index), sub_fields in sorted(grouped.items(), key=lambda item: item[0]):
        record, failures = _build_record(attr, form, sub_fields)
        if report is not None:
            report.validation_failures.extend(failures)
        if record is not None:
            getattr(form, attr).append(record)


def _build_record(target: str, form, fields) -> tuple[object | None, list[ValidationFailure]]:
    # Pydantic keeps the item type on model_fields; reach in via the class.
    item_cls = type(form).model_fields[target].annotation.__args__[0]
    values = {f.field_path: f.value for f in fields if _is_meaningful(f.value)}
    candidate = {k: v for k, v in values.items() if k in item_cls.model_fields}
    failures: list[ValidationFailure] = []

    # The LLM returns free text for every field, but some (dates, enums,
    # floats) are typed strictly on the record model — e.g. "15/03/2024"
    # fails pydantic's ISO-8601 date parser. One bad field used to take down
    # the whole compiled-form endpoint (a raised ValidationError here was
    # unhandled all the way up through review.py). Instead, drop only the
    # field(s) that actually fail to validate — recorded in `failures` for
    # the extraction report rather than silently discarded — and keep
    # retrying, so a record still gets built from whatever *did* come back
    # well-formed.
    while candidate:
        try:
            return item_cls(**candidate), failures
        except ValidationError as exc:
            bad_keys = {err["loc"][0] for err in exc.errors() if err["loc"]} & candidate.keys()
            if not bad_keys:
                return None, failures  # a required field is missing outright; no amount of pruning fixes that
            for key in bad_keys:
                reason = next(
                    (err["msg"] for err in exc.errors() if err["loc"] and err["loc"][0] == key), "invalid value"
                )
                failures.append(ValidationFailure(field_path=f"{target}.{key}", value=candidate.get(key), reason=reason))
                candidate.pop(key, None)
    return None, failures
