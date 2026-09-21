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
        # Result-table rows live on Document.tests_json, NOT among doc.fields,
        # so the loop below could never see them and the compiled form came
        # back with no table at all on documents whose table WAS read
        # correctly. Carried here before the `if not fields: continue` guard
        # on purpose: a pure results document (a lab report with a big table
        # and almost no header metadata) can have zero schema-mapped fields
        # and still be the only thing in the application worth showing.
        _collect_test_rows(form, doc)

        fields = [f for f in doc.fields if not accepted_only or f.accepted]
        if not fields:
            continue
        all_fields_seen.extend(fields)

        # Open-ended fields (documents/extractor.py's extract_open_fields*)
        # never had a known schema slot to begin with — route them straight
        # into extra_fields by their own invented name, bypassing
        # LIST_TARGET/FLAT_MERGE_TARGETS entirely so they can never collide
        # with (overwrite) a real schema attribute via _merge_flat_fields'
        # hasattr/setattr below. A document whose only fields are open ones
        # (e.g. doc_type "other") still contributes them here even though it
        # then hits the same no-op dispatch below as it always has.
        open_fields = [f for f in fields if getattr(f, "source", "llm") == "open_extraction"]
        for f in open_fields:
            if _is_meaningful(f.value):
                form.extra_fields[f.field_path] = f.value
        fields = [f for f in fields if f not in open_fields]
        if not fields:
            continue

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


# A unit glued onto the end of a reference range ("<= 1 mg/dL", "3.50-5.00
# g/dl") or of the result itself ("3.5 %"). Used only to fill a unit the
# extractor left empty — never to overwrite one it found.
_TRAILING_UNIT = re.compile(
    r"(?:^|[\s\d\)\]])("
    r"(?:mg|g|kg|ug|µg|mcg|ng|pg|IU|mIU|mmol|umol|µmol|nmol|mEq|cfu|CFU|ppm|ppb|"
    r"kcal|cal|mL|ml|L|dL|dl|cm|mm|sec|min)"
    r"(?:\s*/\s*(?:d?[lL]|m?[lL]|g|kg|mg|hr|h|day|m3|mm3|uL|µL|cumm|cu\s*mm))?"
    r"|%"
    r")\s*$"
)

_TEST_ROW_COLUMNS = ("test_name", "result", "unit", "reference_range", "method",
                     "flag", "specimen", "section", "panel_name", "s_no",
                     "page_number", "sample_date")


def recover_unit(row: dict) -> str | None:
    """The unit for a row whose `unit` column came back empty, when the unit
    is visible elsewhere in the SAME row.

    Deliberately narrow. MEASURED across 1,267 real extracted rows: 414 have
    no unit, and 356 of those are qualitative results (NEG / Absent / A2A2 /
    pH) that correctly have none. Only rows carrying the unit in their own
    reference range or result text can be filled without inventing anything,
    so that is all this does — it does not guess a unit from the test name,
    which would be fabrication dressed up as recovery.

    The bigger unit gap is a whole table column being missed at extraction
    time (19 documents came back with 0% units), and no post-processing can
    fix that — the data is not in the row to recover. See the extraction
    prompt, not here."""
    if row.get("unit") and str(row["unit"]).strip():
        return None
    for source in ("reference_range", "result"):
        text = str(row.get(source) or "").strip()
        if not text:
            continue
        match = _TRAILING_UNIT.search(text)
        if match:
            return match.group(1).strip()
    return None


def split_chart_series(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Separate points read off a trend CHART from real results-table rows.

    Returns (table_rows, chart_rows).

    A results table states each analyte ONCE and gives it a reference range.
    A chart series repeats one analyte many times, each point carrying its
    own date and no range — so the rule is: three or more rows sharing a test
    name, none with a reference range, every one with a DISTINCT date.

    All three conditions are required, and the counter-examples are why.
    MEASURED across 1,267 real rows: this moves 33 (2.6%), every one of them
    the HbA1c scatter chart in 001_Lab-report.png. It correctly leaves alone
    'uric acid' repeated 3x in AH3500627 (it has ranges, no dates) and a test
    report number repeated 13x in certificate.pdf (no ranges, but only 2
    distinct dates, so not one-point-per-date).

    Deliberately conservative: misfiling a real table row as a chart point
    would hide a genuine result, which is far worse than leaving a chart
    point in the table.
    """
    by_name: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_name[str(row.get("test_name") or "").strip().lower()].append(row)

    chart_ids: set[int] = set()
    for name, members in by_name.items():
        if not name or len(members) < 3:
            continue
        if any(str(m.get("reference_range") or "").strip() for m in members):
            continue
        dates = [str(m.get("sample_date") or "").strip() for m in members]
        if not all(dates) or len(set(dates)) != len(members):
            continue
        chart_ids.update(id(m) for m in members)

    table_rows = [r for r in rows if id(r) not in chart_ids]
    chart_rows = [r for r in rows if id(r) in chart_ids]
    return table_rows, chart_rows


def _collect_test_rows(form, doc) -> None:
    """Append one document's result-table rows to form.test_results, and any
    trend-chart points to form.chart_readings instead.

    Tagged with the source document so a multi-report application does not
    silently merge two labs' tables into one list a reviewer cannot tell
    apart."""
    # getattr, not attribute access: callers pass anything with `fields` and
    # `doc_type` — the tests use SimpleNamespace stand-ins, and a document
    # that predates tests_json has None. Neither is an error, and neither
    # should take down compiling the rest of the form.
    rows = getattr(doc, "tests_json", None) or []
    if not isinstance(rows, list) or not hasattr(form, "test_results"):
        return
    rows = [r for r in rows if isinstance(r, dict)]
    table_rows, chart_rows = split_chart_series(rows)
    name = getattr(doc, "filename", None)

    def _carry(row: dict) -> dict | None:
        carried = {k: v for k, v in row.items() if k in _TEST_ROW_COLUMNS and _is_meaningful(v)}
        if not carried:
            return None
        recovered = recover_unit(row)
        if recovered:
            carried["unit"] = recovered
            carried["unit_source"] = "derived_from_reference_range"
        if name:
            carried["source_document"] = name
        return carried

    for row in table_rows:
        carried = _carry(row)
        if carried is not None:
            form.test_results.append(carried)

    if not hasattr(form, "chart_readings"):
        return
    for row in chart_rows:
        carried = _carry(row)
        if carried is None:
            continue
        # Named so a reader can see at a glance why this point is not in the
        # results table: it is a plotted historical reading, not this
        # report's own measurement.
        carried["reading_source"] = "trend_chart"
        carried["series_name"] = row.get("test_name")
        form.chart_readings.append(carried)


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
