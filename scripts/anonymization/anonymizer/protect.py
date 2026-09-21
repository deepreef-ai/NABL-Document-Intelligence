"""The false-positive guard: laboratory content that must never be redacted.

This module exists because the expensive failure mode of an anonymiser is not
a missed identifier — verification catches those and marks the file for review
— but a redacted haemoglobin value, which is silent, permanent in the output
copy, and makes the anonymised corpus useless for the purpose it was created
for.

Every detection is offered to `is_protected()` before it is accepted. The rule
is deliberately one-directional: protection always wins over detection.
"""
from __future__ import annotations

import re

# Units, analytes and report furniture. Matched case-insensitively against the
# whole candidate value, so this does not suppress "Ramesh Kumar" merely
# because "kumar" is not in the list — it suppresses a value that IS one of
# these things.
_UNITS = {
    "mg/dl", "g/dl", "mg/l", "g/l", "ug/dl", "µg/dl", "mcg/dl", "ng/ml",
    "pg/ml", "iu/l", "u/l", "miu/l", "mmol/l", "umol/l", "µmol/l", "meq/l",
    "cells/cumm", "cells/cu.mm", "/cumm", "/cu.mm", "million/cumm", "lakhs/cumm",
    "thousand/cumm", "fl", "pg", "%", "ratio", "index", "sec", "seconds",
    "mm/hr", "mmhg", "kpa", "ppm", "ppb", "cfu/g", "cfu/ml", "kcal", "kcal/100g",
    "g/100g", "mg/100g", "mg/kg", "g/kg", "ml", "l", "mg", "g", "kg", "ug", "ng",
    "iu", "cm", "mm", "nm", "copies/ml", "titre", "titer",
}

_ANALYTES = {
    "haemoglobin", "hemoglobin", "hb", "rbc", "wbc", "platelet", "platelets",
    "hct", "pcv", "mcv", "mch", "mchc", "rdw", "esr", "neutrophils",
    "lymphocytes", "monocytes", "eosinophils", "basophils", "glucose",
    "creatinine", "urea", "bun", "uric acid", "sodium", "potassium",
    "chloride", "calcium", "phosphorus", "magnesium", "bilirubin", "sgot",
    "sgpt", "ast", "alt", "alp", "ggt", "albumin", "globulin", "protein",
    "cholesterol", "triglycerides", "hdl", "ldl", "vldl", "tsh", "t3", "t4",
    "ft3", "ft4", "hba1c", "vitamin", "vitamin d", "vitamin b12", "ferritin",
    "iron", "tibc", "crp", "psa", "amylase", "lipase", "ldh", "ck", "ckmb",
    "troponin", "inr", "pt", "aptt", "fibrinogen", "moisture", "ash", "fat",
    "energy", "carbohydrate", "sodium chloride", "acidity", "ph", "density",
    "peroxide value", "acid value", "saponification value", "iodine value",
    "aflatoxin", "pesticide residue", "heavy metals", "lead", "arsenic",
    "mercury", "cadmium", "e. coli", "salmonella", "yeast", "mould", "mold",
    "total plate count", "coliform",
}

_REPORT_TERMS = {
    "test", "tests", "result", "results", "unit", "units", "method",
    "reference", "reference range", "biological reference interval",
    "normal range", "range", "specimen", "sample", "sample type",
    "investigation", "investigations", "parameter", "parameters", "analyte",
    "observation", "observed value", "department", "panel", "profile",
    "interpretation", "comments", "remarks", "note", "notes", "page",
    "report", "final report", "status", "verified", "approved", "authorised",
    "authorized", "signature", "end of report", "normal", "abnormal", "high",
    "low", "positive", "negative", "reactive", "non reactive", "detected",
    "not detected", "within normal limits", "nad", "absent", "present",
    "nil", "trace", "clear", "turbid", "pale yellow", "yellow", "serum",
    "plasma", "whole blood", "urine", "stool", "edta", "fluoride", "citrate",
    "accredited", "accreditation", "nabl", "iso", "protocol", "sop",
    "collected", "received", "reported", "registered", "conclusion",
}

PROTECTED_TERMS = _UNITS | _ANALYTES | _REPORT_TERMS

# A reference range: "12.5-16.0", "3.5 – 5.1", "< 200", "Up to 40", "0.4-4.0 mIU/L"
_RANGE = re.compile(
    r"^\s*(?:<|>|≤|≥|up\s*to|upto|less\s*than|greater\s*than)?\s*"
    r"\d+(?:\.\d+)?\s*(?:[-–—to]+\s*\d+(?:\.\d+)?)?\s*[A-Za-zµ%/().]*\s*$",
    re.IGNORECASE,
)

# A measurement: a number, optionally signed/decimal, optionally with a unit
# attached. "5.2", "12.4 g/dl", "1,200", "<0.01", "3.5x10^6"
_MEASUREMENT = re.compile(
    r"^\s*[<>≤≥]?\s*\d[\d,]*(?:\.\d+)?(?:\s*[xX×]\s*10\s*\^?\s*-?\d+)?"
    r"\s*[A-Za-zµ%/().\-]*\s*$"
)

# A date on its own. Dates are redacted only when a DOB label points at them;
# collection and report dates are laboratory metadata and must survive.
_DATE = re.compile(
    r"^\s*\d{1,2}[-/.][A-Za-z0-9]{1,9}[-/.]\d{2,4}(?:[\s,]+\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AaPp]\.?[Mm]\.?)?)?\s*$"
)


def looks_like_reference_range(text: str) -> bool:
    return bool(_RANGE.match(text)) and any(c.isdigit() for c in text)


def looks_like_measurement(text: str) -> bool:
    return bool(_MEASUREMENT.match(text)) and any(c.isdigit() for c in text)


def looks_like_date(text: str) -> bool:
    return bool(_DATE.match(text))


class ProtectedVocabulary:
    """Holds the protected set, plus any site-specific additions from config."""

    def __init__(self, extra_terms: tuple[str, ...] = ()):
        self.terms = {t.strip().lower() for t in PROTECTED_TERMS}
        self.terms |= {t.strip().lower() for t in extra_terms if t.strip()}

    def is_protected_term(self, text: str) -> bool:
        """True when the WHOLE value is laboratory vocabulary."""
        cleaned = re.sub(r"[\s:;,.]+$", "", text.strip().lower())
        if not cleaned:
            return True  # nothing to redact
        if cleaned in self.terms:
            return True
        # "Haemoglobin (Hb)" / "Glucose - Fasting": strip qualifiers and retry.
        base = re.split(r"[(\-–—/]", cleaned)[0].strip()
        return bool(base) and base in self.terms

    def is_protected(self, kind: str, text: str) -> tuple[bool, str]:
        """Should this detection be discarded? Returns (protected, reason)."""
        value = text.strip()
        if not value:
            return True, "empty"
        if self.is_protected_term(value):
            return True, "laboratory vocabulary"
        if looks_like_reference_range(value) and kind not in ("PHONE", "AADHAAR"):
            return True, "reference range"

        # Numeric identifiers are the dangerous overlap with lab values.
        # AADHAAR and PHONE survive this check because they have already
        # cleared a checksum or a strict shape rule.
        if kind in ("PATIENT_ID", "VOTER_ID", "PASSPORT") and looks_like_measurement(value):
            return True, "looks like a measured value"

        # A context-detected name that is really a date or a number is a label
        # that ran past its value into the next column.
        if kind in ("PATIENT_NAME", "DOCTOR_NAME"):
            if looks_like_date(value) or looks_like_measurement(value):
                return True, "value is a date or measurement, not a name"
            if not any(c.isalpha() for c in value):
                return True, "no letters in a name field"
        return False, ""
