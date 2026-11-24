"""Deterministic PII patterns and their validators.

Everything here is pure: string in, verdict out. No I/O, no model, no state —
which is what makes the detection layer testable and reviewable by someone who
needs to sign off on what the tool will and will not remove.

A pattern alone is not enough for the numeric identifiers. "123456789012" is
the shape of an Aadhaar number and also the shape of a specimen barcode, and a
lab report is full of the latter. Where a checksum exists (Aadhaar's Verhoeff,
IFSC's structure) it is applied, because the cost of a false positive here is
a destroyed laboratory value.
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------
# Verhoeff — the checksum the UIDAI uses for Aadhaar numbers.
# Without it, any 12-digit run on the page is a candidate; with it, roughly
# 9 in 10 accidental 12-digit numbers are rejected.
# --------------------------------------------------------------------------
_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def verhoeff_valid(digits: str) -> bool:
    """True when `digits` carries a valid Verhoeff check digit."""
    if not digits.isdigit():
        return False
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def is_aadhaar(candidate: str, require_checksum: bool = True) -> bool:
    """Aadhaar numbers are 12 digits and never start with 0 or 1."""
    d = re.sub(r"\D", "", candidate)
    if len(d) != 12 or d[0] in "01":
        return False
    # An all-same-digit run (000000000000, 111111111111) is placeholder text on
    # a form, not somebody's identity number.
    if len(set(d)) == 1:
        return False
    return verhoeff_valid(d) if require_checksum else True


def is_pan(candidate: str) -> bool:
    """Permanent Account Number: AAAAA9999A.

    The 4th character encodes holder type; 'P' is an individual. Other letters
    are companies, trusts and firms, which are organisation data rather than
    personal data — but we treat all of them as PII, since an accountant's firm
    PAN on a lab report is still an identifier that does not belong there.
    """
    return bool(re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", candidate.strip().upper()))


def is_ifsc(candidate: str) -> bool:
    """Bank IFSC: 4 letters, then '0', then 6 alphanumerics."""
    return bool(re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", candidate.strip().upper()))


# --------------------------------------------------------------------------
# Shape patterns. Each is anchored with word boundaries so a match cannot
# begin mid-token; validation beyond shape happens in the functions above.
# --------------------------------------------------------------------------
PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    # Indian mobile numbers start 6-9. The optional +91/0 prefix is matched so
    # the country code is redacted with the number rather than left behind.
    "PHONE": re.compile(r"(?<![\d\-/])(?:\+?91[\-\s]?|0)?[6-9]\d{9}(?![\d\-/])"),
    "AADHAAR": re.compile(r"(?<!\d)\d{4}[\s\-]?\d{4}[\s\-]?\d{4}(?!\d)"),
    "PAN": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    "GSTIN": re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]\b"),
    "IFSC": re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
    "PASSPORT": re.compile(r"\b[A-PR-WY][0-9]{7}\b"),
    "VOTER_ID": re.compile(r"\b[A-Z]{3}[0-9]{7}\b"),
    "DRIVING_LICENCE": re.compile(r"\b[A-Z]{2}[\s\-]?\d{2}[\s\-]?(?:19|20)\d{2}[\s\-]?\d{7}\b"),
}

# Validators gate a shape match. A kind with no validator is accepted on shape.
VALIDATORS = {
    "PAN": is_pan,
    "IFSC": is_ifsc,
}

# Config flag name per detector kind, so the runner can switch each on or off.
KIND_TO_FLAG = {
    "EMAIL": "email", "PHONE": "phone", "AADHAAR": "aadhaar", "PAN": "pan",
    "GSTIN": "gstin", "IFSC": "ifsc", "PASSPORT": "passport",
    "VOTER_ID": "voter_id", "DRIVING_LICENCE": "driving_licence",
}

# --------------------------------------------------------------------------
# Context labels. These drive the detectors that cannot work on shape alone —
# a person's name looks like any other pair of capitalised words.
# --------------------------------------------------------------------------
PATIENT_NAME_LABELS = (
    "patient name", "patient's name", "patients name", "name of patient",
    "patient", "pt name", "pt. name", "client name", "name of the patient",
)
DOCTOR_NAME_LABELS = (
    "referred by", "referring physician", "referring doctor", "consultant",
    "ref by", "ref. by", "referrer", "doctor", "physician", "reffered by",
)
ADDRESS_LABELS = (
    "address", "residential address", "patient address", "addr",
    "correspondence address", "permanent address",
)
DOB_LABELS = (
    "dob", "d.o.b", "date of birth", "birth date", "birthdate",
)
PATIENT_ID_LABELS = (
    "uhid", "u.h.id", "mrn", "mr no", "mr number", "medical record",
    "patient id", "patient no", "patient number", "registration no",
    "registration number", "reg no", "reg. no", "accession no",
    "accession number", "abha", "abha id", "health id", "ip no", "op no",
)
AGE_SEX_LABELS = ("age", "age/sex", "age / sex", "sex", "gender")

# A bare "name" label is ambiguous: "Test Name", "Method Name" and "Sample
# Name" are all laboratory content, not identity. When the token preceding the
# label is one of these, the label is not a patient-name label.
NAME_LABEL_NEGATIVE_PREFIXES = (
    "test", "tests", "method", "sample", "product", "lab", "laboratory",
    "investigation", "parameter", "analyte", "panel", "profile", "department",
    "file", "report", "hospital", "centre", "center", "clinic", "company",
    "instrument", "equipment", "brand", "user", "login",
)

# Value tokens that mean "this field was left blank" — redacting them adds no
# privacy and loses the information that the field was empty.
NULL_VALUES = frozenset({
    "-", "--", "n/a", "na", "nil", "none", "not applicable", "not available",
    "unknown", "self", "...", ".", "_", "__", "___", "xxx", "xx",
})


def looks_like_null(text: str) -> bool:
    return text.strip().lower() in NULL_VALUES


def normalise_label(text: str) -> str:
    """Lower-case, strip trailing punctuation and collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(":.- \t")
