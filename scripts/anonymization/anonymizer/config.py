"""Configuration for the lab-report anonymiser.

Every threshold, pattern toggle and output choice lives here rather than being
scattered through the code, because the correct settings are site-specific: a
lab that prints Aadhaar numbers on reports needs different handling from one
that prints only an internal UHID, and neither should require a code change.

Defaults are deliberately conservative. Where a choice trades recall (catching
more PII) against precision (not destroying lab data), the default favours
catching the PII, because a missed identifier is a disclosure while an
over-redacted field is a re-run.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Files we will attempt to process. Anything else in the tree is ignored
# rather than copied, so the output directory never becomes a partial mirror
# of unrelated content.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
PDF_SUFFIXES = (".pdf",)


@dataclass
class DetectionConfig:
    """Which detectors run, and how aggressive each is."""

    # Self-validating identifiers — matched by shape AND a checksum or a
    # structural rule, so they carry high confidence with no context needed.
    aadhaar: bool = True
    pan: bool = True
    email: bool = True
    phone: bool = True
    passport: bool = True
    voter_id: bool = True
    driving_licence: bool = True
    ifsc: bool = True
    gstin: bool = False  # a LAB identifier, not a patient one — off by default

    # Context-driven detectors — a label such as "Patient Name:" followed by a
    # value. Higher recall, and the main source of false positives, which is
    # why the protected-term list in protect.py gates them.
    patient_name: bool = True
    doctor_name: bool = True
    address: bool = True
    date_of_birth: bool = True
    patient_ids: bool = True  # UHID / MRN / registration / accession numbers
    age_sex: bool = False  # usually clinically relevant; off by default

    # Aadhaar shape (12 digits) collides with specimen and accession numbers.
    # Requiring the Verhoeff checksum removes almost all of that collision;
    # turning it off will redact more real Aadhaars and far more lab numbers.
    aadhaar_require_checksum: bool = True

    # How many tokens after a label may be swallowed as its value. Names are
    # short; addresses run long. Capping this stops a missing line-break from
    # redacting the rest of the page.
    max_value_tokens_name: int = 5
    max_value_tokens_address: int = 25


@dataclass
class RedactionConfig:
    """How a detected span is removed from the page."""

    # black_box  — cover with a filled rectangle and delete the underlying text
    # label      — same removal, but stamp "[PATIENT_NAME]" in the gap
    # pseudonym  — same removal, but stamp a stable fake value
    mode: str = "label"

    box_colour: tuple[float, float, float] = (0.0, 0.0, 0.0)
    label_colour: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Detected boxes are grown slightly before redacting. OCR boxes in
    # particular sit tight against the glyphs, and a box flush to the ink can
    # leave a readable sliver of the first and last characters.
    padding_px: float = 1.5

    # Pages with no usable text layer are rasterised, painted and rebuilt as an
    # image. Higher DPI keeps the report legible; it also grows the output.
    raster_dpi: int = 200

    # Salt for pseudonym mode. Set this to a stable secret to make pseudonyms
    # consistent across runs; leave empty to get a per-run salt, which means
    # the same person gets a different pseudonym in each run.
    pseudonym_salt: str = ""


@dataclass
class OcrConfig:
    enabled: bool = True
    # A page with fewer than this many characters of extractable text is
    # treated as scanned. 20 is enough to clear stray page furniture ("Page 1
    # of 3") without mistaking a sparse cover page for a real text layer.
    min_page_text_chars: int = 20
    dpi: int = 200
    # OCR lines below this confidence are still redacted if they match a
    # detector, but they are additionally flagged in the report, because a
    # low-confidence read is also a low-confidence *miss*.
    low_confidence: float = 0.55

    # A page can have a perfectly good text layer AND carry PII inside an
    # embedded image — the letterhead case, where the lab's block of name,
    # address, phone and email is one raster. Reading only the text layer misses
    # it entirely. MEASURED on a real report: an email address sat in the
    # letterhead image of a page with 1,607 characters of extractable text, was
    # never detected, and was caught only by the verification pass.
    # When on, such pages are additionally OCR'd and any words falling outside
    # the text layer are folded into detection.
    augment_text_pages: bool = True
    # Only bother when images cover at least this fraction of the page, so a
    # small signature stamp or logo does not trigger a full OCR pass.
    image_area_threshold: float = 0.02


@dataclass
class VerificationConfig:
    """The post-anonymisation pass that decides OK vs FAILED_REVIEW."""

    enabled: bool = True
    # Re-read the output's text layer and re-run the detectors. Cheap, and it
    # catches the classic redaction failure: a black rectangle drawn over text
    # that is still selectable underneath.
    check_text_layer: bool = True
    # Re-OCR the rendered output. Expensive, and the only way to catch PII that
    # was never detected in the first place.
    check_ocr: bool = True
    # Only these detector kinds can fail a document. Context-driven detections
    # are noisy enough that failing on them would mark most documents for
    # review; the self-validating identifiers are the ones that must not
    # survive.
    failing_kinds: tuple[str, ...] = (
        "AADHAAR", "PAN", "EMAIL", "PHONE", "PASSPORT",
        "VOTER_ID", "DRIVING_LICENCE", "IFSC",
    )


@dataclass
class AnonymizerConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    redaction: RedactionConfig = field(default_factory=RedactionConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)

    # Reports record WHERE PII was found and what kind, never the value itself.
    # Turning this on writes a truncated sample ("Ra… (8 chars)") to help tune
    # rules — which makes the report itself sensitive. Off by default.
    include_pii_samples: bool = False

    extra_protected_terms: tuple[str, ...] = ()
    extra_patient_labels: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnonymizerConfig":
        def build(klass, key):
            section = data.get(key) or {}
            known = {f for f in klass.__dataclass_fields__}
            unknown = set(section) - known
            if unknown:
                raise ValueError(f"unknown setting(s) in '{key}': {', '.join(sorted(unknown))}")
            return klass(**section)

        top_known = {"detection", "redaction", "ocr", "verification",
                     "include_pii_samples", "extra_protected_terms", "extra_patient_labels"}
        unknown = set(data) - top_known
        if unknown:
            raise ValueError(f"unknown top-level setting(s): {', '.join(sorted(unknown))}")

        return cls(
            detection=build(DetectionConfig, "detection"),
            redaction=build(RedactionConfig, "redaction"),
            ocr=build(OcrConfig, "ocr"),
            verification=build(VerificationConfig, "verification"),
            include_pii_samples=bool(data.get("include_pii_samples", False)),
            extra_protected_terms=tuple(data.get("extra_protected_terms") or ()),
            extra_patient_labels=tuple(data.get("extra_patient_labels") or ()),
        )

    @classmethod
    def load(cls, path: str | Path | None) -> "AnonymizerConfig":
        """Load YAML or JSON. Returns defaults when path is None."""
        if path is None:
            return cls()
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"config file not found: {p}")
        raw = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            import yaml  # imported lazily so JSON-only users need no PyYAML

            data = yaml.safe_load(raw) or {}
        else:
            data = json.loads(raw or "{}")
        if not isinstance(data, dict):
            raise ValueError(f"config must be a mapping, got {type(data).__name__}")
        return cls.from_dict(data)


VALID_REDACTION_MODES = ("black_box", "label", "pseudonym")
