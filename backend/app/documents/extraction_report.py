"""Structured report alongside the compiled form: what was extracted and
from where, what's still missing, what conflicted across sources, and what
Pydantic rejected — kept separate from the compiled JSON so the existing
review UI / compile_form() output doesn't change shape."""
from dataclasses import asdict, dataclass, field


@dataclass
class FieldConflict:
    field_path: str
    values: list[str | None]
    sources: list[str]  # parallel to `values`, e.g. ["rule_based", "llm"]


@dataclass
class ValidationFailure:
    field_path: str
    value: str | None
    reason: str


@dataclass
class ExtractionReport:
    extracted_fields: list[dict] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    conflicts: list[FieldConflict] = field(default_factory=list)
    validation_failures: list[ValidationFailure] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "extracted_fields": self.extracted_fields,
            "missing_fields": self.missing_fields,
            "conflicts": [asdict(c) for c in self.conflicts],
            "validation_failures": [asdict(v) for v in self.validation_failures],
        }
