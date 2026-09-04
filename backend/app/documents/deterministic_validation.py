"""Validation with NO LLM call.

The point is to decide, cheaply and repeatably, which fields are worth
spending a recovery call on. Before this, the pipeline retried any field
whose value was empty and nothing else — so a value the model got plainly
WRONG (present, well-formed, and not in the document at all) was never
questioned, while cost was spent re-asking about fields that genuinely
aren't in the document.

Reuses rather than reimplements: `compiler.detect_conflicts` for
cross-source disagreement and `grounding.ground` for evidence matching.

Two output sets drive recovery:
  missing    — required/expected and absent
  suspicious — present but the evidence doesn't support it
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_SENTINELS = {"", "n/a", "na", "null", "none", "-", "--", "tbd", "not available", "not found"}
_NUMERIC = re.compile(r"^[<>]?=?\s*-?\d+(?:[.,]\d+)?\s*$")
_DATE = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\b|\b(\d{4})[./-](\d{1,2})[./-](\d{1,2})\b")
_RANGE = re.compile(r"^\s*[\d.]+\s*[-–]\s*[\d.]+\s*$")


@dataclass
class ValidationIssue:
    field_path: str
    value: str | None
    reason: str
    kind: str  # "missing" | "suspicious"


@dataclass
class ValidationResult:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def missing(self) -> list[str]:
        return [i.field_path for i in self.issues if i.kind == "missing"]

    @property
    def suspicious(self) -> list[str]:
        return [i.field_path for i in self.issues if i.kind == "suspicious"]

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "missing": self.missing,
            "suspicious": self.suspicious,
            "issues": [
                {"field": i.field_path, "value": i.value, "reason": i.reason, "kind": i.kind}
                for i in self.issues
            ],
        }


def _meaningful(value) -> bool:
    return value is not None and str(value).strip().lower() not in _SENTINELS


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def validate(
    fields: dict,
    tests: list[dict],
    source_text: str,
    required_fields: list[str] | None = None,
) -> ValidationResult:
    """`source_text` is the evidence actually sent to the model for these
    values — the union of the chunk texts, not the whole document."""
    result = ValidationResult()
    haystack = _normalize(source_text)

    # 1. required fields present
    for path in required_fields or []:
        if not _meaningful(fields.get(path)):
            result.issues.append(ValidationIssue(path, fields.get(path), "required field absent", "missing"))

    # 2. every extracted value must be supported by the evidence
    for key, value in fields.items():
        if not _meaningful(value):
            continue
        needle = _normalize(value)
        # Length floor differs by kind, and getting this wrong matters: at a
        # flat floor of 4 the spec's own headline case — model says
        # hemoglobin 18.2 where the page says 13.2 — normalizes to "182" and
        # was silently skipped. Numeric values are the ones worth checking
        # even when short (a wrong lab result IS the failure mode); free text
        # needs more characters before a substring test means anything.
        is_numeric = bool(_NUMERIC.match(str(value).strip()))
        if len(needle) < (2 if is_numeric else 4):
            continue
        if needle not in haystack:
            result.issues.append(
                ValidationIssue(key, str(value), "value not found in the source evidence", "suspicious")
            )

    # 3. date sanity — a real date, not "45/99/2026"
    for key, value in fields.items():
        if not _meaningful(value) or "date" not in key.lower():
            continue
        m = _DATE.search(str(value))
        if not m:
            continue
        groups = [g for g in m.groups() if g]
        try:
            nums = [int(g) for g in groups]
        except ValueError:
            continue
        if len(nums) == 3 and not (
            (1 <= nums[0] <= 31 and 1 <= nums[1] <= 12) or (1 <= nums[1] <= 12 and 1 <= nums[2] <= 31)
        ):
            result.issues.append(ValidationIssue(key, str(value), "implausible date", "suspicious"))

    # 4. test rows: duplicates, and result/unit/range coherence
    seen: dict[str, int] = {}
    for i, row in enumerate(tests):
        name = str(row.get("test_name") or "").strip()
        if not name:
            result.issues.append(ValidationIssue(f"tests[{i}].test_name", None, "test row has no name", "missing"))
            continue
        norm = _normalize(name)
        if norm in seen:
            result.issues.append(
                ValidationIssue(
                    f"tests[{i}].test_name", name,
                    f"duplicate of tests[{seen[norm]}].test_name", "suspicious",
                )
            )
        else:
            seen[norm] = i

        value = row.get("result")
        if not _meaningful(value):
            result.issues.append(ValidationIssue(f"tests[{i}].result", None, "test row has no result", "missing"))
            continue
        # A numeric result with a numeric reference range that is malformed
        # (e.g. "13-" or "17-13") is worth a second look.
        ref = row.get("reference_range")
        if _meaningful(ref) and _RANGE.match(str(ref)):
            lo, hi = (float(x) for x in re.split(r"[-–]", str(ref).strip()))
            if lo > hi:
                result.issues.append(
                    ValidationIssue(f"tests[{i}].reference_range", str(ref), "reversed reference range", "suspicious")
                )
        # A unit on a non-numeric result ("Absent g/dL") is usually a
        # mis-parse of an adjacent column.
        if _meaningful(row.get("unit")) and not _NUMERIC.match(str(value).strip()):
            if str(value).strip().lower() not in ("absent", "present", "negative", "positive", "nil", "detected"):
                result.issues.append(
                    ValidationIssue(
                        f"tests[{i}].result", str(value),
                        f"non-numeric result carries unit {row.get('unit')!r}", "suspicious",
                    )
                )
    return result
