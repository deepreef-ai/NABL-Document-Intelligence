"""Per-document LLM call metering and hard caps.

Before this existed nothing counted Nova calls and nothing capped them:
MEASURED 2026-09-03, a 1-page certificate cost 3 (classify + schema +
unified) and a long completed_application_form cost 11-21 (one call per
schema section, plus one retry per section, plus an unconditional unified
pass). Cost was invisible because Bedrock's `usage` block was discarded too.

Design notes:

- `spend()` RETURNS FALSE when exhausted rather than raising. A budget
  ceiling is an expected operating condition, not an error: the pipeline
  must return the partial result it already has (with a reason recorded),
  exactly like it already does when one schema section's provider is
  rate-limited. Raising would turn "we stopped early" into "the upload
  failed".
- One instance per document, passed explicitly. No module-level counter:
  the upload endpoint runs requests concurrently in a thread pool (see
  routers/documents.py), so shared mutable state would attribute one
  document's calls to another.
- Categories are metered separately AND against a shared total, so a
  document cannot spend its recovery allowance on extraction calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.config import get_settings

CLASSIFICATION = "classification"
EXTRACTION = "extraction"
RECOVERY = "recovery"
VISION = "vision"

_CATEGORIES = (CLASSIFICATION, EXTRACTION, RECOVERY, VISION)


@dataclass
class TokenUsage:
    """Real token counts, when the provider reports them (Bedrock's Converse
    API does; see llm/providers.py's NovaProvider). Zero means "not
    reported", not "free" — don't present it as a measured cost."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens


@dataclass
class CallBudget:
    """Meters one document's LLM calls against configured caps."""

    document_id: str = "unknown"
    max_classification: int = 0
    max_extraction: int = 0
    max_recovery: int = 0
    max_total: int = 0
    counts: dict[str, int] = field(default_factory=lambda: {c: 0 for c in _CATEGORIES})
    usage: TokenUsage = field(default_factory=TokenUsage)
    # Why the pipeline stopped asking for more calls, if it did. Surfaced to
    # the caller so "we stopped at the ceiling" is never mistaken for "the
    # document genuinely had nothing more in it".
    stop_reason: str | None = None

    @classmethod
    def from_settings(cls, document_id: str = "unknown") -> CallBudget:
        s = get_settings()
        return cls(
            document_id=document_id,
            max_classification=s.max_classification_calls,
            max_extraction=s.max_initial_extraction_calls,
            max_recovery=s.max_recovery_calls,
            max_total=s.max_total_llm_calls,
        )

    @property
    def total_calls(self) -> int:
        return sum(self.counts.values())

    def _cap_for(self, category: str) -> int | None:
        # Vision calls are deliberately capped by the recovery allowance
        # rather than getting their own budget line: a vision call IS a
        # recovery attempt (see documents/recovery.py's escalation policy),
        # and giving it a separate pool would let one document quietly spend
        # double what the total suggests.
        return {
            CLASSIFICATION: self.max_classification,
            EXTRACTION: self.max_extraction,
            RECOVERY: self.max_recovery,
            VISION: self.max_recovery,
        }.get(category)

    def can_spend(self, category: str, count: int = 1) -> bool:
        if category not in _CATEGORIES:
            raise ValueError(f"unknown call category: {category!r}")
        if self.total_calls + count > self.max_total:
            return False
        cap = self._cap_for(category)
        if cap is None:
            return True
        spent = self.counts[category]
        if category in (RECOVERY, VISION):
            # Shared allowance — count them together against max_recovery.
            spent = self.counts[RECOVERY] + self.counts[VISION]
        return spent + count <= cap

    def spend(self, category: str, count: int = 1) -> bool:
        """Reserve `count` calls. Returns False (and records why) when the
        budget cannot cover them; the caller must then return what it has."""
        if not self.can_spend(category, count):
            if self.stop_reason is None:
                if self.total_calls + count > self.max_total:
                    self.stop_reason = (
                        f"total LLM call budget reached ({self.total_calls}/{self.max_total})"
                    )
                else:
                    self.stop_reason = (
                        f"{category} call budget reached "
                        f"({self.counts[category]}/{self._cap_for(category)})"
                    )
            return False
        self.counts[category] += count
        return True

    def record_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.usage.add(input_tokens, output_tokens)

    def as_dict(self) -> dict:
        """The per-document call log."""
        return {
            "document_id": self.document_id,
            "classification_calls": self.counts[CLASSIFICATION],
            "extraction_calls": self.counts[EXTRACTION],
            "recovery_calls": self.counts[RECOVERY],
            "vision_calls": self.counts[VISION],
            "total_llm_calls": self.total_calls,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "total_tokens": self.usage.total_tokens,
            "stop_reason": self.stop_reason,
        }
