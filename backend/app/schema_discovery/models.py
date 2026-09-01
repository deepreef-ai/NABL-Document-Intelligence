"""Output structures for the schema-discovery stage: one DocumentKeys per
sampled document (this stage's per-document deliverable, matching the
{document_id, candidate_domain, keys} shape this stage was asked for), plus
a DiscoveryIndexEntry per sampled document (this run's audit trail, mirroring
dataset_normalization/models.py's IndexEntry) that pipeline.py appends to
discovery_index.jsonl.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DocumentKeys:
    document_id: str
    candidate_domain: str
    keys: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiscoveryIndexEntry:
    document_id: str
    original_filename: str
    heuristic_domain: str  # sampling-only bucket, see domains.py — not authoritative
    candidate_domain: str | None  # the LLM's own answer; None if the call failed
    status: str  # "processed" | "skipped" | "failed"
    key_count: int
    error: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)
