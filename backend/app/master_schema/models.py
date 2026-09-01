"""Output structure for the key-mapping file (Step 4, requirement 4's
explicit deliverable): one entry per canonical-key cluster, exactly the
shape requested — canonical_key/aliases/domain/status — plus field_role and
total_frequency, which Step 5 and a human reviewer both need.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class KeyMappingEntry:
    canonical_key: str
    aliases: list[str]
    domain: str
    status: str  # "approved" | "review"
    field_role: str  # "document_field" | "table_column" | "parameter"
    total_frequency: int

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)
