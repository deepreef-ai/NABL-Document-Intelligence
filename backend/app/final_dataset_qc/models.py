from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FinalDatasetQCReport:
    total_documents: int = 0
    valid_documents: int = 0
    invalid_documents: int = 0
    duplicate_documents: int = 0
    invalid_annotations: int = 0
    unexpected_keys: int = 0
    missing_values: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)
