from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class FieldFailure:
    document_id: str
    key: str
    ground_truth: Any
    prediction: Any
    error_type: str  # "missing" | "wrong_value" | "wrong_key" | "extra" | "wrong_unit"

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)
