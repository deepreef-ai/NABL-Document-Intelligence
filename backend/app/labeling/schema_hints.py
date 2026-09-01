"""Builds compact per-domain hint lists from Step 4's master_schema.json for
the labeling prompt — a small, representative sample of each domain's known
fields/parameters, not its full (hundreds-strong, long-tail-heavy) canonical
key list. The prompt tells the LLM to use these names when they match, but
to still report any other clearly-labeled field/parameter actually present —
the hints steer naming consistency, they don't gate what can be extracted.
"""
from __future__ import annotations

import json
from pathlib import Path

MAX_DOCUMENT_FIELDS_HINT = 15
MAX_PARAMETERS_HINT = 20


def load_master_schema(master_schema_dir: Path) -> dict:
    path = master_schema_dir / "master_schema.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def build_domain_hints(master_schema: dict) -> dict[str, dict[str, list[str]]]:
    """{domain: {"fields": [top document-level keys], "parameters": [top
    measured-parameter keys]}} — key_details is already sorted by
    -total_frequency (see master_schema/pipeline.py), so taking a prefix
    keeps the most commonly-seen names."""
    hints: dict[str, dict[str, list[str]]] = {}
    for domain, info in master_schema.get("domains", {}).items():
        details = info.get("key_details", [])
        document_fields = [d["canonical_key"] for d in details if d.get("field_role") == "document_field"]
        parameters = [d["canonical_key"] for d in details if d.get("field_role") == "parameter"]
        hints[domain] = {
            "fields": document_fields[:MAX_DOCUMENT_FIELDS_HINT],
            "parameters": parameters[:MAX_PARAMETERS_HINT],
        }
    return hints


def format_hints_block(hints: dict[str, dict[str, list[str]]]) -> str:
    lines = []
    for domain, h in hints.items():
        fields_part = ", ".join(h["fields"]) or "(none known yet)"
        params_part = ", ".join(h["parameters"]) or "(none known yet)"
        lines.append(f"- {domain}: fields=[{fields_part}]; parameters=[{params_part}]")
    return "\n".join(lines)
