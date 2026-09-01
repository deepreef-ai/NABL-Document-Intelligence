"""Builds the master schema (Step 4) from Step 3's domain_keys.json:
per-domain canonical keys (after duplicate/synonym clustering), common keys
shared across domains, domain-specific keys, and a field_role for each
canonical key so Step 5 knows whether it belongs in a document's flat
`fields` or as a column/row of its `tests` table.

Writes two files:
  - master_schema.json — the schema itself, grouped by domain (this stage's
    main deliverable, used for annotation)
  - key_mapping.json   — one entry per canonical-key cluster, with its
    aliases and "approved"/"review" status (requirement 4's deliverable)

Makes no LLM calls — purely deterministic clustering over keys Step 3
already discovered.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.master_schema.clustering import cluster_domain_keys
from app.master_schema.field_roles import classify_field_role
from app.master_schema.models import KeyMappingEntry
from app.schema_discovery.domains import CANONICAL_DOMAINS


def build_master_schema(schema_discovery_dir: Path) -> tuple[dict, list[KeyMappingEntry]]:
    domain_keys_path = schema_discovery_dir / "domain_keys.json"
    domain_keys = json.loads(domain_keys_path.read_text(encoding="utf-8")) if domain_keys_path.exists() else {}

    mapping_entries: list[KeyMappingEntry] = []
    domain_canonical_keys: dict[str, set[str]] = {}
    domains_output: dict[str, dict] = {}

    for domain in CANONICAL_DOMAINS:
        key_frequency = domain_keys.get(domain, {}).get("key_frequency", {})
        clusters = cluster_domain_keys(key_frequency)
        canonical_keys = [c.canonical_key for c in clusters]
        domain_canonical_keys[domain] = set(canonical_keys)

        key_details = []
        for cluster in clusters:
            role = classify_field_role(cluster.canonical_key)
            mapping_entries.append(KeyMappingEntry(
                canonical_key=cluster.canonical_key, aliases=cluster.aliases,
                domain=domain, status=cluster.status, field_role=role,
                total_frequency=cluster.total_frequency,
            ))
            key_details.append({
                "canonical_key": cluster.canonical_key, "aliases": cluster.aliases,
                "status": cluster.status, "field_role": role,
                "total_frequency": cluster.total_frequency,
            })

        domains_output[domain] = {
            "sample_count": domain_keys.get(domain, {}).get("sample_count", 0),
            "keys": sorted(canonical_keys),
            "key_details": sorted(key_details, key=lambda d: (-d["total_frequency"], d["canonical_key"])),
        }

    # Common vs domain-specific: exact canonical-string match across domains.
    # A near-miss like "unit" (settled on in one domain) vs "units" (settled
    # on in another) that wasn't already unified WITHIN a domain stays
    # visible as domain-specific rather than being guessed at across
    # domains — see clustering.py's docstring on preferring under-merging to
    # a wrong automatic merge.
    domain_counts: dict[str, int] = {}
    for keys in domain_canonical_keys.values():
        for key in keys:
            domain_counts[key] = domain_counts.get(key, 0) + 1

    common_keys = sorted(key for key, count in domain_counts.items() if count > 1)
    common_keys_by_domain = {
        key: sorted(domain for domain, keys in domain_canonical_keys.items() if key in keys)
        for key in common_keys
    }
    domain_specific_keys = {
        domain: sorted(key for key in keys if domain_counts[key] == 1)
        for domain, keys in domain_canonical_keys.items()
    }

    master_schema = {
        "domains": domains_output,
        "common_keys": common_keys,
        "common_keys_by_domain": common_keys_by_domain,
        "domain_specific_keys": domain_specific_keys,
    }
    return master_schema, mapping_entries


def write_master_schema(schema_discovery_dir: Path, output_dir: Path) -> tuple[dict, list[KeyMappingEntry]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    master_schema, mapping_entries = build_master_schema(schema_discovery_dir)

    (output_dir / "master_schema.json").write_text(
        json.dumps(master_schema, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "key_mapping.json").write_text(
        json.dumps([entry.to_json_dict() for entry in mapping_entries], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return master_schema, mapping_entries


def print_summary(master_schema: dict, mapping_entries: list[KeyMappingEntry]) -> None:
    print()
    print("=== Master Schema Summary ===")
    for domain, info in master_schema["domains"].items():
        review_count = sum(1 for e in mapping_entries if e.domain == domain and e.status == "review")
        print(
            f"{domain:10s}: {len(info['keys']):4d} canonical keys "
            f"(from {info['sample_count']} sampled docs, {review_count} clusters need review)"
        )
    print(f"Common keys across >=2 domains: {len(master_schema['common_keys'])}")
    approved = sum(1 for e in mapping_entries if e.status == "approved")
    review = sum(1 for e in mapping_entries if e.status == "review")
    print(f"Total key-mapping entries: {len(mapping_entries)} ({approved} approved, {review} review)")
